import numpy as np
import tensorflow as tf
import pickle

from keras.layers import Input, Dense, Lambda, Concatenate
from keras.models import Model
from keras.callbacks import *
from keras import optimizers
import keras.backend as K


class Function(object):
    def __init__(self):
        raise NotImplemented

    def __call__(self, x):
        raise NotImplemented


class Domain(object):
    '''
    Domain only supports
    (0) continuous hyper rectangles in R^d
    (1) discrete and finite set of items
    '''
    CONTINUOUS = 0
    DISCRETE = 1
    TYPES = [CONTINUOUS, DISCRETE]

    def __init__(self, space_type, domain):
        assert (space_type in self.TYPES)
        self.space_type = space_type

        # check validity of domain
        if space_type == self.CONTINUOUS:
            domain = np.array(domain)
            assert (domain.ndim == 2)
            assert ((domain[1] - domain[0] > 0).all())
        elif space_type == self.DISCRETE:
            assert (type(domain) is list)

        self.domain = domain


class DiscreteObjFcn(Function):
    def __init__(self, y_values):
        self.domain = Domain(space_type=1, domain=range(len(y_values)))
        self.y_values = y_values
        self.fg = None

    def __call__(self, x_idx):
        return self.y_values[x_idx]


class ContinuousObjFcn(Function):
    def __init__(self, domain):
        self.domain = Domain(space_type=0, domain=domain)
        self.fg = None

    def __call__(self, x):
        raise NotImplemented


def cosine_activation(x):
    return K.cos(x)


def SplitLayer_for_each_function(fcn_idx):
    # given a fcn idx, return the x values for the fcn
    def func(x):
        return x[:, fcn_idx, :]  # first dim is for n_data

    return Lambda(func)


def Squeeze(axis):
    def func(x):
        return K.squeeze(x, axis)

    return Lambda(func)


class FeatureFcn(Function):
    def __init__(self, dim_x, n_fcns):
        self.sess = tf.Session()
        self.dim_x = dim_x
        self.n_fcns = n_fcns
        self.make_network()
        self.feat_fcn.summary()

    def make_network(self):
        self.dense_num = 2048
        # input data is of the shape
        # (data_idx,fcn_idx,dim_x)

        x_input = Input(shape=(self.dim_x,), name='x', dtype='float32')
        phi = Dense(self.dense_num, activation=cosine_activation, use_bias=True)(x_input)
        last_layers = []
        for i in range(self.n_fcns):
            last_layers.append(Dense(1, activation='linear', name='W_' + str(i), use_bias=False)(phi))
        outputs = Concatenate(axis=-1)(last_layers)

        self.score_fcn = Model(input=[x_input], output=outputs)
        self.feat_fcn = Model(input=[x_input], output=phi)

        adam = optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.score_fcn.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])

    def train_feature_function(self, xvals, Dprior, dirsave):
        Ys = Dprior.transpose()  # make it n_arms by p_inst
        Xs = xvals

        filepath = dirsave + "/feat_fcn_fcn.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')
        callbacks_list = [earlystop]
        self.score_fcn.fit(Xs, Ys,
                           epochs=10000, batch_size=32,
                           callbacks=callbacks_list,
                           validation_split=0.1)
        self.feat_fcn.save_weights(filepath)

    def analytically_solve_for_Ws(self, Xs, Dprior, dirsave):
        i = 0
        Ws = []
        losses = []
        for Ys in Dprior:
            phiX = self.feat_fcn.predict(Xs)
            X_TX = np.dot(phiX.T, phiX)
            print "inverting..."
            print '%d/%d' % (i, Dprior.shape[0])
            i += 1
            W_i = np.dot(np.dot(np.linalg.inv(X_TX + 0.00001 * np.eye(self.dense_num, self.dense_num)), phiX.T), Ys)
            Ws.append(W_i)
            losses.append(np.mean(np.abs(np.dot(phiX, W_i) - Ys)))
        pickle.dump(Ws, open(dirsave + '/Ws.pkl', 'wb'))
        return Ws, losses

    def train(self, Xs, Dprior, dirsave):
        LOAD_TRAINED_WEIGHTS = True
        if LOAD_TRAINED_WEIGHTS:
            feat_fcn_path = dirsave + "/feat_fcn_fcn.best.hdf5"
            self.feat_fcn.load_weights(feat_fcn_path)
            self.Ws = pickle.load(open(dirsave + '/Ws.pkl', 'r'))
        else:
            self.train_feature_function(Xs, Dprior, dirsave)
            self.Ws, losses = self.analytically_solve_for_Ws(Xs, Dprior, dirsave)



class AutomaticZetaUCB(Function):
    # this is the negative version of GP-UCB (for consistency of minimizing)
    # this UCB uses zeta as used in the paper to prove regret bounds
    def __init__(self, N, delta, gp):
        '''
        N: the number of training datasets (number of rows in Y)
        delta: w.p. 1-delta the regret bound holds. Preferred range: (0, 0.05)
        '''
        self.N = N
        self.delta = delta
        self.gp = gp
        self.zeta = None

    def set_zeta(self):
        t = len(self.gp.evaled_x)
        delta = self.delta
        N = self.N
        iota = np.sqrt(6. * (N - 2 + t + 2 * np.sqrt(t * np.log(6. / delta)) \
                             + 2. * np.log(6. / delta)) / (delta * N * (N - t - 2)))

        b = np.log(6. / delta) / (N - t - 1)
        self.zeta = (iota + np.sqrt(2 * np.log(3. / delta))) / \
                    np.sqrt(1 - 2 * np.sqrt(b))

    def __call__(self, x):
        self.set_zeta()
        if len(x) == 1: x = x[None, :]
        mu, var = self.gp.predict(x)
        return -mu - self.zeta * np.sqrt(var)  # helper function minimizes

    def fg(self, x):
        # returns function value and gradient value at x
        mu, var = self.gp.predict(x)
        dmdx, dvdx = self.gp.predictive_gradients(x)
        dmdx = dmdx[0, :, 0]
        dvdx = dvdx[0, :]
        f = -mu - self.zeta * np.sqrt(var)
        g = -dmdx - 0.5 * dvdx / np.sqrt(var)
        return f[0, 0], g[0, :]


class UCB(Function):
    # this is the negative version of GP-UCB (for consistency of minimizing)
    def __init__(self, zeta, gp):
        self.zeta = zeta
        self.gp = gp

    def __call__(self, x):
        mu, var = self.gp.predict(x)
        return -mu - self.zeta * np.sqrt(var)

    def fg(self, x):
        # returns function value and gradient value at x
        mu, var = self.gp.predict(x)
        dmdx, dvdx = self.gp.predictive_gradients(x)
        dmdx = dmdx[0, :, 0]
        dvdx = dvdx[0, :]
        f = -mu - self.zeta * np.sqrt(var)
        g = -dmdx - 0.5 * dvdx / np.sqrt(var)
        return f[0, 0], g[0, :]


class ProbImprovement(Function):
    def __init__(self, target_val, gp):
        self.target_val = target_val
        self.gp = gp

    def __call__(self, x):
        mu, var = self.gp.predict(x)

        if np.any(var == 0):
            var += 0.00000001
        return (self.target_val - mu) / np.sqrt(var)

    def fg(self, x):
        # returns function value and gradient value at x
        mu, var = self.gp.predict(x)
        dmdx, dvdx = self.gp.predictive_gradients(x)
        dmdx = dmdx[0, :, 0]
        dvdx = dvdx[0, :]
        f = (self.target_val - mu) / np.sqrt(var)
        g = (-np.sqrt(var) * dmdx - 0.5 * (self.target_val - mu) * dvdx / np.sqrt(var)) / var
        return f[0, 0], g[0, :]
