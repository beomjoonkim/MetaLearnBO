import numpy as np

#import bloscpack as bp
import cPickle as pickle

import os
#import GPy
import scipy.stats as stats
from scipy import linalg, optimize

from sklearn import linear_model
from plot_tools import *


MAX_ITER = 100


def group_lasso(X, y, alpha, groups, max_iter=MAX_ITER, rtol=1e-6,
             verbose=False):
    """
    Linear least-squares with l2/l1 regularization solver.

    Solves problem of the form:

               .5 * |Xb - y| + n_samples * alpha * Sum(w_j * |b_j|)

    where |.| is the l2-norm and b_j is the coefficients of b in the
    j-th group. This is commonly known as the `group lasso`.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Design Matrix.

    y : array of shape (n_samples,)

    alpha : float or array
        Amount of penalization to use.

    groups : array of shape (n_features,)
        Group label. For each column, it indicates
        its group apertenance.

    rtol : float
        Relative tolerance. ensures ||(x - x_) / x_|| < rtol,
        where x_ is the approximate solution and x is the
        true solution.

    Returns
    -------
    x : array
        vector of coefficients

    References
    ----------
    "Efficient Block-coordinate Descent Algorithms for the Group Lasso",
    Qin, Scheninberg, Goldfarb
    """

    # .. local variables ..
    X, y, groups, alpha = map(np.asanyarray, (X, y, groups, alpha))
    if len(groups) != X.shape[1]:
        raise ValueError("Incorrect shape for groups")
    w_new = np.zeros(X.shape[1], dtype=X.dtype)
    alpha = alpha * X.shape[0]

    # .. use integer indices for groups ..
    group_labels = [np.where(groups == i)[0] for i in np.unique(groups)]
    H_groups = [np.dot(X[:, g].T, X[:, g]) for g in group_labels]
    eig = map(linalg.eigh, H_groups)
    Xy = np.dot(X.T, y)
    initial_guess = np.zeros(len(group_labels))

    def f(x, qp2, eigvals, alpha):
        return 1 - np.sum( qp2 / ((x * eigvals + alpha) ** 2))
    def df(x, qp2, eigvals, penalty):
        # .. first derivative ..
        return np.sum((2 * qp2 * eigvals) / ((penalty + x * eigvals) ** 3))

    if X.shape[0] > X.shape[1]:
        H = np.dot(X.T, X)
    else:
        H = np.dot(X.T, X)
    #    H = None

    for n_iter in range(max_iter):
        w_old = w_new.copy()
        for i, g in enumerate(group_labels):
            # .. shrinkage operator ..
            eigvals, eigvects = eig[i]
            w_i = w_new.copy()
            w_i[g] = 0.
            if H is not None:
                X_residual = np.dot(H[g], w_i) - Xy[g]
            else:
                X_residual = np.dot(X.T, np.dot(X[:, g], w_i)) - Xy[g]
            qp = np.dot(eigvects.T, X_residual)
            if len(g) < 2:
                # for single groups we know a closed form solution
                w_new[g] = - np.sign(X_residual) * max(abs(X_residual) - alpha, 0)
            else:
                if alpha < linalg.norm(X_residual, 2):
                    initial_guess[i] = optimize.newton(f, initial_guess[i], df, tol=.5,
                                args=(qp ** 2, eigvals, alpha))
                    w_new[g] = - initial_guess[i] * np.dot(eigvects /  (eigvals * initial_guess[i] + alpha), qp)
                else:
                    w_new[g] = 0.


        # .. dual gap ..
        max_inc = linalg.norm(w_old - w_new, np.inf)
        if True: #max_inc < rtol * np.amax(w_new):
            residual = np.dot(X, w_new) - y
            group_norm = alpha * np.sum([linalg.norm(w_new[g], 2)
                         for g in group_labels])
            if H is not None:
                norm_Anu = [linalg.norm(np.dot(H[g], w_new) - Xy[g]) \
                           for g in group_labels]
            else:
                norm_Anu = [linalg.norm(np.dot(H[g], residual)) \
                           for g in group_labels]
            if np.any(norm_Anu > alpha):
                nnu = residual * np.min(alpha / norm_Anu)
            else:
                nnu = residual
            primal_obj =  .5 * np.dot(residual, residual) + group_norm
            dual_obj   = -.5 * np.dot(nnu, nnu) - np.dot(nnu, y)
            dual_gap = primal_obj - dual_obj
            if verbose:
                print 'Relative error: %s' % (dual_gap / dual_obj)
            if np.abs(dual_gap / dual_obj) < rtol:
                break

    return w_new


def check_kkt(A, b, x, penalty, groups):
    """Check KKT conditions for the group lasso

    Returns True if conditions are satisfied, False otherwise
    """
    group_labels = [groups == i for i in np.unique(groups)]
    penalty = penalty * A.shape[0]
    z = np.dot(A.T, np.dot(A, x) - b)
    safety_net = 1e-1 # sort of tolerance
    for g in group_labels:
        if linalg.norm(x[g]) == 0:
            if not linalg.norm(z[g]) < penalty + safety_net:
                return False
        else:
            w = - penalty * x[g] / linalg.norm(x[g], 2)
            if not np.allclose(z[g], w, safety_net):
                return False
    return True





def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0,x)
def linear(x):
    return x
def multivariate_normal_nloglik(yy, mu, cov):
    covinv = np.linalg.pinv(cov)
    diffpred = yy - mu
    return diffpred[0]/2*np.log(2*np.pi) + 0.5*np.log(np.linalg.det(cov)) + 0.5*np.dot(diffpred.T, np.dot(covinv, diffpred))
def tp_array(array):
    return np.array([array]).T

class rdgp(object):
    # currently support 1-dim input
    def __init__(self, bfunc, wmean = [.0], wstd=[.1], nfeature=20, sigma=0.1, phiscale=1.):
        assert(len(wmean) == len(wstd))
        self.nfeature = nfeature*len(wstd)
        self.wstd = wstd # standard deviation for weights
        self.wmean = wmean
        self.sigma = sigma
        self.bfunc = bfunc
        self.amean = np.zeros((self.nfeature,1))
        self.acov = np.identity(self.nfeature)
        self.phiscale = phiscale
        self.sample_w()
    def phi(self, x):
        # nfeature x data
        phi = self.phiscale * np.sqrt(2.0/self.nfeature)*self.bfunc(self.w*x.T+np.tile(self.c,( 1, x.shape[0] ) ) )
        return phi
    def set_paras(self, wstd=None, phiscale=None, sigma=None, nfeature=None):
        if wstd:
            #self.w = self.w*wstd[0]/self.wstd[0]
            self.wstd = wstd
            self.sample_w()
        if phiscale:
            self.phiscale = phiscale
        if sigma:
            self.sigma = sigma
        if nfeature:
            self.nfeature = nfeature*len(self.wstd)
            self.amean = np.zeros((self.nfeature,1))
            self.acov = np.identity(self.nfeature)
            self.sample_w()
    def sample_w(self):
        n = len(self.wstd)
        self.w = np.zeros((self.nfeature,1))
        self.c = np.random.uniform(-np.pi,np.pi, (self.nfeature,1))
        nf = self.nfeature/n
        wcnt = 0
        for (wm,ws) in zip(self.wmean, self.wstd):
            self.w[wcnt:wcnt+nf] = np.random.normal(wm, ws, (nf,1))
            wcnt += nf
    def sample_a(self):
        self.a = np.random.multivariate_normal(self.amean.T[0], self.acov)
        self.a = tp_array(self.a)
        print self.a.shape
    def gram_matrix(self,xvec):
        Z = self.phi(xvec) # nfeature x ndata
        return np.dot(Z.T,Z)
    def sample_K_plot(self):
        self.sample_w()
        xvec = np.array([np.linspace(-10,10,500)]).T
        init_plotting()
        K = self.gram_matrix(xvec)
        cs=sns.color_palette("Set1", n_colors=8)
        for i in range(8):
            y = np.random.multivariate_normal([0]*xvec.shape[0], K)
            plt.plot(xvec.T[0], y.T, color=cs[i], alpha=0.8)
        #plt.show()
        plt.xlabel(r'$a$')
        plt.ylabel(r'$Q_s(a)$')
        plt.xlim([-10,10])
        plt.tight_layout()
        plt.savefig('figs' + '/' + self.bfunc.__name__ + 'nf' + str(self.nfeature) + '_wstd' + str(self.wstd) +'.eps',format='eps', dpi=1000)

    def sample_a_plot(self):
        self.sample_w()
        xvec = np.array([np.linspace(-10,10,500)]).T
        init_plotting()
        
        cs=sns.color_palette("Set1", n_colors=8)
        for i in range(1):
            a = self.sample_a()
            Z = self.phi(xvec)
            y = np.dot(a,Z)
            plt.plot(xvec.T[0], y, color=cs[i], alpha=0.8, label='The real f(x)')
        #plt.show()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$f(x)$')
        plt.xlim([-10,10])
        plt.tight_layout()
        plt.savefig('figs' + '/' + self.bfunc.__name__ + 'a_nf' + str(self.nfeature) + '_wstd' + str(self.wstd) +'_wmean' + str(self.wmean)+'.eps',format='eps', dpi=1000)
        pick = np.random.choice(xvec.shape[0],40,False)
        return xvec[pick,:],y[pick]
    def plot(self, xx=None, yy=None):
        xvec = tp_array(np.linspace(-15,15,500))
        Zq = self.phi(xvec)
        mu = self.predict(xvec)

        #import pdb; pdb.set_trace()
        var = np.einsum('ij,ij->j', Zq, np.dot(self.acov, Zq))
        sig = tp_array(np.sqrt(var)) + self.sigma

        print mu.shape, xvec.shape, sig.shape
        cs=sns.color_palette("Set1", n_colors=8)
        plt.clf()
        
        plt.plot(xvec,mu,color=cs[3], label='mean')
        plt.plot(xvec,mu+2*sig,color=cs[1])
        plt.plot(xvec,mu-2*sig,color=cs[1])
        if xx is not None:
            plt.scatter(xx.T[0],yy.T[0],c=cs[2],marker='o', s=25, label='data')
        plt.xlim([-15,15])
        plt.xlabel(r'$x$')
        plt.ylabel(r'$f(x)$')
        plt.tight_layout()
        plt.legend(loc='best')

    def l1conditional(self, xx, yy, alpha=0.1, isplot=0):
        if not hasattr(self,'w'):
            self.sample_w()
        sigma = self.sigma

        Z = self.phi(xx)
        '''
        B = np.dot(Z, Z.T) + np.identity(self.nfeature)*(self.sigma**2)
        Binv = np.array(np.linalg.pinv(B))
        tmp = np.dot(Binv,np.dot(Z, yy))
        self.acov = Binv*(sigma**2)
        # fit amean
        reg = linear_model.Lasso(alpha=alpha)
        #import pdb; pdb.set_trace()
        reg.fit(Z.T, yy)
        #print reg.coef_.shape
        self.amean = tp_array(reg.coef_)
        '''
        
        reg = linear_model.LassoCV(alphas=np.logspace(-4,-0.5,30), random_state=0)
        reg.fit(Z.T, yy)
        self.amean = tp_array(reg.coef_)
        '''
        tmp = []
        for i in range(10):
            tmp += [i]*100 
        groups = np.array(tmp)
        print groups.shape, self.w.shape
        yy = np.array(yy).T[0]
        Z = np.array(Z.T)
        print Z.shape, yy.shape

        tmp = group_lasso(Z, yy, 1e-5, groups)
        print tmp.shape
        self.amean = tp_array(tmp)
        '''
        '''
        amean = self.amean
        for i in range(1000):
            #print amean[0:5]
            minusterm = np.dot(Binv, np.sign(amean))
            compare = tmp/minusterm
            #print compare.shape
            if np.any(compare>0):
                lam = np.min(compare[compare>0])
            else:
                lam = 0
            #print lam
            amean = tmp - lam*minusterm
        self.amean = amean
        # print amean
        '''
        if isplot:
            self.plot(xx, yy)
    def conditional(self, xx, yy, isplot=0):
        if not hasattr(self,'w'):
            self.sample_w()
        sigma = self.sigma
        Z = self.phi(xx)
        B = np.dot(Z, Z.T)/(self.sigma**2) + np.identity(self.nfeature)
        Binv = np.linalg.pinv(B)
        self.amean = np.dot(Binv,np.dot(Z, yy))/(sigma**2)
        self.acov = Binv

        if isplot:
            self.plot(xx, yy)
    def predict(self, x):
        # datasize x 1
        return np.dot(self.phi(x).T, self.amean)
    def predict_datacov(self, x):
        Z = self.phi(x)
        return np.dot(np.dot(Z.T, self.acov),Z)
        #plt.savefig('figs' + '/' + self.bfunc.__name__ + 'cond_nf' + str(self.nfeature) + '_wstd' + str(self.wstd) + '_wmean' + str(self.wmean)+'.eps',format='eps', dpi=1000)
    def nloglik(self, xx, yy):
        # The smaller nloglik, the smaller the function class we have shrinked
        #print 'Negative log likelihood'
        mu = self.predict(xx)
        cov = self.predict_datacov(xx)# + np.identity(xx.shape[0])*(self.sigma**2)
        return multivariate_normal_nloglik(yy, mu, cov)

    def loss(self, xx, yy):
        Z = self.phi(xx)
        u, s, v = np.linalg.svd(Z,0) # nfeature x min(ndata, nfeature), min x min, min x ndata
        print 'u shape is ', u.shape, '. s shape is ', s.shape, '. v shape is ', v.shape
        s2 = s**2
        sigma2 = self.sigma**2
        sig_s = s2*sigma2/(s2 + sigma2)
        pseudo_det = np.prod(sig_s)
        print 'log pseudo det = ', np.log(pseudo_det)
        lik_s = sigma2/( np.multiply(s2, s2 + sigma2) )
        print 'lik_s > 0?', np.all(lik_s > 0)
        tmp = np.dot(np.dot(v.T,np.diag(lik_s)),v)
        print 'vt like_s v >=0?', np.all(np.linalg.eigvals(tmp) >= 0)
        log_lik = np.dot(np.dot(yy.T, tmp), yy)
        ztz = np.dot(Z.T,Z)
        #print np.linalg.pinv( np.dot(ztz,ztz/sigma2+np.identity(ztz.shape[0])) )
        
        print 'ztz == vt s2 v?', np.allclose(ztz, np.dot(np.dot(v.T, np.diag(s2)), v))
        print 'vt v = I?', np.allclose(np.dot(v.T,v), np.identity(v.shape[1]))
        cutoff = 1e-15*np.maximum.reduce(s2)
        #print np.maximum.reduce(s2)
        s2inv = s2
        for i in range(len(s2)):
            if s2[i] > cutoff:
                s2inv[i] = 1./s2[i]
            else:
                s2inv[i] = 0.

        #print s2inv
        #s2inv[np.where(s2inv > 1e8)] = 0
        #print s2inv
        #print s2inv
        #print np.linalg.pinv(ztz)
        #print np.dot(v.T, np.dot(np.diag(s2inv), v))
        a,b,c = np.linalg.svd(ztz, 0)
        print 'a.T == c?', np.allclose(a.T, c)
        print 'b == s2?', np.allclose(b, s2)
        print 'c == v?', np.allclose(c,v)
        print 'pinv(ztz) - vt s2inv v = ', np.sum(np.absolute(np.linalg.pinv(ztz) - np.dot(v.T, np.dot(np.diag(s2inv), v)))) /np.min(np.absolute(np.linalg.pinv(ztz)))
        print 'pinv(ztz(ztz + sigma^2I)) == vt lik_s v?', np.allclose(np.linalg.pinv( np.dot(ztz,ztz/sigma2+np.identity(ztz.shape[0]) ) ), np.dot(np.dot(v.T,np.diag(lik_s)),v))
        log_lik2 = np.dot(np.dot(yy.T,np.linalg.pinv( np.dot(ztz,ztz/sigma2+np.identity(ztz.shape[0])) ) ), yy)
        print log_lik, log_lik2
        print np.sum(np.absolute(log_lik - log_lik2))
        return log_lik + pseudo_det

class frdgp(rdgp):
    # sampled function of rdgp
    def __init__(self, bfunc=np.cos, wmean = [.0], wstd=[.1], nfeature=20, sigma=0.1, phiscale=1.):
        rdgp.__init__(self, bfunc, wmean, wstd, nfeature, sigma, phiscale)
        self.sample_w()
        self.sample_a()
    def predict(self, x):
        return np.dot(self.phi(x).T, self.a)
    def plot(self, filepath):
        dirnm = os.path.dirname(filepath)
        try:
            os.makedirs(dirnm)
        except OSError:
            if not os.path.isdir(dirnm):
                raise
        init_plotting()
        xvec = np.array([np.linspace(-10,10,500)]).T
        yvec = self.predict(xvec)
        plt.clf()
        plt.plot(xvec.T[0], yvec)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$f(x)$')
        plt.xlim([-10,10])
        plt.tight_layout()
        plt.savefig(filepath + '.eps', format='eps', dpi=1000)
    def generate_train_data(self, N):
        self.xtrain = tp_array(np.random.uniform(-10, 10, N))
        self.ytrain = self.predict(self.xtrain)

def diffnorm(fr1, fr2, isplot=1):
    # compute norm of the difference between two frdgp 
    xvec = np.array([np.linspace(-10,10,500)]).T
    yfr1 = fr1.predict(xvec)
    if len(yfr1) == 2:
        yfr1 = yfr1[0]
    yfr2 = fr2.predict(xvec)
    if len(yfr2) == 2:
        yfr2 = yfr2[0]
    if isplot:
        plt.plot(xvec, yfr1, color='red')
        plt.plot(xvec, yfr2, color='green')
        plt.show()
    return np.sum(np.absolute(yfr1 - yfr2))/xvec.shape[0]



def generate_test_functions(N):
    for i in range(N):
        wstd = np.random.uniform(-2., 2.)
        wstd = np.exp(wstd)
        nfeature = np.random.randint(10, 1000)
        print wstd, nfeature
        fr = frdgp(np.cos, [0.], [wstd], nfeature)
        fr.generate_train_data(100)
        fr.plot('test_functions/testfunc_'+str(i))
        save_obj(fr, 'test_functions/testfunc_' + str(i) + '.pkl')
def fit_gpy(gpy_func, xx, yy):
    gpy_func.set_XY(xx, yy)
    #gpy_func['Gaussian_noise.variance'].constrain_bounded(0.01,1, warning=False)
    gpy_func.optimize()

def gpy_nloglik(gpy_func, xx, yy):
    mu, cov = gpy_func.predict(xx, full_cov=True)
    return multivariate_normal_nloglik(yy, mu, cov)
def fit_rdgp_l2(rdgp_func, xx, yy, wstd, phiscale, sigma, nfeature=None):
    rdgp_func.set_paras(wstd, phiscale, sigma, nfeature)
    # rdgp_func.sample_w()
    rdgp_func.conditional(xx, yy, 0)

def fit_rdgp_l1(rdgp_func, xx, yy, alpha=0.1):
    #rdgp_func.set_paras(wstd, phiscale, sigma, nfeature)
    #rdgp_func.sample_w()
    rdgp_func.l1conditional(xx, yy, alpha=alpha, isplot=0)

def fit_function(func_index, isplot=1):
    fr = load_obj('test_functions/testfunc_' + str(func_index) + '.pkl')
    #gpy_func = GPy.models.GPRegression(fr.xtrain[:2, :], fr.ytrain[:2, :], GPy.kern.RBF(1))
    rdgp_funcs = []
    for nfeature in [10, 50, 100, 500, 1000]:
        rdgp_funcs.append(rdgp(np.cos, nfeature=nfeature))
    wstd = np.random.uniform(-2., 2., 10)
    wstd = np.exp(wstd)
    rdgp_funcl1 = rdgp(np.cos, wmean=[0.]* 10, wstd=wstd, nfeature=100)
    rdgp_funcl1.sample_w()

    result = []

    for train_cnt in range(10,100,10):
        res = []
        xx, yy = fr.xtrain[:train_cnt, :], fr.ytrain[:train_cnt, :]
        #fit_gpy(gpy_func, xx, yy) 
        #print gpy_func
        #gpy_func.plot()
        #lossgpy, loglikgpy = diffnorm(fr, gpy_func), gpy_func.log_likelihood()
        #res.append((lossgpy, loglikgpy))
        fit_rdgp_l1(rdgp_funcl1, xx, yy)
        if isplot:
            plt.clf()
            plt.scatter(xx, yy)
        diff = diffnorm(fr, rdgp_funcl1, isplot)
        print diff
        #res.append( (diff, rdgp_funcl1.nloglik(xx, yy)[0,0]) )

        # sample random features from gpy_func
        # fit_rdgp(rdgp_func, xx, yy, [1.0], 1., 0.1)
        # print 'before ', diffnorm(rdgp_func, fr), rdgp_func.nloglik(xx, yy)
        #gpyls, gpykvar = gpy_func.kern.lengthscale[0], gpy_func.kern.variance[0]
        # print gpyls, gpykvar
        #wstd, phiscale = 1./gpyls, ((2*np.pi)**0.25) * ((gpyls*gpykvar)**0.5)
        # print wstd, phiscale
        '''
        for i in range(len([10, 50, 100, 500, 1000])):
            fit_rdgp_l2(rdgp_funcs[i], xx, yy, [wstd], phiscale, gpy_func.Gaussian_noise.variance**0.5)
            diff = diffnorm(fr, rdgp_funcs[i])
            print diff
            res.append( (diff, rdgp_funcs[i].nloglik(xx, yy)[0,0]) )
        
        print res
        '''
        result.append(res)
    # print fr.wmean, fr.wstd, fr.nfeature, rdgp_func.wstd
    return result
def fit_function_rd(func_index, train_cnt):
    fr = load_obj('test_functions/testfunc_' + str(func_index) + '.pkl')
    xx, yy = fr.xtrain[:train_cnt, :].T, fr.ytrain[:train_cnt, :].T
    reg_coeff = 0.01
    # dimension of each layer
    hdims = [1, 100, 100, 1]
    w = [np.zeros( (hdims[i], hdims[i-1]) ) for i in range(1, len(hdims))]
    z = [np.zeros((hdims[i], train_cnt)) for i in range(1,len(hdims)), yy]
    h = [xx] + [np.zeros((hdims[i], train_cnt)) for i in range(1,len(hdims) - 1)]
    tmax = 10

    for t in range(tmax):
        for k in range(len(h) - 1, 0, -1):
            dum = np.random.normal(0,1, w[k].shape)
            z[k-1] = bfunc_inv(h[k] * dum)
        print 'backward prop end'


def w_forward(xx, h, z, w, reg_coeff, bfunc):
    for k in range(len(h) - 1):
        sigma_w = np.linalg.pinv(np.dot(h[k].T, h[k]) + np.identity(h[k].shape[1] * reg_coeff) ) 
        mu_w = np.dot( np.dot(z[k].T, h[k]), sigma_w)
        w[k] = mu_w + np.random.multivariate_normal(np.zeros((1,sigma.shape[0])))
        h[k + 1] = bfunc(np.dot(h[k], w[k].T))

if __name__ == "__main__":
    #generate_test_functions(100)
    bigres = []
    for i in range(1,100):
        res = fit_function(i)
        bigres.append(res)
    # res2 is for rescaling w
    # res3 is for resamping w
    save_obj(bigres, 'res4.dat')
