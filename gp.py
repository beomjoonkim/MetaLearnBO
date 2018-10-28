import numpy as np

import GPy
import sklearn
import time


class StandardDiscreteGP(object):
    def __init__(self, Dprior, xvals, problem_name):
        self.xvals = xvals
        xdim = xvals.shape[1]
        if problem_name == 'gpb':
            # Implement this additive kernel
            g_ker = GPy.kern.RBF(58, active_dims=np.arange(58))
            b_ker = GPy.kern.RBF(3, active_dims=range(58, 61))
            p_ker = GPy.kern.RBF(3, active_dims=range(61, 64))
            self.kern = g_ker + b_ker + p_ker
        else:
            self.kern = GPy.kern.RBF(xdim)
        self.model = None

    def predict(self, x):
        if self.model is None:
            return np.array([1]), np.array([1])
        x = self.xvals[x]
        stime = time.time()
        mu, sig = self.model.predict(x)
        print "Prediction time %.2f" % (time.time() - stime)
        mu[self.evaled_x] = -np.inf
        sig[self.evaled_x] = 0
        return mu, sig

    def update(self, evaled_x, evaled_y):
        self.evaled_x = np.array(evaled_x)
        self.evaled_y = np.array(evaled_y)
        if len(evaled_x) == 0:
            return
        evaled_x = np.array(evaled_x)
        evaled_y = np.array(evaled_y)
        evaled_x = np.array(self.xvals[evaled_x])
        evaled_y = np.array(evaled_y)[:, None]
        self.model = GPy.models.GPRegression(evaled_x, evaled_y, kernel=self.kern)
        self.model.optimize(messages=False, max_f_eval=1000)


class CommonRS(StandardDiscreteGP):
    # Common response surface strategy. Paper:
    # https://pdfs.semanticscholar.org/75f2/6734972ebaffc6b43d45abd3048ef75f15a5.pdf
    def __init__(self, Dprior, xvals, problem_name):
        self.xvals = xvals
        xdim = xvals.shape[1]

        if problem_name == 'gpb':
            # Implement this additive kernel
            g_ker = GPy.kern.RBF(58, active_dims=np.arange(58))
            b_ker = GPy.kern.RBF(3, active_dims=range(58, 61))
            p_ker = GPy.kern.RBF(3, active_dims=range(61, 64))
            self.kern = g_ker + b_ker + p_ker
        else:
            self.kern = GPy.kern.RBF(xdim)

        scaler = sklearn.preprocessing.StandardScaler()
        D_ = scaler.fit_transform(Dprior.T)  # function-wise standardization
        D_ = D_.T
        super(CommonRS, self).__init__(D_, xvals, problem_name)
        n_fcns = D_.shape[0]
        n_evals = D_.shape[1]
        y_prior = D_.reshape(n_fcns * n_evals, 1)
        x_prior = xvals.repeat(n_fcns, 0)

        y_prior = y_prior[:1000, :]
        x_prior = x_prior[:1000, :]
        self.y_prior = y_prior
        self.x_prior = x_prior
        stime = time.time()
        self.model = GPy.models.SparseGPRegression(x_prior, y_prior, kernel=self.kern)
        self.model.optimize(messages=False, max_f_eval=1)
        print "GPy initial training took %.2f" % (time.time() - stime)

    def update(self, evaled_x, evaled_y):
        self.evaled_x = np.array(evaled_x)
        self.evaled_y = np.array(evaled_y)
        if len(evaled_x) == 0:
            return
        evaled_x = np.array(self.xvals[evaled_x])
        evaled_y = np.array(evaled_y)[:, None]

        # find the response value
        evaled_y = (evaled_y - np.mean(self.evaled_y))
        if np.std(self.evaled_y) > 0:
            evaled_y /= np.std(self.evaled_y)

        evaled_x = np.r_[self.x_prior, evaled_x]
        evaled_y = np.r_[self.y_prior, evaled_y]
        stime = time.time()
        self.model = GPy.models.SparseGPRegression(evaled_x, evaled_y, kernel=self.kern)
        # self.model.optimize(messages=False,max_f_eval = 1)
        print "GPy initial training took %.2f" % (time.time() - stime)

    def predict(self, x):
        if self.model is None:
            return np.array([1]), np.array([1])
        x = self.xvals[x]
        stime = time.time()
        mu, sig = self.model.predict(x)
        if len(self.evaled_x) > 0:
            mu[self.evaled_x] = -np.inf
            sig[self.evaled_x] = 0
        return mu, sig


class ContinuousCommonRS(CommonRS):
    # Common response surface strategy. Paper:
    # https://pdfs.semanticscholar.org/75f2/6734972ebaffc6b43d45abd3048ef75f15a5.pdf
    def __init__(self, Dprior, xvals, problem_name):
        self.xvals = xvals
        xdim = xvals.shape[1]
        print xdim

        if problem_name == 'gpb':
            # Implement this additive kernel
            g_ker = GPy.kern.RBF(58, active_dims=np.arange(58))
            b_ker = GPy.kern.RBF(3, active_dims=range(58, 61))
            p_ker = GPy.kern.RBF(3, active_dims=range(61, 64))
            self.kern = g_ker + b_ker + p_ker
        else:
            self.kern = GPy.kern.RBF(xdim)

        scaler = sklearn.preprocessing.StandardScaler()
        D_ = scaler.fit_transform(Dprior.T)  # function-wise standardization
        D_ = D_.T
        super(CommonRS, self).__init__(D_, xvals, problem_name)
        n_fcns = D_.shape[0]
        n_evals = D_.shape[1]
        y_prior = D_.reshape(n_fcns * n_evals, 1)
        x_prior = xvals.repeat(n_fcns, 0)

        # reduce the prior data points to 1000 from 1000*1000
        y_prior = y_prior[:1000, :]
        x_prior = x_prior[:1000, :]
        self.y_prior = y_prior
        self.x_prior = x_prior
        stime = time.time()
        self.model = GPy.models.SparseGPRegression(x_prior, y_prior, kernel=self.kern)
        self.model.optimize(messages=False, max_f_eval=1)
        print "GPy initial training took %.2f" % (time.time() - stime)

    def update(self, evaled_x, evaled_y):
        self.evaled_x = np.array(evaled_x)
        self.evaled_y = np.array(evaled_y)
        if len(evaled_x) == 0:
            return
        evaled_x = np.array(evaled_x)
        evaled_y = np.array(evaled_y)[:, None]

        # find the response value
        evaled_y = (evaled_y - np.mean(self.evaled_y))
        if np.std(self.evaled_y) > 0:
            evaled_y /= np.std(self.evaled_y)

        evaled_x = np.r_[self.x_prior, evaled_x]
        evaled_y = np.r_[self.y_prior, evaled_y]
        stime = time.time()
        self.model = GPy.models.SparseGPRegression(evaled_x, evaled_y, kernel=self.kern)
        print "GPy initial training took %.2f" % (time.time() - stime)

    def predict(self, x):
        if self.model is None:
            return np.array([1]), np.array([1])
        if len(x.shape) == 1: x = x[None, :]
        stime = time.time()
        mu, sig = self.model.predict(x)
        return mu, sig


class StandardContinuousGP():
    def __init__(self, Dprior, xvals, problem_name):
        self.xvals = xvals
        xdim = xvals.shape[1]
        self.model = None
        """
        if problem_name=='cbelt':
          ker1 = GPy.kern.RBF(3,active_dims=np.arange(3))
          ker2 = GPy.kern.RBF(3,active_dims=range(3,6))
          ker3 = GPy.kern.RBF(3,active_dims=range(6,9))
          ker4 = GPy.kern.RBF(3,active_dims=range(9,12))
          ker5 = GPy.kern.RBF(3,active_dims=range(12,15))
          self.kern = ker1+ker2+ker3+ker4+ker5
        else:
        """
        self.kern = GPy.kern.RBF(xdim, variance=200) # 500 was too good

    def predict(self, x):
        if self.model is None:
            return np.array([1]), np.array([1])
        if len(x.shape) == 1: x = x[None, :]
        mu, sig = self.model.predict(x)
        return mu, sig

    def update(self, evaled_x, evaled_y):
        self.evaled_x = np.array(evaled_x)
        self.evaled_y = np.array(evaled_y)
        if len(evaled_x) == 0:
            return
        evaled_x = np.array(evaled_x)
        evaled_y = np.array(evaled_y)[:, None]
        self.model = GPy.models.GPRegression(evaled_x, evaled_y, kernel=self.kern)
        #self.model.optimize(messages=False, max_f_eval = 1) # maximum likelihood maximization
        #import pdb;pdb.set_trace()


class GP(object):
    def __init__(self, Dprior):
        self.Dprior = Dprior

    def predict(self, x):
        raise NotImplemented

    def update(self, evaled_x, evaled_y):
        self.evaled_x = evaled_x;
        self.evaled_y = evaled_y
        stime = time.time()
        self.update_mu(evaled_x, evaled_y)
        self.update_cov(evaled_x)
        print "Update time %f" % (time.time() - stime)


class PriorEstDiscreteGP(GP):
    def __init__(self, Dprior, prior_est_alg):
        GP.__init__(self, Dprior)
        if prior_est_alg == "zbk":
            self.compute_prior_with_point_est()
        elif prior_est_alg == "niw":
            self.compute_prior_with_niw()

    def predict(self, x):
        return self.mu_updated[x], np.diag(self.cov_updated)[x]

    def compute_prior_with_point_est(self):
        self.mu = np.mean(self.Dprior, axis=0)
        self.cov = np.cov(self.Dprior.transpose())

    def compute_prior_with_niw(self):
        # zero for mu0 and random priors
        xdim = self.Dprior.shape[1]
        mu_0 = np.zeros(xdim)[:, None]
        sigma_0 = np.eye(xdim, xdim) + np.random.random((xdim, xdim)) * 0.1
        lambda_0 = 1
        dof_0 = xdim

        # Sigma prior:
        #   W^{-1}(sigma0,dof0)
        # Mu prior:
        #   N(mu0,1/lambda0 * sigma0)

        # Compute posterior
        # formula source: https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution
        ndata = self.Dprior.shape[0]
        ybar = np.mean(self.Dprior, axis=0)[:, None]

        mu_n = (lambda_0 * mu_0 + ndata * ybar) / (lambda_0 + ndata)
        S = np.cov(self.Dprior.transpose()) * (ndata - 1)
        sigma_n = sigma_0 \
                  + S \
                  + (lambda_0 * ndata / (lambda_0 + ndata)) \
                  * np.dot((ybar - mu_0), (ybar - mu_0).T)
        self.mu = mu_n.squeeze()
        self.cov = sigma_n

    def update_mu(self, observed_x, observed_y):
        mu = self.mu
        cov = self.cov

        eval_idxs = observed_x
        uneval_idxs = list((set(range(len(mu)))).difference(set(eval_idxs)))

        # ix_ function builds cross products of elements of first and second inputs
        cov_uneval_eval = cov[np.ix_(uneval_idxs, eval_idxs)]
        cov_eval_eval = cov[np.ix_(eval_idxs, eval_idxs)]
        eval_diff = np.asmatrix(observed_y - self.mu[eval_idxs]).transpose()

        reg_term = 0  # 0.000001*np.eye( len(eval_idxs) )
        mu_update = np.asmatrix(mu[uneval_idxs]).transpose() \
                    + np.dot(np.dot(cov_uneval_eval, np.linalg.inv(cov_eval_eval + reg_term)), \
                             eval_diff)
        mu_update = np.asarray(mu_update).reshape((mu_update.shape[0],))

        # Evaluated arms get -inf value
        self.mu_updated = np.ones(self.mu.shape) * -np.inf
        self.mu_updated[uneval_idxs] = mu_update

    def update_cov(self, eval_idxs):
        cov = self.cov
        mu = self.mu

        uneval_idxs = list((set(range(len(cov[0, :])))).difference(set(eval_idxs)))
        cov_eval_uneval = cov[np.ix_(eval_idxs, uneval_idxs)]
        cov_eval_eval = cov[np.ix_(eval_idxs, eval_idxs)]
        cov_uneval_uneval = cov[np.ix_(uneval_idxs, uneval_idxs)]
        cov_uneval_eval = cov[np.ix_(uneval_idxs, eval_idxs)]

        reg_term = 0  # 0.000001*np.eye( len(eval_idxs) )
        cov_update = cov_uneval_uneval - \
                     np.dot(np.dot(cov_uneval_eval, np.linalg.inv(cov_eval_eval + reg_term)),
                            cov_eval_uneval)

        # Evaluated arms get zero variance
        #   this is easier than keeping track of which arm was evaluated and then
        #   trying to coordinate that with domain's indices
        self.cov_updated = np.zeros(self.cov.shape)
        self.cov_updated[np.ix_(uneval_idxs, uneval_idxs)] = cov_update


class PriorEstContinuousGP(GP):
    def __init__(self, Dprior, prior_est_alg, feat_fcn, Ws):
        super(PriorEstContinuousGP, self).__init__(Dprior)
        self.feat_fcn = feat_fcn
        self.Ws = np.array(Ws).squeeze()
        if prior_est_alg == "zbk":
            self.compute_prior_with_point_est()
        elif prior_est_alg == "niw":
            self.compute_prior_with_niw()

    def predict(self, x):
        if len(x.shape) == 1:
            x = x[None, :]
        feat_x = self.feat_fcn.feat_fcn.predict(x)
        mu_x = np.dot(feat_x, self.muW_prime)
        var_x = np.diag(np.dot(np.dot(feat_x, self.covW_prime), feat_x.T))
        return mu_x, var_x

    def compute_prior_with_point_est(self):
        self.muW = np.mean(self.Ws, axis=0)
        self.covW = np.cov(np.array(self.Ws).T)

    def compute_prior_with_niw(self):
        # zero for mu0 and random priors
        xdim = self.Dprior.shape[1]
        mu_0 = np.zeros(xdim)[:, None]
        sigma_0 = np.eye(xdim, xdim) + np.random.random((xdim, xdim)) * 0.1
        lambda_0 = 1
        dof_0 = xdim

        # Sigma prior:
        #   W^{-1}(sigma0,dof0)
        # Mu prior:
        #   N(mu0,1/lambda0 * sigma0)

        # Compute posterior
        # formula source: https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution
        ndata = self.Dprior.shape[0]
        ybar = np.mean(self.Dprior, axis=0)[:, None]

        mu_n = (lambda_0 * mu_0 + ndata * ybar) / (lambda_0 + ndata)
        S = np.cov(self.Dprior.transpose()) * (ndata - 1)
        sigma_n = sigma_0 \
                  + S \
                  + (lambda_0 * ndata / (lambda_0 + ndata)) \
                  * np.dot((ybar - mu_0), (ybar - mu_0).T)
        self.mu = mu_n.squeeze()
        self.cov = sigma_n

    def update_mu(self, obs_x, obs_y):
        if len(obs_x) == 0:
            self.muW_prime = self.muW
            return
        obs_x = np.array(obs_x)
        if len(obs_x.shape) == 1:
            obs_x = obs_x[None, :]

        # TODO:  fix this madness with feat_fcn
        self.phi = self.feat_fcn.feat_fcn.predict(obs_x)

        # self.phi = self.feat_fcn.feat_fcn.predict(obs_x)
        self.phi_sigma_phi_T = np.dot(np.dot(self.phi, self.covW), self.phi.T)
        self.phi_sigma_phi_T += 0.00001 * np.eye(self.phi_sigma_phi_T.shape[0])
        self.phi_sigma_phi_T_inv = np.linalg.inv(self.phi_sigma_phi_T)
        self.sigma_phi = np.dot(self.covW, self.phi.T)
        self.phi_T_u = np.dot(self.phi, self.muW)
        self.phi_sigma = np.dot(self.phi, self.covW)

        self.muW_prime = self.muW + np.dot(np.dot(self.sigma_phi, self.phi_sigma_phi_T_inv), obs_y - self.phi_T_u)

    def update_cov(self, obs_x):
        if len(obs_x) == 0:
            self.covW_prime = self.covW
            return
        obs_x = np.array(obs_x)
        if len(obs_x.shape) == 1:
            obs_x = obs_x[None, :]

        """
        phi = self.feat_fcn.feat_fcn.predict(obs_x)
        phi_sigma_phi_T = np.dot(np.dot(phi,self.covW),phi.T)
        phi_sigma_phi_T += 0.00001*np.eye(phi_sigma_phi_T.shape[0])
        sigma_phi       = np.dot(self.covW,phi.T)
        phi_sigma       = np.dot(phi,self.covW)
        """

        t = len(obs_x)
        N = self.Dprior.shape[0]
        try:
            self.covW_prime = float(N - 1.0) / (N - t - 1) * (self.covW -
                                                              np.dot(np.dot(self.sigma_phi, self.phi_sigma_phi_T_inv),
                                                                     self.phi_sigma) + 0.0001)
        except:
            import pdb;
            pdb.set_trace()


class DiscreteGPIterative(PriorEstDiscreteGP):
    def predict(self, x):
        return self.mu[x], np.diag(self.cov)[x]

    def update_mu(self, observed_x, observed_y):
        mu = self.mu
        cov = self.cov

        if len(observed_x) > 0:
            eval_idxs = [observed_x[-1]]
            observed_y = observed_y[-1]
        else:
            eval_idxs = observed_x
        uneval_idxs = list((set(range(len(mu)))).difference(set(eval_idxs)))

        # ix_ function builds cross products of elements of first and second inputs
        cov_uneval_eval = cov[np.ix_(uneval_idxs, eval_idxs)]
        cov_eval_eval = cov[np.ix_(eval_idxs, eval_idxs)]
        eval_diff = np.asmatrix(observed_y - self.mu[eval_idxs]).transpose()

        reg_term = 0.001  # 0.000001*np.eye( len(eval_idxs) )
        if cov_eval_eval == 0:
            cov_eval_eval += reg_term
        mu_update = np.asmatrix(mu[uneval_idxs]).transpose() + \
                    np.dot(np.dot(cov_uneval_eval, np.linalg.inv(cov_eval_eval)),
                           eval_diff)
        mu_update = np.asarray(mu_update).reshape((mu_update.shape[0],))

        # Evaluated arms get -inf value
        self.mu_updated = np.ones(self.mu.shape) * -np.inf
        self.mu_updated[uneval_idxs] = mu_update

        self.mu = self.mu_updated

    def update_cov(self, observed_x):
        cov = self.cov
        mu = self.mu

        if len(observed_x) > 0:
            eval_idxs = [observed_x[-1]]
        else:
            eval_idxs = observed_x
        uneval_idxs = list((set(range(len(cov[0, :])))).difference(set(eval_idxs)))
        cov_eval_uneval = cov[np.ix_(eval_idxs, uneval_idxs)]
        cov_eval_eval = cov[np.ix_(eval_idxs, eval_idxs)]
        cov_uneval_uneval = cov[np.ix_(uneval_idxs, uneval_idxs)]
        cov_uneval_eval = cov[np.ix_(uneval_idxs, eval_idxs)]

        reg_term = 0.001
        if cov_eval_eval == 0: cov_eval_eval += reg_term

        t = len(observed_x)
        N = self.Dprior.shape[0]
        cov_update = float(N - 1.0) / (N - t - 1) * (cov_uneval_uneval -
                                                     np.dot(np.dot(cov_uneval_eval, np.linalg.inv(cov_eval_eval)),
                                                            cov_eval_uneval))

        # Evaluated arms get zero variance
        #   this is easier than keeping track of which arm was evaluated and then
        #   trying to coordinate that with domain's indices
        self.cov_updated = np.zeros(self.cov.shape)
        self.cov_updated[np.ix_(uneval_idxs, uneval_idxs)] = cov_update
        self.cov = self.cov_updated
