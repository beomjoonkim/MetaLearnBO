import pickle
import scipy.io as sio
import numpy as np
import argparse
import os
import time
import sklearn
import sys

from functions import DiscreteObjFcn, \
    AutomaticZetaUCB, ProbImprovement, FeatureFcn

try:
    from functions import CbeltObjFcn
    from prior_est import mat_comp as mc
    from fancyimpute import SoftImpute
except:
    pass
from gp_sample import SynthObjFcn

from gp import DiscreteGPIterative, StandardDiscreteGP, \
    StandardContinuousGP, PriorEstContinuousGP, \
    CommonRS, ContinuousCommonRS
from rand_search import KtimesRandomSearch, KtimesContRandomSearch
from bo import BO


#### Helper functions for creating or formatting data ####

def format_gpb_xvals(thetas):
    grasps = thetas['g_']
    placements = thetas['p_'][:, 0:3, -1]
    baseposes = thetas['b_']
    return np.c_[grasps, placements, baseposes]


def create_synth_data():
    n_features = 1000
    domain = [[-10., -10.], [10., 10.]]
    stime = time.time()
    x, y = np.meshgrid(np.linspace(-10, 10, 10), np.linspace(-10, 10, 10))

    x = np.random.uniform(-10, 10, (1000,))
    y = np.random.uniform(-10, 10, (1000,))
    xy = np.c_[x, y]
    # xy = np.array([x.ravel(),y.ravel()]).T
    z_vals = []
    for rnd_seed in range(1000):
        syn_fcn = SynthObjFcn(domain, rnd_seed, n_features, noise_sigma=0.01)
        z = syn_fcn(xy)
        print rnd_seed
        z_vals.append(z)
        if len(z_vals) % 10 == 0:
            pickle.dump([xy, np.array(z_vals)], open('./data/synthetic/synth.pkl', 'wb'))
    return xy, z
###################################################################################

#### Helper functions for partially missing data points ####
def randomly_mask_entries(X, p_missing):
    # mask entries with probability p_missing
    missing_mask = np.random.rand(*X.shape) < p_missing
    X_incomplete = X.copy()
    X_incomplete[missing_mask] = np.nan
    return X_incomplete, missing_mask


def complete_matrix(X):
    simpute = SoftImpute()
    X_completed = simpute.complete(X)
    return X_completed
###################################################################################


#### Helper functions for loading prior data and saving results ####
def load_data(domain):
    print "Running " + domain
    if domain == 'ag':
        xvals = sio.loadmat('./data/ag/1800_rrt.mat')['standard_selected_thetas']
        xvals = sklearn.preprocessing.StandardScaler().fit_transform(xvals)
        yvals = sio.loadmat('./data/ag/1800_rrt.mat')['standard_selected_rewards']
    elif domain == 'gpb':
        xvals = format_gpb_xvals(sio.loadmat('./data/gpb/thetas.mat'))
        xvals = sklearn.preprocessing.StandardScaler().fit_transform(xvals)
        yvals = sio.loadmat('./data/gpb/1500_rrt.mat')['standard_selected_rewards']
    elif domain == 'synth':
        if os.path.isfile('./data/synthetic/synth.pkl'):
            xvals, yvals = pickle.load(open('./data/synthetic/synth.pkl', 'r'))
        else:
            xvals, yvals = create_synth_data()
    else:
        print "Wrong domain name"
        return -1
    return xvals, yvals


def setup_prior_data(args):
    xvals, score_mat = load_data(args.domain)
    print 'Total score mat shape', score_mat.shape
    test_idx = args.pidx  # rules out the test problem instance; irrelevant for cont domain
    test_vals = score_mat[test_idx, :]
    if args.domain != 'synth':
        Dprior = np.delete(score_mat, test_idx, axis=0)
    Dprior = score_mat

    Nportion = args.Nportion
    n_prior_data = int(Nportion * Dprior.shape[0])
    print 'n prior data %d' % (n_prior_data)
    print 'n prior data %d' % (n_prior_data)
    Dprior = Dprior[:n_prior_data, :]
    return Dprior, xvals, test_vals


def save_results(evaled, ordered, pidx, savedir):
    fname = str(pidx) + '.pkl'
    print "Saving results to ", savedir + fname
    pickle.dump({'evaled': evaled, 'ordered': ordered}, open(savedir + fname, 'wb'))


def make_save_dir(args):
    portion = args.Nportion
    if args.domain == 'synth':
        if args.algorithm != 'zbk':
            dirsave = './test_results/' + args.domain + \
                      '/' + args.algorithm + \
                      '/' + args.bo + '/'
        else:
            dirsave = './test_results/' + args.domain + \
                      '/' + args.algorithm + \
                      '/' + args.bo + \
                      '/portion_' + str(portion) + '/'
    else:
        if args.algorithm == 'rand':
            dirsave = './test_results/' + args.domain + \
                      '/' + 'full' + \
                      '/' + args.algorithm + \
                      '/portion_' + str(portion) + '/'
        else:
            dirsave = './test_results/' + args.domain + \
                  '/' + 'full' + \
                  '/' + args.algorithm + \
                  '/' + args.bo + \
                  '/portion_' + str(portion) + '/'
    dirsave += '/trial_' + str(args.trial) + "/"

    if not os.path.isdir(dirsave):
        os.makedirs(dirsave)

    return dirsave
###################################################################################


def get_optimizer_for_discrete_domain(args, xvals, test_vals, Dprior, obj_fcn):
    """
    :param args: contains configurations for
    :param xvals: evaluated x values
    :param test_vals: test function values
    :param Dprior: prior function evaluations on xvals
    :param obj_fcn: true target function
    :return: optimizer for obj_fcn. It will be either BO or random search.
    """
    N = Dprior.shape[0]
    if args.algorithm == 'zbk' or args.algorithm == 'niw':
        gp = DiscreteGPIterative(Dprior, args.algorithm)
    elif args.algorithm == 'plain':
        gp = StandardDiscreteGP(Dprior, xvals, args.domain)
    elif args.algorithm == 'commonrs':
        gp = CommonRS(Dprior, xvals, args.domain)
    elif args.algorithm == 'rand':
        optimizer = KtimesRandomSearch(obj_fcn, k=1)
    else:
        print "wrong algorithm name"
        sys.exit(-1)

    if args.bo == 'gpucb' and args.algorithm != 'rand':
        acq_fcn = AutomaticZetaUCB(N=N, delta=0.01, gp=gp)
        optimizer = BO(gp, obj_fcn, acq_fcn)
    elif args.bo == 'pi' and args.algorithm != 'rand':
        acq_fcn = ProbImprovement(target_val=max(test_vals), gp=gp)
        optimizer = BO(gp, obj_fcn, acq_fcn)

    return optimizer


def get_optimizer_for_continuous_domain(args, xvals, Dprior, obj_fcn, savedir):
    """
    arguments and return values similar to the discrete domain
    """
    N = Dprior.shape[0]
    if args.algorithm == 'zbk':
        feat_fcn = FeatureFcn(dim_x=xvals.shape[1], n_fcns=N)
        feat_fcn.train(xvals, Dprior, savedir)
        gp = PriorEstContinuousGP(Dprior, args.algorithm, feat_fcn, feat_fcn.Ws)
    elif args.algorithm == 'rand':
        optimizer = KtimesContRandomSearch(obj_fcn, k=1)  # Change this to uniform rand search
    elif args.algorithm == 'commonrs':
        gp = ContinuousCommonRS(Dprior, xvals, args.domain)
    elif args.algorithm == 'plain':
        gp = StandardContinuousGP(Dprior, xvals, args.domain)

    if args.bo == 'gpucb' and args.algorithm != 'rand':
        acq_fcn = AutomaticZetaUCB(N=N, delta=0.01, gp=gp)
        optimizer = BO(gp, obj_fcn, acq_fcn)
    elif args.bo == 'pi' and args.algorithm != 'rand':
        acq_fcn = ProbImprovement(target_val=np.max(Dprior), gp=gp)
        optimizer = BO(gp, obj_fcn, acq_fcn)

    return optimizer


def run_discrete_domain(args, xvals, test_vals, Dprior,  savedir):
    test_idx = args.pidx
    print 'Max value = %f' % max(test_vals)
    p = args.Nportion

    if os.path.isfile(savedir + '/' + str(test_idx) + '.pkl'):
        print 'Already done'
        return

    if args.data_type == 'partial' and args.algorithm == 'zbk':
        Dprior_masked, missing_mask = randomly_mask_entries(Dprior, p)
        Dprior = complete_matrix(Dprior_masked)

    obj_fcn = DiscreteObjFcn(test_vals)
    optimizer = get_optimizer_for_discrete_domain(args, xvals, test_vals, Dprior, obj_fcn)
    n_arms = Dprior.shape[1]
    perform_BO(optimizer, obj_fcn, test_idx, savedir, n_arms)


def run_continuous_domain(args, xvals, Dprior, savedir):
    test_idx = args.pidx

    is_result_file_already_exist = os.path.isfile(savedir + '/' + str(test_idx) + '.pkl')
    if is_result_file_already_exist:
        print 'Already done'
        return

    n_features = 1000
    domain = [[-10., -10.], [10., 10.]]
    obj_fcn = SynthObjFcn(domain, np.random.randint(10000), n_features, noise_sigma=0.01)
    optimizer = get_optimizer_for_continuous_domain(args, xvals, Dprior, obj_fcn, savedir)
    perform_BO(optimizer, obj_fcn, test_idx, savedir, n_arms=np.inf)


def perform_BO(optimizer, obj_fcn, test_idx, savedir, n_arms):
    evaled = []
    vals = []

    n_eval = min(n_arms, 97)
    for (x, y) in optimizer.generate_evals(n_eval):
        evaled.append(x)
        vals.append(obj_fcn(x))
        print 'Best reward', max(vals)
        print '%d/%d' % (len(evaled), n_eval)
        print vals

    ordered = [np.max(vals[0:t]) for t in range(1, len(vals))]
    save_results(evaled, ordered, test_idx, savedir)


def main():
    parser = argparse.ArgumentParser(description='Process openrave domain configurations')
    parser.add_argument('-domain', default='ag') # ag, gpb, or synth
    parser.add_argument('-data_type', default='full')  # full or partial. If partial, -Nportion option is in effect
    parser.add_argument('-pidx', type=int, default=0)  # function to rule out. LOOCV for discrete domains.
    parser.add_argument('-Nportion', type=float, default=1.0) # portion to rule out if we have missing entries
    parser.add_argument('-trial', type=int, default=0)
    parser.add_argument('-algorithm', type=str, default='zbk')  # rand, plain, commonrs, or zbk
    parser.add_argument('-bo', type=str, default='gpucb')  # gpucb or pi
    parser.add_argument('--load_feat', help='loads the feature fcn', action="store_true")
    args = parser.parse_args()

    if (args.algorithm == 'rand' or args.algorithm == 'plain' or args.algorithm == 'commonrs') and args.Nportion != 1.0:
        print 'plain, rand and commonrs only works with portion 1.0'
        return

    Dprior, xvals, test_vals = setup_prior_data(args)
    savedir = make_save_dir(args)

    if args.domain == 'ag' or args.domain == 'gpb':
        run_discrete_domain(args, xvals, test_vals, Dprior, savedir)
    elif args.domain == 'cbelt' or args.domain == 'synth':
        run_continuous_domain(args, xvals, Dprior, savedir)
    else:
        print 'wrong domain name'


if __name__ == '__main__':
    main()
