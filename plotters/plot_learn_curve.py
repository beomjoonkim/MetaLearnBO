from plot_utils import *
import argparse
import pickle
import itertools
from plotters.plot_vs_n_eval import algo_names, bo_names

# def load_test_results( fcn_type,domain, dtype, entry_type, search_fcn, algo, p_miss=None)
"""
algo_names ={
'zbk':'Pebo',
'plain':'Plain',
'commonrs':'Tlsmbo'
}
bo_names={
'gpucb':'UCB',
'pi':'PI'
}
"""


def load_test_results(alg, domain, portions, bo):
    folder_name = './test_results/' \
                  + domain + '/full/' \
                  + alg + '/' \
                  + bo + '/'
    result_list = []
    for portion in portions:
        results = []
        if (alg != 'rand') and (alg != 'plain') and (alg != 'commonrs'):
            p_folder = folder_name + '/portion_' + str(portion) + '/'
        else:
            p_folder = folder_name + '/portion_1.0/'
        if domain == 'synth':
            if alg.find('zbk') != -1:
                p_folder += '/n_output_100/trial_1/'
            else:
                p_folder += '/trial_1/'
        if domain == 'cbelt':
            if alg == 'zbk':
                p_folder += '/n_output_200/trial_1/'
            elif alg == 'plain':
                p_folder += '/n_output_200/trial_1/'
            else:
                p_folder += '/trial_1/'

        print p_folder
        for fin in os.listdir(p_folder):
            if fin.find('.pkl') == -1: continue
            if fin.find('Ws.pkl') != -1: continue
            evaled = np.array(pickle.load(open(p_folder + fin, 'r'))['ordered'])
            results.append(evaled[5])
            if len(results) > 49:
                break
        print len(results)
        result_list.append(results)
    return np.array(result_list).T

def load_test_results_for_continuous_domain(alg, BO, portions):
    if alg != 'zbk':
        dirsave = './test_results/synth' + \
                  '/' + alg + \
                  '/' + str(BO) + '/'
    else:
        dirsave = './test_results/synth' + \
                  '/' + alg + \
                  '/' + str(BO) + '/'

    n_evals = 97

    results_across_portions = []
    for portion in portions:
        results_across_trials = []
        if alg == 'zbk':
            folder_name = dirsave + '/portion_' + str(portion) + '/'
        else:
            folder_name = dirsave
        for trial_folder in os.listdir(folder_name):
            if trial_folder.find('trial') == -1:
                continue
            for fin in os.listdir(folder_name + trial_folder):
                if fin.find('hdf5') != -1:
                    continue
                if fin.find('Ws.') != -1:
                    continue
                evaled = np.array(pickle.load(open(folder_name + trial_folder + '/' + fin, 'r'))['ordered'])
                results_across_trials.append(evaled[10])
        results_across_portions.append(results_across_trials)

    return np.array(results_across_portions).T

def main():
    parser = argparse.ArgumentParser(description='Process openrave domain configurations')
    parser.add_argument('-domain', default='synth')
    parser.add_argument('-data_type', default='full')
    args = parser.parse_args()

    palette = itertools.cycle(sns.color_palette())
    rand_color = next(palette)
    plain_color = next(palette)
    zbk_color = next(palette)
    tlsm_color = next(palette)

    if args.domain == 'ag':
        portions = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
    elif args.domain == 'gpb':
        portions = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
    else:
        portions = [0.1, 0.3, 0.5, 0.7, 0.9]
    alg_list = ['zbk', 'plain']
    bo_list = ['gpucb']
    for bo in bo_list:
        for alg, color in zip(alg_list, [zbk_color, plain_color]):
            if args.domain == 'synth':
                results = load_test_results_for_continuous_domain(alg, bo, portions)
            else:
                results = load_test_results(alg, args.domain, portions, bo)
            print [np.mean(r) for r in results]
            print np.mean(results, axis=0)
            print len(results)
            plot_learn_curve(portions, results, algo_names[alg] + '-' + bo_names[bo], color=color)
    plt.savefig('./plots/l_curve_' + args.domain + '.pdf')


if __name__ == '__main__':
    main()
