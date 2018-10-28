from plot_utils import *
import argparse
import pickle
import itertools

algo_names = {
    'zbk': 'PEM-BO',
    'zbk_biased': 'PEM-BO',
    'plain': 'Plain',
    'commonrs': 'TLSM-BO',
    'rand': 'rand'
}
bo_names = {
    'gpucb': 'UCB',
    'pi': 'PI'
}


# def load_test_results( fcn_type,domain, dtype, entry_type, search_fcn, algo, p_miss=None)

def load_test_results(alg, BO, domain, Nportion):
    if alg == 'rand':
        folder_name = './test_results/' + domain + '/full/' + alg + '/portion_1.0/'
    elif alg == 'zbk':
        folder_name = './test_results/' + domain + '/full/' + alg + '/' + BO + '/portion_' + str(Nportion) + '/'
    else:
        folder_name = './test_results/' + domain + '/full/' + alg + '/' + BO + '/portion_1.0/'

    results = []
    if domain == 'ag':
        n_evals = 161
    elif domain == 'gpb':
        n_evals = 30
    else:
        n_evals = 99

    print folder_name
    for fin in os.listdir(folder_name):
        if fin.find('hdf5') != -1:
            continue
        if fin.find('Ws.') != -1:
            continue
        evaled = np.array(pickle.load(open(folder_name + fin, 'r'))['ordered'])
        results.append(evaled[0:n_evals])
    results = np.array(results)
    return results


def load_test_results_for_continuous_domain(alg, BO):
    if alg != 'zbk':
        dirsave = './test_results/synth' + \
                  '/' + alg + \
                  '/' + str(BO) + '/'
    else:
        dirsave = './test_results/synth' + \
                  '/' + alg + \
                  '/' + str(BO) + \
                  '/portion_1.0/'

    n_evals = 97
    results = []
    for trial_folder in os.listdir(dirsave):
        if trial_folder.find('trial') == -1:
            continue
        for fin in os.listdir(dirsave + trial_folder):
            if fin.find('hdf5') != -1:
                continue
            if fin.find('Ws.') != -1:
                continue
            evaled = np.array(pickle.load(open(dirsave + trial_folder + '/' + fin, 'r'))['ordered'])
            results.append(evaled[0:n_evals])
    return np.array(results)


def main():
    parser = argparse.ArgumentParser(description='Process openrave domain configurations')
    parser.add_argument('-domain', default='synth')
    parser.add_argument('-Nportion', type=float, default=1.0)
    args = parser.parse_args()

    palette = itertools.cycle(sns.color_palette())
    rand_color = next(palette)
    plain_color = next(palette)
    zbk_color = next(palette)
    tlsm_color = next(palette)

    name = 'Random'
    if args.domain == 'synth':
        results = load_test_results_for_continuous_domain('rand', 'gpucb')
    else:
        results = load_test_results('rand', 'gpucb', args.domain, args.Nportion)

    print len(results)
    plot_score_vs_n_evals(range(1, results.shape[1] + 1), results, name, color=rand_color)

    bo_list = ['gpucb']

    alg_list = ['plain', 'zbk', 'commonrs']
    for alg, plot_color in zip(alg_list, [plain_color, zbk_color, tlsm_color]):
        for bo in bo_list:
            if args.domain == 'synth':
                results = load_test_results_for_continuous_domain(alg, bo)
            else:
                results = load_test_results(alg, bo, args.domain, args.Nportion)
            print len(results)
            name = algo_names[alg] + '-' + bo_names[bo] if alg != 'rand' else algo_names[alg]
            plot_score_vs_n_evals(range(1, results.shape[1] + 1), results, name, color=plot_color)
    plt.savefig('./plots/' + args.domain + '.pdf', dpi=1000)


if __name__ == '__main__':
    main()
