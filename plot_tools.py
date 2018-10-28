import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cPickle as pickle
def init_plotting():
    #plt.rcParams['figure.figsize'] = (10, 4)
    #plt.rcParams['figure.facecolor'] = 'w'
    plt.rcParams['font.size'] = 20
    #plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']
    plt.rcParams.update({'figure.autolayout': True})
    '''
    plt.rcParams['errorbar.capsize'] = 5
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.minor.size'] = 5
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['xtick.minor.width'] = 2
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['ytick.minor.width'] = 2
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'center left'
    '''
    #plt.rcParams['axes.linewidth'] = 2
    #plt.rcParams['axes.grid'] = False
    
    #plt.gca().spines['right'].set_color('none')
    #plt.gca().spines['top'].set_color('none')
    #plt.gca().xaxis.set_ticks_position('bottom')
    #plt.gca().yaxis.set_ticks_position('left')
    #plt.gca().spines['left'].set_visible(True)
    #plt.gca().spines['bottom'].set_visible(True)

    matplotlib.rc('lines', lw=3, mew=3)
    sns.set_style("ticks")

def save_obj(obj, filenm):
    with open(filenm, 'wb') as outputf:
        pickle.dump(obj, outputf, pickle.HIGHEST_PROTOCOL)
def load_obj(filenm):
    with open(filenm, 'rb') as inputf:
        return pickle.load(inputf)

def plot_res(filenm):
    init_plotting()
    plt.clf()
    alphas = [1e-5, 1e-4, 3e-4, 7e-4, 1e-3, 0.005, 0.01, 0.05, 0.07, 0.1]
    nfeatures = [10, 50, 100, 500, 1000]
    labels = ['GPy'] + [ ('alpha=' + str(a)) for a in alphas] + [('Nfeature=' + str(f)) for f in nfeatures ]
    bigres = load_obj(filenm)
    bigdiff = []
    for funccnt in range(0,100):
        funcres = bigres[funccnt]
        alldiff = [[a[0] for a in res] for res in funcres]

        if 0:
            alldiff = np.array(alldiff)
            plt.clf()
            colors=sns.color_palette("Set1", n_colors=8)
            for i in [0] + range(11, 16):
                plt.plot(range(2,100), alldiff[:, i], label=labels[i])
            plt.legend()
            plt.savefig('test_functions' + '/' + 'testfunc_' + str(funccnt) + '_nf' +'.eps',format='eps', dpi=1000)

            plt.clf()
            #colors = plt.cm.rainbow(np.linspace(0,1,11))
            colors=sns.color_palette("Set1", n_colors=8)
            cnt = 0
            for i in [0] + range(3,10):
                plt.plot(range(2,100), alldiff[:, i], label=labels[i], c=colors[cnt])
                cnt += 1
            plt.legend()
            plt.savefig('test_functions' + '/' + 'testfunc_' + str(funccnt) + '_l1' +'.eps',format='eps', dpi=1000)
        bigdiff.append(alldiff)
    bigdiff = np.array(bigdiff)
    meandiff = np.mean(bigdiff, axis=0)
    print meandiff
    print np.std(bigdiff, axis=0)
    colors=sns.color_palette("Set1", n_colors=8)
    print meandiff.shape
    for i in [0] + range(11, 16):
        plt.plot(range(2,100), meandiff[:, i], label=labels[i])
    plt.legend()
    plt.show()
    cnt = 0

    for i in [0] + range(3,10):
        plt.plot(range(2,100), meandiff[:, i], label=labels[i], c=colors[cnt])
        cnt += 1
    plt.legend()
    plt.show()

    meandiff = np.std(bigdiff, axis=0)
    colors=sns.color_palette("Set1", n_colors=8)
    print meandiff.shape
    for i in [0] + range(11, 16):
        plt.plot(range(2,100), meandiff[:, i], label=labels[i])
    plt.legend()
    plt.show()
    cnt = 0

    for i in [0] + range(3,10):
        plt.plot(range(2,100), meandiff[:, i], label=labels[i], c=colors[cnt])
        cnt += 1
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_res('res3.dat')