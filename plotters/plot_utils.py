import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import seaborn as sns


def load_test_results(domain, dtype):
    if dtype == 'partial':
        assert p is not None


def plot_score_vs_n_evals(xvals, yvals, algo, color):
    ax = sns.tsplot(yvals, xvals, ci=90, condition=algo, color=color)
    ax.set_xlim(-xvals[-1] * 0.03, )
    plt.xlabel('Number of evaluations', fontsize=14)
    plt.ylabel('Rewards', fontsize=14)
    plt.legend(fontsize=14)

def plot_learn_curve(xvals, vals, algo, color):
    ax = sns.tsplot(vals, xvals, ci=90, condition=algo, color=color)
    ax.set_xlim(0, )
    # ax.set_xticklabels(xvals)
    plt.xlabel('Portions of N',fontsize=14)
    plt.ylabel('Rewards',fontsize=14)
    plt.legend(fontsize=14)
