import os
import sys
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

from src.lfr.benchmark_results import BenchmarkResults
from src.utils.plotting import init_plot_style

ResultInfo = namedtuple('ResultInfo', ['path', 'label', 'linestyle'])


def plot_ami_vs_mu(avg_degree, n, save_figure=True, plot_uncertainty=True, plot_legend=True,
                   base_dir='../results/lfr/ami/', fig_dir='../figures/'):
    """Plot LFR results as a function of the mixing parameter.

    Plot benchmark AMI results for a fixed average degree and network size as a function of the mixing parameter.

    Parameters
    ----------
    avg_degree : int
        Fixed average degree of the benchmark series.
    n : int
        Fixed network size of the benchmark series.
    save_figure : bool
        If True, save the figure as .pdf in ´fig_dir´
    plot_uncertainty : bool
        If True, plot uncertainty as a margin of one standard deviation.
    base_dir : str
        Directory containing the LFR results as .pkl files.
    fig_dir : str
        Output directory for storing generated figures
    """

    # assemble a list of results to plot (path + plot formatting) -> add to list what you want to plot
    path_suffix = f'/{avg_degree}deg/{n}n.pkl'
    info = [ResultInfo(base_dir + 'infomap' + path_suffix, 'Infomap', 'x:'),
            ResultInfo(base_dir + 'synwalk' + path_suffix, 'SynWalk', '^:'),
            ResultInfo(base_dir + 'walktrap' + path_suffix, 'Walktrap', 's:')]

    # plot the results
    plt.figure(figsize=(12, 9))

    for entry in info:
        results = BenchmarkResults.load(entry.path)
        xdata = results.get_var_list()
        ydata = results.get_mean_scores()
        plt.plot(xdata, ydata, entry.linestyle, label=entry.label)

        if plot_uncertainty:
            upper = np.clip(ydata + results.get_score_std(), 0.0, 1.0)
            lower = np.clip(ydata - results.get_score_std(), 0.0, 1.0)
            plt.fill_between(xdata, upper, lower, alpha=0.25)

    plt.xlabel(r'Mixing parameter, $\mu$')
    plt.ylabel(r'AMI, $\mathcal{I}^{adj}(\mathcal{Y}, \mathcal{Y}_{true})$')
    if plot_legend:
        plt.legend(loc='best')
    plt.tight_layout()

    # save figure as .pdf
    if save_figure:
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = fig_dir + f'lfr_ami_vs_mu_{avg_degree}k_{n}n.pdf'
        plt.savefig(fig_path, dpi=600, format='pdf')
        plt.close()


def plot_ami_vs_rho(avg_degree, mu, save_figure=True, plot_uncertainty=True, plot_legend=True,
                    base_dir='../results/lfr/ami/', fig_dir='../figures/'):
    """Plot LFR results as a function of the network density.

    Plot benchmark AMI results for a fixed average degree and mixing parameter as a function of the network density.

    Parameters
    ----------
    avg_degree : int
        Fixed average degree of the benchmark series.
    mu : float
        Fixed mixing parameter of the benchmark series.
    save_figure : bool
        If True, save the figure as .pdf in ´fig_dir´
    plot_uncertainty : bool
        If True, plot uncertainty as a margin of one standard deviation.
    base_dir : str
        Directory containing the LFR results as .pkl files.
    fig_dir : str
        Output directory for storing generated figures
    """
    # assemble a list of results to plot (path + plot formatting) -> add to list what you want to plot
    path_suffix = f'/{avg_degree}deg/{int(100 * mu)}mu.pkl'
    info = [ResultInfo(base_dir + 'infomap' + path_suffix, 'Infomap', 'x:'),
            ResultInfo(base_dir + 'synwalk' + path_suffix, 'SynWalk', '^:'),
            ResultInfo(base_dir + 'walktrap' + path_suffix, 'Walktrap', 's:')]

    # plot the results
    plt.figure(figsize=(12, 9))

    for entry in info:
        results = BenchmarkResults.load(entry.path)
        n = results.get_var_list()
        xdata = 2.0 * avg_degree / (n - 1)  # network density
        ydata = results.get_mean_scores()
        plt.plot(xdata, ydata, entry.linestyle, label=entry.label)

        if plot_uncertainty:
            upper = np.clip(ydata + results.get_score_std(), 0.0, 1.0)
            lower = np.clip(ydata - results.get_score_std(), 0.0, 1.0)
            plt.fill_between(xdata, upper, lower, alpha=0.25)

    plt.xlabel(r'Network density, $\rho$')
    plt.ylabel(r'AMI, $\mathcal{I}^{adj}(\mathcal{Y}, \mathcal{Y}_{true})$')
    plt.xlim([0.0014, 0.4])
    plt.ylim([-0.05, 1.05])
    plt.semilogx()
    if plot_legend:
        plt.legend(loc=5)
    plt.tight_layout()

    # save figure as .pdf
    if save_figure:
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = fig_dir + f'lfr_ami_vs_nd_{avg_degree}k_{int(100 * mu)}mu.pdf'
        plt.savefig(fig_path, dpi=600, format='pdf')
        plt.close()


def plot_synwalk_error_vs_mu(avg_degree, n, save_figure=True, plot_legend=True,
                             base_dir='../results/lfr/synwalk_error/',
                             fig_dir='../figures/'):
    """Plot the mismatch in Synwalk objective as a function of the mixing parameter.

    Plot the relative deviation of the Synwalk objective for a predicted clustering and the ground truth clustering of
    a graph.

    Parameters
    ----------
    avg_degree : int
        Fixed average degree of the benchmark series.
    n : int
        Fixed network size of the benchmark series.
    save_figure : bool
        If True, save the figure as .pdf in ´fig_dir´
    base_dir : str
        Directory containing the LFR results as .pkl files.
    fig_dir : str
        Output directory for storing generated figures
    """
    # assemble a list of results to plot (path + plot formatting) -> add to list what you want to plot
    path_suffix = f'/{avg_degree}deg/{n}n.pkl'
    info = [ResultInfo(base_dir + 'synwalk' + path_suffix, 'SynWalk', 'r+:')]

    # plot the results
    plt.figure(figsize=(12, 9))

    lines = []
    labels = []
    for entry in info:
        results = BenchmarkResults.load(entry.path)
        # plot mean deviation
        xdata = results.get_var_list()
        ydata = results.get_mean_scores()
        plt.plot([0.2, 0.8], [0, 0], 'k')
        ln, = plt.plot(xdata, ydata, entry.linestyle, label=entry.label)
        lines.append(ln)
        labels.append(entry.label + ' - Mean Mismatch')
        # plot individual sample deviations
        for dp in results.datapoints:
            ln = plt.scatter([dp.var] * dp.num_samples, dp.scores, marker='x', c='b', alpha=0.3)

        lines.append(ln)
        labels.append(entry.label + ' - Synwalk Objective Mismatch')

    plt.xlabel(r'Mixing parameter, $\mu$')
    plt.ylabel(r'Synwalk Deviation, $\frac{J(\mathcal{Y}) - J(\mathcal{Y}_{true})}{J(\mathcal{Y}_{true})}$')
    if plot_legend:
        plt.legend(lines, labels, loc=0)
    plt.tight_layout()

    # save figure as .pdf
    if save_figure:
        os.makedirs(fig_dir, exist_ok=True)
        fig_path = fig_dir + f'lfr_synwalk_err_{avg_degree}k_{n}n.pdf'
        plt.savefig(fig_path, dpi=600, format='pdf')
        plt.close()


def main(argv):
    """
    Generate all figures with AMI results.

    Parameters
    ----------
    argv : list
        List holding the command line arguments. The first argument is the directory containing the AMI results. The
        second argument is gives the target directory for the figures.
    """
    if len(argv) < 2:
        print('Usage: ' + os.path.basename(__file__) + ' [results_dir] [fig_dir]')
        return

    # get results_dir from arguments
    results_dir = argv[0]
    if results_dir[-1] != '/':
        results_dir += '/'

    # get fig_dir from arguments
    fig_dir = argv[1]
    if fig_dir[-1] != '/':
        fig_dir += '/'

    init_plot_style()

    # plot ami vs mu
    avg_degrees = [15, 25, 50]  # average node degrees
    network_sizes = [300, 600, 1200]  # network sizes

    for avg_degree in avg_degrees:
        for n in network_sizes:
            plot_ami_vs_mu(avg_degree, n, base_dir=results_dir, fig_dir=fig_dir)

    # plot ami vs rho
    avg_degrees = [15, 25, 50]  # average node degrees
    mixing_parameters = [0.35, 0.45, 0.55]  # network sizes

    for avg_degree in avg_degrees:
        for mu in mixing_parameters:
            plot_ami_vs_rho(avg_degree, mu, base_dir=results_dir, fig_dir=fig_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
