from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import ScalarFormatter


class CustomSciFormatter(ScalarFormatter):
    """A custom formatter for plot tick labels.

    Restrain tick labels to a given number of decimal places, regardless of float or scientific representation.

    Attributes
    ----------
    fmt: str
        Format string for the tick labels.
    """

    def __init__(self, fmt='%.2f'):
        super(CustomSciFormatter, self).__init__()
        self.fmt = fmt
        self.set_powerlimits((-1, 3))

    def _set_format(self):
        self.format = self.fmt


def init_plot_style():
    """Initialize the plot style for pyplot.
    """
    plt.rcParams.update({'lines.linewidth': 2})
    plt.rcParams.update({'lines.markersize': 8})
    plt.rcParams.update({'lines.markeredgewidth': 2})
    plt.rcParams.update({'axes.labelpad': 20})
    plt.rcParams.update({'xtick.major.width': 2.5})
    plt.rcParams.update({'xtick.major.size': 15})
    plt.rcParams.update({'xtick.minor.size': 10})
    plt.rcParams.update({'ytick.major.width': 2.5})
    plt.rcParams.update({'ytick.major.size': 15})
    plt.rcParams.update({'ytick.minor.size': 10})

    # for font settings see also https://stackoverflow.com/questions/2537868/sans-serif-math-with-latex-in-matplotlib
    plt.rcParams.update({'font.size': 30})
    plt.rcParams.update({'font.family': 'sans-serif'})
    plt.rcParams.update({'text.usetex': True})
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath,amssymb,amsfonts,amsthm}',
        r'\usepackage{siunitx}',  # i need upright \micro symbols, but you need...
        r'\sisetup{detect-all}',  # ...this to force siunitx to actually use your fonts
        r'\usepackage{helvet}',  # set the normal font here
        r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
        r'\sansmath'  # <- tricky! -- gotta actually tell tex to use!
    ]


def plot_histogram(ax: plt.Subplot, data: List, labels: List, n_bins=20, normalization='pmf', log_scale=False,
                   bin_edges=None, tick_fmt='%.2f'):
    """Plot a histogram.

    Plot a histogram with selectable normalization, scale and bin edges, etc. All data vectors share the same bin
    structure and are plotted side by side.

    Parameters
    ----------
    ax : plt.Subplot
        The target matplotlib subplot.
    data : List
        List of data vectors (lists, arrays,..).
    labels : List
        List of data labels (strings).
    n_bins : int
        Number of bins in the histogram.
    normalization : str
        Either 'pmf' for probability mass, 'pdf' for probability density or not normalized for other options.
    log_scale : bool
        Plot data values (x-axis) on a log scale. Bins are spaced logarithmically.
    bin_edges : list
        Optional custom bin edges. If given, `n_bins` is ignored.
    tick_fmt : str
        Format string for the data tick labels (x-axis).
    """
    xmax = -np.infty
    xmin = np.infty
    for d in data:
        xmin = min(xmin, np.min(d))
        xmax = max(xmax, np.max(d))

    if bin_edges is None:
        if log_scale:
            bin_edges = np.logspace(np.log10(xmin), np.log10(xmax), n_bins + 1)
        else:
            bin_edges = np.linspace(xmin, xmax, n_bins + 1)
    else:
        n_bins = len(bin_edges) - 1

    bin_sizes = bin_edges[1:] - bin_edges[:-1]
    bin_centers = bin_edges[:-1] + bin_sizes * 0.5

    plt_bin_widths = bin_sizes * 0.9
    col_widths = plt_bin_widths / len(data)
    plt_bin_centers = bin_centers - 0.5 * plt_bin_widths + col_widths * 0.5

    for i, (d, l) in enumerate(zip(data, labels)):
        counts, _ = np.histogram(d, bins=bin_edges)
        if normalization == 'pmf' or normalization == 'pdf':
            counts = counts / np.sum(counts)  # normalize to get pmf for bins
        if normalization == 'pdf':
            counts /= bin_sizes  # normalize to get pdf

        ax.bar(plt_bin_centers + i * col_widths, counts, width=col_widths, label=l)

    if log_scale:
        ax.set_xscale('log')
    else:
        ax.xaxis.set_minor_locator(FixedLocator(bin_centers))
        ax.set_xticks(bin_centers[::int(np.ceil(n_bins / 10))])
        ax.xaxis.set_major_formatter(CustomSciFormatter(tick_fmt))
