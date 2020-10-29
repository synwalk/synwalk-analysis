from collections import namedtuple
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from clusim.clustering import Clustering
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
        self.set_powerlimits((-2, 3))

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
                   bin_edges=None, tick_fmt='%.2f', xmin=None, xmax=None):
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
    xmin : float
        Leftmost bin edge. If none the minimum entry in all data is taken instead.
    xmax : float
        Rightmost bin edge. If none the maximum entry in all data is taken instead.
    """
    if xmin is None:
        xmin = np.infty
        for d in data:
            xmin = min(xmin, np.min(d))

    if xmax is None:
        xmax = -np.infty
        for d in data:
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


# compute node positions for communities placed in a circular manner
def community_layout(clu: Clustering):
    """Create a community layout for plotting.

    Creates a layout where nodes are grouped according to the given clustering and clusters are arranged in a
    circular fashion.

    Parameters
    ----------
    clu : Clustering
        The clustering.

    Returns
    ------
    dict
        Dictionary of node positions. The keys are the node ids and the values are tuples of x/y coordinates.
    """
    CartCoordinate = namedtuple('CartCoordinate', ['x', 'y'])

    cluster_lists = clu.to_cluster_list()
    clu_sizes = np.asarray([len(cluster_list) for cluster_list in cluster_lists])

    clu_sectors = clu_sizes * 2.0 * np.pi / clu.n_elements  # sector (in rad) each cluster occupies
    clu_radii = clu_sizes * 1.0 / clu.n_elements  # radii of the clusters

    # compute cluster centers on a circle with radius 1
    cum_sectors = 0
    clu_centers = []  # list of cluster center polar coordinates
    for i, clu_sector in enumerate(clu_sectors):
        axis = clu_sector / 2 + cum_sectors
        cum_sectors += clu_sector
        clu_center = CartCoordinate(np.cos(axis), np.sin(axis))
        clu_centers.append(clu_center)

    # place nodes randomly in a neighborhood of their respective cluster center
    node_positions = {}
    for radius, size, center, node_ids in zip(clu_radii, clu_sizes, clu_centers, cluster_lists):
        node_radii = 2.0 * radius * np.sqrt(np.random.uniform(0, 1, size))
        node_angles = np.random.uniform(0, 2 * np.pi, size)
        for node_id, r, phi in zip(node_ids, node_radii, node_angles):
            node_positions[node_id] = (r * np.cos(phi) + center[0], r * np.sin(phi) + center[1])

    return node_positions


def plot_graph(graph_file, membership_list, node_positions, ax, misclassified_nodes=None, cmap=plt.get_cmap('Set3')):
    """Plot a graph for a given layout and clustering.

    Parameters
    ----------
    graph_file : str
        Path to the graph file in edge list format.
    membership_list : list
        List of cluster memberships of each node.
    node_positions : dict
        List of coordinate tuples of node positions.
    ax : plt.Subplot
        The target matplotlib subplot.
    misclassified_nodes : list
        List of misclassified nodes.
    cmap : colormap
        The colormap to use for node coloring.
    """
    if misclassified_nodes is None:
        misclassified_nodes = []

    # plotting params
    node_size = 250
    edge_width = 0.05

    graph = nx.read_edgelist(graph_file, nodetype=int)
    node_colors = cmap(membership_list)

    # convenience wrapper for drawing
    def draw_nodes(nodes, node_shape='o', border=False):
        lw = 2.5 if border else 0
        nx.draw_networkx_nodes(graph, pos=node_positions, ax=ax, node_color=node_colors[nodes], nodelist=nodes,
                               node_size=node_size, node_shape=node_shape, edgecolors='black', linewidths=lw)

    # clear frame and ticks
    ax.set_frame_on(False)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_xticks([])

    # draw edges first
    nx.draw_networkx_edges(graph, node_positions, ax=ax, width=edge_width, edge_color='gray', alpha=0.5)

    # then draw correct and then misclassified nodes
    correctly_classified_nodes = set(graph.nodes)
    correctly_classified_nodes.difference_update(misclassified_nodes)

    draw_nodes(list(correctly_classified_nodes))
    draw_nodes(misclassified_nodes, border=True)
