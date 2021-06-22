import csv

import graph_tool
from clusim.clustering import Clustering
from graph_tool import Graph
from graph_tool.inference import minimize_blockmodel_dl
from graph_tool.stats import remove_self_loops


def read_graph(filepath) -> Graph:
    """Reads a graph from an edge list file and returns a graph_tool Graph object.

    Parameters
    ----------
    filepath : str
        Path to the edge list file.

    Returns
    ------
    Graph
        A graph_tool Graph object.
    """
    with open(filepath) as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(128))
        delim = dialect.delimiter
    graph = graph_tool.load_graph_from_csv(filepath, directed=False, skip_first=False, csv_options={'delimiter': delim})
    remove_self_loops(graph)

    return graph


def sbm_inference(filepath, num_models=1):
    """Wrapper around graph_tool's SBM inference method.

    Parameters
    ----------
    filepath : str
        Path to a edge list file.
    num_models : int
        Number of SBM models to infer and select the best model from.

    Returns
    ------
    Clustering
        A clusim Clustering object.
    """
    graph = read_graph(filepath)
    best_state = minimize_blockmodel_dl(graph, deg_corr=True)
    min_entropy = best_state.entropy()
    for _ in range(num_models - 1):
        state = minimize_blockmodel_dl(graph, deg_corr=True)
        entropy = state.entropy()
        if entropy < min_entropy:
            best_state = state
            min_entropy = entropy

    member_list = best_state.get_blocks().get_array()
    clu = Clustering().from_membership_list(member_list)
    return clu.relabel_clusters_by_size()
