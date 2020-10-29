from collections import defaultdict

import numpy as np
from clusim.clustering import Clustering
from igraph import Graph


def contingency_table(clu1: Clustering, clu2: Clustering):
    """Builds a contingency table for two given clusterings.

    Parameters
    ----------
    clu1 : Clustering
        First clustering.
    clu2 : Clustering
        Second clustering.

    Returns
    ------
    numpy.ndarray
        A (m x n) 2D array of integers representing the contingency table between clu1 and clu2,
        where m is the number of clusters in clu1 and n is the number of clusters in clu2.
    """
    ctable = np.zeros((clu1.n_clusters, clu2.n_clusters), dtype=int)
    for id1, members1 in clu1.to_clu2elm_dict().items():
        for id2, members2 in clu2.to_clu2elm_dict().items():
            ctable[id1, id2] = len(set(members1).intersection(set(members2)))
    return ctable


def match_clusters(clu_true: Clustering, clu_pred: Clustering):
    """Matches cluster labels between two clusterings.

    The matching is based on the contingency table between the two clusterings. In every step, a cluster mapping is
    added by greedily picking the clusters with maximum overlap from the contingency table. The procedure stops whenever
    all clusters of any input clustering have a match. The residual clusters obtain no mapping.

    Parameters
    ----------
    clu_true : Clustering
        First clustering.
    clu_pred : Clustering
        Second clustering.

    Returns
    ------
    dict
        A dictionary mapping from clu_true cluster ids to clu_pred cluster ids.
    """
    ctable = contingency_table(clu_true, clu_pred)
    mapping = {}
    for _ in range(min(clu_true.n_clusters, clu_pred.n_clusters)):
        ctrue, cpred = None, None
        while ctrue is None or ctrue in mapping.keys() or cpred in mapping.values():
            argmax_flattened = np.argmax(ctable)
            ctrue, cpred = np.unravel_index(argmax_flattened, ctable.shape)
            ctable[ctrue, cpred] = -1
        mapping[ctrue] = cpred

    return mapping


def normalized_local_degrees(graph: Graph, clu: Clustering):
    """Computes the normalized local degrees for a given graph and clustering.

    Parameters
    ----------
    graph : Graph
        The graph.
    clu : Clustering
        The clustering.

    Returns
    ------
    numpy.array
        An array containing the normalized local degrees.
    """
    graph.vs['node_id'] = graph.vs.indices
    node_degrees = graph.degree()
    nlds = np.zeros((len(graph.vs)))
    for i, cluster_list in enumerate(clu.to_cluster_list()):
        cluster = graph.subgraph(cluster_list)
        max_cluster_edges = (len(cluster.vs) * (len(cluster.vs) - 1)) / 2.0
        # avoid devide by zero
        if max_cluster_edges == 0:
            continue
        nlds[cluster.vs['node_id']] = [node_degrees[i] / max_cluster_edges for i in cluster.vs['node_id']]

    return nlds


def get_misclassfied_nodes(clu_true: Clustering, clu_pred: Clustering):
    """Returns the correct and misclassified nodes of a predicted clustering w.r.t. some ground truth.

    We align the predicted clusters to their respective ground truth clusters. Nodes in the intersection of the
    ground truth modules with their aligned counterparts are considered as correctly classified, whereas the residual
    set of nodes form the group of misclassified nodes.

    Parameters
    ----------
    clu_true : Clustering
        The ground truth clustering.
    clu_pred : Clustering
        The predicted clustering.

    Returns
    ------
    set, set
        The sets of correctly classified and misclassified nodes.
    """
    mapping = match_clusters(clu_true, clu_pred)

    correctly_classified = set()
    for ctrue, cpred in mapping.items():
        true_cluster = set(clu_true.to_clu2elm_dict()[ctrue])
        matched_cluster = set(clu_pred.to_clu2elm_dict()[cpred])
        correctly_classified.update(true_cluster.intersection(matched_cluster))

    misclassified = set(range(clu_pred.n_elements))
    misclassified.difference_update(correctly_classified)

    return correctly_classified, misclassified


def matched_memberships(clu_true: Clustering, clu_pred: Clustering):
    """Returns a matched membership list for a predicted clustering.

    Returns the membership list for a predicted clustering when matched to a ground truth clustering. Clusters without
    a matching ground truth cluster are aggregated into a single "residual" cluster.

    Parameters
    ----------
    clu_true : Clustering
        The ground truth clustering.
    clu_pred : Clustering
        The predicted clustering.

    Returns
    ------
    list
        The matched membership list with a residual cluster.
    """
    residual_nodes_cluster = max(clu_true.clusters) + 1
    mapping = match_clusters(clu_true, clu_pred)  # mapping from true to predicted clusters
    inv_mapping = defaultdict(lambda: residual_nodes_cluster)
    inv_mapping.update({v: k for k, v in mapping.items()})  # mapping from predicted to true clusters

    modified_membership_list = [inv_mapping[cpred] for cpred in clu_pred.to_membership_list()]
    return residual_nodes_cluster, modified_membership_list


def conductance(subgraph: Graph):
    """Computes the conductance for a given cluster/subgraph.

    Parameters
    ----------
    subgraph : Graph
        The subgraph.

    Returns
    ------
    float
        The conductance.
    """
    node_degrees = subgraph.vs['degree']
    int_degrees = np.array(subgraph.degree())
    ext_degrees = node_degrees - int_degrees
    return np.sum(ext_degrees) / (2.0 * np.sum(int_degrees) + np.sum(ext_degrees)) if np.sum(ext_degrees) > 0 else 0.0


def cut_ratio(graph: Graph, subgraph: Graph):
    """Computes the cut ratio for a given cluster/subgraph.

    Parameters
    ----------
    subgraph : Graph
        The subgraph.

    Returns
    ------
    float
        The cut ratio.
    """
    node_degrees = subgraph.vs['degree']
    int_degrees = np.array(subgraph.degree())
    ext_degrees = node_degrees - int_degrees
    return np.sum(ext_degrees) / (len(subgraph.vs) * (len(graph.vs) - len(subgraph.vs)))


def get_cluster_properties(graph: Graph, clu: Clustering, feature='size', ignore_smaller_than=3):
    """Computes cluster properties for a given graph and clustering.

    Parameters
    ----------
    graph : Graph
        The graph.
    clu : Clustering
        The clustering.
    feature : str
        Cluster property to be analyzed.
    ignore_smaller_than : int
        Minimum cluster size for clusters to be considered in the analysis.

    Returns
    ------
    list
        List of computed cluster properties.
    """
    # annotate nodes with their degree as this is needed for computing the conductance and cut ratio
    graph.vs['degree'] = graph.degree()

    statistics = []
    for nodes in clu.to_cluster_list():
        if len(nodes) < ignore_smaller_than:
            continue

        cluster = graph.subgraph(nodes)
        if feature == 'size':
            statistics.append(len(nodes))
        elif feature == 'density':
            statistics.append(cluster.density())
        elif feature == 'clustering_coefficient':
            statistics.append(cluster.transitivity_undirected(mode='zero'))
        elif feature == 'conductance':
            statistics.append(conductance(cluster))
        elif feature == 'cut_ratio':
            statistics.append(cut_ratio(graph, cluster))
        else:
            print('Invalid feature!')
            return []

    return statistics


def get_node_properties(graph: Graph, clu: Clustering, feature='mixing_parameter', ignore_smaller_than=3):
    """Computes node properties for a given graph and clustering.

    Parameters
    ----------
    graph : Graph
        The graph.
    clu : Clustering
        The clustering.
    feature : str
        Node property to be analyzed.
    ignore_smaller_than : int
        Minimum cluster size for member nodes to be considered in the analysis.

    Returns
    ------
    list
        List of computed node properties.
    """
    # annotate nodes with their degree
    graph.vs['degree'] = graph.degree()

    statistics = []
    for nodes in clu.to_cluster_list():
        if len(nodes) < ignore_smaller_than:
            continue

        cluster = graph.subgraph(nodes)
        if feature == 'mixing_parameter':
            int_degrees = np.array(cluster.degree())
            statistics.extend([1.0 - kint / k for kint, k in zip(int_degrees, cluster.vs['degree']) if k > 2])
            pass
        elif feature == 'nld':
            max_cluster_edges = (len(cluster.vs) * (len(cluster.vs) - 1)) / 2.0
            statistics.extend(np.array(cluster.vs['degree']) / max_cluster_edges)
        else:
            print('Invalid feature!')
            return []

    return statistics
