import networkx as nx
import numpy as np
from clusim.clustering import Clustering
from networkx import Graph
from sklearn.metrics import adjusted_mutual_info_score


def ami_score(clu_file_pred, clu_file_true, graph_file=None):
    """Compute adjusted mutual (AMI) information between two clusterings.

    This method computes the AMI using the scikit-learn method 'adjusted_mutual_info_score',
    because it is significantly faster than it's clusim counterpart.

    Parameters
    ----------
    clu_file_pred : str
        Path to clusim Clustering file (.json) containing the predicted clustering.
    clu_file_true : str
        Path to clusim Clustering file (.json) containing the ground truth clustering.
    graph_file : str
        Interface placeholder. Not used.

    Returns
    ------
    float
        Adjusted mutual information between the two input clusterings.
    """
    clu_pred = Clustering().load(clu_file_pred)
    clu_true = Clustering().load(clu_file_true)
    labels_pred = clu_pred.to_membership_list()
    labels_true = clu_true.to_membership_list()
    return adjusted_mutual_info_score(labels_pred, labels_true, average_method='arithmetic')


def synwalk_error(clu_file_pred, clu_file_true, graph_file):
    """Compute the difference in synwalk objective for two clusterings.

    This method computes the AMI using the scikit-learn method 'adjusted_mutual_info_score',
    because it is significantly faster than it's clusim counterpart.

    Parameters
    ----------
    clu_file_pred : str
        Path to clusim Clustering file (.json) containing the predicted clustering.
    clu_file_true : str
        Path to clusim Clustering file (.json) containing the ground truth clustering.
    graph_file : str
        Path to the underlying graph file in edge list format.

    Returns
    ------
    float
        Relative error in synwalk objective of the predicted clustering w.r.t ground truth.
    """
    # load graph and clusterings
    clu_pred = Clustering().load(clu_file_pred)
    clu_true = Clustering().load(clu_file_true)
    graph = nx.read_edgelist(graph_file, nodetype=int)

    node_flows, node_transitions = compute_node_distributions(graph)
    objective_pred = synwalk_objective(graph, clu_pred, node_flows, node_transitions)
    objective_true = synwalk_objective(graph, clu_true, node_flows, node_transitions)
    return objective_pred / objective_true - 1.0


def plogq(p, q):
    """Compute p * log(q) where log has base 2.

        Edge case: 0 *  log(0) = 0

        Parameters
        ----------
        p : float
            First function argument.
        q : float
            Second function argument.

        Returns
        ------
        float
            p * log(q)
    """
    if p < 1e-18:
        return 0.0

    if q < 1e-18:
        print(f'Unexpected zero operand in plogq: p={p}, q={q}\n.')
        return -np.inf

    return p * np.log2(q)


def plogp(p):
    """Compute p * log(p) where log has base 2.

        Edge case: 0 *  log(0) = 0

        Parameters
        ----------
        p : float
            Function argument.

        Returns
        ------
        float
            p * log(p)
    """
    return plogq(p, p)


def compute_node_distributions(graph: Graph):
    """Compute the stationary distribution and transition probabilities over nodes for a given graph.

        Parameters
        ----------
        graph : Graph
            A networkx graph object.

        Returns
        ------
        numpy array, numpy array
            The stationary distribution over nodes and the transition probability matrix.
    """
    # stationary distribution over nodes
    p = np.fromiter(nx.pagerank_scipy(graph, alpha=0.99).values(), dtype=float)
    # transition probability matrix
    P = nx.google_matrix(graph, alpha=0.99, nodelist=sorted(graph))
    return p, P


def synwalk_objective(graph: Graph, clu: Clustering, node_flows=None, node_transitions=None):
    """Compute the synwalk objective for a given graph and clustering.

        Parameters
        ----------
        graph : Graph
            A networkx Graph object.
        clu : Clustering
            A clusim Clustering object.
        node_flows: numpy array
            The stationary distribution over nodes. Computed from graph if None.
        node_transitions: numpy array
            The transition probability matrix of the graph. Computed from graph if None.

        Returns
        ------
        float
            The resulting synwalk objective.
    """
    # compute node distributions if not given
    if node_flows is None:
        # stationary distribution over nodes
        node_flows = np.fromiter(nx.pagerank_scipy(graph, alpha=0.99).values(), dtype=float)
    if node_transitions is None:
        # transition probability matrix
        node_transitions = nx.google_matrix(graph, alpha=0.99, nodelist=sorted(graph))

    # compute module distributions
    membership_list = clu.to_membership_list()
    module_flows = np.zeros((clu.n_clusters,))  # stationary distribution over modules
    module_stay_flows = np.zeros((clu.n_clusters,))  # joint probabilities for staying within a specific module
    for node in graph:
        module_idx = membership_list[node]
        module_flows[module_idx] += node_flows[node]
        for neighbor in graph.neighbors(node):
            if membership_list[neighbor] == membership_list[node]:
                module_stay_flows[module_idx] += node_flows[node] * node_transitions[node, neighbor]

    # compute synwalk objective
    objective = 0.0
    for module_flow, module_stay in zip(module_flows, module_stay_flows):
        # check corner cases
        epsilon = 1e-18  # vicinity threshold for numerical stability
        if (module_flow <= epsilon) or (module_flow + epsilon >= 1.0):
            continue

        module_exit = module_flow - module_stay  # joint probability of leaving a specific module
        objective += plogp(module_stay) \
                     - 2.0 * plogq(module_stay, module_flow) \
                     + plogp(module_exit) \
                     - plogq(module_exit, module_flow) \
                     - plogq(module_exit, 1.0 - module_flow)

    return objective
