from clusim.clustering import Clustering
from igraph import Graph


def read_graph(filepath, format='edgelist') -> Graph:
    """Reads a graph from an edge list file and returns an igraph Graph object.

    Parameters
    ----------
    filepath : str
        Path to the edge list file.
    format : str
        Format of the input file, e.g. 'edgelist', 'pajek',...

    Returns
    ------
    Graph
        An igraph Graph object.
    """
    graph = Graph.Read(filepath, format=format)
    graph.to_undirected()  # igraph returns a directed graph by default
    graph.simplify(combine_edges='sum')

    # handle one-based vertex numbering: if edge list numbers vertices starting from 1,
    # the first vertex (vertex '0') will be an artifact
    if graph.vs[0].degree() == 0:
        graph.delete_vertices([0])

    return graph


def walktrap(filepath):
    """Wrapper around igraph's walktrap implementation.

    Parameters
    ----------
    filepath : str
        Path to a edge list file.

    Returns
    ------
    Clustering
        A clusim Clustering object.
    """
    graph = read_graph(filepath)
    member_list = graph.community_walktrap().as_clustering().membership
    clu = Clustering().from_membership_list(member_list)
    return clu.relabel_clusters_by_size()


def spinglass(filepath):
    """Wrapper around igraph's spinglass implementation.

    Parameters
    ----------
    filepath : str
        Path to a edge list file.

    Returns
    ------
    Clustering
        A clusim Clustering object.
    """
    graph = read_graph(filepath)
    member_list = graph.community_spinglass().membership
    clu = Clustering().from_membership_list(member_list)
    return clu.relabel_clusters_by_size()


def label_propagation(filepath):
    """Wrapper around igraph's label propagation implementation.

    Parameters
    ----------
    filepath : str
        Path to a edge list file.

    Returns
    ------
    Clustering
        A clusim Clustering object.
    """
    graph = read_graph(filepath)
    member_list = graph.community_label_propagation().membership
    clu = Clustering().from_membership_list(member_list)
    return clu.relabel_clusters_by_size()


def louvain(filepath):
    """Wrapper around igraph's multilevel implementation.

    Parameters
    ----------
    filepath : str
        Path to a edge list file.

    Returns
    ------
    Clustering
        A clusim Clustering object.
    """
    graph = read_graph(filepath)
    member_list = graph.community_multilevel().membership
    clu = Clustering().from_membership_list(member_list)
    return clu.relabel_clusters_by_size()


def edge_betweenness(filepath):
    """Wrapper around igraph's edge betweenness implementation.

    Parameters
    ----------
    filepath : str
        Path to a edge list file.

    Returns
    ------
    Clustering
        A clusim Clustering object.
    """
    graph = read_graph(filepath)
    member_list = graph.community_edge_betweenness().as_clustering().membership
    clu = Clustering().from_membership_list(member_list)
    return clu.relabel_clusters_by_size()
