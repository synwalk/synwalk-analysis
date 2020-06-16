import networkx as nx
import numpy as np
from clusim.clustering import Clustering
from networkx.generators.community import LFR_benchmark_graph


def generate_lfr_params(num_nodes, mixing_param):
    """Generates a dictionary of LFR benchmark parameters for a given network size and mixing parameter.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    mixing_param : float
        Global mixing parameter for the graph.

    Returns
    ------
    dict
        A dictionary of LFR benchmark parameters.
    """

    # generic LFR parameters
    max_community = int(0.2 * num_nodes)
    min_community = int(max_community * 0.25)
    max_degree = int(max_community * 0.3)
    min_degree = int(min_community * 0.4)
    gamma = 3.5  # Power law exponent for the degree distribution
    beta = 1.1  # Power law exponent for the community size distribution

    params = {'n': num_nodes, 'tau1': gamma, 'tau2': beta, 'mu': mixing_param, 'min_degree': min_degree,
              'max_degree': max_degree, 'max_community': max_community, 'min_community': min_community}
    return params


def generate_benchmark_graphs(storage_dir, num_graphs, params):
    """Generates and stores LFR benchmark graphs and clustering information.

    Parameters
    ----------
    storage_dir : str
        Directory where the generated graphs are stored. Path should end with '/'.
    num_graphs : int
        Number of graphs to generate.
    params : dict
        LFR benchmark params as returned by 'generate_lfr_params()'
    """
    # generate random seeds for benchmark generation reproducibility
    int32 = np.iinfo(np.int32)

    # generate 'num_graphs' benchmark graphs
    seeds = set()
    graph_cnt = 0
    while graph_cnt < num_graphs:
        seed = np.random.randint(int32.max)
        if seed in seeds:
            continue

        # generate benchmark graph
        try:
            graph = LFR_benchmark_graph(**params, seed=seed)
        except nx.ExceededMaxIterations as err:
            # print(f'Failed to generate network: ', err)
            continue

        # update seeds + increase graph count
        seeds.add(seed)
        graph_cnt += 1

        # store graph
        nx.write_edgelist(graph, storage_dir + 'graph_' + str(seed) + '.txt', data=False)

        # extract and store ground truth communities
        communities = {frozenset(graph.nodes[v]['community']) for v in graph}
        clu = Clustering()
        clu.from_cluster_list(communities)
        clu.relabel_clusters_by_size()
        clu.save(storage_dir + 'clustering_' + str(seed) + '.json')

    # store random seeds
    with open(storage_dir + 'seeds.txt', mode='w') as f:
        f.write(f'# The benchmark graphs in this directory are generated '
                f'with the following {num_graphs} random seeds:\n')
        f.write('\n'.join(map(str, seeds)))
