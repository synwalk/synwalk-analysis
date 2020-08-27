import csv

import networkx as nx
import numpy as np
from clusim.clustering import Clustering
from networkx.generators.community import LFR_benchmark_graph

from src.wrappers.lfr_generator import LFRGenerator


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


def generate_benchmark_graphs_networkx(storage_dir, num_graphs, params):
    """Generates and stores LFR benchmark graphs and clustering information.

    This method generated LFR graphs using the networkx function 'LFR_benchmark_graph'. WARNING: this generator
    is FLAWED, as it will not produce graphs that match the desired mixing parameter!

    Parameters
    ----------
    storage_dir : str
        Directory where the generated graphs are stored. Path should end with '/'.
    num_graphs : int
        Number of graphs to generate.
    params : dict
        LFR benchmark params as returned by 'generate_lfr_params()'
    """
    # maximum random seed
    max_seed = np.iinfo(np.int32).max

    # generate 'num_graphs' benchmark graphs
    seeds = set()
    graph_cnt = 0
    while graph_cnt < num_graphs:
        # generate random seeds in range [0, max_seed] for benchmark generation reproducibility
        seed = np.random.randint(max_seed)
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


def generate_benchmark_graphs(output_dir, num_graphs, graph_size, mixing_parameter, avg_degree):
    """Generates and stores LFR benchmark graphs and clustering information.

    Parameters
    ----------
    output_dir : str
        Directory where the generated graphs are stored. Path must end with '/'.
    num_graphs : int
        Number of graphs to generate.
    graph_size : int
        Number of nodes in the generated graphs.
    mixing_parameter : float
        Global mixing parameter of the generated graphs.
    avg_degree : int
        Average node degree.
    """
    # maximum random seed
    max_seed = np.iinfo(np.int32).max

    generator = LFRGenerator()

    # generate 'num_graphs' benchmark graphs
    seeds = set()
    graph_cnt = 0
    while graph_cnt < num_graphs:
        # generate random seeds in range [0, max_seed] for benchmark generation reproducibility
        seed = np.random.randint(max_seed)
        if seed in seeds:
            continue

        # generate benchmark graph
        status = generator.generate(graph_size, mixing_parameter, avg_degree, output_dir, seed)
        if status:
            print(f'Failed to generate network with exit code {status}.')
            continue

        # update seeds + increase graph count
        seeds.add(seed)
        graph_cnt += 1

        # extract and store ground truth communities as Clusim clustering
        with open(output_dir + f'membership_{seed}.txt') as f:
            data = csv.DictReader(f, delimiter='\t', fieldnames=['node', 'community'])
            membership_list = [int(row['community']) for row in data]

        clu = Clustering()
        clu.from_membership_list(membership_list)
        clu.relabel_clusters_by_size()
        clu.save(output_dir + 'clustering_' + str(seed) + '.json')

    # store random seeds
    with open(output_dir + 'seeds.txt', mode='w') as f:
        f.write(f'# The benchmark graphs in this directory are generated with the parameter string ' +
                generator.get_param_string(graph_size, mixing_parameter, avg_degree) +
                f' and with the following {num_graphs} random seeds:\n')
        f.write('\n'.join(map(str, seeds)))
