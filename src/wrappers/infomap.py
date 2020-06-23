import os

import networkx as nx
import numpy as np
import pandas as pd
from clusim.clustering import Clustering


class Infomap:
    """Wrapper class around the infomap/synwalk cmd-line tool.

    Attributes
    ----------
    workspace_path : str
        Path to a temporary workspace directory. Directory is created if does not exist.
    infomap_path : str
        Path to the Infomap binary (should include synwalk extension).
    """

    def __init__(self, workspace_path='../workspace/', infomap_path='../infomap/Infomap'):
        """Read infomap/synwalk results from .tree file and return a clusim clustering.

            Parameters
            ----------
            workspace_path : str
                Path to a temporary workspace directory. Directory is created if does not exist.
            infomap_path : str
                Path to the Infomap binary (should include synwalk extension).
        """
        self.workspace_path = workspace_path
        self.infomap_path = infomap_path

    def infomap(self, input_file, additional_args=''):
        """Wrapper around infomap cmd line tool. Runs infomap on the given input file.

            Parameters
            ----------
            input_file : str
                Path to the input file in edge list format.
            additional_args : str
                Additional arguments for the infomap cmd line tool.

            Returns
            ------
            Clustering
                A clusim Clustering object holding the results from infomap.
        """
        os.makedirs(self.workspace_path, exist_ok=True)

        # construct argument string
        args = ' --two-level --undirected --zero-based-numbering' \
               ' --input-format link-list --out-name infomap_out'
        args += additional_args

        # run infomap
        os.system(self.infomap_path + ' ' + input_file + ' ' + self.workspace_path + ' ' + args)

        # read clustering from output file
        clu = Infomap.read_communities_from_tree_file(self.workspace_path + 'infomap_out.tree')
        return clu

    def synwalk(self, input_file, additional_args=''):
        """Wrapper around synwalk cmd line tool. Runs synwalk on the given input file.

            Parameters
            ----------
            input_file : str
                Path to the input file in edge list format.
            additional_args : str
                Additional arguments for the synwalk cmd line tool.

            Returns
            ------
            Clustering
                A clusim Clustering object holding the results from synwalk.
        """
        os.makedirs(self.workspace_path, exist_ok=True)

        # construct argument string
        args = ' --two-level --undirected --zero-based-numbering --altmap' \
               ' --input-format link-list --out-name synwalk_out'
        args += additional_args

        # run synwalk
        os.system(self.infomap_path + ' ' + input_file + ' ' + self.workspace_path + ' ' + args)

        # read clustering from output file
        clu = Infomap.read_communities_from_tree_file(self.workspace_path + 'synwalk_out.tree')
        return clu

    @classmethod
    def read_communities_from_tree_file(cls, filepath) -> Clustering:
        """Read infomap/synwalk results from .tree file and return a clusim clustering.

            Parameters
            ----------
            filepath : str
                Path to the .tree file.

            Returns
            ------
            Clustering
                A clusim Clustering object holding the results from the .tree file.
        """
        # read dataframe from .tree file + clean data
        df = pd.read_csv(filepath, sep=' ', header=1)
        df.columns = ['community', 'flow', 'name', 'node', 'trash']
        df = df.drop(['flow', 'trash'], axis=1)
        df['community'] = df['community'].apply(lambda x: x.split(':')[0])

        # extract community membership list
        node_ids = df['name'].tolist()
        min_label = int(df['community'].min())
        labels = df['community'].apply(lambda x: int(x) - min_label).tolist()
        membership_list = [cluster_id for _, cluster_id in sorted(zip(node_ids, labels))]

        # generate clustering from membership list
        clu = Clustering()
        clu.from_membership_list(membership_list)
        clu.relabel_clusters_by_size()
        return clu

    @classmethod
    def plogq(cls, p, q):
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

    @classmethod
    def plogp(cls, p):
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
        return cls.plogq(p, p)

    @classmethod
    def compute_synwalk_objective(cls, graph_file, clustering_file):
        """Compute the synwalk objective for a given graph and clustering.

            Parameters
            ----------
            graph_file : str
                Path to a graph file in edge list format.
            clustering_file : str
                Path to clusim clustering file (.json).

            Returns
            ------
            float
                The resulting synwalk objective.
        """
        # read graph and clustering
        graph = nx.read_edgelist(graph_file, nodetype=int)
        clu = Clustering().load(clustering_file)
        membership_list = clu.to_membership_list()

        # compute node distributions
        p = np.fromiter(nx.pagerank_scipy(graph, alpha=0.99).values(),
                        dtype=float)  # stationary distribution over nodes
        P = nx.google_matrix(graph, alpha=0.99, nodelist=sorted(graph))  # transition probability matrix

        # compute module distributions
        module_flows = np.zeros((clu.n_clusters,))  # stationary distribution over modules
        module_stay_flows = np.zeros((clu.n_clusters,))  # joint probabilities for staying within a specific module
        for node in graph:
            module_idx = membership_list[node]
            module_flows[module_idx] += p[node]
            for neighbor in graph.neighbors(node):
                if membership_list[neighbor] == membership_list[node]:
                    module_stay_flows[module_idx] += p[node] * P[node, neighbor]

        # compute synwalk objective
        synwalk_objective = 0.0
        for module_flow, module_stay in zip(module_flows, module_stay_flows):
            # check corner cases
            epsilon = 1e-18  # vicinity threshold for numerical stability
            if (module_flow <= epsilon) or (module_flow + epsilon >= 1.0):
                continue

            module_exit = module_flow - module_stay  # joint probability for leaving a specific module
            synwalk_objective += cls.plogp(module_stay) \
                                 - 2.0 * cls.plogq(module_stay, module_flow) \
                                 + cls.plogp(module_exit) \
                                 - cls.plogq(module_exit, module_flow) \
                                 - cls.plogq(module_exit, 1.0 - module_flow)

        return synwalk_objective
