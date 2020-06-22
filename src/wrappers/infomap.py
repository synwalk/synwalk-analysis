import os

import pandas as pd
from clusim.clustering import Clustering


class Infomap:
    """Wrapper class around the infomap/synwalk cmd-line tool.

    ...

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
        if not os.path.exists(self.workspace_path):
            os.mkdir(self.workspace_path)

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
        if not os.path.exists(self.workspace_path):
            os.mkdir(self.workspace_path)

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
