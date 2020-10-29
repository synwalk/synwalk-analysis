from collections import defaultdict
from typing import Dict, Callable

CleanerDict = Dict[str, Callable]


class DataCleaner:
    """Data cleaner for converting raw network files into clean edge list files.

    Attributes
    ----------
    raw_dir : str
        Directory containing subfolders with the raw network data.
    cleaned_dir : str
        Output directory for the clean edge list file.
    cleaners : dict
        Dictionary of network identifiers and their respective cleaning methods.
    """

    def __init__(self, raw_dir, cleaned_dir):
        """
        Parameters
        ----------
        raw_dir : str
            Directory containing the raw network file.
        cleaned_dir : str
            Output directory for the clean edge list file.
        """
        self.raw_dir = raw_dir
        self.cleaned_dir = cleaned_dir
        self.cleaners: CleanerDict = dict({'dblp': self.clean_dblp,
                                           'facebook': self.clean_facebook,
                                           'github': self.clean_github,
                                           'lastfm-asia': self.clean_lastfm_asia,
                                           'pennsylvania-roads': self.clean_pennsylvania_roads,
                                           'wordnet': self.clean_wordnet})

    def clean(self, network: str):
        """Cleans a raw network file and generates a cleaned edge list file.

        Parameters
        ----------
        network : str
            For options/implemented cleaners see self.cleaners.keys()
        """
        if network not in self.cleaners:
            print('No cleaning method for network \'' + network + '\' available!')
            return
        cleaner = self.cleaners[network]
        cleaner(network)

    def clean_all(self):
        """Cleans all raw networks with implemented cleaners.
        """
        for network, cleaner in self.cleaners.items():
            cleaner(network)

    def clean_dblp(self, network: str):
        """Cleans the raw dblp network data.

        Parameters
        ----------
        network : str
            Directory name in self.raw_dir where the raw network file is located. Will also be used for the cleaned
            file name.
        """
        raw_file = self.raw_dir + network + '/com-dblp.ungraph.txt'
        cleaned_file = self.cleaned_dir + network + '.txt'
        self.clean_edgelist(raw_file, cleaned_file, drop_first=4)

    def clean_facebook(self, network: str):
        """Cleans the raw facebook network data.

        Parameters
        ----------
        network : str
            Directory name in self.raw_dir where the raw network file is located. Will also be used for the cleaned
            file name.
        """
        raw_file = self.raw_dir + network + '/musae_facebook_edges.csv'
        cleaned_file = self.cleaned_dir + network + '.txt'
        self.clean_edgelist(raw_file, cleaned_file, drop_first=1, sep=',')

    def clean_github(self, network: str):
        """Cleans the raw github network data.

        Parameters
        ----------
        network : str
            Directory name in self.raw_dir where the raw network file is located. Will also be used for the cleaned
            file name.
        """
        raw_file = self.raw_dir + network + '/musae_git_edges.csv'
        cleaned_file = self.cleaned_dir + network + '.txt'
        self.clean_edgelist(raw_file, cleaned_file, drop_first=1, sep=',')

    def clean_lastfm_asia(self, network: str):
        """Cleans the raw lastfm-asia network data.

        Parameters
        ----------
        network : str
            Directory name in self.raw_dir where the raw network file is located. Will also be used for the cleaned
            file name.
        """
        raw_file = self.raw_dir + network + '/lastfm_asia_edges.csv'
        cleaned_file = self.cleaned_dir + network + '.txt'
        self.clean_edgelist(raw_file, cleaned_file, drop_first=1, sep=',')

    def clean_pennsylvania_roads(self, network: str):
        """Cleans the raw pennsylvania roads network data.

        Parameters
        ----------
        network : str
            Directory name in self.raw_dir where the raw network file is located. Will also be used for the cleaned
            file name.
        """
        raw_file = self.raw_dir + network + '/roadNet-PA.txt'
        cleaned_file = self.cleaned_dir + network + '.txt'
        self.clean_edgelist(raw_file, cleaned_file, drop_first=4)

    def clean_wordnet(self, network: str):
        """Cleans the raw wordnet network data.

        Parameters
        ----------
        network : str
            Directory name in self.raw_dir where the raw network file is located. Will also be used for the cleaned
            file name.
        """
        raw_file = self.raw_dir + network + '/out.wordnet-words'
        cleaned_file = self.cleaned_dir + network + '.txt'
        self.clean_edgelist(raw_file, cleaned_file, drop_first=2)

    @classmethod
    def clean_edgelist(cls, raw_file, cleaned_file, drop_first=0, sep=None):
        """Cleans a raw edgelist file.

        Reads the raw file line per line and drops the first `drop_first` lines. Then writes edges into the target file.

        Parameters
        ----------
        raw_file : str
            Path to the raw edgelist file.
        cleaned_file : str
            Path to the target/cleaned edgelist file.
        drop_first : int
            Number of lines to drop from the raw file.
        sep : str
            Separator between nodes. Defaults to None.
        """
        node_map = defaultdict(lambda: len(node_map))
        with open(raw_file, 'r') as rf:
            with open(cleaned_file, 'w') as cf:
                # drop the first x lines
                for _ in range(drop_first):
                    rf.readline()

                # read adjacency list from raw file and correct to zero-based node numbering
                line = rf.readline().strip()
                while line:
                    node1, node2 = list(map(int, line.split(sep=sep)))
                    cf.write(f'{node_map[node1]} {node_map[node2]}\n')
                    line = rf.readline().strip()
