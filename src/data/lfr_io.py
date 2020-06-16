import os
import re


def get_benchmark_files(benchmark_dir):
    """Extract benchmark file paths from a given directory.

    Parameters
    ----------
    benchmark_dir : str
        Path to the directory containing the benchmark files. The directory should contain pairs of
        graph files (graph_<seed>.txt) and clustering files (clustering_<seed>.json).

    Returns
    ------
    dict, dict
        Two dictionaries (graph files, clustering files), mapping from seeds to the corresponding file paths.
    """

    # read files from benchmark folder
    files = [entry for entry in os.scandir(benchmark_dir) if entry.is_file()]

    graph_files = dict()
    clustering_files = dict()
    # iterate over files and store filepath according to content
    for file in files:
        filename = os.path.basename(file)
        tokens = re.split('[_.]', filename)  # split into [identifier, seed, type]
        if tokens[0] == 'graph':
            graph_files[int(tokens[1])] = os.path.abspath(file)
        if tokens[0] == 'clustering':
            clustering_files[int(tokens[1])] = os.path.abspath(file)

    return graph_files, clustering_files
