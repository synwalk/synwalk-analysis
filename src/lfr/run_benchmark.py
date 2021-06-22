import os

from src.data.lfr_io import get_benchmark_files
from src.lfr.benchmark_results import *
from src.wrappers.igraph import *
from src.wrappers.infomap import Infomap
from src.wrappers.graph_tool import sbm_inference
from src.wrappers.metrics import ami_score, synwalk_error

# method dict
methods = {'infomap': Infomap().infomap,
           'synwalk': Infomap().synwalk,
           'walktrap': walktrap,
           'louvain': louvain,
           'graphtool': sbm_inference,
           'label_propagation': label_propagation}

# metrics dict
metrics = {'ami': ami_score,
           'synwalk_error': synwalk_error}


def run_benchmark(benchmark_dir, results_dir, clustering_method, max_samples=100, overwrite_results=False):
    """Run a given clustering method on LFR benchmark graphs.

    We assume benchmark graphs with a fixed network size and variable mixing parameter mu. The results are stored as
    clusim Clusterings (clustering_<seed>_<clustering_method>.json). This enables flexibility in later evaluation.

    Parameters
    ----------
    benchmark_dir : str
        Path to a directory containing a set of benchmark graphs, ordered in subdirectories.
        E.g. ´benchmark_dir = ../data/lfr_benchmark/15deg/300n/´.
    results_dir : str
        Path to a directory where to store the resulting clusterings. These will be stored in subdirectories reflecting
        the directory structure in benchmark_dir.
        E.g. ´results_dir = ../results/lfr_benchmark/clustering/infomap/15deg/300n/´.
    clustering_method : str
        String identifier for a clustering method that should be applied to the benchmark graphs. Available clustering
        methods are 'infomap', 'synwalk', 'walktrap', 'louvain', 'graphtool' and 'label_propagation'.
    max_samples : int
        Maximum number of benchmark graphs processed from each subdirectory in ´benchmark_dir´.
    overwrite_results : bool
        Overwrites existing clustering files if True. Skips evaluation if False.
    """
    # check if clustering method is valid
    if clustering_method not in methods:
        print('Invalid clustering method: ' + clustering_method)
        return

    method = methods[clustering_method]

    # iterate over subfolders in the benchmark directory
    folders = [entry for entry in os.scandir(benchmark_dir) if entry.is_dir()]
    for folder_idx, folder in enumerate(folders):
        # create results storage dir if necessary
        var_dir = results_dir + os.path.basename(folder) + '/'
        os.makedirs(var_dir, exist_ok=True)

        # iterate over graph files and evaluate
        graph_files, _ = get_benchmark_files(folder)
        num_samples = min(len(graph_files), max_samples)
        for sample_idx, seed in enumerate(graph_files.keys()):
            if sample_idx >= num_samples:
                break

            # assemble results file path and check existence
            clu_path = var_dir + 'clustering_' + str(seed) + '_' + clustering_method + '.json'
            if not overwrite_results and os.path.exists(clu_path):
                print('\rResults in ' + clu_path + ' already exist. Skipping evaluation...')
                continue

            # evaluate method and store clustering results
            clu = method(graph_files[seed])
            clu.save(clu_path)

        print(f'\rCompleted {(folder_idx + 1)}/{len(folders)} data points.', end='')
    print('')


def evaluate_clustering_results(benchmark_dir, results_dir, metric='ami', variable='mu') -> BenchmarkResults:
    """Run a given clustering method on a set of LFR benchmark graphs.

    Parameters
    ----------
    benchmark_dir : str
        Path to a directory containing a set of ground truth clusterings, ordered in subdirectories.
        E.g. ´benchmark_dir = ../data/lfr_benchmark/15deg/300n/´. The subdirectories are named according to ´<x>mu/´ or
        ´<x>n/´, where <x> are the first two decimal digits of the mixing parameter in the first case. In the latter
        case, <x> is the network size.
    results_dir : str
        Path to a directory with the same structure as benchmark_dir containing the predicted clusterings.
        E.g. ´results_dir = ../results/lfr_benchmark/clustering/infomap/15deg/300n/´.
    metric : str
        Clustering metric used to compare the predicted and the ground truth clusterings. Available metrics are 'ami'
        and 'synwalk_error'.
    variable : str
        The variable parameter of the benchmark set reflected in the subdirectory naming, i.e. 'n' for the network size
        or 'mu' for the mixing parameter.

    Returns
    ------
    BenchmarkResults
        A BenchmarkResults object containing the achieved scores (according to metric) for every graph processed.
    """
    if metric not in metrics:
        print('Invalid metric - cancelling evaluation.')
        return BenchmarkResults()

    if variable not in {'mu', 'n'}:
        print('Invalid variable - cancelling evaluation.')
        return BenchmarkResults()

    metric = metrics[metric]
    results = BenchmarkResults()

    # iterate over subfolders in the benchmark directory
    folders = [entry for entry in os.scandir(results_dir) if entry.is_dir()]
    for folder_idx, folder_pred in enumerate(folders):
        folder_name = os.path.basename(folder_pred)
        var = float(folder_name[0:2]) / 100 if variable == 'mu' else int(folder_name[:-1])
        folder_true = benchmark_dir + folder_name + '/'

        # get clustering file paths
        _, clu_files_pred = get_benchmark_files(folder_pred)
        graph_files, clu_files_true = get_benchmark_files(folder_true)
        # iterate over files and evaluate
        dp = DataPoint(var, len(clu_files_pred))
        for seed, clu_file_pred in clu_files_pred.items():
            assert (seed in clu_files_true)  # there should be a ground truth
            score = metric(clu_file_pred, clu_files_true[seed], graph_files[seed])
            dp.add_sample(seed, score)

        print(f'\rCompleted {(folder_idx + 1)}/{len(folders)} data points.', end='')
        results.add_datapoint(dp)
    print('')
    return results
