import os

from clusim.clustering import Clustering

from src.data.lfr_io import get_benchmark_files
from src.lfr.benchmark_results import *
from src.wrappers.metrics import ami_score


def run_benchmark(benchmark_path, clustering_method, max_samples=100, metric=ami_score) -> BenchmarkResults:
    """Run a given clustering method on LFR benchmark graphs.

    We assume benchmark graphs with a fixed network size and variable mixing parameter mu.

    Parameters
    ----------
    benchmark_path : str
        Path to a directory containing subdirectories named according to <xx>mu. The subdirectories
        contain benchmark graphs where <xx> represents the first two decimals of their mixing parameter.
    clustering_method : callable
        Clustering method that should be applied to the benchmark graphs. The function must accept a single
        argument, i.e. the path to an edge list file.
    max_samples : int
        Maximum number of benchmark graphs processed from each subdirectory.
    metric : callable
        Clustering metric used to compare the clustering results to the ground truth clustering. The function
        must accept two clusim Clustering objects as arguments.

    Returns
    ------
    BenchmarkResults
        A BenchmarkResults object containing the achieved scores (according to metric) for every graph processed.
    """
    results = BenchmarkResults()
    clu_true = Clustering()

    folders = [entry for entry in os.scandir(benchmark_path) if entry.is_dir()]
    for folder_idx, folder in enumerate(folders):
        mu = float(os.path.basename(folder)[0:2]) / 100

        graph_files, clustering_files = get_benchmark_files(folder)
        num_samples = min(len(graph_files), max_samples)
        dp = DataPoint(mu, num_samples)
        for sample_idx, seed in enumerate(graph_files.keys()):
            if sample_idx >= num_samples:
                break
            clu = clustering_method(graph_files[seed])
            clu_true.load(clustering_files[seed])
            score = metric(clu, clu_true)
            dp.add_sample(seed, score)

        print(f'\rCompleted {(folder_idx + 1)}/{len(folders)} data points.', end='')
        results.add_datapoint(dp)
    print('')
    return results
