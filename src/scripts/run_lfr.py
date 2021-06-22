import argparse
import os

from src.lfr.run_benchmark import run_benchmark


def run_lfr_k_n_mu(avg_degree_list, n_list, methods, overwrite_results=False, base_path='./'):
    """ Run selected community detection methods on a specified set of generated LFR benchmark graphs. This is a
    convenience wrapper around src.lfr.run_benchmark.run_benchmark for running large-scale benchmarks with multiple
    methods and benchmark configurations.

    Parameters
    ----------
    avg_degree_list : list
        List of average degrees specifying the relevant set of benchmark graphs to be used. Assumes a proper directory
        structure of the dataset, see `base_path`.
    n_list : list
        List of network sizes specifying the relevant set of benchmark graphs to be used. Assumes a proper directory
        structure of the dataset, see `base_path`.
    methods : list
        List of string identifiers of clustering methods that should be run on the benchmark graphs. Available methods
         are 'infomap', 'synwalk', 'walktrap', 'louvain', 'graphtool' and 'label_propagation'.
    overwrite_results : bool
        Overwrites existing clustering files if True. Skips evaluation if False.
    base_path : str
        Path to a directory containing a set of benchmark graphs, ordered in subdirectories.
        E.g. ´benchmark_dir = ../data/lfr_benchmark/<avg_degree>deg/<mu>mu/´.
    """
    base_path = base_path + '/' if base_path[-1] != '/' else base_path

    i = 0
    num_benchmark_sets = len(n_list) * len(avg_degree_list)
    for avg_degree in avg_degree_list:
        for n in n_list:
            i += 1
            print(f'Starting benchmark for benchmark set {i}/{num_benchmark_sets}...')
            benchmark_dir = base_path + f'data/lfr_benchmark/{avg_degree}deg/{n}n/'

            for j, method in enumerate(methods):
                print(f'Testing method {j + 1}/{len(methods)} ({method})...')

                # create results directory if necessary
                results_dir = base_path + 'results/lfr/clustering/' + method + f'/{avg_degree}deg/{n}n/'
                os.makedirs(results_dir, exist_ok=True)

                run_benchmark(benchmark_dir, results_dir, method, overwrite_results=overwrite_results)


def run_lfr_k_mu_n(avg_degree_list, mu_list, methods, overwrite_results=False, base_path='./'):
    """ Run selected community detection methods on a specified set of generated LFR benchmark graphs. This is a
    convenience wrapper around src.lfr.run_benchmark.run_benchmark for running large-scale benchmarks with multiple
    methods and benchmark configurations.

    Parameters
    ----------
    avg_degree_list : list
        List of average degrees specifying the relevant set of benchmark graphs to be used. Assumes a proper directory
        structure of the dataset, see `base_path`.
    mu_list : list
        List of mixing parameters specifying the relevant set of benchmark graphs to be used. Assumes a proper directory
        structure of the dataset, see `base_path`.
    methods : list
        List of string identifiers of clustering methods that should be run on the benchmark graphs. Available methods
         are 'infomap', 'synwalk', 'walktrap', 'louvain', 'graphtool' and 'label_propagation'.
    overwrite_results : bool
        Overwrites existing clustering files if True. Skips evaluation if False.
    base_path : str
        Path to a directory containing a set of benchmark graphs, ordered in subdirectories.
        E.g. ´benchmark_dir = ../data/lfr_benchmark/<avg_degree>deg/<mu>mu/´.
    """
    base_path = base_path + '/' if base_path[-1] != '/' else base_path

    i = 0
    num_benchmark_sets = len(mu_list) * len(avg_degree_list)
    for avg_degree in avg_degree_list:
        for mu in mu_list:
            i += 1
            print(f'Starting benchmark for benchmark set {i}/{num_benchmark_sets}...')
            benchmark_dir = base_path + f'data/lfr_benchmark/{avg_degree}deg/{int(mu * 100)}mu/'

            for j, method in enumerate(methods):
                print(f'Testing method {j + 1}/{len(methods)} ({method})...')

                # create results directory if necessary
                results_dir = base_path + 'results/lfr/clustering/' + method + f'/{avg_degree}deg/{int(mu * 100)}mu/'
                os.makedirs(results_dir, exist_ok=True)

                run_benchmark(benchmark_dir, results_dir, method, overwrite_results=overwrite_results)


def main(argd):
    """ Run selected community detection methods on generated LFR benchmark graphs.

    Parameters
    ----------
    argd : dict
        Dictionary of parsed command line arguments.
    """
    if argd['network_sizes'] is not None:
        run_lfr_k_n_mu(argd['avg_degrees'][0], argd['network_sizes'][0], argd['methods'][0], argd['overwrite_results'],
                       argd['base_path'])
    else:
        run_lfr_k_mu_n(argd['avg_degrees'][0], argd['mixing_params'][0], argd['methods'][0], argd['overwrite_results'],
                       argd['base_path'])


# parse arguments when run from shell
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Run community detection methods on generated LFR benchmark graphs.')
    parser.add_argument('--avg-degrees', action='append', nargs='+', type=int,
                        help='List of average degrees of the benchmark graphs')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--network-sizes', action='append', nargs='+', type=int,
                       help='List of network-sizes of the benchmark graphs')
    group.add_argument('--mixing-params', action='append', nargs='+', type=float,
                       help='List of mixing-params of the benchmark graphs')
    parser.add_argument('-m', '--methods', action='append', nargs='+', type=str,
                        help='List of methods to be benchmarked.')
    parser.add_argument('--overwrite-results', action='store_true')
    parser.add_argument('--base-path', type=str, default='./', help='Base path of the project directory')

    args = parser.parse_args()
    main(vars(args))
