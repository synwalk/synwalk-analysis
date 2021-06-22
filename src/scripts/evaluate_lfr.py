import argparse
import os

from src.lfr.run_benchmark import evaluate_clustering_results


def eval_lfr_k_n_mu(avg_degree_list, n_list, methods, metric='ami', base_path='./'):
    base_path = base_path + '/' if base_path[-1] != '/' else base_path

    i = 0
    num_benchmark_sets = len(n_list) * len(avg_degree_list)
    for avg_degree in avg_degree_list:
        for n in n_list:
            i += 1
            print(f'Evaluating predicted clusterings for benchmark set {i}/{num_benchmark_sets}...')
            benchmark_dir = base_path + f'data/lfr_benchmark/{avg_degree}deg/{n}n/'

            for j, method in enumerate(methods):
                print(f'Evaluating method {j + 1}/{len(methods)} ({method})...')
                pred_dir = base_path + 'results/lfr/clustering/' + method + f'/{avg_degree}deg/{n}n/'

                results = evaluate_clustering_results(benchmark_dir, pred_dir, metric, variable='mu')

                results_dir = base_path + 'results/lfr/' + metric + '/' + method + f'/{avg_degree}deg/'
                os.makedirs(results_dir, exist_ok=True)
                results.save(results_dir + f'{n}n.pkl')


def eval_lfr_k_mu_n(avg_degree_list, mu_list, methods, metric='ami', base_path='./'):
    base_path = base_path + '/' if base_path[-1] != '/' else base_path

    i = 0
    num_benchmark_sets = len(mu_list) * len(avg_degree_list)
    for avg_degree in avg_degree_list:
        for mu in mu_list:
            i += 1
            print(f'Evaluating predicted clusterings for benchmark set {i}/{num_benchmark_sets}...')
            benchmark_dir = base_path + f'data/lfr_benchmark/{avg_degree}deg/{int(100 * mu)}mu/'

            for j, method in enumerate(methods):
                print(f'Evaluating method {j + 1}/{len(methods)} ({method})...')
                pred_dir = base_path + 'results/lfr/clustering/' + method + f'/{avg_degree}deg/{int(100 * mu)}mu/'

                results = evaluate_clustering_results(benchmark_dir, pred_dir, metric, variable='n')

                results_dir = base_path + 'results/lfr/' + metric + '/' + method + f'/{avg_degree}deg/'
                os.makedirs(results_dir, exist_ok=True)
                results.save(results_dir + f'{int(100 * mu)}mu.pkl')


def main(argd):
    """ Run selected community detection methods on generated LFR benchmark graphs.

    Parameters
    ----------
    argd : dict
        Dictionary of parsed command line arguments.
    """
    if argd['network_sizes'] is not None:
        eval_lfr_k_n_mu(argd['avg_degrees'][0], argd['network_sizes'][0], argd['methods'][0], argd['metric'],
                        argd['base_path'])
    else:
        eval_lfr_k_mu_n(argd['avg_degrees'][0], argd['mixing_params'][0], argd['methods'][0], argd['metric'],
                        argd['base_path'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluate detected LFR clusterings.')
    parser.add_argument('--avg-degrees', action='append', nargs='+', type=int,
                        help='List of average degrees of the benchmark graphs')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--network-sizes', action='append', nargs='+', type=int,
                       help='List of network-sizes of the benchmark graphs')
    group.add_argument('--mixing-params', action='append', nargs='+', type=float,
                       help='List of mixing-params of the benchmark graphs')
    parser.add_argument('-m', '--methods', action='append', nargs='+', type=str,
                        help='List of methods to be benchmarked.')
    parser.add_argument('--metric', type=str, default='ami', help='Evaluation metric')
    parser.add_argument('--base-path', type=str, default='./', help='Base path of the project directory')

    args = parser.parse_args()
    main(vars(args))
