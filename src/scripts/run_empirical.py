import os
import sys

from src.wrappers.igraph import walktrap
from src.wrappers.infomap import Infomap


def detect_communities(network: str, base_path='./'):
    """ Run community detection (Walktrap, Infomap, Synwalk) on a given network.

    Parameters
    ----------
    network : str
        Network identifier of the network to be analyzed.
    base_path : str
        Path to the project root.
    """
    # check if network file exists
    graph_file = base_path + 'data/empirical/clean/' + network + '.txt'
    if not os.path.exists(graph_file):
        print('Graph file for network ' + network + ' does not exist at ' + graph_file + '. Abort detection.')
        return

    # assemble paths
    workspace = base_path + 'workspace/'
    results_dir = base_path + 'results/empirical/' + network + '/'
    os.makedirs(results_dir, exist_ok=True)

    print('Detecting communities on ' + network + ' with Infomap...', flush=True)
    infomap_path = base_path + 'Infomap'
    clu = Infomap(workspace, infomap_path).infomap(graph_file)
    clu.save(results_dir + 'clustering_infomap.json')

    print('Detecting communities on ' + network + ' with SynWalk...', flush=True)
    clu = Infomap(workspace, infomap_path).synwalk(graph_file)
    clu.save(results_dir + 'clustering_synwalk.json')

    print('Detecting communities on ' + network + ' with Walktrap...', flush=True)
    clu = walktrap(graph_file)
    clu.save(results_dir + 'clustering_walktrap.json')


def main(argv):
    """ Run community detection (Walktrap, Infomap, Synwalk) on a given set of networks.

    Parameters
    ----------
    argv : list
        List holding the command line arguments. The first argument is the project base path. The
        second argument is gives the network identifiers for the analysis.
    """
    if len(argv) < 2:
        print('Usage: ' + os.path.basename(__file__) + ' [base_path] [network_identifiers]')
        return

    # get synwalk base path from arguments
    base_path = argv[0]
    if base_path[-1] != '/':
        base_path += '/'

    for network in argv[1:]:
        detect_communities(network, base_path)


if __name__ == "__main__":
    main(sys.argv[1:])
