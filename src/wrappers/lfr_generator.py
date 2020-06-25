import os


class LFRGenerator:
    """Wrapper class around the LFR benchmark generator cmd-line tool.

    Attributes
    ----------
    bin_path : str
        Path to the generator binary.
    """

    def __init__(self, bin_path='../lfr_generator/benchmark'):
        """ Parameters
            ----------
            bin_path : str
                Path to the generator binary.
        """
        self.bin_path = bin_path

    @classmethod
    def get_param_string(cls, num_nodes, mixing_param, avg_degree):
        """Generates an argument string of LFR benchmark parameters for a given network size and mixing parameter.

        Parameters
        ----------
        num_nodes : int
            Number of nodes in the graph.
        mixing_param : float
            Global mixing parameter for the graph.
        avg_degree : int
            Average node degree.

        Returns
        ------
        str
            An argument string for the generator binary.
        """

        # generic LFR parameters
        max_community = int(0.2 * num_nodes)
        min_community = int(max_community * 0.25)
        max_degree = int(max_community * 0.95)
        gamma = 2.0  # Power law exponent for the degree distribution
        beta = 1.0  # Power law exponent for the community size distribution

        arg_str = f' -N {num_nodes} -mu {mixing_param} -k {avg_degree} -maxk {max_degree} -t1 {gamma} -t2 {beta} '
        arg_str += f'-minc {min_community} -maxc {max_community} '
        return arg_str

    def generate(self, num_nodes, mixing_parameter, avg_degree, output_dir='./', seed=0):
        """Generate a benchmark graph with the given parameters in output_dir.

            Parameters
            ----------
            num_nodes : int
                Number of nodes in the generated graph.
            mixing_parameter : float
                Global mixing parameter of the graph.
            avg_degree : int
                Average node degree.
            output_dir : str
                Path to the output directory. Must end with '/'.
            seed : int
                Random seed used for graph generation.

        """
        # make output directory if necessary
        os.makedirs(output_dir, exist_ok=True)

        # run generator
        return os.system(self.bin_path + self.get_param_string(num_nodes, mixing_parameter, avg_degree)
                         + f'-s {seed} -o ' + output_dir)
