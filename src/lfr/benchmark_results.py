import pickle

import numpy as np


class DataPoint:
    """Class representing a data point in the LFR benchmark.

        A data point contains the scores achieved on number of graph realizations which
        are described by a fixed parameter pair (n, mu), i.e. network size and mixing parameter
        respectively.

        Attributes
        ----------
        var : float
            The variable value associated with a data point, e.g. mu = 0.5.
        num_samples : int
            Number of graph realizations for this data point.
        scores : numpy array
            Scores corresponding to the graph realizations.
        seeds : list
            Integer seeds corresponding to the graph realizations.
    """

    def __init__(self, var, num_samples):
        """
            Parameters
            ----------
            var : float
                The variable value associated with a data point, e.g. mu = 0.5.
            num_samples : int
                Number of graph realizations for this data point.
        """
        self.var = var
        self.num_samples = num_samples  # number of graph realizations per data point
        self.scores = np.zeros((self.num_samples,))
        self.seeds = [-1] * self.num_samples
        self._cursor = -1

    def add_sample(self, seed, score):
        """Adds a (seed, score) pair associated with a graph realization to the data point.

            Parameters
            ----------
            seed : int
                Seed corresponding to the graph realization.
            score : float
                Score achieved when testing on the graph with generated with seed.
        """
        if self._cursor < self.num_samples - 1:
            self._cursor += 1
            self.seeds[self._cursor] = seed
            self.scores[self._cursor] = score


class BenchmarkResults:
    """Stores LFR benchmark results as a list of data points.

        Attributes
        ----------
        datapoints : DataPoint
            List of data points.
    """

    def __init__(self):
        self.datapoints = []

    def add_datapoint(self, dp: DataPoint):
        """Adds a data point to the list of data points.

            Parameters
            ----------
            dp : DataPoint
                Data point to add.
        """
        self.datapoints.append(dp)

    def get_var_list(self):
        """Returns a list of variables corresponding to the stored data points.

            Returns
            ----------
            list
                List of variables corresponding to the stored data points.
        """
        return [dp.var for dp in self.datapoints]

    def get_mean_scores(self):
        """Returns a list of mean scores corresponding to the stored data points.

            Returns
            ----------
            list
                List of mean scores corresponding to the stored data points. Each
                entry is a mean value across all values stored within a data point.
        """
        return np.array([np.mean(dp.scores) for dp in self.datapoints])

    def get_score_std(self):
        """Returns a list of score standard deviations corresponding to the stored data points.

            Returns
            ----------
            list
                List of score standard deviations corresponding to the stored data points. Each
                entry is represents the standard deviation of the scores stored within a data point.
        """
        return np.array([np.std(dp.scores, ddof=1) for dp in self.datapoints])

    def save(self, path):
        """Stores a data point as pickle dump.

            Parameters
            ----------
            path : str
                File path where to store the data point.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> 'BenchmarkResults':
        """Loads a data point from a pickle dump.

            Parameters
            ----------
            path : str
                File path to the pickle dump.

            Returns
            ----------
            BenchmarkResults
                A new BenchmarkResults object containing the pickle dump.
        """
        with open(path, 'rb') as f:
            results = pickle.load(f)
        return results
