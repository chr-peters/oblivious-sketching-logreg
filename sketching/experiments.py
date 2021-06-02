import abc

import numpy as np
import pandas as pd

from . import optimizer
from .datasets import Dataset

_rng = np.random.default_rng()


class BaseExperiment(abc.ABC):
    def __init__(self, dataset: Dataset, results_filename):
        self.dataset = dataset
        self.results_filename = results_filename

    @abc.abstractmethod
    def get_reduced_matrix_and_weights(self, config):
        pass

    @abc.abstractmethod
    def get_config_grid(self):
        """
        Returns a list of configurations that are used to run the experiments.
        """
        pass

    def run(self):
        Z = self.dataset.get_Z()
        beta_opt = self.dataset.get_beta_opt()
        objective_function = optimizer.get_objective_function(Z)
        f_opt = objective_function(beta_opt)

        results = []
        for cur_config in self.get_config_grid():
            reduced_matrix, weights = self.get_reduced_matrix_and_weights(cur_config)
            cur_beta_opt = optimizer.optimize(Z=reduced_matrix, w=weights).x
            cur_ratio = objective_function(cur_beta_opt) / f_opt
            results.append({**cur_config, "ratio": cur_ratio})

        df = pd.DataFrame(results)
        df.to_csv(self.results_filename, index=False)


class UniformSamplingExperiment(BaseExperiment):
    def __init__(
        self,
        dataset: Dataset,
        results_filename,
        min_size,
        max_size,
        step_size,
        num_runs,
    ):
        super().__init__(dataset=dataset, results_filename=results_filename)
        self.min_size = min_size
        self.max_size = max_size
        self.step_size = step_size
        self.num_runs = num_runs

    def get_config_grid(self):
        grid = []
        for size in np.arange(
            start=self.min_size,
            stop=self.max_size + self.step_size,
            step=self.step_size,
        ):
            for run in range(1, self.num_runs + 1):
                grid.append({"run": run, "size": size})

        return grid

    def get_reduced_matrix_and_weights(self, config):
        Z = self.dataset.get_Z()
        n = self.dataset.get_n()
        size = config["size"]

        row_indices = _rng.choice(n, size=size, replace=False)
        reduced_matrix = Z[row_indices]
        weights = np.ones(size)

        return reduced_matrix, weights
