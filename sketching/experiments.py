import abc
from time import perf_counter

import numpy as np
import pandas as pd

from . import optimizer
from .datasets import Dataset
from .sketch import Sketch

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

    def optimize(self, reduced_matrix, weights):
        return optimizer.optimize(Z=reduced_matrix, w=weights).x

    def run(self):
        Z = self.dataset.get_Z()
        beta_opt = self.dataset.get_beta_opt()
        objective_function = optimizer.get_objective_function(Z)
        f_opt = objective_function(beta_opt)

        results = []
        for cur_config in self.get_config_grid():
            start_time = perf_counter()

            reduced_matrix, weights = self.get_reduced_matrix_and_weights(cur_config)
            sampling_time = perf_counter() - start_time

            cur_beta_opt = self.optimize(reduced_matrix, weights)
            total_time = perf_counter() - start_time

            cur_ratio = objective_function(cur_beta_opt) / f_opt
            results.append(
                {
                    **cur_config,
                    "ratio": cur_ratio,
                    "sampling_time_s": sampling_time,
                    "total_time_s": total_time,
                }
            )

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


class ObliviousSketchingExperiment(BaseExperiment):
    def __init__(
        self,
        dataset: Dataset,
        results_filename,
        min_size,
        max_size,
        step_size,
        num_runs,
        h_max,
        kyfan_percent,
    ):
        super().__init__(dataset=dataset, results_filename=results_filename)
        self.min_size = min_size
        self.max_size = max_size
        self.step_size = step_size
        self.num_runs = num_runs
        self.h_max = h_max
        self.kyfan_percent = kyfan_percent

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
        d = self.dataset.get_d() + 1
        size = config["size"]

        # divide by (h_max + 2) + to get one more block for unif sampling
        N = max(int(size / (self.h_max + 2)), 1)
        b = (n / N) ** (1.0 / self.h_max)
        actual_sketch_size = N * (self.h_max + 1)

        unif_block_size = max(size - actual_sketch_size, 1)

        sketch = Sketch(self.h_max, b, N, n, d)
        for j in range(0, n):
            sketch.insert(Z[j])
        reduced_matrix = sketch.get_reduced_matrix()
        weights_sketch = sketch.get_weights()

        # do the unif sampling
        rows = _rng.choice(n, unif_block_size, replace=False)
        unif_sample = Z[rows]

        # concat the sketch and the uniform sample
        reduced_matrix = np.vstack([reduced_matrix, unif_sample])

        weights_unif = np.ones(unif_block_size) * n / unif_block_size

        weights = np.concatenate([weights_sketch, weights_unif])
        weights = weights / np.sum(weights)

        self.cur_kyfan_k = int(N * self.kyfan_percent)
        self.cur_kyfan_max_len = actual_sketch_size
        self.cur_kyfan_block_size = N

        return reduced_matrix, weights

    def optimize(self, reduced_matrix, weights):
        return optimizer.optimize(
            reduced_matrix,
            weights,
            block_size=self.cur_kyfan_block_size,
            k=self.cur_kyfan_k,
            max_len=self.cur_kyfan_max_len,
        ).x
