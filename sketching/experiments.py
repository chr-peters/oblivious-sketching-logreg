import abc
import logging
from time import perf_counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import SGDClassifier

from . import optimizer, settings
from .datasets import Dataset
from .l2s_sampling import l2s_sampling
from .sketch import Sketch

logger = logging.getLogger(settings.LOGGER_NAME)

_rng = np.random.default_rng()


class BaseExperiment(abc.ABC):
    def __init__(
        self,
        num_runs,
        min_size,
        max_size,
        step_size,
        dataset: Dataset,
        results_filename,
    ):
        self.num_runs = num_runs
        self.min_size = min_size
        self.max_size = max_size
        self.step_size = step_size
        self.dataset = dataset
        self.results_filename = results_filename

    @abc.abstractmethod
    def get_reduced_matrix_and_weights(self, config):
        pass

    def get_config_grid(self):
        """
        Returns a list of configurations that are used to run the experiments.
        """
        grid = []
        for size in np.arange(
            start=self.min_size,
            stop=self.max_size + self.step_size,
            step=self.step_size,
        ):
            for run in range(1, self.num_runs + 1):
                grid.append({"run": run, "size": size})

        return grid

    def optimize(self, reduced_matrix, weights):
        return optimizer.optimize(Z=reduced_matrix, w=weights).x

    def run(self, parallel=False, n_jobs=4):
        Z = self.dataset.get_Z()
        beta_opt = self.dataset.get_beta_opt()
        objective_function = optimizer.get_objective_function(Z)
        f_opt = objective_function(beta_opt)

        logger.info("Running experiments...")

        def job_function(cur_config):
            logger.info(f"Current experimental config: {cur_config}")

            start_time = perf_counter()

            reduced_matrix, weights = self.get_reduced_matrix_and_weights(cur_config)
            sampling_time = perf_counter() - start_time

            cur_beta_opt = self.optimize(reduced_matrix, weights)
            total_time = perf_counter() - start_time

            cur_ratio = objective_function(cur_beta_opt) / f_opt
            return {
                **cur_config,
                "ratio": cur_ratio,
                "sampling_time_s": sampling_time,
                "total_time_s": total_time,
            }

        if parallel:
            results = Parallel(n_jobs=n_jobs)(
                delayed(job_function)(cur_config)
                for cur_config in self.get_config_grid()
            )
        else:
            results = [
                job_function(cur_config) for cur_config in self.get_config_grid()
            ]

        logger.info(f"Writing results to {self.results_filename}")

        df = pd.DataFrame(results)
        df.to_csv(self.results_filename, index=False)

        logger.info("Done.")


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
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
        )

    def get_reduced_matrix_and_weights(self, config):
        Z = self.dataset.get_Z()
        n = self.dataset.get_n()
        size = config["size"]

        row_indices = _rng.choice(n, size=size, replace=False)
        reduced_matrix = Z[row_indices]
        weights = np.ones(size)

        return reduced_matrix, weights


class ObliviousSketchingExperiment(BaseExperiment):
    """
    WARNING: This implementation is not thread safe!!!
    """

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
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
        )
        self.h_max = h_max
        self.kyfan_percent = kyfan_percent

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


class L2SExperiment(BaseExperiment):
    def __init__(
        self,
        dataset: Dataset,
        results_filename,
        min_size,
        max_size,
        step_size,
        num_runs,
    ):
        super().__init__(
            num_runs=num_runs,
            min_size=min_size,
            max_size=max_size,
            step_size=step_size,
            dataset=dataset,
            results_filename=results_filename,
        )

    def get_reduced_matrix_and_weights(self, config):
        Z = self.dataset.get_Z()
        size = config["size"]

        reduced_matrix, weights = l2s_sampling(Z, size=size)

        return reduced_matrix, weights


class SGDExperiment(BaseExperiment):
    def __init__(
        self,
        num_runs,
        dataset: Dataset,
        results_filename,
    ):
        n = dataset.get_n()
        super().__init__(
            num_runs=num_runs,
            min_size=n,
            max_size=n,
            step_size=0,
            dataset=dataset,
            results_filename=results_filename,
        )

    def get_config_grid(self):
        grid = []

        for run in range(1, self.num_runs + 1):
            grid.append({"run": run})

        return grid

    def get_reduced_matrix_and_weights(self, config):
        # For SGD, no reduction is performed
        return None, None

    def optimize(self, reduced_matrix, weights):
        """Performs a run of stochastic gradient descent."""
        X = self.dataset.get_X()
        y = self.dataset.get_y()

        learner = SGDClassifier(
            loss="log", alpha=0, learning_rate="adaptive", eta0=0.01, max_iter=1
        )

        learner.partial_fit(X, y, classes=np.unique(y))

        beta_opt = np.concatenate((learner.coef_[0], learner.intercept_))

        return beta_opt
