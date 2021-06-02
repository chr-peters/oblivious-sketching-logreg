import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from sketching.datasets import Dataset
from sketching.experiments import UniformSamplingExperiment


class ExampleDataset(Dataset):
    def __init__(self):
        super().__init__(use_caching=False)

    def get_name(self):
        return "example_name"

    def load_X_y(self):
        X = np.array([[1, 0], [0.1, 1], [-0.1, 1], [-1, 0], [0, -1]])
        y = np.array([1, -1, -1, 1, -1])
        return X, y


def test_uniform_sampling_experiment(tmp_path):
    dataset = ExampleDataset()
    results_filename = tmp_path / "results.csv"
    experiment = UniformSamplingExperiment(
        dataset,
        results_filename=results_filename,
        min_size=1,
        max_size=5,
        step_size=2,
        num_runs=3,
    )
    experiment.run()

    df = pd.read_csv(results_filename)

    run_unique, run_counts = np.unique(df["run"], return_counts=True)
    assert_array_equal(run_unique, [1, 2, 3])
    assert_array_equal(run_counts, [3, 3, 3])
    assert np.sum(df["ratio"].isna()) == 0


def test_uniform_sampling_reduction(tmp_path):
    dataset = ExampleDataset()
    results_filename = tmp_path / "results.csv"
    experiment = UniformSamplingExperiment(
        dataset,
        results_filename=results_filename,
        min_size=1,
        max_size=5,
        step_size=1,
        num_runs=1,
    )

    for cur_config in experiment.get_config_grid():
        cur_matrix, cur_weights = experiment.get_reduced_matrix_and_weights(cur_config)
        assert_array_equal(cur_weights, np.ones(cur_config["size"]))
        assert cur_matrix.shape[0] == cur_config["size"]
        assert cur_matrix.shape[1] == dataset.get_d() + 1
