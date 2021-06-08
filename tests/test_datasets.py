import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from sketching.datasets import Dataset, NoisyDataset

X = np.array([[1, 0], [0.1, 1], [-0.1, 1], [-1, 0], [0, -1]])
y = np.array([1, -1, -1, 1, -1])
Z = np.array([[1, 0, 1], [-0.1, -1, -1], [0.1, -1, -1], [-1, 0, 1], [0, 1, -1]])
beta_opt = np.array([0.0, -0.60493802, -0.30248661])


class ExampleDataset(Dataset):
    def __init__(self, use_caching, cache_dir=None):
        super().__init__(use_caching=use_caching, cache_dir=cache_dir)

    def get_name(self):
        return "example_name"

    def load_X_y(self):
        return X, y


def test_abstract_dataset_no_caching():
    """
    Use a small toy example to test the abstract dataset base class.
    """
    dataset = ExampleDataset(use_caching=False)
    assert dataset.get_name() == "example_name"
    assert_array_equal(dataset.get_X(), X)
    assert_array_equal(dataset.get_y(), y)
    assert_array_equal(dataset.get_Z(), Z)
    assert_array_almost_equal(dataset.get_beta_opt(), beta_opt, decimal=4)
    assert dataset.get_n() == 5
    assert dataset.get_d() == 2


def test_abstract_dataset_caching(tmp_path):
    """
    Use a small toy example to test the abstract dataset base class.
    """

    # run the tests twice to simulate a cache hit
    for i in range(2):
        dataset = ExampleDataset(use_caching=True, cache_dir=tmp_path)
        assert dataset.get_name() == "example_name"
        assert_array_equal(dataset.get_X(), X)
        assert_array_equal(dataset.get_y(), y)
        assert_array_equal(dataset.get_Z(), Z)
        assert_array_almost_equal(dataset.get_beta_opt(), beta_opt, decimal=4)
        assert dataset.get_n() == 5
        assert dataset.get_d() == 2


def test_noisy_dataset():
    dataset = ExampleDataset(use_caching=False)
    noisy_dataset = NoisyDataset(dataset=dataset, percentage=0.4, std=100)

    assert noisy_dataset.get_name() == "example_name_noisy"
    assert dataset.get_n() == 5
    assert dataset.get_d() == 2
    assert dataset.get_Z().shape[0] == 5
    assert dataset.get_Z().shape[1] == 3

    # make sure that exactly 40% of the rows (2 rows) are affected by noise
    Z_original = dataset.get_Z()
    Z_noisy = noisy_dataset.get_Z()

    assert np.sum(np.all(np.equal(Z_original, Z_noisy), axis=1)) == 3
