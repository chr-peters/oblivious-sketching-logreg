import numpy as np

from sketching import optimizer


def test_keep_k_simple():
    test_vec = np.array([5, 3, -10, 6, 7, 2, 8, 6, 9])
    test_block_size = 3

    results, indices = optimizer.only_keep_k(
        test_vec, test_block_size, k=1, biggest=True
    )

    assert np.allclose(
        results,
        np.array([5, 7, 9]),
    )

    assert np.allclose(indices, np.array([0, 4, 8]))

    results, indices = optimizer.only_keep_k(
        test_vec, test_block_size, k=2, biggest=True
    )

    assert np.allclose(results, np.array([5, 3, 7, 6, 9, 8]))

    assert np.allclose(indices, np.array([0, 1, 4, 3, 8, 6]))

    results, indices = optimizer.only_keep_k(
        test_vec, test_block_size, k=3, biggest=True
    )

    assert np.allclose(results, np.array([5, 3, -10, 6, 7, 2, 8, 6, 9]))

    assert np.allclose(indices, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))

    # test keep smallest
    results, indices = optimizer.only_keep_k(
        test_vec, test_block_size, k=1, biggest=False
    )

    assert np.allclose(
        results,
        np.array([-10, 2, 6]),
    )

    assert np.allclose(indices, np.array([2, 5, 7]))

    results, indices = optimizer.only_keep_k(
        test_vec, test_block_size, k=2, biggest=False
    )

    assert np.allclose(results, np.array([-10, 3, 2, 6, 6, 8]))

    assert np.allclose(indices, np.array([2, 1, 5, 3, 7, 6]))

    results, indices = optimizer.only_keep_k(
        test_vec, test_block_size, k=3, biggest=False
    )

    assert np.allclose(results, np.array([5, 3, -10, 6, 7, 2, 8, 6, 9]))

    assert np.allclose(indices, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))


def test_keep_k_do_not_touch():
    """
    Do not touch means: Don't touch the last elements of the vector.
    Here in the test: 1, 2, 3, 4, 5 are the last elements.
    """
    test_vec = np.array([5, 3, -10, 6, 7, 2, 8, 6, 9, 1, 2, 3, 4, 5])
    test_block_size = 3
    test_max_len = 9

    results, indices = optimizer.only_keep_k(
        test_vec, test_block_size, k=1, max_len=test_max_len, biggest=True
    )

    assert np.allclose(
        results,
        np.array([5, 7, 9, 1, 2, 3, 4, 5]),
    )

    assert np.allclose(indices, np.array([0, 4, 8, 9, 10, 11, 12, 13]))

    results, indices = optimizer.only_keep_k(
        test_vec, test_block_size, k=2, max_len=test_max_len, biggest=True
    )

    assert np.allclose(results, np.array([5, 3, 7, 6, 9, 8, 1, 2, 3, 4, 5]))

    assert np.allclose(indices, np.array([0, 1, 4, 3, 8, 6, 9, 10, 11, 12, 13]))

    results, indices = optimizer.only_keep_k(
        test_vec, test_block_size, k=3, max_len=test_max_len, biggest=True
    )

    assert np.allclose(results, np.array([5, 3, -10, 6, 7, 2, 8, 6, 9, 1, 2, 3, 4, 5]))

    assert np.allclose(
        indices, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    )

    # test keep smallest
    results, indices = optimizer.only_keep_k(
        test_vec, test_block_size, k=1, max_len=test_max_len, biggest=False
    )

    assert np.allclose(
        results,
        np.array([-10, 2, 6, 1, 2, 3, 4, 5]),
    )

    assert np.allclose(indices, np.array([2, 5, 7, 9, 10, 11, 12, 13]))

    results, indices = optimizer.only_keep_k(
        test_vec, test_block_size, k=2, max_len=test_max_len, biggest=False
    )

    assert np.allclose(results, np.array([-10, 3, 2, 6, 6, 8, 1, 2, 3, 4, 5]))

    assert np.allclose(indices, np.array([2, 1, 5, 3, 7, 6, 9, 10, 11, 12, 13]))

    results, indices = optimizer.only_keep_k(
        test_vec, test_block_size, k=3, max_len=test_max_len, biggest=False
    )

    assert np.allclose(results, np.array([5, 3, -10, 6, 7, 2, 8, 6, 9, 1, 2, 3, 4, 5]))

    assert np.allclose(
        indices, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    )
