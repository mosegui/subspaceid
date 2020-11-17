import numpy as np

from blkhank import hankel_matrix


def test_hankel_matrix_1D():
    x = np.arange(7).reshape(-1, 1)
    expected = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]])
    x_hankel = hankel_matrix(x, 3, 4)

    np.testing.assert_allclose(x_hankel, expected)


def test_hankel_matrix_ND():
    x = np.vstack((np.arange(10), np.arange(0, 20, 2))).T

    expected = np.array([[0, 1, 2, 3], [0, 2, 4, 6], [1, 2, 3, 4], [2, 4, 6, 8], [2, 3, 4, 5], [4, 6, 8, 10]])
    x_hankel = hankel_matrix(x, 3, 4)

    np.testing.assert_allclose(x_hankel, expected)
