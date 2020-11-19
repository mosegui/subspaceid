import numpy as np
import scipy.signal
import pytest

from det_stat import subid_det

"""
These are not very good tests since they are not completely deterministic. Still makes sense
to use them in the code porting, to start with, as these are the algorithm test cases that
the original book authors used to validate algorithm behavior.

The present tests for a first-order linear system:
 - test_white_noise_output
 - test_white_noise_input_n_output
 - test_coloured_noise_output
are described in book section 2.3.3: Notes on noisy measurements. Pages 47 trough 49.
"""


INPUT_LENGTH = 1000


@pytest.fixture(scope="module")
def first_order_system():
    """Generates and returns the input and output signals of a
    one-dimensional linear system. Also returns the expected values for the
    system's state and feedthrough matrices for testing comparison, as well
    as the identification system order and number of block rows.
    """
    a = 0.85
    b = 0.5
    c = -0.3
    d = 0.1

    bf, af = scipy.signal.butter(2, 0.3, 'low')

    input_gen = scipy.signal.lfilter(bf, af, np.random.randn(INPUT_LENGTH))
    ur = (input_gen + 0.1 * np.random.randn(INPUT_LENGTH)).reshape(-1, 1)
    dt = 1

    syst = (a, b, c, d, dt)
    t, yr, x = scipy.signal.dlsim(syst, ur)

    i = 4
    n = 1

    return ur, yr, a, d, i, n


def test_white_noise_output(first_order_system):
    """Tests deterministic subspace identification for a one-dimensional
    linear system with white noise in its outputs (white process noise)
    """
    ur, yr, expected_a, expected_d, i, n = first_order_system

    list_of_As = []
    list_of_Ds = []

    num_exps = 200

    for k in range(num_exps):
        y = yr + 0.1 * np.random.randn(INPUT_LENGTH).reshape(-1, 1)
        u = ur

        A, B, C, D, ss = subid_det(y, i, n, u)
        list_of_As.append(A[0][0])
        list_of_Ds.append(D[0][0])

    np.testing.assert_allclose(np.mean(list_of_As), expected_a, atol=1e-3)
    np.testing.assert_allclose(np.mean(list_of_Ds), expected_d, atol=1e-2)


def test_white_noise_input_n_output(first_order_system):
    """Tests deterministic subspace identification for a one-dimensional
    linear system with white noises in both its inputs and outputs
    (white measurement and process noises)
    """
    ur, yr, expected_a, expected_d, i, n = first_order_system

    list_of_As = []
    list_of_Ds = []

    num_exps = 200

    for k in range(num_exps):
        y = yr + 0.1 * np.random.randn(INPUT_LENGTH).reshape(-1, 1)
        u = ur + 0.1 * np.random.randn(INPUT_LENGTH).reshape(-1, 1)

        A, B, C, D, ss = subid_det(y, i, n, u)
        list_of_As.append(A[0][0])
        list_of_Ds.append(D[0][0])

    expected_offset_a = 0.005
    expected_offset_d = -0.015

    np.testing.assert_allclose(np.mean(list_of_As), expected_a + expected_offset_a, atol=1e-3)
    np.testing.assert_allclose(np.mean(list_of_Ds), expected_d + expected_offset_d, atol=5e-3)


def test_coloured_noise_output(first_order_system):
    """Tests deterministic subspace identification for a one-dimensional
    linear system with coloured noise in its outputs (coloured process noise)
    """
    ur, yr, expected_a, expected_d, i, n = first_order_system

    list_of_As = []
    list_of_Ds = []

    num_exps = 200

    bc = [0.02, -0.041]
    ac = [1, -0.85]

    for k in range(num_exps):
        output_gen = scipy.signal.lfilter(bc, ac, np.random.randn(INPUT_LENGTH))
        y = yr + output_gen.reshape(-1, 1)
        u = ur

        A, B, C, D, ss = subid_det(y, i, n, u)
        list_of_As.append(A[0][0])
        list_of_Ds.append(D[0][0])

    np.testing.assert_allclose(np.mean(list_of_As), expected_a, atol=1e-3)
    np.testing.assert_allclose(np.mean(list_of_Ds), expected_d, atol=1e-3)


@pytest.fixture(scope="module")
def second_order_system():
    """Second-order system, one input, one output with identity matrix
    as state matrix"""
    a = np.array([[1., 0.],
                  [0., 1.]])
    b = np.array([[1.], [1.]])
    c = np.array([[1., 1.]])
    d = 0.

    bf, af = scipy.signal.butter(2, 0.3, 'low')

    input_gen = scipy.signal.lfilter(bf, af, np.random.randn(INPUT_LENGTH))
    ur = (input_gen + 0.1 * np.random.randn(INPUT_LENGTH)).reshape(-1, 1)

    dt = 1

    syst = (a, b, c, d, dt)
    t, yr, x = scipy.signal.dlsim(syst, ur)

    i = 6
    n = 2

    return ur, yr, a, d, i, n

def test_diagonal_state_matrix(second_order_system):
    """ Tests the average found state matrix is mostly diagonal"""
    ur, yr, expected_a, expected_d, i, n = second_order_system

    avg_A = None

    num_exps = 200

    for k in range(num_exps):
        y = yr + 0.1 * np.random.randn(INPUT_LENGTH).reshape(-1, 1)

        A, _, _, _, _ = subid_det(y, i, n, ur)

        if avg_A is None:
            avg_A = A
        else:
            avg_A = (k * avg_A + A)/(k + 1)

    avg_A_diag = np.diag(np.diagonal(avg_A))
    avg_a_offdiag = avg_A - avg_A_diag

    assert np.linalg.norm(avg_A_diag)/np.linalg.norm(avg_a_offdiag) >= 1e2  # check diagonal terms are orders of magnitude larger than off-diagonal ones.
    assert np.abs(avg_A_diag.max()/avg_A_diag.min()) <= 1e1  # Check diagonal elements are of same order of magnitude
