import numpy as np
import scipy.signal
import pytest

from det_stat import subid_det

"""
These are not very good tests since they are not completely deterministi. Still makes sense
to use them in the code porting, to start with, as these are the algorithm test cases that
the original book authors used to validate algorithm behavior.

Tests described in book section 2.3.3: Notes on noisy measurements. Pages 47 trough 49.
"""

@pytest.fixture(scope="module")
def generate_system_signals():
    """Generates and returns the input and output signals of a
    one-dimensional linear system. Also returns the expected values for the
    system's state and feedthrough matrices for testing comparison, as well
    as the identifaction system order and number of block rows.
    """
    a = 0.85
    b = 0.5
    c = -0.3
    d = 0.1

    bf, af = scipy.signal.butter(2, 0.3, 'low')

    input_gen = scipy.signal.lfilter(bf, af, np.random.randn(1000))
    ur = (input_gen + 0.1 * np.random.randn(1000)).reshape(-1, 1)
    dt = 1

    syst = (a, b, c, d, dt)
    t, yr, x = scipy.signal.dlsim(syst, ur)

    i = 4
    n = 1

    return ur, yr, a, d, i, n


def test_white_noise_output_1D(generate_system_signals):
    """Tests deterministic subspace identification for a one-dimensional
    linear system with white noise in its outputs (white process noise)
    """
    ur, yr, expected_a, expected_d, i, n = generate_system_signals

    list_of_As = []
    list_of_Ds = []

    num_exps = 200

    for k in range(num_exps):
        y = yr + 0.1 * np.random.randn(1000).reshape(-1, 1)
        u = ur

        A, B, C, D, ss = subid_det(y, i, n, u)
        list_of_As.append(A[0][0])
        list_of_Ds.append(D[0][0])

    np.testing.assert_allclose(np.mean(list_of_As), expected_a, atol=1e-3)
    np.testing.assert_allclose(np.mean(list_of_Ds), expected_d, atol=5e-3)


def test_white_noise_input_n_output_1D(generate_system_signals):
    """Tests deterministic subspace identification for a one-dimensional
    linear system with white noises in both its inputs and outputs
    (white measurement and process noises)
    """
    ur, yr, expected_a, expected_d, i, n = generate_system_signals

    list_of_As = []
    list_of_Ds = []

    num_exps = 200

    for k in range(num_exps):
        y = yr + 0.1 * np.random.randn(1000).reshape(-1, 1)
        u = ur + 0.1 * np.random.randn(1000).reshape(-1, 1)

        A, B, C, D, ss = subid_det(y, i, n, u)
        list_of_As.append(A[0][0])
        list_of_Ds.append(D[0][0])

    expected_offset_a = 0.005
    expected_offset_d = -0.015

    np.testing.assert_allclose(np.mean(list_of_As), expected_a + expected_offset_a, atol=1e-3)
    np.testing.assert_allclose(np.mean(list_of_Ds), expected_d + expected_offset_d, atol=5e-3)


def test_coloured_noise_output_1D(generate_system_signals):
    """Tests deterministic subspace identification for a one-dimensional
    linear system with coloured noise in its outputs (coloured process noise)
    """
    ur, yr, expected_a, expected_d, i, n = generate_system_signals

    list_of_As = []
    list_of_Ds = []

    num_exps = 200

    bc = [0.02, -0.041]
    ac = [1, -0.85]

    for k in range(num_exps):
        output_gen = scipy.signal.lfilter(bc, ac, np.random.randn(1000))
        y = yr + output_gen.reshape(-1, 1)
        u = ur

        A, B, C, D, ss = subid_det(y, i, n, u)
        list_of_As.append(A[0][0])
        list_of_Ds.append(D[0][0])

    np.testing.assert_allclose(np.mean(list_of_As), expected_a, atol=1e-3)
    np.testing.assert_allclose(np.mean(list_of_Ds), expected_d, atol=1e-3)
