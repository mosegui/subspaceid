import numpy as np
import scipy.signal

from subid import subid

def test_interation():

    a = np.array([[0.603, 0.603, 0, 0],
                  [-0.603, 0.603, 0, 0],
                  [0, 0, -0.603, -0.603],
                  [0, 0, 0.603, -0.603]])

    b = np.array([[1.1650, -0.6965],
                  [0.6268, 1.6961],
                  [0.0751, 0.0591],
                  [0.3516, 1.7971]])

    c = np.array([[0.2641, -1.4462, 1.2460, 0.5774],
                  [0.8717, -0.7012, -0.6390, -0.3600]])

    d = np.array([[-0.1356, -1.2704],
                  [-1.3493, 0.9846]])

    m = 2  # Number of inputs
    l = 2  # Number of outputs

    N = 1000
    t = np.arange(1, N + 1)
    u = np.array([np.sin(t).T, np.sin(t + 3.14/2).T]).T.reshape(-1, 2)
    dt = 1

    syst = (a, b, c, d, dt)
    t, y, x = scipy.signal.dlsim(syst, u)

    max_order = 8

    i = 2 * max_order // l

    n = 4

    A, B, C, D, K, Ro, ss = subid(y, i, n, u=u)

    exp_A = np.array([[-2.41229427e-01, 7.33840936e-02, -9.86304719e-03, -2.30022000e-03],
                      [1.03535439e+00, -4.60890219e-01, -4.14974362e-03, 1.73673059e-02],
                      [1.62935269e+01, -1.74125293e+01, 4.81451957e-01, -3.99165265e-01],
                      [-5.04117360e+01, -2.66187842e+01, -6.53704582e-02, 3.25934186e-01]])

    np.testing.assert_allclose(exp_A, A)
