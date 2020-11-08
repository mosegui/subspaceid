import numpy as np
import scipy.signal

from subid import subid
from bode import bode_plot

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
# u = np.sin(t).reshape(-1,1)
dt = 1

syst = (a, b, c, d, dt)
t, y, x = scipy.signal.dlsim(syst, u)

bode_plot(*syst)

max_order = 8

i = 2 * max_order // l

n = 4

A, B, C, D, K, Ro, ss = subid(y, i, n, u=u)

syst2 = (A, B, C, D, dt)

bode_plot(*syst2)
