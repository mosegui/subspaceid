import numpy as np
import scipy.signal

from subid import subid
import matplotlib.pyplot as plt

def bode(a, b, c, d, dt):

    if b.shape[-1] != d.shape[-1]:
        raise ValueError('D and B matrices must have same number of columns')

    number_of_inputs = b.shape[-1]
    number_of_outputs = c.shape[0]

    ws = []
    mags = []
    phases = []
    labels = []

    for col in range(number_of_inputs):
        for row in range(number_of_outputs):
            c_row = c[row].reshape(1, -1)
            b_col = b[:, col].reshape(-1,1)
            d_col = d[row, col].reshape(-1,1)

            w_col, mag_col, phase_col = scipy.signal.dbode((a, b_col, c_row, d_col, dt))
            ws.append(w_col)
            mags.append(mag_col)
            phases.append(phase_col)
            labels.append(f"I{col+1} -> O{row+1}")

    fig, (ax_mag, ax_phase) = plt.subplots(nrows=2)

    for idx, w in enumerate(ws):
        ax_mag.semilogx(w, mags[idx], label=labels[idx])
        ax_phase.semilogx(w, phases[idx], label=labels[idx])
    ax_mag.legend()
    ax_phase.legend()

# Consider a multivariable fourth order system a,b,c,d with two inputs and two outputs:

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

bode(*syst)

max_order = 8

i = 2 * max_order // l

n = 4

A, B, C, D, K, Ro, ss = subid(y, i, n, u=u)

syst2 = (A, B, C, D, dt)

bode(*syst2)
