import numpy as np


def hankel_matrix(y, i, j):
    l, nd = y.shape
    if nd < l:
        y = y.conj().T
        l, nd = y.shape

    if i < 0:
        raise ValueError("i should be positive")
    if j < 0:
        raise ValueError("j should be positive")
    if j > nd - i + 1:
        raise ValueError("j too big")

    H = np.zeros((l*i, j))
    for k in range(i):
        # H[k*l : (k+1)*l, :] = y[:, k : k+j]
        # H[(k - 1) * l + 1: k * l, :] = y[:, k: k + j - 1]
        H[k * l: (k + 1) * l, :] = y[:, k:k + j]

    return H