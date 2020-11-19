"""
The content of this module is based on the original work:

    "Subspace Identification for Linear Systems, Theory, Implementation, Applications"
                           Van Overschee P., De Moor B.
                        Kluwer Academic Publishers, (1996)
"""

import numba
import numpy as np

@numba.njit()
def shifted_tile(y, i, j):
    l, nd = y.shape
    if nd < l:
        y = y.conj().T
        l, nd = y.shape

    H = np.zeros((l*i, j))
    for k in range(i):
        H[k * l: (k + 1) * l, :] = y[:, k:k + j]
    return H


def hankel_matrix(y, i, j):

    assert i >= 0, "i should be non-negative"
    assert j >= 0, "j should be non-negative"

    observations, dimensions = y.shape
    _msg = "data must be passed in column format, rows being the data observations and columns the" \
           "data dimensions"
    assert observations > dimensions, _msg

    if j > observations - i + 1:
        raise ValueError("j too big")

    return shifted_tile(y, i, j)