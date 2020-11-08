import numpy as np

from solvric import solvric

def gl2kr(A, G, C, L0):
    if (G == []) | (L0 == []):
        K = []
        R = []
    else:
        # Solve the Riccati equation
        P, flag = solvric(A, G, C, L0)
        if flag == 1:  # Riccati equation had no solution
            print('Warning: Non positive real covariance model => K = R = []')
            K = []
            R = []
        else:
            # Make output (Page 63 for instance)
            R = L0 - C @ P @ C.conj().T
            K = (G - A @ P @ C.conj().T) @ np.linalg.inv(R)

    return K, R