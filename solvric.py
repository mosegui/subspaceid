import numpy as np
import scipy as sp

def solvric(A, G, C, L0):
    if (G == []) | (L0 == []):
        P = []
        flag = 0

    else:

        n = len(A)  # Dimensions
        L0i = np.linalg.inv(L0)  # Compute the inverse once

        #   Set up the matrices for the eigenvalue decomposition

        A1 = np.hstack((A.conj().T - C.conj().T @ L0i @ G.conj().T, np.zeros((n, n))))
        A2 = np.hstack((-G @ L0i @ G.conj().T, np.eye(n)))
        AA = np.vstack((A1, A2))

        B1 = np.hstack((np.eye(n), -C.conj().T @ L0i @ C))
        B2 = np.hstack((np.zeros((n, n)), A - G @ L0i @ C))
        BB = np.vstack((B1, B2))

        #   Compute the eigenvalue decomposition

        ew, v = sp.linalg.eig(AA, BB, right=True)

        # If there's an eigenvalue on the unit circle => no solution
        flag = 0
        if np.prod(np.abs(np.abs(ew) - np.ones((2 * n, 1))) > 1e-9) < 1:
            flag = 1

        # Sort the eigenvalues
        I = np.argsort(np.abs(ew))

        # Compute P
        P = np.real(v[n:2 * n, I[:n]] / v[:n, I[:n]])

    return P, flag
