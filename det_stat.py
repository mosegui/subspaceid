"""
The content of this module is based on the original work:

    "Subspace Identification for Linear Systems, Theory, Implementation, Applications"
                           Van Overschee P., De Moor B.
                        Kluwer Academic Publishers, (1996)
"""

import numpy as np

from algebra import matdiv, project_on_perpendicular
from blkhank import hankel_matrix


def subid_det(y, i, n, u, w="SV"):

    input_observations, input_dimensions = u.shape  # nu, m
    output_observations, output_dimensions = y.shape  # ny, l

    assert output_observations == input_observations, "Number of data points different in input and output"
    assert i > 0, "Number of block rows should be positive"
    assert (output_observations - 2 * i + 1) >= (2 * output_dimensions * i), "Not enough data points"


    j = input_observations - (2 * i - 1)  # Determine the number of columns in the Hankel matrices

    Y = hankel_matrix(y / np.sqrt(j), 2*i, j)
    U = hankel_matrix(u / np.sqrt(j), 2 * i, j)

    m = input_dimensions
    l = output_dimensions

    # Compute the R factor
    R = np.triu(np.linalg.qr(np.vstack((U,Y)).conj().T, mode='complete')[1]).conj().T  # R factor
    R = R[: 2 * i * (m + l), : 2 * i * (m + l)]

    # Calculate oblique projection
    Rf = R[(2*m+l)*i:, :]  # Future outputs
    Rp = np.vstack((R[:m * i,:], R[2 * m * i: (2 * m + l) * i, :]))  # Past(inputs and) outputs
    Ru = R[m * i:2 * m * i, :]  # Future inputs

    Rfp = project_on_perpendicular(Rf, Ru)  # Perpendicular Future outputs
    Rpp = project_on_perpendicular(Rp, Ru)  # Perpendicular Past

    # The oblique projection "obl/Ufp = Yf/Ufp * pinv(Wp/Ufp) * Wp" Computed as 6.1 on page 166
    Ob = matdiv(Rfp, Rpp) @ Rp

    # # Funny rankv check (SVD takes too long). This check is needed to avoid rank deficiency warnings
    # if np.linalg.norm(Rpp[:, (2 * m+l) * i - 2 * l: (2 * m + l) * i], 'fro') < 1e-10:
    #     Ob = (Rfp @ np.linalg.pinv(Rpp.conj().T).conj().T) @ Rp  # Oblique projection
    # else:
    #     Ob = matdiv(Rfp, Rpp) @ Rp

    # Compute the SVD
    U, S, V = np.linalg.svd(Ob)
    ss = np.diag(S)

    # Determine the order from the singular values
    U1 = U[:, :n]

     # Determine gam and gamm
    gam = U1 @ np.diag(np.sqrt(np.diag(ss[:n])))
    gamm = gam[:l*(i-1), :]

    # Compute Obm (the second oblique projection)
    Rf = R[(2 * m + l) * i + l:, :]
    Rp = np.vstack((R[:m * (i + 1), :], R[2 * m * i: (2 * m + l) * i + l, :]))
    Ru = R[m * i + m:2 * m * i, :]

    Rfp = project_on_perpendicular(Rf, Ru)
    Rpp = project_on_perpendicular(Rp, Ru)

    Obm = matdiv(Rfp, Rpp) @ Rp

    # # Funny rankv check (SVD takes too long). This check is needed to avoid rank deficiency warnings
    # if np.linalg.norm(Rpp[:, (2 * m+l) * i - 2 * l: (2 * m + l) * i], 'fro') < 1e-10:
    #     Ob = (Rfp @ np.linalg.pinv(Rpp.conj().T).conj().T) @ Rp  # Oblique projection
    # else:
    #     Ob = matdiv(Rfp, Rpp) @ Rp

    # Determine the states Xi and Xip
    Xi  = np.linalg.pinv(gam) @ Ob
    Xip = np.linalg.pinv(gamm) @ Obm

    # Solve linear system of equations
    Rhs = np.vstack((Xi, R[m * i: m * (i + 1), :]))
    Lhs = np.vstack((Xip, R[(2 * m + l) * i: (2 * m + l) * i + l, :]))

    sol = matdiv(Lhs, Rhs)

    # Extract the system matrices
    A = sol[:n, :n]
    B = sol[:n, n:n+m]
    C = sol[n:n+l, :n]
    D = sol[n:n+l, n:n+m]

    return A, B, C, D, ss
