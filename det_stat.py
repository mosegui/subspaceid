import numpy as np

from blkhank import hankel_matrix


def matdiv(mat1, mat2):
    """
    Relevant links:
        https://de.mathworks.com/help/fixedpoint/ref/embedded.fi.mrdivide.html
        https://stackoverflow.com/questions/1001634/array-division-translating-from-matlab-to-python
    """
    A = np.asmatrix(mat1)
    B = np.asmatrix(mat2)

    X = A * B.T * (B * B.T).I

    return np.asarray(X)

def project(mat1, mat2):
    """Orthogonal projection of mat1 on mat2 along a direction
    perpendicular to mat2
    """
    return matdiv(mat1, mat2) @ mat2

def project_on_perpendicular(mat1, mat2):
    """Projects mat1 onto the orthogonal component of the row space
    of mat2
    """
    return mat1 - project(mat1, mat2)



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

    ######################################################
    #                       STEP 1                       #
    ######################################################
    # Calculate oblique projection
    Rf = R[(2*m+l)*i:, :]  # Future outputs
    Rp = np.vstack((R[:m * i,:], R[2 * m * i: (2 * m + l) * i, :]))  # Past(inputs and) outputs
    Ru = R[m * i:2 * m * i, :]  # Future inputs
    # Ru = R[m * i:2 * m * i, :mi2]  # Future inputs

    # Perpendicular Future outputs
    Rfp = project_on_perpendicular(Rf, Ru)
    # Rfp = np.hstack((project_on_perpendicular(Rf[:, :mi2], Ru[:, :mi2]), Rf[:, mi2:]))

    # Perpendicular Past
    Rpp = project_on_perpendicular(Rp, Ru)
    # Rpp = np.hstack((project_on_perpendicular(Rp[:, :mi2], Ru[:, :mi2]), Rp[:, mi2:]))

    # The oblique projection "obl/Ufp = Yf/Ufp * pinv(Wp/Ufp) * Wp" Computed as 6.1 on page 166
    Ob = matdiv(Rfp, Rpp) @ Rp

    # # Funny rankv check (SVD takes too long). This check is needed to avoid rank deficiency warnings
    # if np.linalg.norm(Rpp[:, (2 * m+l) * i - 2 * l: (2 * m + l) * i], 'fro') < 1e-10:
    #     Ob = (Rfp @ np.linalg.pinv(Rpp.conj().T).conj().T) @ Rp  # Oblique projection
    # else:
    #     Ob = matdiv(Rfp, Rpp) @ Rp

    # ######################################################
    # #                       STEP 2                       #
    # ######################################################
    # Compute the SVD
    U, S, V = np.linalg.svd(Ob)
    ss = np.diag(S)

    # ######################################################
    # #                       STEP 3                       #
    # ######################################################
    # Determine the order from the singular values
    U1 = U[:, :n]

    # ######################################################
    # #                       STEP 4                       #
    # ######################################################
     # Determine gam and gamm
    gam = U1 @ np.diag(np.sqrt(np.diag(ss[:n])))
    gamm = gam[:l*(i-1), :]

    # ######################################################
    # #                       STEP 5                       #
    # ######################################################
    # Compute Obm (the second oblique projection)
    Rf = R[(2 * m + l) * i + l:, :]
    Rp = np.vstack((R[:m * (i + 1), :], R[2 * m * i: (2 * m + l) * i + l, :]))
    Ru = R[m * i + m:2 * m * i, :]

    Rfp = project_on_perpendicular(Rf, Ru)
    # Rfp = np.hstack((project_on_perpendicular(Rf[:, :mi2], Ru[:, :mi2]), Rf[:, mi2:]))

    Rpp = project_on_perpendicular(Rp, Ru)
    # Rpp = np.hstack((project_on_perpendicular(Rp[:, :mi2], Ru[:, :mi2]), Rp[:, mi2:]))

    Obm = matdiv(Rfp, Rpp) @ Rp

    # # Funny rankv check (SVD takes too long). This check is needed to avoid rank deficiency warnings
    # if np.linalg.norm(Rpp[:, (2 * m+l) * i - 2 * l: (2 * m + l) * i], 'fro') < 1e-10:
    #     Ob = (Rfp @ np.linalg.pinv(Rpp.conj().T).conj().T) @ Rp  # Oblique projection
    # else:
    #     Ob = matdiv(Rfp, Rpp) @ Rp

    # Determine the states Xi and Xip
    Xi  = np.linalg.pinv(gam) @ Ob
    Xip = np.linalg.pinv(gamm) @ Obm

    # ######################################################
    # #                       STEP 6                       #
    # ######################################################

    Rhs = np.vstack((Xi, R[m * i: m * (i + 1), :]))
    Lhs = np.vstack((Xip, R[(2 * m + l) * i: (2 * m + l) * i + l, :]))

    sol = matdiv(Lhs, Rhs)

    # Extract the system matrices
    A = sol[:n, :n]
    B = sol[:n, n:n+m]
    C = sol[n:n+l, :n]
    D = sol[n:n+l, n:n+m]

    return A, B, C, D, ss
