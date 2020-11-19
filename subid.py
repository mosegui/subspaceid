import numpy as np
from scipy import linalg as sp_linalg

from blkhank import hankel_matrix
from gl2kr import gl2kr

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


def subid_det(y, i, n, u, w="SV"):

    input_observations, input_dimensions = u.shape  # nu, m
    output_observations, output_dimensions = y.shape  # ny, l

    assert output_observations == input_observations, "Number of data points different in input and output"
    assert i > 0, "Number of block rows should be positive"
    assert (output_observations - 2 * i + 1) >= (2 * output_dimensions * i), "Not enough data points"


    j = output_observations - (2 * i - 1)  # Determine the number of columns in the Hankel matrices

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

    mi2 = 2 * m * i

    Rf = R[(2*m+l)*i:2*(m+l)*i, :]  # Future outputs
    Rp = np.vstack((R[:m * i,:], R[2 * m * i: (2 * m + l) * i,:]))  # Past(inputs and) outputs

    Ru = R[m * i:2 * m * i, :mi2]  # Future inputs
    # Perpendicular Future outputs
    temp1 = matdiv(Rf[:, :mi2], Ru)
    Rfp = np.hstack((Rf[:,:mi2] - temp1 @ Ru, Rf[:, mi2:2*(m+l)*i]))
    # Perpendicular Past
    # temp2 = np.linalg.solve(Ru,Rp[:,:mi2])
    temp2 = matdiv(Rp[:, :mi2], Ru)
    Rpp = np.hstack((Rp[:,:mi2] - temp2 @ Ru, Rp[:, mi2:2*(m+l)*i]))

    # The oblique projection: Computed as 6.1 on page 166
    # obl/Ufp = Yf/Ufp * pinv(Wp/Ufp) * (Wp/Ufp)
    # The extra projection on Ufp (Uf perpendicular) tends to give better numerical conditioning (see algo page 131)
    # And it is needed both for CVA as MOESP

    # Funny rankv check (SVD takes too long)
    # This check is needed to avoid rank deficiency warnings
    if np.linalg.norm(Rpp[:, (2 * m+l) * i - 2 * l: (2 * m + l) * i], 'fro') < 1e-10:
        Ob = (Rfp @ np.linalg.pinv(Rpp.conj().T).conj().T) @ Rp  # Oblique projection
    else:
        Ob = matdiv(Rfp, Rpp) @ Rp

    ######################################################
    #                       STEP 2                       #
    ######################################################
    # Compute the SVD
    # Compute the matrix WOW we want to take an SVD of

    # Extra projection of Ob on Uf perpendicular
    temp3 = matdiv(Ob[:, :mi2], Ru)
    WOW = np.hstack((Ob[:, : mi2] - temp3 @ Ru, Ob[:, mi2: 2 * (m + l) * i]))

    if w == "CVA":
        W1i = np.triu(np.linalg.qr(Rf.conj().T, mode='complete')[1])
        W1i = W1i[:l*i, l*i].conj().T
        WOW = np.linalg.solve(W1i, WOW)

    U, S, V = np.linalg.svd(WOW)

    if w == "CVA":
        U = W1i @ U

    ss = np.diag(S)

    ######################################################
    #                       STEP 3                       #
    ######################################################

    U1 = U[:, :n]

    ######################################################
    #                       STEP 4                       #
    ######################################################

     # Determine gam and gamm
    gam = U1 @ np.diag(np.sqrt(np.diag(ss[:n])))
    gamm = U1[:l*(i-1), :] @ np.diag(np.sqrt(np.diag(ss[:n])))
    # and their pseudo inverses
    gam_inv = np.linalg.pinv(gam)
    gamm_inv = np.linalg.pinv(gamm)

    ######################################################
    #                       STEP 5                       #
    ######################################################

    # Determine the matrices A and C
    Rhs = np.vstack((np.hstack((gam_inv @ R[(2 * m + l) * i: 2 * (m + l) * i, : (2 * m + l) * i], np.zeros((n, l)))), R[m * i: 2 * m * i, : (2 * m + l) * i + l]))
    Lhs = np.vstack((gamm_inv @ R[(2 * m + l) * i + l: 2 * (m + l) * i, : (2 * m + l) * i + l], R[(2 * m + l) * i: (2 * m + l) * i + l, : (2 * m + l) * i + l]))

    #  Solve least squares
    sol = matdiv(Lhs, Rhs)
    # sol = np.linalg.lstsq(Rhs, Lhs)[0].T

    #   Extract the system matrices
    A = sol[0:n, 0:n]
    C = sol[n:n + l, 0:n]
    res = Lhs - sol @ Rhs  # Residuals

    ### RECOMPUTE gamm FROM A AND C

    gam = C
    for k in range(i-1):
        gam = np.concatenate((gam, gam[-2:] @ A), axis=0)
        ## TEST THIS! test case in: /home/mosegui/PycharmProjects/SSI/matrix_expansion.py

    gamm = gam[:l * (i - 1), :]

    gam_inv = np.linalg.pinv(gam)
    gamm_inv = np.linalg.pinv(gamm)

    ### RECOMPUTE THE STATES WITH THE NEW gamma

    Rhs = np.vstack((np.hstack((gam_inv @ R[(2 * m + l) * i: 2 * (m + l) * i, : (2 * m + l) * i], np.zeros((n, l)))), R[m * i: 2 * m * i, : (2 * m + l) * i + l]))
    Lhs = np.vstack((gamm_inv @ R[(2 * m + l) * i + l: 2 * (m + l) * i, : (2 * m + l) * i + l], R[(2 * m + l) * i: (2 * m + l) * i + l, : (2 * m + l) * i + l]))

    ######################################################
    #                       STEP 6                       #
    ######################################################

    # P and Q as on page 125
    P = Lhs - np.vstack((A, C)) @ Rhs[:n, :]
    P = P[:, :2*m*i]
    Q = R[m*i:2*m*i,:2*m*i]  # Future inputs

    # L1, L2, M as on page 119
    L1 = A @ gam_inv
    L2 = C @ gam_inv
    M = np.hstack((np.zeros((n, l)), gamm_inv))
    X = np.vstack(
        (
            np.hstack(
                (
                    np.eye(l),
                    np.zeros((l, n))
                )
            ),
            np.hstack(
                (
                    np.zeros((l*(i-1), l)),
                    gamm
                )
            )
        )
    )

    totm = 0
    for k in range(i):
        # Calculate N and the Kronecker products (page 126)
        N = np.vstack(
            (
                np.hstack(
                    (
                        M[:, k * l: l * i] - L1[:, k * l: l * i],
                        np.zeros((n,k*l))
                    )
                ),
                np.hstack(
                    (
                        -L2[:, k * l: l * i],
                        np.zeros((l,k*l))
                    )
                )
            )
        )

        if k == 0:
            N[n: n + l, : l] = np.eye(l) + N[n: n + l, : l]

        N = N @ X

        totm += np.kron(Q[k * m: (k + 1) * m, :].conj().T, N)


    # Solve least squares
    P = P.flatten("F").reshape(-1,1)
    sol = np.linalg.lstsq(totm, P)[0].reshape(-1,1)

    sol_bd = sol.reshape((n + l, m))
    D = sol_bd[:l, :]
    B = sol_bd[l:l+n, :]

    ######################################################
    #                       STEP 7                       #
    ######################################################
    if np.linalg.norm(res) > 1e-10:
        # edtermine QSR from the residuals
        cov = res @ res.conj().T

        Qs = cov[:n, :n]
        Ss = cov[:n, n:n+l]
        Rs = cov[n:n+l, n:n+l]

        sig = sp_linalg.solve_discrete_lyapunov(A, Qs)
        G = A @ sig @ C.conj().T + Ss
        L0 = C @ sig @ C.conj().T + Rs

        # Determine K and Ro
        K, Ro = gl2kr(A,G,C,L0)
    else:
        Ro = []
        K = []

    return A, B, C, D, K, Ro, ss


def subid_sto(y, i, n, w="CVA"):

    if i < 0:
        raise ValueError("Number of block rows should be positive")

    m = 0  # stochastic has no input

    # Turn the data into row vectors and check
    l, ny = y.shape
    if ny < l:
        y = y.conj().T
        l, ny = y.shape

    if (ny - 2 * i + 1) < ( 2 * l * i):
        raise ValueError("Not enough data points")

    # # Check the weight to be used
    # wn = 0
    # if len(w) == 2:
    #     if w.lower() == "sv":
    #         wn = 1
    #         if id_type == "deterministic":
    #             waux = 2
    #         else:
    #             waux = 3
    # elif len(w) == 3:
    #     if w.lower() == "cva":
    #         wn = 2
    #         if id_type == "deterministic":
    #             waux = 3
    #         else:
    #             waux = 1
    #
    # if wn == 0:
    #     # ERROR: w should be SV or CVS
    #     w = wn

    # Determine the number of columns in the Hankel matrices
    j = ny - 2 * i + 1

    # Compute the R factor
    Y = hankel_matrix(y.conj().T / np.sqrt(j), 2*i, j)

    R = np.triu(np.linalg.qr(Y.conj().T, mode='complete')[1]).conj().T
    R = R[: 2 * i * (m +l), : 2 * i * (m +l)]

    ######################################################
    #                       STEP 1                       #
    ######################################################

    mi2 = 2 * m * i

    Rf = R[(2*m+l)*i:2*(m+l)*i, :]  # Future outputs
    Rp = np.vstack((R[:m * i,:], R[2 * m * i: (2 * m + l) * i,:]))  # Past(inputs and) outputs

    # Ob = np.linalg.solve(Rp,Rf) @ Rp  which is the same as
    Ob = np.hstack((Rf[:, :l*i], np.zeros(l*i, l*i)))


    ######################################################
    #                       STEP 2                       #
    ######################################################
    # Compute the SVD
    # Compute the matrix WOW we want to take an SVD of

    WOW = Ob

    if w == "CVA":
        W1i = np.triu(np.linalg.qr(Rf.conj().T, mode='complete')[1])
        W1i = W1i[:l*i, l*i].conj().T
        WOW = np.linalg.solve(W1i, WOW)

    U, S, V = np.linalg.svd(WOW)

    if w == "CVA":
        U = W1i @ U

    ss = np.diag(S)

    ######################################################
    #                       STEP 3                       #
    ######################################################

    U1 = U[:, :n]

    ######################################################
    #                       STEP 4                       #
    ######################################################

     # Determine gam and gamm
    gam = U1 @ np.diag(np.sqrt(np.diag(ss[:n])))
    gamm = U1[:l*(i-1), :] @ np.diag(np.sqrt(np.diag(ss[:n])))
    # and their pseudo inverses
    gam_inv = np.linalg.pinv(gam)
    gamm_inv = np.linalg.pinv(gamm)

    ######################################################
    #                       STEP 5                       #
    ######################################################

    # Determine the matrices A and C
    Rhs = np.vstack((np.hstack((gam_inv @ R[(2 * m + l) * i: 2 * (m + l) * i, : (2 * m + l) * i], np.zeros((n, l)))), R[m * i: 2 * m * i, : (2 * m + l) * i + l]))
    Lhs = np.vstack((gamm_inv @ R[(2 * m + l) * i + l: 2 * (m + l) * i, : (2 * m + l) * i + l], R[(2 * m + l) * i: (2 * m + l) * i + l, : (2 * m + l) * i + l]))

    #  Solve least squares
    sol = matdiv(Lhs, Rhs)
    # sol = np.linalg.lstsq(Rhs, Lhs)[0].T

    #   Extract the system matrices
    A = sol[0:n, 0:n]
    C = sol[n:n + l, 0:n]
    res = Lhs - sol @ Rhs  # Residuals

    ### RECOMPUTE gamm FROM A AND C

    gam = C
    for k in range(i-1):
        gam = np.concatenate((gam, gam[-2:] @ A), axis=0)
        ## TEST THIS! test case in: /home/mosegui/PycharmProjects/SSI/matrix_expansion.py

    gamm = gam[:l * (i - 1), :]

    gam_inv = np.linalg.pinv(gam)
    gamm_inv = np.linalg.pinv(gamm)

    ######################################################
    #                       STEP 6                       #
    ######################################################

    B = []
    D = []

    ######################################################
    #                       STEP 7                       #
    ######################################################
    if np.linalg.norm(res) > 1e-10:
        # edtermine QSR from the residuals
        cov = res @ res.conj().T

        Qs = cov[:n, :n]
        Ss = cov[:n, n:n+l]
        Rs = cov[n:n+l, n:n+l]

        sig = sp_linalg.solve_discrete_lyapunov(A, Qs)
        G = A @ sig @ C.conj().T + Ss
        L0 = C @ sig @ C.conj().T + Rs

        # Determine K and Ro
        K, Ro = gl2kr(A,G,C,L0)
    else:
        Ro = []
        K = []

    return A, B, C, D, K, Ro, ss
