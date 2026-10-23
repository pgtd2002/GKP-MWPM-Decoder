import numpy as np



def square_lattice_qpqp(n):
    """
    Generate generator matrix for square GKP lattice
    ordering: (qpqp)
    """

    I = np.eye(n)

    M = np.block([
        [np.sqrt(2)*I, np.zeros((n,n))],
        [np.zeros((n,n)), np.sqrt(2)*I]
    ])

    return M


import numpy as np

def hex_lattice_generator_n_modes(n, scale=1.0):
    """
    Block-diagonal hex lattice generator for n modes.
    """

    prefactor = 3 ** (-0.25)

    M_hex = prefactor * np.array([
        [2.0, 0.0],
        [1.0, np.sqrt(3.0)]
    ])

    single = scale * M_hex

    M = np.zeros((2*n, 2*n))

    for i in range(n):
        M[2*i:2*i+2, 2*i:2*i+2] = single

    return M


def Omega_qpqp(n):
    """
    Omega = I_N ⊗ [[0,1],[-1,0]]

    Ordering:
    (q1,p1,q2,p2,...)
    """

    omega = np.array([
        [0, 1],
        [-1, 0]
    ])

    return np.kron(np.eye(n), omega)





def Omega_qqpp(n):
    """
    Omega for ordering:
    (q1,...,qn,p1,...,pn)
    """

    I = np.eye(n)
    Z = np.zeros((n, n))

    Omega = np.block([
        [Z, I],
        [-I, Z]
    ])

    return Omega



def qqpp_to_qpqp_permutation(n):
    """
    Permutation matrix converting:

    (q1,...,qn,p1,...,pn)
        →
    (q1,p1,q2,p2,...)
    """

    P = np.zeros((2*n, 2*n))

    for i in range(n):

        P[2*i, i] = 1
        P[2*i+1, n+i] = 1

    return P

def qpqp_to_qqpp_permutation(n):
    return np.linalg.inv(qqpp_to_qpqp_permutation(n))



def dual_lattice(M, ordering="qpqp"):

    n = M.shape[0] // 2
    if ordering == "qpqp":
        Om = Omega_qpqp(n)

    else:
        Om = Omega_qqpp(n)


    A = M @ Om @ M.T

    return Om @ np.linalg.inv(A) @ M


def symplectic_checker(S, ordering="qpqp", tol=1e-9):
    """
    Check if matrix S is symplectic.

    ordering:
        "qpqp" or "qqpp"
    """

    n = S.shape[0] // 2

    if ordering == "qpqp":
        Om = Omega_qpqp(n)

    elif ordering == "qqpp":
        Om = Omega_qqpp(n)

    else:
        raise ValueError("ordering must be 'qpqp' or 'qqpp'")

    return np.allclose(S @ Om @ S.T, Om, atol=tol)




def symplectic_error(S, ordering="qpqp"):
    """
    Return Frobenius norm of symplectic violation.
    """

    n = S.shape[0] // 2

    if ordering == "qpqp":
        Om = Omega_qpqp(n)
    else:
        Om = Omega_qqpp(n)

    diff = S @ Om @ S.T - Om

    return np.linalg.norm(diff)


import numpy as np

def is_in_stabilizer_lattice(correction, M, tol=1e-6):
    """
    Check whether a correction vector lies in the stabilizer lattice Λ(M).

    Parameters
    ----------
    correction : ndarray, shape (n,)
        Correction vector c.
    M : ndarray, shape (n, k)
        Stabilizer lattice generator matrix.
    tol : float
        Numerical tolerance for integer check.

    Returns
    -------
    bool
        True if correction ∈ Λ(M), else False.
    z : ndarray
        Integer lattice coordinates (rounded).
    """

    # Compute pseudoinverse
    M_pinv = np.linalg.pinv(M)

    # Solve for lattice coordinates
    z = M_pinv @ correction

    # Check if coordinates are integers
    z_round = np.round(z)

    is_integer = np.all(np.abs(z - z_round) < tol)

    return is_integer






def is_in_stabilizer_lattice_square_fast(correction, M, tol=1e-9):

    z = np.linalg.solve(M.T, correction)

    z_round = np.round(z)

    is_integer = np.all(np.abs(z - z_round) < tol)

    return is_integer








import numpy as np


def direct_sum(A, B, dtype=int):
    """
    Compute the direct sum of two matrices.

    Parameters
    ----------
    A : ndarray (m1, n1)
    B : ndarray (m2, n2)
    dtype : data type (default int)

    Returns
    -------
    ndarray (m1+m2, n1+n2)

    [ A  0 ]
    [ 0  B ]
    """

    m1, n1 = A.shape
    m2, n2 = B.shape

    top = np.hstack([
        A,
        np.zeros((m1, n2), dtype=dtype)
    ])

    bottom = np.hstack([
        np.zeros((m2, n1), dtype=dtype),
        B
    ])

    return np.vstack([
        top,
        bottom
    ])




def is_it_anti_symmetric_integer_valued_matrix(A, tol=1e-9):
    A = np.array(A)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False

    # Integer-valued up to tolerance
    if not np.allclose(A, np.round(A), atol=tol):
        return False

    # Anti-symmetric up to tolerance
    if not np.allclose(A.T, -A, atol=tol):
        return False

    return True