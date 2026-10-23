import numpy as np






def build_concatenated_gkp_algorithm7(G_binary):
    """
    Algorithm 7 from the paper.

    Input:
        G_binary : (m × 2N) stabilizer generator matrix over GF(2)

    Output:
        M : (2N × 2N) lattice generator (may be rank deficient)
    """
    G = G_binary.copy() % 2
    m, total_cols = G.shape
    N = total_cols // 2

    # Start with separable lattice
    M = np.sqrt(2) * np.eye(2 * N)

    replaced = np.zeros(2 * N, dtype=bool)

    # for each stabilizer generator
    for j in range(m):

        gj = G[j]

        # scan for first available nonzero position
        for l in range(2 * N):

            if gj[l] != 0:

                if not replaced[l]:

                    M[l, :] = gj / np.sqrt(2)

                    replaced[l] = True

                    break

    return np.array(M)













# =========================
# GF(2) operations
# =========================

def gf2_swap_rows(M, i, j):
    M[[i, j]] = M[[j, i]]


def gf2_swap_cols(M, i, j):
    M[:, [i, j]] = M[:, [j, i]]


def gf2_row_add(M, src, dst):
    M[dst] = (M[dst] + M[src]) % 2


def gf2_independent_row_basis(M):
    """
    Return an independent row basis of M over GF(2).
    Redundant rows are removed so downstream rank assumptions hold.
    """
    A = M.copy() % 2
    m, n = A.shape
    r = 0

    for col in range(n):
        pivot = None
        for row in range(r, m):
            if A[row, col] == 1:
                pivot = row
                break

        if pivot is None:
            continue

        if pivot != r:
            gf2_swap_rows(A, pivot, r)

        for row in range(m):
            if row != r and A[row, col] == 1:
                gf2_row_add(A, r, row)

        r += 1
        if r == m:
            break

    return A[:r].copy(), r



# ==========================================================
# Correct stabilizer standard form
# ==========================================================

def stabilizer_standard_form(G):

    G = G.copy() % 2
    G, _ = gf2_independent_row_basis(G)

    m, total_cols = G.shape
    N = total_cols // 2

    col_perm = list(range(total_cols))

    # ======================================================
    # STEP 1 — X elimination
    # ======================================================

    r = 0

    for col in range(N):

        pivot = None

        for row in range(r, m):
            if G[row, col] == 1:
                pivot = row
                break

        if pivot is None:
            continue

        if pivot != r:
            gf2_swap_rows(G, pivot, r)

        for row in range(m):
            if row != r and G[row, col] == 1:
                gf2_row_add(G, r, row)

        if col != r:
            gf2_swap_cols(G, col, r)
            col_perm[col], col_perm[r] = (
                col_perm[r],
                col_perm[col],
            )

        r += 1

        if r == m:
            break

    # parameters

    n2 = m - r
    pivot_row = r

    # ======================================================
    # STEP 2 — Z elimination with column placement
    # ======================================================

    for i in range(n2):

        target_col = N + r + i

        pivot_col = None
        pivot_row_found = None

        # search pivot anywhere in Z

        for col in range(N, 2 * N):

            for row in range(pivot_row, m):

                if G[row, col] == 1:

                    pivot_col = col
                    pivot_row_found = row
                    break

            if pivot_col is not None:
                break

        if pivot_col is None:
            raise ValueError("Z block rank deficiency")

        # move pivot row

        if pivot_row_found != pivot_row:
            gf2_swap_rows(G, pivot_row_found, pivot_row)

        # clear column

        for row in range(m):
            if row != pivot_row and G[row, pivot_col] == 1:
                gf2_row_add(G, pivot_row, row)

        # move pivot column into canonical position

        if pivot_col != target_col:

            gf2_swap_cols(G, pivot_col, target_col)

            col_perm[pivot_col], col_perm[target_col] = (
                col_perm[target_col],
                col_perm[pivot_col],
            )

        pivot_row += 1

    info = dict(
        N=N,
        m=m,
        r=r,
        n2=n2,
        k=N - m,
        column_permutation=col_perm,
    )

    return G, r, col_perm



def build_concatenated_gkp_generator_qqpp(G_binary):

    G_std, r, col_perm = stabilizer_standard_form(G_binary)

    m, total_cols = G_std.shape
    N = total_cols // 2

    k  = N - m
    n2 = m - r   # = N - k - r

    n1 = r
    n3 = k

    # Column block offsets (in permuted basis)
    b1 = 0          # q: I    width r
    b2 = r          # q: A₁   width n2   ← Block 5 goes here
    b3 = r + n2     # q: A₂   width k    ← Block 3 goes here
    b4 = N          # p: B/D  width r    ← Block 4 goes here
    b5 = N + r      # p: 0/I  width n2   (already in stab rows)
    b6 = N + r + n2 # p: C/E  width k    ← Block 6 goes here

    M = np.zeros((2 * N, 2 * N), dtype=float)
    row = 0

    # Block 1: r rows — [I | A₁ | A₂ | B | 0 | C]
    for i in range(r):
        M[row] = G_std[i]
        row += 1

    # Block 2: n2 rows — [0 | 0 | 0 | D | I | E]
    for i in range(n2):
        M[row] = G_std[r + i]
        row += 1

    # Block 3: k rows — [0 | 0 | 2I | 0 | 0 | 0]  (q-side at b3)
    for i in range(n3):
        M[row, b3 + i] = 2
        row += 1

    # Block 4: r rows — [0 | 0 | 0 | 2I | 0 | 0]  (p-side at b4)
    for i in range(n1):
        M[row, b4 + i] = 2
        row += 1

    # Block 5: n2 rows — [0 | 2I | 0 | 0 | 0 | 0]  (q-side at b2)  ← YOUR BUG WAS HERE
    for i in range(n2):
        M[row, b2 + i] = 2
        row += 1

    # Block 6: k rows — [0 | 0 | 0 | 0 | 0 | 2I]  (p-side at b6)
    for i in range(n3):
        M[row, b6 + i] = 2
        row += 1

    assert row == 2 * N, f"Row count mismatch: {row} != {2*N}"
    # ... [existing block logic] ...

    assert row == 2 * N, f"Row count mismatch: {row} != {2 * N}"
    M = M / np.sqrt(2)


    # # --- NEW: Reverse the column permutation ---
    # # col_perm[i] tells us the physical qubit index that is currently sitting at column i
    # M_physical = np.zeros_like(M)
    # for permuted_idx, physical_idx in enumerate(col_perm):
    #     M_physical[:, physical_idx] = M[:, permuted_idx]

    info = {
        "N": N,
        "k": k,
        "num_stabilizers": m,
        "r": r,
        "n2": n2,
        "column_permutation": col_perm,
    }

    # Return the un-permuted physical lattice matrix
    return M, info


    return M, info




####################################
####################################
####################################






def gf2_rref_no_col_swaps(H):
    """
    Computes RREF over GF(2) using ONLY row operations.
    Returns the RREF matrix and a list of pivot column indices.
    """
    A = H.copy() % 2
    m, n = A.shape
    pivots = []
    r = 0

    for c in range(n):
        if r >= m:
            break

        pivot_row = None
        for i in range(r, m):
            if A[i, c] == 1:
                pivot_row = i
                break

        if pivot_row is not None:
            if pivot_row != r:
                A[[r, pivot_row]] = A[[pivot_row, r]]

            for i in range(m):
                if i != r and A[i, c] == 1:
                    A[i] = (A[i] + A[r]) % 2

            pivots.append(c)
            r += 1

    return A, pivots


def build_css_gkp_rref(Hx, Hz):
    """
    Topologically safe lattice builder for CSS codes (including higher-genus).

    Inputs:
        Hx : (m_X × N) binary matrix for X stabilizers
        Hz : (m_Z × N) binary matrix for Z stabilizers

    Output:
        M : (2N × 2N) full-rank GKP lattice matrix
    """
    mx, N = Hx.shape
    mz, N_z = Hz.shape

    assert N == N_z, "Hx and Hz must have the same number of columns (N)."

    # 1. Process X and Z blocks independently without column swaps
    Hx_rref, pivots_x = gf2_rref_no_col_swaps(Hx)
    Hz_rref, pivots_z = gf2_rref_no_col_swaps(Hz)

    if len(pivots_x) < mx or len(pivots_z) < mz:
        raise ValueError("Either Hx or Hz is not full rank over GF(2). Remove dependent rows first.")

    # 2. Start with the completely separable GKP base lattice
    M = np.sqrt(2) * np.eye(2 * N)

    # 3. Embed the X stabilizers into the q-quadrature (left half of M)
    for i in range(mx):
        pivot_col = pivots_x[i]
        M[pivot_col, :N] = Hx_rref[i] / np.sqrt(2)

    # 4. Embed the Z stabilizers into the p-quadrature (right half of M)
    # Note: Z pivots are offset by N in the 2N dimensional phase space
    for i in range(mz):
        pivot_col = pivots_z[i]
        M[N + pivot_col, N:] = Hz_rref[i] / np.sqrt(2)

    return M

