import numpy as np
import pymatching
from scipy.sparse import csc_matrix
import numba as nb

# =====================================================================
# NUMBA JIT COMPILED CORE MATH
# (Fuses operations and eliminates NumPy temporary array allocations)
# =====================================================================
SQRT_PI = np.sqrt(np.pi)
INV_SQRT_PI = 1.0 / SQRT_PI

@nb.njit(fastmath=True)
def compute_gkp_costs_numba(t_prime, out_costs, offset):
    """Iterates directly over the contiguous array using a column offset."""
    rows, cols = out_costs.shape
    for i in range(rows):
        for j in range(cols):
            t = t_prime[i, j + offset]
            c = np.round(t)
            w = c - 1.0 if t < c else c + 1.0
            out_costs[i, j] = (w - t) ** 2 - (c - t) ** 2

@nb.njit(fastmath=True)
def apply_corrections_numba(t_prime, flips_batch, out_corrected, offset):
    rows, cols = flips_batch.shape
    for i in range(rows):
        for j in range(cols):
            t = t_prime[i, j + offset]
            c = np.round(t)
            if flips_batch[i, j] == 1:
                shift = -1.0 if t < c else 1.0
                out_corrected[i, j] = c + shift
            else:
                out_corrected[i, j] = c


# =====================================================================
# THE OPTIMIZED DECODER CLASS
# =====================================================================


class GKPSurfaceDecoder:
    def __init__(self, G: np.ndarray):
        self.G = np.array(G)
        self.N = self.G.shape[1] // 2

        # 1. Pre-extract g_q and g_p
        q_rows = np.any(self.G[:, : self.N] != 0, axis=1)
        self.g_q = np.ascontiguousarray(self.G[q_rows, : self.N])

        p_rows = np.any(self.G[:, self.N :] != 0, axis=1)
        self.g_p = np.ascontiguousarray(self.G[p_rows, self.N :])

        # 2. Convert to scipy sparse CSC format once
        self.H_q = csc_matrix(self.g_q)
        self.H_p = csc_matrix(self.g_p)

        # 3. Instantiate base matching objects once
        self.matching_q = pymatching.Matching.from_check_matrix(self.H_q)
        self.matching_p = pymatching.Matching.from_check_matrix(self.H_p)

        # 4. Initialize buffer tracking variables
        self._max_batch_size = 0

    def _check_and_allocate_buffers(self, batch_size):
        """Ensures we only allocate RAM once per dynamic pool size."""
        if batch_size > self._max_batch_size:
            self._costs_p = np.empty((batch_size, self.N), dtype=np.float64)
            self._costs_q = np.empty((batch_size, self.N), dtype=np.float64)
            self._p_flips = np.empty((batch_size, self.N), dtype=np.uint8)
            self._q_flips = np.empty((batch_size, self.N), dtype=np.uint8)
            self._chi_prime_q_corrected = np.empty((batch_size, self.N), dtype=np.float64)
            self._chi_prime_p_corrected = np.empty((batch_size, self.N), dtype=np.float64)
            self._max_batch_size = batch_size

    def decode_analog_batch(self, t_batch_T: np.ndarray) -> np.ndarray:
        batch_size = t_batch_T.shape[0]

        # Ensure pre-allocated memory pools are large enough
        self._check_and_allocate_buffers(batch_size)

        # Grab exact slices of the pre-allocated buffers
        costs_p = self._costs_p[:batch_size]
        costs_q = self._costs_q[:batch_size]
        p_flips = self._p_flips[:batch_size]
        q_flips = self._q_flips[:batch_size]
        chi_prime_q_corr = self._chi_prime_q_corrected[:batch_size]
        chi_prime_p_corr = self._chi_prime_p_corrected[:batch_size]

        # Normalize t globally (remains perfectly contiguous in memory!)
        t_prime = t_batch_T * INV_SQRT_PI

        # Compute chi_prime with contiguous memory
        chi_prime = np.rint(t_prime)
        chi_prime_q = chi_prime[:, : self.N]
        chi_prime_p = chi_prime[:, self.N :]

        # 1. BATCHED Syndromes (Using NumPy OpenBLAS)
        syndrome_q_batch = np.mod(chi_prime_p @ self.g_q.T, 2).astype(np.uint8)
        syndrome_p_batch = np.mod(chi_prime_q @ self.g_p.T, 2).astype(np.uint8)

        # 2. NUMBA FUSED Analog Costs (Pass the offset instead of a slice)
        compute_gkp_costs_numba(t_prime, costs_p, offset=self.N)
        compute_gkp_costs_numba(t_prime, costs_q, offset=0)

        # 3. SEQUENTIAL PyMatching
        for i in range(batch_size):
            matching_q = pymatching.Matching(self.H_q, weights=costs_p[i])
            p_flips[i] = matching_q.decode(syndrome_q_batch[i])

            matching_p = pymatching.Matching(self.H_p, weights=costs_q[i])
            q_flips[i] = matching_p.decode(syndrome_p_batch[i])

        # 4. NUMBA FUSED Corrections
        apply_corrections_numba(t_prime, q_flips, chi_prime_q_corr, offset=0)
        apply_corrections_numba(t_prime, p_flips, chi_prime_p_corr, offset=self.N)

        # Combine results
        chi_prime_prime = np.hstack([chi_prime_q_corr, chi_prime_p_corr])
        return chi_prime_prime * SQRT_PI
