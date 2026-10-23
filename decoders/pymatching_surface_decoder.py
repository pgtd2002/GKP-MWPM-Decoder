import numpy as np
import pymatching
from scipy.sparse import csc_matrix

_DECODER_CACHE = {}


class GKPSurfaceDecoder:
    def __init__(self, G: np.ndarray):
        """
        Initializes the decoder and structures the check matrices.
        """
        self.G = np.array(G)
        self.N = self.G.shape[1] // 2

        # 1. Pre-extract g_q and g_p
        q_rows = np.any(self.G[:, :self.N] != 0, axis=1)
        self.g_q = np.ascontiguousarray(self.G[q_rows, :self.N])

        p_rows = np.any(self.G[:, self.N:] != 0, axis=1)
        self.g_p = np.ascontiguousarray(self.G[p_rows, self.N:])

        # 2. Convert to scipy sparse CSC format once
        self.H_q = csc_matrix(self.g_q)
        self.H_p = csc_matrix(self.g_p)

        # 3. Instantiate base matching objects once to reuse graph structures
        self.matching_q = pymatching.Matching.from_check_matrix(self.H_q)
        self.matching_p = pymatching.Matching.from_check_matrix(self.H_p)

    def decode_analog_batch(self, t_batch_T: np.ndarray) -> np.ndarray:
        """
        Hybrid approach: Vectorizes all math and cost calculations, but loops
        the graph generation since GKP analog weights change every single trial.
        """
        batch_size = t_batch_T.shape[0]
        t_prime = t_batch_T / np.sqrt(np.pi)

        t_q = t_prime[:, :self.N]
        t_p = t_prime[:, self.N:]

        chi_prime_q = np.rint(t_q)
        chi_prime_p = np.rint(t_p)

        # 1. BATCHED Syndromes
        syndrome_q_batch = np.mod(chi_prime_p @ self.g_q.T, 2).astype(np.uint8)
        syndrome_p_batch = np.mod(chi_prime_q @ self.g_p.T, 2).astype(np.uint8)

        # 2. BATCHED Analog Costs
        c_p = np.rint(t_p)
        w_p = np.where(t_p < c_p, c_p - 1.0, c_p + 1.0)
        costs_p_batch = (w_p - t_p) ** 2 - (c_p - t_p) ** 2

        c_q = np.rint(t_q)
        w_q = np.where(t_q < c_q, c_q - 1.0, c_q + 1.0)
        costs_q_batch = (w_q - t_q) ** 2 - (c_q - t_q) ** 2

        # 3. SEQUENTIAL PyMatching (Required for dynamic weights)
        p_flips_batch = np.zeros((batch_size, self.N), dtype=np.uint8)
        q_flips_batch = np.zeros((batch_size, self.N), dtype=np.uint8)

        for i in range(batch_size):
            # PyMatching v2 is highly optimized to build from CSC matrices instantly
            matching_q = pymatching.Matching(self.H_q, weights=costs_p_batch[i])
            p_flips_batch[i] = matching_q.decode(syndrome_q_batch[i])

            matching_p = pymatching.Matching(self.H_p, weights=costs_q_batch[i])
            q_flips_batch[i] = matching_p.decode(syndrome_p_batch[i])

        # 4. BATCHED Corrections
        chi_prime_q_corrected = np.rint(t_q)
        mask_q = (q_flips_batch == 1)
        if np.any(mask_q):
            shift_q = np.where(t_q[mask_q] < chi_prime_q_corrected[mask_q], -1.0, 1.0)
            chi_prime_q_corrected[mask_q] += shift_q

        chi_prime_p_corrected = np.rint(t_p)
        mask_p = (p_flips_batch == 1)
        if np.any(mask_p):
            shift_p = np.where(t_p[mask_p] < chi_prime_p_corrected[mask_p], -1.0, 1.0)
            chi_prime_p_corrected[mask_p] += shift_p

        # Combine results
        chi_prime_prime = np.hstack([chi_prime_q_corrected, chi_prime_p_corrected])
        return chi_prime_prime * np.sqrt(np.pi)


def _decoder_cache_key(G: np.ndarray):
    G_binary = np.ascontiguousarray(np.asarray(G, dtype=np.uint8) % 2)
    return (G_binary.shape, G_binary.tobytes()), G_binary


def decode_surface_gkp(t, G):
    key, G_binary = _decoder_cache_key(G)
    decoder = _DECODER_CACHE.get(key)

    if decoder is None:
        decoder = GKPSurfaceDecoder(G_binary)
        _DECODER_CACHE[key] = decoder

    return decoder.decode(np.asarray(t))