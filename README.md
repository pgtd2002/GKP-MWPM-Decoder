# GKP-MWPM-Decoder

Analog-information-assisted minimum-weight perfect matching (MWPM) decoding for concatenated GKP–topological codes, with a Monte Carlo framework for estimating logical fidelity and thresholds under Gaussian displacement noise.

The code supports rotated surface-GKP codes and **multi-genus rotated toric-GKP codes**, a family of higher-genus toric codes that encode several logical qubits in one connected architecture. It was developed for my M.Sc. thesis at IIT Madras and runs from a laptop up to multi-node jobs on the PARAM Pravega supercomputer.

## How the decoder works

The decoder is a closest-point decoder in the framework of [Lin, Chamberland & Brown, *PRX Quantum* **4**, 040334 (2023)](https://doi.org/10.1103/PRXQuantum.4.040334). For each Monte Carlo trial:

1. **Noise.** Sample an i.i.d. Gaussian displacement ξ ~ N(0, σ²) on every quadrature of the n GKP modes (code-capacity model, ideal GKP measurements).
2. **Syndrome.** Compute the continuous GKP syndrome s = (√(2π) M Ωᵀ ξ) mod 2π from the lattice generator matrix M of the concatenated code, then map it to the target vector t.
3. **Hard decision.** Rescale t′ = t/√π and round each component to its nearest integer. The parities of these integers give the binary X- and Z-syndromes of the outer code.
4. **Analog edge weights.** Each mode becomes one edge in the matching graph, weighted by the difference between its squared distances to the next-nearest and nearest lattice points:

   w = (t′ − next-nearest integer)² − (t′ − nearest integer)²

   Modes measured close to a lattice point get a large weight and are unlikely to be flipped. Modes near the midpoint get a weight near zero.
5. **Matching.** PyMatching decodes the X and Z checks separately. Because the weights change on every trial, a new weighted graph is built per shot.
6. **Correction.** Each mode flagged by the matching is moved to its next-nearest integer. The result, rescaled by √π, is the estimated closest lattice point χ.
7. **Success check.** The trial succeeds if the residual ξ − t + χ lies in the code's stabilizer lattice. This is tested by solving against M and checking that the coefficients are integers.

Logical fidelity is the fraction of successful trials at each (distance, σ).

## Repository layout

```text
decoders/
  pymatching_toric_decoder.py   Main decoder: batched, Numba-compiled weight and
                                  correction kernels + PyMatching; used by all drivers
  pymatching_surface_decoder.py   Same decoder with vectorised NumPy kernels (no Numba)
  surface_decoder.py              Reference implementation using NetworkX
                                  max-weight matching (slow; for validation)
functions/
  binary_stabilizer_generators.py Rotated surface-code checks; loader for .mat check matrices
  concatenation.py                GKP lattice generators for concatenated CSS codes (GF(2) RREF)
  lattice.py                      Square/hexagonal GKP lattices and symplectic forms
  noise_simulator.py              Gaussian displacement sampling
stabilizer-matrices/              Precomputed CSS check matrices (Hx, Hz, Lx, Lz) as .mat
surface_hpc.py                    Rotated surface-GKP simulation (single node)
toric_hpc.py                      Multi-genus rotated toric-GKP simulation (single node)
toric_hpc_param_pravega.py        Multi-node version of toric_hpc.py for Slurm clusters
slurm-files/                      Example Slurm job scripts for PARAM Pravega
slurm_node_forecast.py            Helper that reports free and forecast Slurm CPU chunks
```

### Stabilizer matrices

Files are named `stabilizer-matrices/<prefix>_g<g>/<prefix>_g<g>_d<d>.mat`. Each holds `Hx`, `Hz`, `Lx` and `Lz`.

| Folder | Code | Genera | Distances |
| --- | --- | --- | --- |
| `rt_g<g>` | Multi-genus rotated toric code: n = g·d² modes, 2g logical qubits | 2, 4, 5, 8, 11–15 | even, 6–30 (some also 40, 44) |
| `rsc_g<g>` | Rotated surface-code-based comparison constructions | 2–10 | odd, 3–29 |

The driver scripts load the `rt_g<g>` matrices. The rotated surface code used by `surface_hpc.py` is generated in code.

## Requirements

Python 3 with:

```bash
pip install numpy scipy pymatching numba networkx matplotlib threadpoolctl
```

PyMatching 2 is assumed. `networkx` is only needed for the reference decoder.

## Usage

Run all scripts from the repository root, because matrix paths are relative.

```bash
# Rotated surface-GKP code, odd distances
python surface_hpc.py --distances 3 5 7 9 --num-trials 100000

# Genus-2 rotated toric-GKP code on one machine
python toric_hpc.py --genus 2 --distances 6 8 10 12 --num-trials 100000

# Multi-node run (inside a Slurm allocation; see slurm-files/)
srun python -u toric_hpc_param_pravega.py --genus 13 --code-type rt \
     --distances 6 8 10 --num-trials 1000000000
```

Common options: `--seed`, `--workers` (defaults to the number of physical cores), `--output-dir` (default `results/`) and `--no-plot`.

The σ grid is set in `make_sigmas()` in each driver. It currently holds a fine grid around the threshold (about 0.60–0.61), and the commented lines restore the wide sweep.

**Outputs.** Each run writes a compressed `.npz` file with the σ values, the distances and one `fidelity_d<d>` array per distance, plus a fidelity-versus-σ plot. Progress is checkpointed to `.pkl` files, so an interrupted run resumes where it stopped.

## Performance notes

- Weight and correction kernels are Numba-compiled with pre-allocated buffers, and syndromes are computed in batches.
- Workers are pinned to physical cores, and BLAS threading is limited to one thread per process to avoid oversubscription.
- Small codes use a dense linear-algebra path. Larger codes use sparse matrices with a SuperLU factorisation of M, so memory stays bounded at high genus and distance.
- In the multi-node driver, trials are sharded across Slurm ranks with independent seeds. Each rank checkpoints its own shard, and rank 0 aggregates the exact success counts.

## Results

Using this code, I found thresholds of about σ ≈ 0.6025–0.6040 for the multi-genus rotated toric-GKP codes across the genera studied, consistently above the rotated surface-GKP threshold. The genus-2 code achieved up to about 20% higher logical fidelity than rotated surface-GKP codes of equal logical capacity.

The full analysis and plots are in the thesis repository: [msc-thesis-topological-gkp-codes](https://github.com/pgtd2002/msc-thesis-topological-gkp-codes). A manuscript is in preparation for *Physical Review A*.

## Author

**Pritam Ghosh**, M.Sc. Physics, IIT Madras.
Supervised by Prof. Shayan S. Garani (IISc Bengaluru) and co-supervised by Prof. Prabha Mandayam (IIT Madras).