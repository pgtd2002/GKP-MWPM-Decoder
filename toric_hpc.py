import argparse
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# MUST be placed before importing numpy to prevent OpenBLAS thread thrashing!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import scipy.sparse as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

from decoders.pymatching_toric_decoder_2 import GKPSurfaceDecoder
from functions.binary_stabilizer_generators import import_gen_matrix
from functions.concatenation import build_css_gkp_rref
from functions.lattice import Omega_qqpp

SQRT_PI = np.sqrt(np.pi)
SQRT_2PI = np.sqrt(2 * np.pi)
INV_SQRT_PI = 1.0 / SQRT_PI
INV_SQRT_2PI = 1.0 / SQRT_2PI
_WORKER = {}


def get_server_core_count(max_workers=None):
    if max_workers is not None:
        return max(1, int(max_workers))
    for name in ("SLURM_CPUS_PER_TASK", "PBS_NP", "NSLOTS"):
        value = os.environ.get(name)
        if value:
            return max(1, int(value))
    total = os.cpu_count() or 2
    return max(1, total // 2)


def get_system_l3_cache_bytes():
    try:
        output = subprocess.check_output(["lscpu"]).decode("utf-8")
        for line in output.split("\n"):
            if "L3 cache:" in line or "L3:" in line:
                parts = line.split(":")[-1].strip().split()
                size = float(parts[0])
                unit = parts[1].upper()
                if "M" in unit:
                    return int(size * 1024 * 1024)
                elif "K" in unit:
                    return int(size * 1024)
    except Exception:
        pass
    return 128 * 1024 * 1024


def get_optimal_batch_size(dim, num_workers):
    total_l3_bytes = get_system_l3_cache_bytes()
    worker_cache_slice = (total_l3_bytes / num_workers) * 0.75
    bytes_per_trial = 5 * dim * 8
    optimal_batch = int(worker_cache_slice // bytes_per_trial)

    # CRITICAL CHANGE: Force a minimum batch of 2000 to crush the Python loop overhead
    return int(np.clip(optimal_batch, a_min=2000, a_max=20000))


def make_sigmas():
    sigmas_low = np.linspace(0.40, 0.52, 4)
    sigmas_mid = np.linspace(0.52, 0.65, 30)
    sigmas_high = np.linspace(0.65, 0.80, 6)
    sigmas_fine = np.arange(0.601,0.607,0.001)
    #return np.concatenate([sigmas_low, sigmas_mid, sigmas_high])
    return sigmas_fine


def prepare_distance_context(distance, folder_prefix):
    G, Hx, Hz = import_gen_matrix(distance, folder_prefix)
    Omega = Omega_qqpp(G.shape[1] // 2)

    # Generate the base matrix
    M_dense = build_css_gkp_rref(Hx, Hz)

    # 1. Convert to Sparse Matrices instantly
    M_csc = sp.csc_matrix(M_dense)
    Omega_csc = sp.csc_matrix(Omega)
    Omega_T_csc = Omega_csc.T.tocsc()

    # 2. Sparse Precomputation: A = sqrt(2pi) * M * Omega^T
    A_csc = M_csc.dot(Omega_T_csc)
    A_csc.data *= SQRT_2PI

    return {
        "G": np.asarray(G, dtype=np.uint8),
        "A_sparse": A_csc,
        "Omega_csc": Omega_csc,  # <--- Added this to the dictionary!
        "M_csc": M_csc,
        "M_T_csc": M_csc.T.tocsc(),
        "dim": M_dense.shape[0],
    }


def init_worker(context):
    from threadpoolctl import threadpool_limits
    from scipy.sparse.linalg import splu
    threadpool_limits(limits=1)

    _WORKER.clear()
    _WORKER.update(context)
    _WORKER["decoder"] = GKPSurfaceDecoder(context["G"])

    dim = context["dim"]

    # DYNAMIC FAST-PATH: Use 100% Dense Matrix math for smaller lattices
    if dim < 1500:
        _WORKER["use_dense"] = True
        # Precompute the exact inverse once
        _WORKER["M_inv"] = np.linalg.inv(context["M_csc"].toarray())
        _WORKER["M_inv_T"] = _WORKER["M_inv"].T

        # FULLY DENSIFY A AND OMEGA TO UNLOCK CPU AVX VECTORIZATION
        _WORKER["A_dense"] = context["A_sparse"].toarray()
        _WORKER["Omega_dense"] = context["Omega_csc"].toarray()
    else:
        _WORKER["use_dense"] = False
        _WORKER["splu_M"] = splu(context["M_csc"])
        _WORKER["splu_MT"] = splu(context["M_T_csc"])


def run_trial_chunk(args):
    sigma, num_trials, seed, batch_size = args
    rng = np.random.default_rng(seed)
    success_count = 0

    decoder = _WORKER["decoder"]
    dim = _WORKER["dim"]
    use_dense = _WORKER["use_dense"]

    # --- ROUTE THE MATRICES ---
    if use_dense:
        A_mat = _WORKER["A_dense"]
        Omega_mat = _WORKER["Omega_dense"]
        M_inv = _WORKER["M_inv"]
        M_inv_T = _WORKER["M_inv_T"]
    else:
        A_mat = _WORKER["A_sparse"]
        Omega_mat = _WORKER["Omega_csc"]
        splu_M = _WORKER["splu_M"]
        splu_MT = _WORKER["splu_MT"]

    TWO_PI = 2 * np.pi

    for i in range(0, num_trials, batch_size):
        current_batch = min(batch_size, num_trials - i)

        xi = rng.standard_normal(size=(dim, current_batch)) * sigma

        # Because A_mat is now purely dense, this triggers raw BLAS vectorization
        s = np.mod(A_mat.dot(xi) + np.pi, TWO_PI) - np.pi

        if use_dense:
            y = M_inv.dot(s)
        else:
            y = splu_M.solve(s)

        t_batch = Omega_mat.dot(y) * INV_SQRT_2PI

        # 2. PYMATCHING DECODE
        t_batch_T = np.ascontiguousarray(t_batch.T)
        cp_batch_T = decoder.decode_analog_batch(t_batch_T)
        cp_batch = cp_batch_T.T

        # 3. VERIFICATION
        residual_error = xi - t_batch + cp_batch
        effective_error = np.rint(residual_error * INV_SQRT_PI) * SQRT_PI

        if use_dense:
            y_verify = M_inv_T.dot(effective_error)
        else:
            y_verify = splu_MT.solve(effective_error)

        z = y_verify * INV_SQRT_2PI

        is_lattice = np.max(np.abs(z - np.rint(z)), axis=0) < 1e-4
        success_count += np.sum(is_lattice)

    return int(success_count)



def split_trials(num_trials, num_chunks):
    num_chunks = max(1, min(num_trials, num_chunks))
    base = num_trials // num_chunks
    remainder = num_trials % num_chunks
    return [base + (i < remainder) for i in range(num_chunks)]


def monte_carlo_for_sigma(executor, sigma, num_trials, num_workers, seed, batch_size):
    MIN_CHUNK_SIZE = 100
    max_useful_chunks = max(1, num_trials // MIN_CHUNK_SIZE)
    actual_chunks = min(num_workers * 4, max_useful_chunks)

    chunk_counts = split_trials(num_trials, actual_chunks)
    seeds = np.random.SeedSequence(seed).generate_state(len(chunk_counts))
    tasks = [
        (sigma, count, int(chunk_seed), batch_size)
        for count, chunk_seed in zip(chunk_counts, seeds)
    ]

    success_count = sum(executor.map(run_trial_chunk, tasks))
    return success_count / num_trials


def get_dynamic_trials(sigma, base_trials):
    if sigma < 0.52:
        return min(base_trials, 5000)
    elif sigma > 0.62:
        return min(base_trials, 5000)
    else:
        return base_trials


def save_results(
    sigmas, distances, all_fidelities, num_trials, output_dir, genus, code_type
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sigma_steps = len(sigmas)
    distance_label = f"d{min(distances)}-{max(distances)}"

    filename = f"{code_type}_toric_gkp_g{genus}_{distance_label}_trials{num_trials}_sigma_steps{sigma_steps}.npz"
    filepath = output_path / filename

    fidelities = np.array(
        [all_fidelities[distance] for distance in distances], dtype=float
    )

    np.savez_compressed(
        filepath,
        sigmas=np.asarray(sigmas, dtype=float),
        distances=np.asarray(distances, dtype=int),
        fidelities=fidelities,
        num_trials=int(num_trials),
        sigma_steps=int(sigma_steps),
        genus=int(genus),
        code_type=str(code_type),
    )
    return filepath


def save_plot(
    sigmas, distances, all_fidelities, output_dir, data_file, genus, code_type
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {
        6: "forestgreen",
        8: "crimson",
        10: "purple",
        12: "teal",
        14: "black",
        16: "gold",
        18: "magenta",
        20: "navy",
        22: "brown",
        24: "olive",
        26: "cyan",
        28: "grey",
        30: "royalblue",
        32: "darkorange",
    }

    for distance in distances:
        fidelities = np.asarray(all_fidelities[distance], dtype=float)
        infidelities = 1.0 - fidelities
        color = colors.get(distance, None)

        ax1.plot(
            sigmas,
            fidelities,
            marker="o",
            linestyle="-",
            color=color,
            label=f"$d={distance}$",
        )
        ax2.plot(
            sigmas,
            infidelities,
            marker="s",
            linestyle="-",
            color=color,
            label=f"$d={distance}$",
        )

    type_label = "Rotated" if code_type == "rt" else "Standard"
    ax1.set_title(
        rf"Genus-{genus} {type_label} Toric GKP Code Fidelity vs. Noise ($\sigma$)"
    )
    ax1.set_xlabel(r"Noise Strength ($\sigma$)")
    ax1.set_ylabel(r"Logical Fidelity ($P_{success}$)")
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.legend()

    ax2.set_title(r"Logical Infidelity vs. Noise Strength")
    ax2.set_xlabel(r"Noise Strength ($\sigma$)")
    ax2.set_ylabel(r"Logical Infidelity ($1 - P_{success}$)")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.legend()

    fig.tight_layout()
    plot_file = Path(output_dir) / f"{data_file.stem}.png"
    fig.savefig(plot_file, dpi=1600)
    plt.close(fig)
    return plot_file


def load_checkpoint(filepath):
    if os.path.exists(filepath):
        print(f"\n[!] Found checkpoint at {filepath}! Resuming progress...\n")
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None


def save_checkpoint(state, filepath):
    temp_file = filepath + ".tmp"
    with open(temp_file, "wb") as f:
        pickle.dump(state, f)
    os.replace(temp_file, filepath)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Server-optimized Sparse GKP Monte Carlo simulation."
    )
    parser.add_argument(
        "--distances", type=int, nargs="+", default=[i for i in range(6, 32, 2)]
    )
    parser.add_argument("--num-trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=461266)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--genus", type=int, default=4, help="Genus of the toric code (e.g., 1, 2, ...)"
    )
    parser.add_argument(
        "--code-type",
        type=str,
        choices=["rt", "st"],
        default="rt",
        help="'rt' for rotated, 'st' for standard",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    distances = list(args.distances)
    sigmas = make_sigmas()
    num_workers = get_server_core_count(args.workers)

    genus = args.genus
    code_type = args.code_type
    folder_prefix = f"{code_type}_g{genus}"
    type_label = "Rotated" if code_type == "rt" else "Standard"

    file_prefix = f"checkpoint_{code_type}_g{genus}_d{min(distances)}-{max(distances)}_trials{args.num_trials}_seed{args.seed}.pkl"
    checkpoint_file = Path(args.output_dir) / file_prefix
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    state = load_checkpoint(checkpoint_file)
    if state:
        all_fidelities = state["all_fidelities"]
        start_d_idx = state["d_idx"]
        start_s_idx = state["s_idx"]
    else:
        all_fidelities = {}
        start_d_idx = 0
        start_s_idx = 0

    print(
        f"Starting SPARSE batched {type_label} Genus-{genus} simulations with {num_workers} physical workers."
    )
    print(f"Loading matrices from folder pattern: '{folder_prefix}'")
    print(f"Distances: {distances}")
    print(f"Sigma steps: {len(sigmas)}")
    print(f"Max Trials per sigma: {args.num_trials}")

    for d_idx in range(start_d_idx, len(distances)):
        distance = distances[d_idx]
        print(f"\n{'=' * 50}")
        print(
            f" INITIALIZING: {type_label.upper()} GENUS-{genus} | DISTANCE d={distance} "
        )
        print(f"{'=' * 50}")

        context = prepare_distance_context(distance, folder_prefix)

        if d_idx == start_d_idx and start_s_idx > 0:
            fidelities_for_d = all_fidelities.get(distance, [])
        else:
            fidelities_for_d = []
            start_s_idx = 0

        batch_size = get_optimal_batch_size(context["dim"], num_workers)

        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(context,),
        ) as executor:
            for s_idx in range(start_s_idx, len(sigmas)):
                sigma = sigmas[s_idx]
                sigma_seed = args.seed + 100000 * distance + s_idx
                current_trials = get_dynamic_trials(float(sigma), args.num_trials)

                print(
                    f"Simulating g={genus}({code_type}), d={distance}, sigma={sigma:.4f} | Trials: {current_trials} | Batch: {batch_size}...",
                    end="",
                    flush=True,
                )

                start_time = time.perf_counter()

                fidelity = monte_carlo_for_sigma(
                    executor=executor,
                    sigma=float(sigma),
                    num_trials=current_trials,
                    num_workers=num_workers,
                    seed=sigma_seed,
                    batch_size=batch_size,
                )

                elapsed = time.perf_counter() - start_time
                fidelities_for_d.append(fidelity)
                all_fidelities[distance] = fidelities_for_d

                checkpoint_state = {
                    "all_fidelities": all_fidelities,
                    "d_idx": d_idx,
                    "s_idx": s_idx + 1,
                }
                save_checkpoint(checkpoint_state, str(checkpoint_file))

                if elapsed < 60:
                    time_str = f"{elapsed:.1f}s"
                elif elapsed < 3600:
                    time_str = f"{int(elapsed // 60)}m {elapsed % 60:.1f}s"
                else:
                    time_str = f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m"

                print(f" -> Fidelity: {fidelity:.8f} | Time: [{time_str}]")

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    data_file = save_results(
        sigmas=sigmas,
        distances=distances,
        all_fidelities=all_fidelities,
        num_trials=args.num_trials,
        output_dir=args.output_dir,
        genus=genus,
        code_type=code_type,
    )
    print(f"\nSaved final data to {data_file}")

    if not args.no_plot:
        plot_file = save_plot(
            sigmas,
            distances,
            all_fidelities,
            args.output_dir,
            data_file,
            genus,
            code_type,
        )
        print(f"Saved plot to {plot_file}")


if __name__ == "__main__":
    main()
