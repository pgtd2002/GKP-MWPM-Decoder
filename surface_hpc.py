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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle

from decoders.pymatching_toric_decoder_2 import GKPSurfaceDecoder
from functions.binary_stabilizer_generators import get_rotated_surface_code_matrix
from functions.concatenation import build_concatenated_gkp_algorithm7
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

    # Auto-detect physical cores! (64 logical threads // 2 = 32 physical cores)
    total = os.cpu_count() or 2
    return max(1, total // 2)


def get_system_l3_cache_bytes():
    """Dynamically asks the Linux kernel for the total L3 cache size in bytes."""
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
    return 128 * 1024 * 1024  # Safe Fallback: 128MB


def get_optimal_batch_size(dim, num_workers):
    """Calculates the exact batch size needed to saturate a worker's L3 cache slice."""
    total_l3_bytes = get_system_l3_cache_bytes()
    worker_cache_slice = (total_l3_bytes / num_workers) * 0.75
    bytes_per_trial = 5 * dim * 8
    optimal_batch = int(worker_cache_slice // bytes_per_trial)
    return int(np.clip(optimal_batch, a_min=200, a_max=10000))


def make_sigmas():
    #sigmas_low = np.linspace(0.40, 0.55, 6)
    sigmas_fine = np.arange(0.599, 0.605, 0.001)
    #sigmas_mid = np.linspace(0.55, 0.65, 10)
    #sigmas_high = np.linspace(0.65, 0.80, 4)

    #return np.concatenate([sigmas_low, sigmas_mid, sigmas_high])
    return sigmas_fine


def prepare_distance_context(distance):
    """Generates matrices for the planar surface code."""
    G, _, _ = get_rotated_surface_code_matrix(distance)
    Omega = Omega_qqpp(G.shape[1] // 2)
    M = build_concatenated_gkp_algorithm7(G)
    M_inv_T = np.linalg.inv(M.T)
    M_perp = Omega @ M_inv_T @ Omega.T

    return {
        "G": np.asarray(G, dtype=np.uint8),
        "A": np.ascontiguousarray(SQRT_2PI * (M @ Omega.T)),
        "B": np.ascontiguousarray(-(1 / SQRT_2PI) * (Omega @ M_perp).T),
        "M_inv_T": np.ascontiguousarray(M_inv_T),
        "dim": M.shape[0],
    }


def init_worker(context):
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1)

    _WORKER.clear()
    _WORKER.update(context)
    _WORKER["decoder"] = GKPSurfaceDecoder(context["G"])


def run_trial_chunk(args):
    sigma, num_trials, seed, batch_size = args
    rng = np.random.default_rng(seed)
    success_count = 0

    decoder = _WORKER["decoder"]
    dim = _WORKER["dim"]
    A = _WORKER["A"]
    B = _WORKER["B"]
    M_inv_T = _WORKER["M_inv_T"]

    TWO_PI = 2 * np.pi

    for i in range(0, num_trials, batch_size):
        current_batch = min(batch_size, num_trials - i)

        # 1. BATCHED LINEAR ALGEBRA
        xi = rng.standard_normal(size=(dim, current_batch)) * sigma
        s = np.mod((A @ xi) + np.pi, TWO_PI) - np.pi
        t_batch = B @ s

        # 2. INSTANTANEOUS C++ BATCHED DECODING
        t_batch_T = np.ascontiguousarray(t_batch.T)
        cp_batch_T = decoder.decode_analog_batch(t_batch_T)
        cp_batch = cp_batch_T.T

        # 3. BATCHED VERIFICATION
        residual_error = xi - t_batch + cp_batch

        effective_error = np.rint(residual_error * INV_SQRT_PI) * SQRT_PI
        z = (M_inv_T @ effective_error) * INV_SQRT_2PI

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
    if sigma < 0.55:
        return min(base_trials, 5000)
    elif sigma > 0.65:
        return min(base_trials, 5000)
    else:
        return base_trials


def save_results(sigmas, distances, all_fidelities, num_trials, output_dir, code_type):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sigma_steps = len(sigmas)
    distance_label = f"d{min(distances)}-{max(distances)}"

    filename = f"{code_type}_surface_gkp_{distance_label}_trials{num_trials}_sigma_steps{sigma_steps}.npz"
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
        code_type=str(code_type),
    )
    return filepath


def save_plot(sigmas, distances, all_fidelities, output_dir, data_file, code_type):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {
        3: "purple",
        5: "teal",
        7: "forestgreen",
        9: "crimson",
        11: "black",
        13: "gold",
        15: "magenta",
        17: "navy",
        19: "brown",
        21: "olive",
        23: "cyan",
        25: "grey",
        27: "royalblue",
        29: "darkorange",
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
    ax1.set_title(rf"{type_label} Surface GKP Code Fidelity vs. Noise ($\sigma$)")
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
        description="Server-optimized GKP Surface Code Monte Carlo simulation."
    )
    # Defaulting to odd distances (3, 5, 7...) as is standard for surface codes
    parser.add_argument(
        "--distances", type=int, nargs="+", default=[i for i in range(3, 30, 2)]
    )
    parser.add_argument("--num-trials", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=4686)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-plot", action="store_true")

    # Kept code-type to distinguish between rotated and standard if you add standard later
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
    code_type = args.code_type
    type_label = "Rotated" if code_type == "rt" else "Standard"

    # --- 1. CHECKPOINT INITIALIZATION ---
    file_prefix = f"checkpoint_{code_type}_surface_d{min(distances)}-{max(distances)}_trials{args.num_trials}_seed{args.seed}.pkl"
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
        f"Starting batched {type_label} Surface Code simulations with {num_workers} physical workers."
    )
    print(f"Distances: {distances}")
    print(f"Sigma steps: {len(sigmas)}")
    print(f"Max Trials per sigma: {args.num_trials}")

    # --- 2. MODIFIED OUTER LOOP (Distances) ---
    for d_idx in range(start_d_idx, len(distances)):
        distance = distances[d_idx]
        print(f"\n{'=' * 50}")
        print(
            f" INITIALIZING: {type_label.upper()} SURFACE CODE | DISTANCE d={distance} "
        )
        print(f"{'=' * 50}")

        context = prepare_distance_context(distance)

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
            # --- 3. MODIFIED INNER LOOP (Sigmas) ---
            for s_idx in range(start_s_idx, len(sigmas)):
                sigma = sigmas[s_idx]
                sigma_seed = args.seed + 100000 * distance + s_idx
                current_trials = get_dynamic_trials(float(sigma), args.num_trials)

                print(
                    f"Simulating ({code_type}), d={distance}, sigma={sigma:.3f} | Trials: {current_trials} | Batch: {batch_size}...",
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

                # --- 4. SAVE PROGRESS AFTER EVERY SIGMA ---
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

                print(f" -> Fidelity: {fidelity:.6f} | Time: [{time_str}]")

    # --- 5. CLEANUP AND FINAL SAVE ---
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    data_file = save_results(
        sigmas=sigmas,
        distances=distances,
        all_fidelities=all_fidelities,
        num_trials=args.num_trials,
        output_dir=args.output_dir,
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
            code_type,
        )
        print(f"Saved plot to {plot_file}")


if __name__ == "__main__":
    main()
