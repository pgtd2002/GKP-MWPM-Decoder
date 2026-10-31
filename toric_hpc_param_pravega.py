import argparse
import gc
import glob
import multiprocessing as mp
import os
import pickle
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# =====================================================================
# 1. LOCK C-LEVEL THREAD POOLS BEFORE IMPORTING NUMPY / SCIPY
# Prevents OpenBLAS/MKL thread thrashing across 48 physical cores
# =====================================================================
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

from decoders.pymatching_toric_decoder_2 import GKPSurfaceDecoder
from functions.binary_stabilizer_generators import import_gen_matrix
from functions.concatenation import build_css_gkp_rref
from functions.lattice import Omega_qqpp

# Precomputed mathematical constants (multiply instead of divide in hot loops)
SQRT_PI = np.sqrt(np.pi)
SQRT_2PI = np.sqrt(2.0 * np.pi)
INV_SQRT_PI = 1.0 / SQRT_PI
INV_SQRT_2PI = 1.0 / SQRT_2PI

_WORKER = {}


# =====================================================================
# 2. PARAM PRAVEGA HARDWARE, NUMA & SLURM TOPOLOGY DETECTION
# =====================================================================
def get_slurm_topology():
    """
    Detects multi-node SLURM rank and total participating nodes.
    Defaults to (0, 1) when run standalone on a single node.
    """
    # A direct Python invocation represents one process.
    # Multi-node execution must be launched using srun.
    if "SLURM_PROCID" not in os.environ:
        return 0, 1

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    return rank, world_size


def get_server_core_count(max_workers=None):
    """
    Detects physical cores per node on PARAM Pravega (2x Xeon Platinum 8268 = 48 cores).
    Prevents SLURM from spawning 96 workers on SMT hyperthreads (48-95) or defaulting to 1.
    """
    if max_workers is not None:
        return max(1, int(max_workers))

    try:
        output = subprocess.check_output(["lscpu"]).decode("utf-8")
        cores_per_socket, sockets = None, None
        for line in output.split("\n"):
            if "Core(s) per socket:" in line:
                cores_per_socket = int(line.split(":")[-1].strip())
            elif "Socket(s):" in line:
                sockets = int(line.split(":")[-1].strip())
        if cores_per_socket and sockets:
            return cores_per_socket * sockets  # 24 * 2 = 48 physical cores on Param Pravega
    except Exception:
        pass

    # Fallback: OS logical CPUs divided by 2 (96 // 2 = 48)
    total = os.cpu_count() or 2
    return max(1, total // 2)


def get_system_l3_cache_bytes():
    """
    Safely parses per-socket L3 cache on CentOS 7 / Cascade Lake ('36,608 KB' or '36608K')
    and multiplies by socket count (2 * 35.75 MB = 71.5 MB total L3 per node).
    """
    try:
        output = subprocess.check_output(["lscpu"]).decode("utf-8")
        l3_bytes = None
        sockets = 2
        for line in output.split("\n"):
            if "Socket(s):" in line:
                sockets = int(line.split(":")[-1].strip())
            elif "L3 cache:" in line or "L3:" in line:
                val_str = line.split(":")[-1].strip().replace(",", "")
                match = re.match(r"([0-9.]+)\s*([KMGkmg]?)", val_str)
                if match:
                    size = float(match.group(1))
                    unit = match.group(2).upper()
                    if "M" in unit:
                        l3_bytes = int(size * 1024 * 1024)
                    elif "K" in unit:
                        l3_bytes = int(size * 1024)
                    elif "G" in unit:
                        l3_bytes = int(size * 1024 * 1024 * 1024)
        if l3_bytes:
            # lscpu on Xeon 8268 reports per-socket L3 (36,608 KB); multiply by 2 sockets
            if l3_bytes < 50 * 1024 * 1024 and sockets > 1:
                l3_bytes *= sockets
            return l3_bytes
    except Exception:
        pass

    # Safe Fallback for Param Pravega: 71.5 MB L3 Cache
    return int(71.5 * 1024 * 1024)


def get_optimal_batch_size(dim, num_workers):
    """
    Calculates L3-optimal batch size while dynamically scaling bounds
    so high-genus (g > 13, d = 32) runs never overflow 192 GB node RAM.
    """
    total_l3_bytes = get_system_l3_cache_bytes()
    worker_cache_slice = (total_l3_bytes / num_workers) * 0.75
    bytes_per_trial = 5 * dim * 8
    optimal_batch = int(worker_cache_slice // bytes_per_trial)

    if dim < 1500:
        # Small lattice (e.g., d=6): large batch amortizes Python-to-C++ call overhead
        return max(2000, min(10000, optimal_batch))
    elif dim < 8000:
        # Medium lattice (d=8..14): balanced cache fit
        return max(400, min(4000, optimal_batch))
    elif dim < 20000:
        # Large lattice (d=16..24): keep working set inside L2/L3 cache
        return max(150, min(1000, optimal_batch))
    else:
        # Extreme lattice (g > 13, d=26..32, dim >= 20,000): protect 192 GB RAM & cache
        return max(64, min(400, optimal_batch))


# =====================================================================
# 3. NOISE SWEEP & LATTICE CONTEXT PREPARATION
# =====================================================================
def mod_2pi(x):
    return np.mod(x + np.pi, 2.0 * np.pi) - np.pi


def make_sigmas():
    sigmas_low = np.linspace(0.40, 0.52, 4)
    sigmas_mid = np.linspace(0.52, 0.65, 30)
    sigmas_high = np.linspace(0.65, 0.80, 6)
    sigmas_fine = np.arange(0.601, 0.607, 0.001)
    # Uncomment the full curve below if doing a wide sweep instead of threshold hunting:
    # return np.concatenate([sigmas_low, sigmas_mid, sigmas_high])
    return sigmas_fine


def get_dynamic_trials(sigma, base_trials):
    """
    Concentrates full 10^9 - 10^10 trial firepower in the threshold crossing region
    while scaling down deep sub/super-threshold points.
    """
    if sigma < 0.52:
        return min(base_trials, 50000)
    elif sigma > 0.65:
        return min(base_trials, 50000)
    else:
        return base_trials


def prepare_distance_context(distance, folder_prefix):
    """
    Builds sparse check and symplectic lattice matrices on the master process.
    Avoids dense np.linalg.inv() on the master process so RAM stays tiny before fork().
    """
    G, Hx, Hz = import_gen_matrix(distance, folder_prefix)
    dim = G.shape[1]
    Omega = Omega_qqpp(dim // 2)
    M = build_css_gkp_rref(Hx, Hz)

    M_csc = sp.csc_matrix(M, dtype=np.float64)
    Omega_csc = sp.csc_matrix(Omega, dtype=np.float64)

    # A = sqrt(2*pi) * M * Omega^T (purely sparse)
    A_sparse = (M_csc @ Omega_csc.T) * SQRT_2PI

    return {
        "G": np.ascontiguousarray(G, dtype=np.uint8),
        "M_csc": M_csc,
        "A_csr": A_sparse.tocsr(),
        "Omega_csr": Omega_csc.tocsr(),
        "dim": dim,
    }


# =====================================================================
# 4. NUMA-PINNED WORKER INITIALIZATION & HYBRID FAST-PATH ENGINE
# =====================================================================
def get_physical_cpu_ids():
    """Discovers physical CPUs actually permitted by SLURM's cgroup mask on this node."""
    try:
        allowed = sorted(os.sched_getaffinity(0))
        physical_cpus = []
        seen_cores = set()
        for cpu in allowed:
            topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            if topo.exists():
                pkg_id = int((topo / "physical_package_id").read_text().strip())
                core_id = int((topo / "core_id").read_text().strip())
                key = (pkg_id, core_id)
                if key not in seen_cores:
                    seen_cores.add(key)
                    physical_cpus.append(cpu)
            else:
                physical_cpus.append(cpu)
        return physical_cpus if physical_cpus else allowed
    except Exception:
        return list(range(48))


def init_worker(context, worker_counter):
    from threadpoolctl import threadpool_limits
    from scipy.sparse.linalg import splu

    threadpool_limits(limits=1)

    with worker_counter.get_lock():
        worker_id = worker_counter.value
        worker_counter.value += 1

    # Safely pin to SLURM-assigned physical cores (catches both AttributeError and OSError)
    try:
        physical_cpus = get_physical_cpu_ids()
        target_cpu = physical_cpus[worker_id % len(physical_cpus)]
        os.sched_setaffinity(0, {target_cpu})
    except (AttributeError, OSError):
        pass


def run_trial_chunk(args):
    sigma, num_trials, seed, batch_size = args
    rng = np.random.default_rng(seed)
    success_count = 0

    dim = _WORKER["dim"]
    decoder = _WORKER["decoder"]
    use_dense = _WORKER["use_dense"]

    if use_dense:
        A_dense = _WORKER["A_dense"]
        B_dense = _WORKER["B_dense"]
        M_inv_T_scaled = _WORKER["M_inv_T_scaled"]
    else:
        A_csr = _WORKER["A_csr"]
        Omega_csr = _WORKER["Omega_csr"]
        splu_M = _WORKER["splu_M"]

    for start_idx in range(0, num_trials, batch_size):
        current_batch = min(batch_size, num_trials - start_idx)

        # 1. Generate analog Gaussian phase-space noise: shape (dim, current_batch)
        xi = rng.standard_normal(size=(dim, current_batch)) * sigma

        # 2. Compute GKP syndrome s and target vector t_batch
        if use_dense:
            s = mod_2pi(A_dense.dot(xi))
            t_batch = B_dense.dot(s)
        else:
            s = mod_2pi(A_csr.dot(xi))
            t_batch = Omega_csr.dot(splu_M.solve(s)) * INV_SQRT_2PI

        # 3. Repackage into C-contiguous layout for C++ PyMatching batch decode
        t_batch_T = np.ascontiguousarray(t_batch.T)
        cp_batch_T = decoder.decode_analog_batch(t_batch_T)
        cp_batch = cp_batch_T.T

        # 4. Exact Chamberland Verification (Version 2 math)
        residual_error = xi - t_batch + cp_batch
        effective_error = np.rint(residual_error * INV_SQRT_PI) * SQRT_PI

        if use_dense:
            z = M_inv_T_scaled.dot(effective_error)
        else:
            z = splu_M.solve(effective_error, trans="T") * INV_SQRT_2PI

        is_lattice = np.max(np.abs(z - np.rint(z)), axis=0) < 1e-4
        success_count += int(np.sum(is_lattice))

    return success_count


# =====================================================================
# 5. MULTI-NODE TRIAL SHARDING & CHECKPOINTING
# =====================================================================
def split_trials(num_trials, num_chunks):
    num_chunks = max(1, min(num_trials, num_chunks))
    base = num_trials // num_chunks
    remainder = num_trials % num_chunks
    return [base + (i < remainder) for i in range(num_chunks)]


def monte_carlo_for_sigma_sharded(
    executor, sigma, total_trials, base_seed, num_workers, batch_size, rank, world_size
):
    """
    Divides total_trials (e.g. 10^9 or 10^10) evenly across all SLURM nodes.
    Returns raw (success_count, node_trials) for exact 64-bit integer aggregation.
    """
    base_node_trials = total_trials // world_size
    rem_node_trials = total_trials % world_size
    node_trials = base_node_trials + (1 if rank < rem_node_trials else 0)

    if node_trials == 0:
        return 0, 0

    MIN_CHUNK_SIZE = max(100, batch_size)
    max_useful_chunks = max(1, node_trials // MIN_CHUNK_SIZE)
    actual_chunks = min(num_workers * 4, max_useful_chunks)
    trial_chunks = split_trials(node_trials, actual_chunks)

    # Cryptographically independent RNG streams per (base_seed, node_rank)
    ss = np.random.SeedSequence([base_seed, rank])
    child_seeds = ss.spawn(len(trial_chunks))

    tasks = [
        (sigma, count, child_seeds[i], batch_size)
        for i, count in enumerate(trial_chunks)
        if count > 0
    ]

    success_count = sum(executor.map(run_trial_chunk, tasks))
    return int(success_count), int(node_trials)


def load_checkpoint(filepath):
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)
    return None


def save_checkpoint(state, filepath):
    temp_file = f"{filepath}.tmp.{os.getpid()}"
    with open(temp_file, "wb") as f:
        pickle.dump(state, f)
    os.replace(temp_file, filepath)


# =====================================================================
# 6. SAVING RESULTS & PUBLICATION PLOTS
# =====================================================================
def save_results(sigmas, distances, all_fidelities, num_trials, output_dir, genus, code_type):
    os.makedirs(output_dir, exist_ok=True)
    d_min, d_max = min(distances), max(distances)
    filename = (
        f"{code_type}_toric_gkp_g{genus}_d{d_min}-{d_max}_"
        f"trials{num_trials}_sigma_steps{len(sigmas)}.npz"
    )
    filepath = os.path.join(output_dir, filename)

    save_dict = {
        "sigmas": np.asarray(sigmas, dtype=np.float64),
        "distances": np.asarray(distances, dtype=np.int32),
        "num_trials": np.int64(num_trials),
        "genus": np.int32(genus),
        "code_type": str(code_type),
    }
    for d in distances:
        save_dict[f"fidelity_d{d}"] = np.asarray(all_fidelities[d], dtype=np.float64)

    np.savez_compressed(filepath, **save_dict)
    print(f"\n[+] Saved aggregated data to {filepath}")
    return filepath


def save_plot(sigmas, distances, all_fidelities, output_dir, data_file, genus, code_type):
    plt.figure(figsize=(9, 6))
    cmap = plt.get_cmap("tab20")

    for idx, d in enumerate(distances):
        plt.plot(
            sigmas,
            all_fidelities[d],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=cmap(idx % 20),
            label=f"d = {d}",
        )

    type_label = "Rotated" if code_type == "rt" else "Standard"
    plt.title(f"Genus-{genus} {type_label} Toric GKP Code Logical Fidelity vs. Noise")
    plt.xlabel(r"GKP Noise Standard Deviation $\sigma$")
    plt.ylabel("Logical Fidelity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    plot_file = os.path.splitext(data_file)[0] + ".png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"[+] Saved threshold plot to {plot_file}")


# =====================================================================
# 7. CLI & MAIN ORCHESTRATION
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="PARAM Pravega Multi-Node & NUMA-Optimized Sparse Toric GKP Simulator."
    )
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=[i for i in range(6, 32, 2)],  # d = 6 to 30 inclusive
        help="Code distances to simulate (default: 6 8 ... 30)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1000000000,
        help="Total Monte Carlo trials per (d, sigma) across all nodes (e.g., 10^9 or 10^10)",
    )
    parser.add_argument("--seed", type=int, default=461266)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--genus",
        type=int,
        default=14,
        help="Genus of the toric code (e.g., 14, 15, ...)",
    )
    parser.add_argument(
        "--code-type",
        type=str,
        choices=["rt", "st"],
        default="rt",
        help="'rt' for rotated, 'st' for standard",
    )
    return parser.parse_args()


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


def main():
    args = parse_args()
    rank, world_size = get_slurm_topology()
    distances = list(args.distances)
    sigmas = make_sigmas()
    num_workers = get_server_core_count(args.workers)

    folder_prefix = f"{args.code_type}_g{args.genus}"
    type_label = "Rotated" if args.code_type == "rt" else "Standard"
    d_min, d_max = min(distances), max(distances)
    run_id = f"{folder_prefix}_d{d_min}-{d_max}_trials{args.num_trials}_seed{args.seed}"

    os.makedirs(args.output_dir, exist_ok=True)
    shard_dir = Path(args.output_dir) / "shards" / run_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    my_shard_file = shard_dir / f"shard_rank_{rank:04d}_of_{world_size:04d}.pkl"

    my_shard_data = load_checkpoint(str(my_shard_file))
    if my_shard_data is not None:
        if rank == 0:
            print(f"\n[!] Found shard checkpoint at {my_shard_file}! Resuming progress...\n")
    else:
        my_shard_data = {}

    if rank == 0:
        print(
            f"Starting PARAM Pravega {type_label} Genus-{args.genus} simulation across "
            f"{world_size} node(s) ({world_size * num_workers} physical cores total)."
        )
        print(f"Loading matrices from folder pattern: '{folder_prefix}'")
        print(f"Distances: {distances}")
        print(f"Sigma steps: {len(sigmas)}")
        print(f"Target Global Trials per sigma: {args.num_trials:,}")

    for d in distances:
        if rank == 0:
            print(f"\n{'=' * 55}")
            print(f"INITIALIZING: {type_label.upper()} GENUS-{args.genus} | DISTANCE d={d}")

        context = prepare_distance_context(d, folder_prefix)
        batch_size = get_optimal_batch_size(context["dim"], num_workers)
        gc.collect()  # Release temporary master arrays before forking 48 workers

        if d not in my_shard_data:
            my_shard_data[d] = {}

        worker_counter = mp.Value("i", 0)
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(context, worker_counter),
        ) as executor:
            for sigma_idx, sigma in enumerate(sigmas):
                sigma_key = round(float(sigma), 6)

                if sigma_key in my_shard_data[d]:
                    if rank == 0:
                        succ, tot = my_shard_data[d][sigma_key]
                        fid = succ / tot if tot > 0 else 0.0
                        print(
                            f"Skipping g={args.genus}({args.code_type}), d={d}, "
                            f"sigma={sigma:.4f} (Already completed: Node Fid={fid:.8f})"
                        )
                    continue

                current_trials = get_dynamic_trials(sigma, args.num_trials)
                step_seed = args.seed + (d * 10000) + (sigma_idx * 100)

                if rank == 0:
                    print(
                        f"Simulating g={args.genus}({args.code_type}), d={d}, "
                        f"sigma={sigma:.4f} | Global Trials: {current_trials:,} | "
                        f"Batch: {batch_size}...",
                        end="",
                        flush=True,
                    )

                t0 = time.perf_counter()
                succ, tot = monte_carlo_for_sigma_sharded(
                    executor=executor,
                    sigma=sigma,
                    total_trials=current_trials,
                    base_seed=step_seed,
                    num_workers=num_workers,
                    batch_size=batch_size,
                    rank=rank,
                    world_size=world_size,
                )
                elapsed = time.perf_counter() - t0

                my_shard_data[d][sigma_key] = (succ, tot)
                save_checkpoint(my_shard_data, str(my_shard_file))

                if rank == 0:
                    local_fid = succ / tot if tot > 0 else 0.0
                    print(
                        f" -> Node0 Fidelity: {local_fid:.8f} | Time: [{format_duration(elapsed)}]",
                        flush=True,
                    )

        # Free distance context before next distance
        del context
        gc.collect()

    # =====================================================================
    # 8. RANK 0 FINAL MULTI-NODE SHARD AGGREGATION
    # =====================================================================
    if rank == 0:
        if world_size > 1:
            print(f"\nWaiting for all {world_size} nodes to flush final shard files...")
            while True:
                shard_files = sorted(
                    glob.glob(str(shard_dir / f"shard_rank_*_of_{world_size:04d}.pkl"))
                )
                if len(shard_files) == world_size:
                    all_complete = True
                    loaded_shards = []
                    for sf in shard_files:
                        try:
                            s_data = load_checkpoint(sf)
                            if not s_data or any(
                                len(s_data.get(d, {})) < len(sigmas) for d in distances
                            ):
                                all_complete = False
                                break
                            loaded_shards.append(s_data)
                        except Exception:
                            all_complete = False
                            break
                    if all_complete:
                        break
                time.sleep(2)
        else:
            loaded_shards = [my_shard_data]

        all_fidelities = {}
        for d in distances:
            fids = []
            for sigma in sigmas:
                sigma_key = round(float(sigma), 6)
                total_succ = sum(s[d][sigma_key][0] for s in loaded_shards)
                total_trials = sum(s[d][sigma_key][1] for s in loaded_shards)
                fids.append(total_succ / total_trials if total_trials > 0 else 0.0)
            all_fidelities[d] = np.array(fids, dtype=np.float64)

        # Save master combined checkpoint & final .npz / .png
        master_ckpt = os.path.join(args.output_dir, f"checkpoint_{run_id}.pkl")
        save_checkpoint(all_fidelities, master_ckpt)

        data_file = save_results(
            sigmas=sigmas,
            distances=distances,
            all_fidelities=all_fidelities,
            num_trials=args.num_trials,
            output_dir=args.output_dir,
            genus=args.genus,
            code_type=args.code_type,
        )
        if not args.no_plot:
            save_plot(
                sigmas=sigmas,
                distances=distances,
                all_fidelities=all_fidelities,
                output_dir=args.output_dir,
                data_file=data_file,
                genus=args.genus,
                code_type=args.code_type,
            )


if __name__ == "__main__":
    main()