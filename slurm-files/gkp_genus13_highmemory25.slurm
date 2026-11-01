#!/bin/bash
#SBATCH --job-name=gkp_g13_hm25
#SBATCH --partition=highmemory
#SBATCH --account=iisc
#SBATCH --nodes=25
#SBATCH --ntasks=25
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pgtd2002@gmail.com
#SBATCH --output=/scratch/esemrin/GKP-MWPM-Decoder/slurm-files/logs/gkp_g13_hm25_%j.out
#SBATCH --error=/scratch/esemrin/GKP-MWPM-Decoder/slurm-files/logs/gkp_g13_hm25_%j.err

set -Eeuo pipefail

# ============================================================
# 1. Project and environment configuration
# ============================================================
PROJECT_DIR="/scratch/esemrin/GKP-MWPM-Decoder"
CONDA_BASE="/scratch/esemrin/miniforge3"
CONDA_ENV="/scratch/esemrin/gkp-conda"
SIM_SCRIPT="${PROJECT_DIR}/toric_hpc_param_pravega.py"
MATRIX_DIR="${PROJECT_DIR}/rt_g13"
OUTPUT_DIR="${PROJECT_DIR}/results_g13_highmemory25"

cd "${PROJECT_DIR}"

# Slurm opens stdout/stderr before the script starts; create logs/ before sbatch.
mkdir -p "${PROJECT_DIR}/slurm-files/logs" "${OUTPUT_DIR}"

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Keep one BLAS/OpenMP thread per worker. Python multiprocessing provides
# the node-level parallelism (48 workers on each allocated node).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ============================================================
# 2. Preflight checks
# ============================================================
echo "============================================================"
echo "Job ID          : ${SLURM_JOB_ID}"
echo "Partition       : ${SLURM_JOB_PARTITION}"
echo "Node list       : ${SLURM_JOB_NODELIST}"
echo "Tasks           : ${SLURM_NTASKS}"
echo "CPUs per task   : ${SLURM_CPUS_PER_TASK}"
echo "Total CPUs      : $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
echo "Start time      : $(date)"
echo "Working dir     : ${PROJECT_DIR}"
echo "Python          : $(command -v python)"
python --version
echo "Output dir      : ${OUTPUT_DIR}"
echo "============================================================"

if [[ ! -f "${SIM_SCRIPT}" ]]; then
    echo "ERROR: Simulation script not found: ${SIM_SCRIPT}" >&2
    exit 1
fi

if [[ ! -d "${MATRIX_DIR}" ]]; then
    echo "ERROR: Genus-13 matrix directory not found: ${MATRIX_DIR}" >&2
    exit 1
fi

python - <<'PY'
import numpy, scipy, matplotlib, threadpoolctl
print("Dependency preflight: NumPy, SciPy, Matplotlib, threadpoolctl OK")
PY

# ============================================================
# 3. Run: one Slurm rank per node, 48 local Python workers per rank
# ============================================================
srun \
    --ntasks="${SLURM_NTASKS}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    --cpu-bind=cores \
    --kill-on-bad-exit=1 \
    python -u "${SIM_SCRIPT}" \
        --distances 6 8 10 12 14 16 18 20 22 24 26 28 30 \
        --num-trials 100000000 \
        --seed 83456 \
        --workers "${SLURM_CPUS_PER_TASK}" \
        --genus 13 \
        --code-type rt \
        --output-dir "${OUTPUT_DIR}"

 echo "============================================================"
 echo "Simulation finished successfully."
 echo "End time: $(date)"
 echo "============================================================"
