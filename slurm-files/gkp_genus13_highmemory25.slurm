#!/bin/bash
#SBATCH --job-name=gkp_g13
#SBATCH --partition=highmemory
#SBATCH --nodes=25
#SBATCH --ntasks=25
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=2:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pgtd2002@gmail.com
#SBATCH --output=/scratch/esemrin/GKP-MWPM-Decoder/logs/gkp_%j.out
#SBATCH --error=/scratch/esemrin/GKP-MWPM-Decoder/logs/gkp_%j.err

set -Eeuo pipefail

# ============================================================
# 1. Project configuration
# ============================================================
PROJECT_DIR="/scratch/esemrin/GKP-MWPM-Decoder"
CONDA_BASE="/scratch/esemrin/miniforge3"
CONDA_ENV="/scratch/esemrin/gkp-conda"
SIM_SCRIPT="toric_hpc_param_pravega.py"
RESULTS_DIR="${PROJECT_DIR}/results_g13_highmemory25"

# ============================================================
# 2. Validate paths and activate the environment
# ============================================================
cd "${PROJECT_DIR}"

if [[ ! -f "${SIM_SCRIPT}" ]]; then
    echo "ERROR: Simulation script not found: ${PROJECT_DIR}/${SIM_SCRIPT}"
    exit 1
fi

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: conda.sh not found under ${CONDA_BASE}"
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Prevent nested BLAS/OpenMP threading; each Slurm task launches 48 workers.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# ============================================================
# 3. Environment and dependency preflight
# ============================================================
echo "============================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Partition    : ${SLURM_JOB_PARTITION}"
echo "Node list    : ${SLURM_JOB_NODELIST}"
echo "Nodes        : ${SLURM_JOB_NUM_NODES}"
echo "Tasks        : ${SLURM_NTASKS}"
echo "CPUs/task    : ${SLURM_CPUS_PER_TASK}"
echo "Total CPUs   : $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
echo "Start time   : $(date)"
echo "Working dir  : ${PROJECT_DIR}"
echo "Python       : $(command -v python)"
python --version
echo "============================================================"

python - <<'PY'
import numpy, scipy, matplotlib, threadpoolctl
from decoders.pymatching_toric_decoder_2 import GKPSurfaceDecoder
from functions.binary_stabilizer_generators import import_gen_matrix
from functions.concatenation import build_css_gkp_rref
from functions.lattice import Omega_qqpp
G, Hx, Hz = import_gen_matrix(6, "rt_g13")
print("Python dependencies and project modules imported successfully.")
print(f"Genus-13 d=6 input matrices loaded: G={G.shape}, Hx={Hx.shape}, Hz={Hz.shape}")
PY

mkdir -p "${RESULTS_DIR}"

# ============================================================
# 4. Run one parent process per node; each parent launches 48 workers
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
        --output-dir "${RESULTS_DIR}"

 echo "============================================================"
echo "Simulation finished successfully."
echo "End time: $(date)"
echo "Results: ${RESULTS_DIR}"
echo "============================================================"
