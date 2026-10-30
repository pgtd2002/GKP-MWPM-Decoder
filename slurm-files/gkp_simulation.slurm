#!/bin/bash
#SBATCH --job-name=gkp_toric
#SBATCH --partition=debug
#SBATCH --nodes=11
#SBATCH --ntasks=11
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --mail-user=pritamghosh1@iisc.ac.in
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/gkp_%j.out
#SBATCH --error=logs/gkp_%j.err

set -Eeuo pipefail

# ============================================================
# 1. Project configuration
# ============================================================

PROJECT_DIR="/scratch/esemrin/GKP-MWPM-Decoder"
CONDA_BASE="/scratch/esemrin/miniforge3"
CONDA_ENV="/scratch/esemrin/gkp-conda"

# Change this to the actual filename of your simulation script.
SIM_SCRIPT="toric_hpc_param_pravega.py"

# ============================================================
# 2. Set up the job environment
# ============================================================

cd "${PROJECT_DIR}"

# Activate the explicitly installed Miniforge environment.
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Prevent nested BLAS/OpenMP threading.
# Python multiprocessing will use the allocated CPUs instead.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Avoid buffering Python output in the SLURM log.
export PYTHONUNBUFFERED=1

# ============================================================
# 3. Verify the environment
# ============================================================

echo "============================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node list    : ${SLURM_JOB_NODELIST}"
echo "CPUs/task    : ${SLURM_CPUS_PER_TASK}"
echo "Start time   : $(date)"
echo "Working dir  : ${PROJECT_DIR}"
echo "Python       : $(which python)"
python --version
echo "============================================================"

# Fail early if the script filename is incorrect.
if [[ ! -f "${SIM_SCRIPT}" ]]; then
    echo "ERROR: Simulation script not found: ${PROJECT_DIR}/${SIM_SCRIPT}"
    exit 1
fi

# ============================================================
# 4. Run the simulation
# ============================================================

srun --ntasks="${SLURM_NTASKS}" --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    python -u "${SIM_SCRIPT}" \
    --num-trials 100000000 \
    --seed 461266 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --genus 4 \
    --code-type rt \
    --output-dir "${PROJECT_DIR}/results"

# ============================================================
# 5. Finish
# ============================================================

echo "============================================================"
echo "Simulation finished successfully."
echo "End time: $(date)"
echo "============================================================"