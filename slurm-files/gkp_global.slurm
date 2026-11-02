#!/bin/bash
#SBATCH --job-name=gkp_g13
#SBATCH --partition=debug
#SBATCH --nodes=11
#SBATCH --ntasks=11
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=1:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pgtd2002@gmail.com
#SBATCH --output=logs/gkp_%j.out
#SBATCH --error=logs/gkp_%j.err

set -Eeuo pipefail

# ============================================================
# 1. Project configuration
# ============================================================

PROJECT_DIR="/scratch/esemrin/GKP-MWPM-Decoder"

CONDA_BASE="/scratch/esemrin/miniforge3"
CONDA_ENV="/scratch/esemrin/gkp-conda"

SIM_SCRIPT="toric_hpc_param_pravega.py"

# Simulation parameters
GENUS=13
CODE_TYPE="rt"
SEED=461776
NUM_TRIALS=100000000

# Automatically generate a separate results directory for each genus
RESULTS_DIR="${PROJECT_DIR}/results_${CODE_TYPE}_g${GENUS}_${SEED}"

# ============================================================
# 2. Set up the job environment
# ============================================================

cd "${PROJECT_DIR}"

# Activate the explicitly installed Miniforge environment
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Prevent nested BLAS/OpenMP threading
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Avoid buffering Python output in the SLURM log
export PYTHONUNBUFFERED=1

# ============================================================
# 3. Verify the environment
# ============================================================

echo "============================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Partition    : ${SLURM_JOB_PARTITION}"
echo "Node list    : ${SLURM_JOB_NODELIST}"
echo "Nodes        : ${SLURM_JOB_NUM_NODES}"
echo "Tasks        : ${SLURM_NTASKS}"
echo "CPUs/task    : ${SLURM_CPUS_PER_TASK}"
echo "Total CPUs   : $((SLURM_NTASKS * SLURM_CPUS_PER_TASK))"
echo "Genus        : ${GENUS}"
echo "Code type    : ${CODE_TYPE}"
echo "Trials       : ${NUM_TRIALS}"
echo "Seed         : ${SEED}"
echo "Start time   : $(date)"
echo "Working dir  : ${PROJECT_DIR}"
echo "Results dir  : ${RESULTS_DIR}"
echo "Python       : $(which python)"

python --version

echo "============================================================"

# Fail early if the simulation script is missing
if [[ ! -f "${SIM_SCRIPT}" ]]; then
    echo "ERROR: Simulation script not found: ${PROJECT_DIR}/${SIM_SCRIPT}"
    exit 1
fi

# Create genus-specific results directory
mkdir -p "${RESULTS_DIR}"

# ============================================================
# 4. Run the simulation
#    One parent process per node, 48 workers per node
# ============================================================

srun \
    --ntasks="${SLURM_NTASKS}" \
    --ntasks-per-node=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
    python -u "${SIM_SCRIPT}" \
        --num-trials "${NUM_TRIALS}" \
        --seed "${SEED}" \
        --workers "${SLURM_CPUS_PER_TASK}" \
        --genus "${GENUS}" \
        --code-type "${CODE_TYPE}" \
        --output-dir "${RESULTS_DIR}"

# ============================================================
# 5. Finish
# ============================================================

echo "============================================================"
echo "Simulation finished successfully."
echo "Genus       : ${GENUS}"
echo "End time    : $(date)"
echo "Results     : ${RESULTS_DIR}"
echo "============================================================"