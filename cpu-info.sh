#!/bin/bash
#SBATCH --job-name=cpu-info
#SBATCH --partition=small72
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:02:00
#SBATCH --error=cpu-info.%J.err
#SBATCH --output=cpu-info.%J.out

cd "$SLURM_SUBMIT_DIR" || exit 1

echo "===== PARAM Pravega Compute Node ====="
echo "Hostname: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node list: $SLURM_JOB_NODELIST"
echo "CPUs on node: $SLURM_CPUS_ON_NODE"

echo
echo "===== CPU Affinity ====="
grep Cpus_allowed_list /proc/self/status

echo
echo "===== CPU and Cache Information ====="
lscpu
lscpu -C