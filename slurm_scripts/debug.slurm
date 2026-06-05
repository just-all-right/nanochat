#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nc_debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%j_debug.out
#SBATCH --error=logs/%j_debug.err

# 快速调试：depth=6 小模型，200 步
set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

echo "=== Job ${SLURM_JOB_ID}: Quick Debug ==="
echo "Start: $(date)"

python -m scripts.base_train \
    --depth=6 \
    --max_seq_len=256 \
    --device_batch_size=4 \
    --num_iterations=200 \
    --run=quick_debug

echo "Done: $(date)"
