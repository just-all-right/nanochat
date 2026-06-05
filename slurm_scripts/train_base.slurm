#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nc_base
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%j_base.out
#SBATCH --error=logs/%j_base.err

# 标准预训练：depth=20，8 GPU DDP
set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

echo "=== Job ${SLURM_JOB_ID}: Base Training depth=20 ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-8}"

torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train \
    --depth=20 \
    --device_batch_size=32 \
    --run=base_d20

echo "Done: $(date)"
