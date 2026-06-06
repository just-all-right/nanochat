#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nc_base
#SBATCH --gres=gpu:2               # 2 张 GPU（QoS 限制）
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/%j_base.out
#SBATCH --error=logs/%j_base.err

# 标准预训练：depth=20，2 GPU DDP
set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

echo "=== Job ${SLURM_JOB_ID}: Base Training depth=20 on 2 GPUs ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "GPUs: ${SLURM_GPUS_ON_NODE:-2}"

torchrun --standalone --nproc_per_node=2 \
    -m scripts.base_train \
    --depth=20 \
    --device-batch-size=32 \
    --run=base_d20

echo "Done: $(date)"
