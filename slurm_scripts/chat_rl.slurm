#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nc_rl
#SBATCH --gres=gpu:2               # 2 张 GPU（QoS 限制）
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/%j_rl.out
#SBATCH --error=logs/%j_rl.err

# 强化学习 GRPO，2 GPU DDP
set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

echo "=== Job ${SLURM_JOB_ID}: RL/GRPO on 2 GPUs ==="
echo "Start: $(date)"

torchrun --standalone --nproc_per_node=2 \
    -m scripts.chat_rl \
    --depth=20 \
    --run=rl_d20

echo "Done: $(date)"
