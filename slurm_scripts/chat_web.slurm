#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nc_web
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%j_web.out
#SBATCH --error=logs/%j_web.err

# Web 聊天服务
set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

NODE_IP=$(hostname -I | awk '{print $1}')
echo "=== Job ${SLURM_JOB_ID}: Chat Web Server ==="
echo "Start: $(date)"
echo "Server: http://${NODE_IP}:8000"

python -m scripts.chat_web --depth=20 --port=8000
