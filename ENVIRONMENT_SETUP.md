# NanoChat 实验环境配置指南

---

> **适用环境**：GPU 集群通过 Slurm 调度使用，禁止裸机直接运行深度学习程序
>
> **最后更新**：2026-06-05

---

## 目录

1. [集群概况](#1-集群概况)
2. [核心规则](#2-核心规则)
3. [环境初始化（一次性）](#3-环境初始化一次性)
4. [Slurm 快速入门](#4-slurm-快速入门)
5. [交互式开发（srun）](#5-交互式开发srun)
6. [批量提交训练任务（sbatch）](#6-批量提交训练任务sbatch)
7. [多 GPU 分布式训练](#7-多-gpu-分布式训练)
8. [各实验的运行方式对照表](#8-各实验的运行方式对照表)
9. [任务监控与管理](#9-任务监控与管理)
10. [常见问题与排错](#10-常见问题与排错)
11. [附录：Slurm 脚本模板库](#11-附录slurm-脚本模板库)

---

## 1. 集群概况

### 1.1 硬件资源

| 项目 | 规格 |
|------|------|
| **节点数** | 1（单节点集群） |
| **节点名** | `user-NF5468M6` |
| **GPU** | 8× NVIDIA RTX PRO 6000 Blackwell Server Edition |
| **单卡显存** | ~96 GB（总计 ~768 GB） |
| **CPU** | 112 核 |
| **内存** | ~386 GB |
| **CUDA 驱动** | 13.2 |
| **Slurm 版本** | 25.05.0 |

### 1.2 分区与 QoS

| 分区名 | 状态 | 时间限制 | 节点数 |
|--------|------|---------|--------|
| `compute` | 活跃（默认） | 1 天（1-00:00:00） | 1 |

**账户信息**：

| 属性 | 值 |
|------|-----|
| 账户名 | `lab_cluster_account` |
| QoS | `gpu2limit`（限制 GPU 使用数量） |
| 公平份额权重 | 1（`RawShares`） |

### 1.3 磁盘与数据

| 路径 | 用途 |
|------|------|
| `/home/dongjunteng/nanochat` | 项目代码（与登录节点共享） |
| `~/.cache/huggingface` | HuggingFace 数据集缓存（注意磁盘空间） |
| `/tmp` | 临时文件（各节点可能独立，不保证持久化） |

---

## 2. 核心规则

> ⚠️ **必须遵守的三条铁律**：

1. **禁止裸机运行**：不要在登录节点直接运行 `python -m scripts.base_train` 等 GPU 训练脚本。所有 GPU 使用必须通过 Slurm（`srun`/`sbatch`）。
2. **登录节点仅做轻量操作**：代码编辑、文件查看、Git 操作、Python 交互式探索（CPU only）、提交任务。
3. **资源按时释放**：训练完毕后确保任务正常退出，不要占用 GPU 不释放。长时间空闲的交互式会话应及时 `scancel`。

### 2.1 什么可以在登录节点做？

| 操作 | 允许 | 说明 |
|------|:----:|------|
| 编辑代码 (vim/code) | ✅ | — |
| Git 操作 | ✅ | — |
| `python -c "..."` (CPU) | ✅ | 仅 CPU 推理，不调用 CUDA |
| 阅读/理解代码 | ✅ | — |
| `sbatch` / `srun` | ✅ | 提交任务本身就是在登录节点操作的 |
| `squeue` / `sinfo` | ✅ | 查看集群状态 |
| 分词器训练（纯 CPU） | ✅ | `python -m scripts.tok_train` 不需要 GPU |

### 2.2 什么必须在计算节点做？

| 操作 | 说明 |
|------|------|
| 模型预训练 (`base_train`) | 需要 GPU |
| 监督微调 (`chat_sft`) | 需要 GPU |
| 强化学习 (`chat_rl`) | 需要 GPU |
| 模型评估 (`base_eval`, `chat_eval`) | 需要 GPU |
| Web 服务 (`chat_web`) | 需要 GPU |
| 任何调用 `model.cuda()` 的操作 | 需要 GPU |

---

## 3. 环境初始化（一次性）

### 3.1 确认项目已就绪

```bash
cd /home/dongjunteng/nanochat

# 确认虚拟环境存在
ls .venv/bin/python

# 确认项目依赖已安装
.venv/bin/python -c "import torch; print(f'PyTorch {torch.__version__}')"
.venv/bin/python -c "from nanochat.gpt import GPT, GPTConfig; print('nanochat OK')"
```

如果 `torch` 不可导入，说明虚拟环境中的 PyTorch 尚未安装或版本不对。检查 `pyproject.toml` 中的 PyTorch 版本要求（`torch==2.9.1`），并按需重新安装：

```bash
source .venv/bin/activate
uv sync
```

### 3.2 验证 Slurm 可用性

```bash
# 检查 Slurm 客户端工具
which sbatch srun squeue scancel sinfo

# 查看分区信息
sinfo

# 查看当前任务
squeue

# 确认自己有权提交任务（应能看到账户信息）
sshare -U $(whoami)
```

### 3.3 测试：最简单的 GPU 任务

创建一个最小测试脚本验证 GPU 可用性：

```bash
# 在登录节点运行以下命令，通过 srun 在计算节点执行
srun --partition=compute \
     --account=lab_cluster_account \
     --gres=gpu:1 \
     --time=00:05:00 \
     --ntasks=1 \
     --cpus-per-task=4 \
     bash -c '
echo "=== Node: $(hostname) ==="
echo "=== CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=== Testing PyTorch ==="
source /home/dongjunteng/nanochat/.venv/bin/activate
python -c "import torch; print(f\"PyTorch {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\"); print(f\"GPU count: {torch.cuda.device_count()}\")"
'
```

如果成功输出 GPU 信息和 PyTorch CUDA 可用性，说明环境配置完毕。

---

## 4. Slurm 快速入门

### 4.1 核心概念

```
登录节点 (login node)          计算节点 (compute node)
┌─────────────────┐            ┌─────────────────────────┐
│  你在这里工作     │            │  你的任务在这里运行       │
│  - 写代码         │   sbatch   │  - python base_train.py │
│  - 提交任务 ──────┼──────────►│  - 可以访问所有 GPU       │
│  - 查看结果       │   srun     │  - 环境变量自动设置       │
│  - 轻量调试       │◄──────────│  - stdout 回传登录节点    │
└─────────────────┘            └─────────────────────────┘
```

### 4.2 两种使用方式

| 方式 | 命令 | 适用场景 |
|------|------|---------|
| **交互式** | `srun --pty bash` | 调试代码、探索性分析、短时间实验 |
| **批量提交** | `sbatch job.sh` | 长时间训练、多组参数扫描、无人值守运行 |

### 4.3 常用 Slurm 参数

| 参数 | 含义 | 本集群可用值 |
|------|------|-------------|
| `--partition` / `-p` | 分区 | `compute` |
| `--account` / `-A` | 账户 | `lab_cluster_account` |
| `--gres` | 通用资源 | `gpu:1` 到 `gpu:8` |
| `--time` / `-t` | 时间限制 | 最大 `1-00:00:00`（1 天） |
| `--ntasks` / `-n` | 任务数 | 通常为 1（单个 Python 进程） |
| `--cpus-per-task` / `-c` | 每任务的 CPU 数 | 建议 8-16 |
| `--mem` | 内存 | 可选，如 `--mem=64G` |
| `--job-name` / `-J` | 任务名 | 方便在 `squeue` 中识别 |
| `--output` / `-o` | stdout 日志文件路径 | 如 `logs/%j.out`（%j = job ID） |
| `--error` / `-e` | stderr 日志文件路径 | 如 `logs/%j.err` |

---

## 5. 交互式开发（srun）

交互式模式适合：调试代码、短时间实验（< 2 小时）、探索性分析、跑单个评估步骤。

### 5.1 申请交互式 Shell

```bash
# 申请 1 张 GPU，获得一个交互式 bash shell
srun --partition=compute \
     --account=lab_cluster_account \
     --gres=gpu:1 \
     --time=02:00:00 \
     --ntasks=1 \
     --cpus-per-task=8 \
     --mem=64G \
     --pty bash
```

成功后，你会进入计算节点的 shell，此时可以直接运行 GPU 程序：

```bash
# 激活虚拟环境
source /home/dongjunteng/nanochat/.venv/bin/activate

# 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"

# 运行实验（例如：小规模训练测试）
python -m scripts.base_train --depth=6 --max_seq_len=256 \
    --device_batch_size=4 --num_iterations=200 \
    --run=interactive_test
```

**退出交互式会话**：输入 `exit` 或按 `Ctrl+D`，GPU 资源立即释放。

### 5.2 在计算节点运行单个命令

不需要完整 shell，可以像 `ssh` 一样直接执行命令：

```bash
# 直接在计算节点运行 Python 脚本
srun --partition=compute \
     --account=lab_cluster_account \
     --gres=gpu:1 \
     --time=01:00:00 \
     --ntasks=1 \
     --cpus-per-task=8 \
     bash -c 'source /home/dongjunteng/nanochat/.venv/bin/activate && python -m scripts.base_eval --depth=6'
```

### 5.3 交互式 Python 开发

如果需要交互式调试（类似 Jupyter 的体验）：

```bash
# 在计算节点启动 Python REPL
srun --partition=compute \
     --account=lab_cluster_account \
     --gres=gpu:1 \
     --time=01:00:00 \
     --cpus-per-task=8 \
     --pty bash -c 'source /home/dongjunteng/nanochat/.venv/bin/activate && python'
```

---

## 6. 批量提交训练任务（sbatch）

批量模式适合：完整预训练、多阶段流水线、夜间训练、参数扫描。

### 6.1 编写 Slurm 脚本

创建一个脚本文件 `slurm_scripts/train_base.slurm`：

```bash
#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nanochat_base
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%j_base_train.out
#SBATCH --error=logs/%j_base_train.err

# ====== 环境配置 ======
set -e  # 遇到错误立即退出
source /home/dongjunteng/nanochat/.venv/bin/activate

# 创建日志目录
mkdir -p logs

# 打印环境信息（方便排查）
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "GPU: ${SLURM_GPUS_ON_NODE:-8}"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "========================================="

# ====== 启动训练 ======
python -m scripts.base_train \
    --depth=20 \
    --device_batch_size=32 \
    --run=slurm_experiment

echo "========================================="
echo "End: $(date)"
echo "========================================="
```

### 6.2 提交任务

```bash
# 先创建日志目录
mkdir -p logs slurm_scripts

# 提交任务
sbatch slurm_scripts/train_base.slurm

# 输出示例：
# Submitted batch job 572
```

### 6.3 Slurm 脚本编写要点

1. **`#SBATCH` 行必须在脚本最前面**（在非注释命令之前），否则参数无效
2. **`source .venv/bin/activate` 必须在计算节点执行**，因为登录节点的虚拟环境可能缺失 GPU 相关包
3. **`set -e`**：任何命令失败立即退出，避免浪费 GPU 时间
4. **环境打印**：记录 `SLURM_JOB_ID`、节点名、PyTorch 版本等，方便后续排查
5. **日志路径**：用 `logs/%j_xxx.out` 格式（`%j` 自动替换为 Job ID），避免覆盖

---

## 7. 多 GPU 分布式训练

NanoChat 使用 PyTorch 原生 DDP（`torchrun`），在 Slurm 下有两种启动方式。

### 7.1 方式一：单任务多 GPU（推荐，适用于单节点）

本集群只有 1 个节点，推荐在单个 Slurm 任务中申请多张 GPU，然后通过 `torchrun` 启动多进程：

```bash
#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nanochat_ddp
#SBATCH --gres=gpu:8             # 申请全部 8 张 GPU
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%j_ddp.out
#SBATCH --error=logs/%j_ddp.err

set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

echo "Starting DDP training on ${SLURM_GPUS_ON_NODE:-8} GPUs..."

# NanoChat 的 base_train 内部通过 env 变量自动检测 DDP
# 关键环境变量由 torchrun 自动设置：RANK, LOCAL_RANK, WORLD_SIZE
torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train \
    --depth=20 \
    --device_batch_size=32 \
    --run=slurm_ddp_experiment
```

### 7.2 方式二：多 Slurm 任务（适用于多节点扩展）

如果你的集群将来扩展到多节点，可以使用以下方式：

```bash
#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nanochat_ddp_multi
#SBATCH --nodes=2                 # 2 个节点
#SBATCH --gres=gpu:8
#SBATCH --time=1-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/%j_ddp.out

set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

# 获取所有节点列表，计算总进程数
export nnodes=$SLURM_NNODES
export nproc_per_node=8
export total_procs=$((nnodes * nproc_per_node))

# 获取主节点地址
export master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export master_port=29500

torchrun --nnodes=$nnodes \
         --nproc_per_node=$nproc_per_node \
         --master_addr=$master_addr \
         --master_port=$master_port \
         -m scripts.base_train \
         --depth=20 \
         --run=slurm_multi_node
```

### 7.3 验证 DDP 是否正常工作

```bash
# 交互式验证：申请 2 个 GPU，启动 2 个进程的 DDP
srun --partition=compute \
     --account=lab_cluster_account \
     --gres=gpu:2 \
     --time=00:30:00 \
     --ntasks=1 \
     --cpus-per-task=16 \
     --pty bash

# 进入 shell 后：
source /home/dongjunteng/nanochat/.venv/bin/activate
torchrun --standalone --nproc_per_node=2 \
    -m scripts.base_train \
    --depth=6 --max_seq_len=256 \
    --device_batch_size=4 --num_iterations=50 \
    --run=ddp_test

# 观察日志中是否出现 2 个 rank 的输出（如 "[rank0]"、"[rank1]"）
```

---

## 8. 各实验的运行方式对照表

根据实验指导书中的 10 个实验，每个实验的运行建议：

| 实验 | 脚本 | GPU 需求 | 建议方式 | 预计时间 |
|------|------|---------|---------|---------|
| 一（分词器） | `tok_train.py` | **无需 GPU** | 登录节点直接运行 | ~10 分钟 |
| 二（模型架构） | 代码走读 + CPU Python | **无需 GPU** | 登录节点 | 4 小时 |
| 三（数据管线） | 代码走读 | **无需 GPU** | 登录节点 | 3 小时 |
| 四（预训练-小） | `base_train.py --depth=6` | 1 GPU | `srun --pty` | ~30 分钟 |
| 四（预训练-标准） | `base_train.py --depth=20` | 8 GPU | `sbatch` | 2-3 小时 |
| 五（优化器） | 代码走读 + 小测试 | 1 GPU | `srun --pty` | 4 小时 |
| 六（SFT） | `chat_sft.py` | 8 GPU | `sbatch` | 1-2 小时 |
| 七（RL） | `chat_rl.py` | 8 GPU | `sbatch` | 2-4 小时 |
| 八（推理引擎） | `chat_web.py` / `chat_cli.py` | 1 GPU | `srun --pty` | 3 小时 |
| 九（评估） | `base_eval.py` / `chat_eval.py` | 1-4 GPU | `srun` | 1-2 小时 |
| 十（全流程） | 整合上述 | 8 GPU | `sbatch` | 4-6 小时 |

---

## 9. 任务监控与管理

### 9.1 查看集群状态

```bash
# 查看分区和节点状态
sinfo

# 查看详细节点信息（GPU 类型、数量）
sinfo -o "%n %P %.11T %.14D %c %m %G"

# 查看所有任务（含已完成的任务信息）
squeue -o "%.10i %.8u %.10P %.10q %.12j %.8T %.10M %.10l %R"
```

### 9.2 查看自己的任务

```bash
# 查看当前排队/运行中的任务
squeue -u $(whoami)

# 查看近期任务（含已完成的）
sacct -u $(whoami) --format=JobID,JobName,Partition,State,Elapsed,ExitCode -S $(date -d '7 days ago' +%Y-%m-%d)
```

### 9.3 查看任务输出

```bash
# 实时查看运行中任务的输出
tail -f logs/<JOB_ID>_base_train.out

# 查看任务错误日志
cat logs/<JOB_ID>_base_train.err
```

### 9.4 取消任务

```bash
# 取消单个任务
scancel <JOB_ID>

# 取消自己的所有任务
scancel -u $(whoami)
```

### 9.5 查看 GPU 使用情况（在计算节点内）

```bash
# 必须先 srun 进入计算节点
srun --partition=compute --account=lab_cluster_account \
     --gres=gpu:1 --time=00:10:00 --pty bash -c '
watch -n 1 nvidia-smi
'
# 或者用一行命令查看
srun --partition=compute --account=lab_cluster_account \
     --gres=gpu:1 --time=00:05:00 \
     bash -c "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv"
```

---

## 10. 常见问题与排错

### 10.1 任务一直排队（PD 状态）不运行

```bash
# 查看排队原因
squeue -u $(whoami) -o "%.10i %.8u %.10P %.12j %.8T %.10M %.10l %.30R"

# 常见原因和解决方法：
# Reason="Resources"       → GPU 被占用，等空闲即可
# Reason="QOSMaxGRESPerUser" → 你的 QoS 限制了最大 GPU 数，减少 --gres=gpu:X
# Reason="Priority"        → 优先级不够，等待或联系管理员
# Reason="Association"     → 账户问题，检查 --account 参数
```

### 10.2 `torch.cuda.is_available()` 返回 False

```bash
# 1. 确认是通过 srun/sbatch 启动的（不是裸机运行）
echo $SLURM_JOB_ID  # 应该有一个数字

# 2. 确认申请了 GPU
echo $CUDA_VISIBLE_DEVICES  # 应该显示 GPU 编号

# 3. 确认 PyTorch 是 GPU 版本
source /home/dongjunteng/nanochat/.venv/bin/activate
python -c "import torch; print(torch.__version__); print('CUDA build:', torch.version.cuda)"
# 如果显示 "+cpu" 则 PyTorch 是 CPU 版本，需要重新安装
```

### 10.3 显存不足（Out of Memory）

```bash
# 减小设备批次大小
--device_batch_size=16  # 默认 32，可以降到 16、8、4

# 减小上下文长度
--max_seq_len=1024  # 默认 2048

# 减小模型深度
--depth=12  # 从 20 降到 12

# 检查当前显存占用
srun --gres=gpu:1 --time=00:05:00 --pty bash -c "nvidia-smi"
```

### 10.4 时间限制问题

最大时间限制为 1 天。对于超长训练，有两种策略：

```bash
# 策略 1：减少训练量（推荐用于学习目的）
--target_param_data_ratio=8  # 从默认 12 降低
--num_iterations=5000        # 直接指定步数

# 策略 2：利用检查点续训（高级用法）
# 训练结束后检查点会自动保存到 checkpoints/ 目录
# 下次启动时指定 --resume_from=checkpoints/xxx.pt 继续
```

### 10.5 `uv sync` 失败

```bash
# 检查网络连接
curl -I https://pypi.org

# 如有代理需求，设置环境变量
export HTTPS_PROXY=http://your-proxy:port
export HTTP_PROXY=http://your-proxy:port

# 如 PyTorch 下载太慢，可以用国内镜像
uv sync --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 11. 附录：Slurm 脚本模板库

### 模板 A：快速调试（1 GPU，30 分钟）

```bash
#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nanochat_debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%j_debug.out
#SBATCH --error=logs/%j_debug.err

set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

echo "Job ${SLURM_JOB_ID} starting on $(hostname) at $(date)"

python -m scripts.base_train \
    --depth=6 \
    --max_seq_len=256 \
    --device_batch_size=4 \
    --num_iterations=200 \
    --run=quick_debug

echo "Done at $(date)"
```

### 模板 B：标准预训练（8 GPU，深度 20）

```bash
#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nc_base_d20
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%j_base_d20.out
#SBATCH --error=logs/%j_base_d20.err

set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

echo "Job ${SLURM_JOB_ID}: base_train depth=20"

torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train \
    --depth=20 \
    --device_batch_size=32 \
    --run=base_d20

echo "Job ${SLURM_JOB_ID} finished at $(date)"
```

### 模板 C：监督微调 SFT（8 GPU）

```bash
#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nc_sft
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%j_sft.out
#SBATCH --error=logs/%j_sft.err

set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

echo "Job ${SLURM_JOB_ID}: chat_sft"

torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_sft \
    --depth=20 \
    --run=sft_d20

echo "Job ${SLURM_JOB_ID} finished at $(date)"
```

### 模板 D：强化学习 GRPO（8 GPU）

```bash
#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nc_rl
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=logs/%j_rl.out
#SBATCH --error=logs/%j_rl.err

set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

echo "Job ${SLURM_JOB_ID}: chat_rl"

torchrun --standalone --nproc_per_node=8 \
    -m scripts.chat_rl \
    --depth=20 \
    --run=rl_d20

echo "Job ${SLURM_JOB_ID} finished at $(date)"
```

### 模板 E：Web 服务部署（1 GPU，持续运行）

```bash
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

set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

echo "Job ${SLURM_JOB_ID}: Starting chat_web server"

# 获取计算节点 IP
NODE_IP=$(hostname -I | awk '{print $1}')
echo "Server will be available at http://${NODE_IP}:8000"

python -m scripts.chat_web --depth=20 --port=8000

# 用 Ctrl+C 或 scancel 停止
```

### 模板 F：模型评估（1 GPU）

```bash
#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=lab_cluster_account
#SBATCH --job-name=nc_eval
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%j_eval.out
#SBATCH --error=logs/%j_eval.err

set -e
source /home/dongjunteng/nanochat/.venv/bin/activate

echo "Job ${SLURM_JOB_ID}: base_eval"

python -m scripts.base_eval --depth=20

echo "=== Now running chat_eval ==="
python -m scripts.chat_eval --depth=20

echo "Job ${SLURM_JOB_ID} finished at $(date)"
```

---

### 快速操作速查表

```bash
# === 最常用的 10 个命令 ===

# 1. 查看集群状态
sinfo

# 2. 查看任务队列
squeue -u $(whoami)

# 3. 提交批量训练
sbatch slurm_scripts/train_base.slurm

# 4. 申请交互式 GPU shell
srun --gres=gpu:1 --time=02:00:00 --cpus-per-task=8 --pty bash

# 5. 在计算节点执行单条命令
srun --gres=gpu:1 --time=01:00:00 bash -c 'source .venv/bin/activate && python -c "..."'

# 6. 查看运行中任务的实时输出
tail -f logs/<JOB_ID>_*.out

# 7. 取消某个任务
scancel <JOB_ID>

# 8. 取消自己所有任务
scancel -u $(whoami)

# 9. 查看近期任务历史
sacct -u $(whoami) --format=JobID,JobName,State,Elapsed -S today

# 10. 检查 GPU 是否空闲（在计算节点）
srun --gres=gpu:1 --time=00:05:00 bash -c "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv"
```

---

*如有环境配置问题，请将错误信息连同 `squeue -u $(whoami)` 和 `sinfo` 的输出一起提交给管理员。*
