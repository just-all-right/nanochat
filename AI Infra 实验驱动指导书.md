## 基于 nanochat + nano-vLLM 的训练与推理系统实验

  

> **适用方向**：AI Infra 工程师、LLM 训练系统工程师、LLM 推理优化工程师、MLOps/模型服务工程师
>
> **核心目标**：不是“跑通一个大模型训练流程”，而是通过短实验理解训练与推理系统的瓶颈、指标、代码路径和工程权衡。
>
> **建议周期**：2–4 周完成基础版；6–8 周完成增强版。
>
> **硬件假设**：2 张高性能 GPU。
>
> **项目组合**：
>
> - `nanochat`：用于训练系统实验，包括 tokenizer、pretrain、SFT、RL、eval、chat。
>
> - `nano-vLLM`：用于推理系统实验，包括 prefill、decode、KV cache、batching、并发、prefix caching、tensor parallel。

  

---

  

## 0. 本版指导书的核心改变

  

第一版的问题在于：它把学习路径设计成了“读很多代码 → 跑完整流程 → 等结果 → 写总结”。

这会导致三个问题：

  

1. **GPU 没有被充分利用**：大量时间花在阅读和等待上，而不是做变量对照实验。

2. **学习反馈太慢**：完整训练跑很久，但只得到一个模糊结果，很难知道自己学到了什么。

3. **岗位导向不够聚焦**：算法、训练、推理、部署、评估全部覆盖，但 AI Infra 最需要的是系统指标、瓶颈分析和工程解释。

  

本版改为：

  

> **提出系统问题 → 改一个变量 → 跑短实验 → 记录指标 → 回代码定位原因 → 形成结论。**

  

你不需要一开始看懂所有代码。

你需要先建立“现象—指标—代码路径”的连接。

  

---

  

## 1. 总体学习目标

  

完成本指导书后，你应该能够回答这些 AI Infra 面试级问题：

  

### 训练侧

  

- batch size 增大时，tokens/s、显存、step time 如何变化？

- sequence length 增大时，为什么显存和耗时会明显上升？

- 单卡训练和双卡 DDP 训练为什么不是严格 2 倍加速？

- gradient accumulation 和真实 batch size 的区别是什么？

- mixed precision 对速度、显存和稳定性有什么影响？

- checkpoint、eval、logging 会如何影响训练吞吐？

- 数据加载慢时，GPU 利用率会出现什么现象？

  

### 推理侧

  

- prefill 和 decode 有什么区别？

- 为什么 TTFT 主要受 prompt 长度影响？

- 为什么 TPOT 主要受 decode 阶段影响？

- KV cache 为什么会成为长上下文和高并发瓶颈？

- continuous batching 为什么能提升吞吐？

- prefix caching 在什么请求模式下有效？

- tensor parallel 为什么有时会让单请求延迟变高？

- 如何用 QPS、tokens/s、TTFT、TPOT、P95 latency 评价推理服务？

  

---

  

## 2. 实验总原则

  

### 2.1 不做“大而慢”的盲跑

  

避免一上来运行完整 speedrun。

完整流程适合最终验收，不适合最初学习。

  

### 2.2 每个实验只改变一个变量

  

例如：

  

- 只改 batch size；

- 只改 sequence length；

- 只改 GPU 数量；

- 只改并发数；

- 只改 prompt length；

- 只改是否开启 prefix caching。

  

这样你才能知道结果变化来自哪里。

  

### 2.3 每个实验都必须记录系统指标

  

训练实验至少记录：

  

| 指标 | 含义 |

|---|---|

| loss | 学习是否正常 |

| step time | 每步训练耗时 |

| tokens/s | 训练吞吐 |

| GPU memory | 显存占用 |

| GPU utilization | GPU 是否吃满 |

| dataloader time | 数据加载是否成为瓶颈 |

| checkpoint time | 保存是否拖慢训练 |

  

推理实验至少记录：

  

| 指标 | 含义 |

|---|---|

| TTFT | 第一个 token 延迟 |

| TPOT | 每个输出 token 时间 |

| total latency | 总响应时间 |

| tokens/s | 输出吞吐 |

| QPS | 每秒请求数 |

| P50/P95/P99 latency | 服务稳定性 |

| GPU memory | 模型权重 + KV cache 显存 |

| concurrent requests | 并发请求数 |

  

---

  

## 3. 环境准备与实验记录

  

### 3.1 项目准备

  

```bash

# nanochat

git clone https://github.com/karpathy/nanochat.git

cd nanochat

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

```

  

```bash

# nano-vLLM

git clone https://github.com/GeeeekExplorer/nano-vllm.git

cd nano-vllm

python -m venv .venv

source .venv/bin/activate

pip install -e .

```

  

具体依赖以项目 README 为准。如果集群要求 Slurm，所有 GPU 程序必须通过 `srun` 或 `sbatch` 使用。

  

### 3.2 实验记录模板

  

每次实验都复制下面模板：

  

```markdown

## 实验编号：

日期：

项目：nanochat / nano-vLLM

代码 commit：

GPU 型号：

GPU 数量：

模型：

数据集：

变量：

固定参数：

运行命令：

  

### 指标记录

- 显存峰值：

- GPU 利用率：

- tokens/s：

- step time / TTFT / TPOT：

- loss / eval score：

- 异常现象：

  

### 观察

我观察到了什么？

  

### 解释

我认为原因是什么？

  

### 代码定位

相关代码文件：

相关函数：

  

### 下一步

为了验证这个解释，我下一步要改什么变量？

```

  

---

  

# 第一部分：nanochat 训练系统实验

  

## 4. 实验 T0：跑通最小训练链路

  

### 目标

  

不是训练出好模型，而是确认：

  

> batch → model → loss → backward → optimizer → log

  

这条训练路径能跑通。

  

### 推荐设置

  

- 小模型；

- 小数据；

- 100–300 step；

- 单卡；

- 不开启复杂功能；

- 先不追求最终效果。

  

### 运行方式示例

  

```bash

python -m scripts.base_train \

--depth=6 \

--max-seq-len=256 \

--device-batch-size=4 \

--num-iterations=100 \

--run=t0_tiny_pretrain

```

  

如果必须通过 Slurm：

  

```bash

srun --gres=gpu:1 --time=00:30:00 --pty bash

source .venv/bin/activate

python -m scripts.base_train \

--depth=6 \

--max-seq-len=256 \

--device-batch-size=4 \

--num-iterations=100 \

--run=t0_tiny_pretrain

```

  

### 必须记录

  

| 指标 | 结果 |

|---|---|

| 初始 loss | |

| 结束 loss | |

| 平均 step time | |

| tokens/s | |

| 显存峰值 | |

| GPU 利用率 | |

  

### 代码阅读任务

  

只看这一条路径：

  

1. 训练入口在哪里？

2. batch 是在哪里产生的？

3. model forward 在哪里？

4. loss 在哪里计算？

5. backward 在哪里？

6. optimizer.step 在哪里？

7. tokens/s 是在哪里统计的？

  

不要读完整项目。

  

---

  

## 5. 实验 T1：batch size 扫描

  

### 问题

  

> batch size 增大时，训练吞吐和显存如何变化？

  

### 实验设计

  

固定：

  

- depth；

- sequence length；

- GPU 数；

- 数据集；

- precision；

- 训练 step 数。

  

只改变：

  

| 实验 | device-batch-size |

|---|---:|

| T1-1 | 2 |

| T1-2 | 4 |

| T1-3 | 8 |

| T1-4 | 16 |

  

### 记录表

  

| 实验 | batch size | 显存峰值 | step time | tokens/s | loss 变化 | 是否 OOM |

|---|---:|---:|---:|---:|---:|---|

| T1-1 | 2 | | | | | |

| T1-2 | 4 | | | | | |

| T1-3 | 8 | | | | | |

| T1-4 | 16 | | | | | |

  

### 你要得出的结论

  

- batch size 增大后，tokens/s 是否上升？

- 显存是否近似线性上升？

- 有没有出现 GPU 利用率更高但 loss 更抖的情况？

- 当前显卡的最佳 batch size 是多少？

  

### 回代码定位

  

关注：

  

- batch shape；

- gradient accumulation；

- loss normalization；

- tokens/s 计算方式；

- 显存分配点。

  

---

  

## 6. 实验 T2：sequence length 扫描

  

### 问题

  

> sequence length 变长时，为什么训练成本上升明显？

  

### 实验设计

  

固定：

  

- depth；

- batch size；

- GPU 数；

- step 数。

  

只改变：

  

| 实验 | max-seq-len |

|---|---:|

| T2-1 | 256 |

| T2-2 | 512 |

| T2-3 | 1024 |

| T2-4 | 2048 |

  

### 记录表

  

| 实验 | seq len | 显存峰值 | step time | tokens/s | GPU 利用率 | 是否 OOM |

|---|---:|---:|---:|---:|---:|---|

| T2-1 | 256 | | | | | |

| T2-2 | 512 | | | | | |

| T2-3 | 1024 | | | | | |

| T2-4 | 2048 | | | | | |

  

### 你要得出的结论

  

- seq len 翻倍，step time 是否翻倍？

- attention 成本是否开始主导？

- 显存增长主要来自 activation 还是参数？

- 长序列下 batch size 是否需要下降？

  

### 回代码定位

  

重点看：

  

- attention 实现；

- causal mask；

- Flash Attention / SDPA 选择；

- activation 保存；

- `max-seq-len` 如何进入模型和 dataloader。

  

---

  

## 7. 实验 T3：单卡 vs 双卡训练

  

### 问题

  

> 两张 GPU 为什么不一定等于两倍速度？

  

### 实验设计

  

固定：

  

- depth；

- global batch 尽量一致；

- seq len；

- step 数。

  

对比：

  

| 实验 | GPU 数 | 并行方式 |

|---|---:|---|

| T3-1 | 1 | 单卡 |

| T3-2 | 2 | DDP / 项目内分布式 |

  

### 记录表

  

| 实验 | GPU 数 | global batch | step time | tokens/s | 加速比 | 每卡显存 | GPU 利用率 |

|---|---:|---:|---:|---:|---:|---:|---:|

| T3-1 | 1 | | | | 1.0x | | |

| T3-2 | 2 | | | | | | |

  

### 你要得出的结论

  

- 加速比是多少？

- 是否接近 2.0x？

- 如果没有，可能瓶颈是什么？

- batch 太小；

- 模型太小；

- 通信开销；

- 数据加载；

- logging/eval/checkpoint；

- GPU 互联带宽。

  

### 回代码定位

  

重点看：

  

- 分布式初始化；

- rank/world_size；

- 每张卡拿到的数据；

- 梯度同步；

- optimizer 分片或 all-reduce；

- checkpoint 只在 rank0 保存还是所有 rank 保存。

  

---

  

## 8. 实验 T4：gradient accumulation

  

### 问题

  

> 显存不够时，gradient accumulation 如何模拟更大的 global batch？

  

### 实验设计

  

对比两组等效 global batch：

  

| 实验 | device batch | grad accumulation | global batch |

|---|---:|---:|---:|

| T4-1 | 8 | 1 | 8 |

| T4-2 | 4 | 2 | 8 |

| T4-3 | 2 | 4 | 8 |

  

### 记录表

  

| 实验 | device batch | grad accum | 显存峰值 | step time | tokens/s | loss 稳定性 |

|---|---:|---:|---:|---:|---:|---|

| T4-1 | 8 | 1 | | | | |

| T4-2 | 4 | 2 | | | | |

| T4-3 | 2 | 4 | | | | |

  

### 你要得出的结论

  

- gradient accumulation 是否降低显存？

- 是否牺牲吞吐？

- 它和真正大 batch 的区别是什么？

- optimizer.step 的频率如何变化？

  

---

  

## 9. 实验 T5：训练过程中的“干扰项”分析

  

### 问题

  

> 训练慢到底是模型慢，还是 eval、logging、checkpoint 慢？

  

### 实验设计

  

对比：

  

| 实验 | eval | checkpoint | logging |

|---|---|---|---|

| T5-1 | 关 | 关 | 低频 |

| T5-2 | 开 | 关 | 低频 |

| T5-3 | 关 | 开 | 低频 |

| T5-4 | 开 | 开 | 高频 |

  

### 记录表

  

| 实验 | 平均 step time | eval 耗时 | checkpoint 耗时 | tokens/s 抖动 |

|---|---:|---:|---:|---|

| T5-1 | | | | |

| T5-2 | | | | |

| T5-3 | | | | |

| T5-4 | | | | |

  

### 你要得出的结论

  

训练系统不是只有 forward/backward。

评估、保存、日志、数据加载都会影响有效吞吐。

  

---

  

# 第二部分：nanochat 后训练实验

  

## 10. 实验 P0：SFT 最小实验

  

### 问题

  

> SFT 和预训练的训练目标有什么不同？

  

### 实验设计

  

跑一个短 SFT：

  

```bash

python -m scripts.chat_sft \

--depth=6 \

--num-iterations=100 \

--run=p0_sft_tiny

```

  

### 必须观察

  

| 观察项 | 说明 |

|---|---|

| 数据格式 | 是否是 role-based conversation |

| loss mask | 是否只对 assistant token 计算 loss |

| 学习率 | 是否低于预训练 |

| 行为变化 | 模型是否更像助手 |

  

### 代码阅读任务

  

只看：

  

- 对话样本如何 render；

- user token 是否 mask；

- assistant token 如何计算 loss；

- SFT checkpoint 如何加载 base checkpoint。

  

---

  

## 11. 实验 P1：RL / GRPO 最小实验

  

### 问题

  

> RL 后训练如何把 reward 转成参数更新？

  

### 实验设计

  

不要先跑长 RL。

先做 10–50 step 的短实验，观察日志。

  

记录：

  

| 指标 | 含义 |

|---|---|

| reward mean | 当前采样平均奖励 |

| reward std | 组内是否有区分度 |

| pass@k | 多采样是否提高通过率 |

| response length | 是否越来越长 |

| sample diversity | 是否模式坍缩 |

| policy loss | 是否稳定 |

  

### 你要特别观察

  

如果一组样本全对或全错，GRPO 的组内相对优势会怎样？

  

| 情况 | advantage |

|---|---|

| 全部 reward=1 | 全为 0，几乎无学习信号 |

| 全部 reward=0 | 全为 0，几乎无学习信号 |

| 一半对一半错 | 有正负 advantage，学习信号强 |

  

### 回代码定位

  

重点看：

  

- 每个 prompt 采样几个 response；

- reward 如何计算；

- group mean 如何计算；

- log probability 如何用于 loss；

- 是否有 KL、clip 或 length penalty。

  

---

  

# 第三部分：nano-vLLM 推理系统实验

  

## 12. 实验 I0：单请求生成

  

### 问题

  

> 一个请求在推理引擎里经过哪些阶段？

  

### 运行

  

用 nano-vLLM 加载一个小模型，先只做单请求生成。

  

记录：

  

| 指标 | 结果 |

|---|---|

| prompt tokens | |

| output tokens | |

| TTFT | |

| total latency | |

| output tokens/s | |

| 显存峰值 | |

  

### 代码阅读路径

  

只看：

  

> request → scheduler → prefill → KV cache → decode → sampling → output

  

不要先看所有优化细节。

  

---

  

## 13. 实验 I1：prompt length 扫描

  

### 问题

  

> prompt 越长，为什么第一个 token 越慢？

  

### 实验设计

  

固定 output length 和并发，只改变 prompt length：

  

| 实验 | prompt tokens |

|---|---:|

| I1-1 | 128 |

| I1-2 | 512 |

| I1-3 | 1024 |

| I1-4 | 2048 |

  

### 记录表

  

| 实验 | prompt tokens | TTFT | TPOT | total latency | 显存 |

|---|---:|---:|---:|---:|---:|

| I1-1 | 128 | | | | |

| I1-2 | 512 | | | | |

| I1-3 | 1024 | | | | |

| I1-4 | 2048 | | | | |

  

### 你要得出的结论

  

- prompt length 主要影响 prefill；

- decode 阶段每次生成一个 token；

- KV cache 大小与上下文长度、层数、KV heads、head dim 有关。

  

---

  

## 14. 实验 I2：output length 扫描

  

### 问题

  

> 输出越长，为什么总延迟几乎线性增加？

  

### 实验设计

  

固定 prompt length，只改变 output length：

  

| 实验 | output tokens |

|---|---:|

| I2-1 | 32 |

| I2-2 | 128 |

| I2-3 | 512 |

  

### 记录表

  

| 实验 | output tokens | TTFT | 平均 TPOT | total latency | tokens/s |

|---|---:|---:|---:|---:|---:|

| I2-1 | 32 | | | | |

| I2-2 | 128 | | | | |

| I2-3 | 512 | | | | |

  

### 代码定位

  

重点看 decode loop 和 sampling。

  

---

  

## 15. 实验 I3：并发请求压测

  

### 问题

  

> 并发增加时，吞吐和延迟如何权衡？

  

### 实验设计

  

固定 prompt/output length，只改变并发：

  

| 实验 | 并发数 |

|---|---:|

| I3-1 | 1 |

| I3-2 | 4 |

| I3-3 | 8 |

| I3-4 | 16 |

| I3-5 | 32 |

  

### 记录表

  

| 并发 | QPS | tokens/s | 平均延迟 | P50 | P95 | P99 | 显存 |

|---:|---:|---:|---:|---:|---:|---:|---:|

| 1 | | | | | | | |

| 4 | | | | | | | |

| 8 | | | | | | | |

| 16 | | | | | | | |

| 32 | | | | | | | |

  

### 你要得出的结论

  

- 并发增加通常提高吞吐；

- 但平均延迟和 P95 可能上升；

- 达到某个并发后，吞吐饱和；

- 饱和后继续加并发只会排队。

  

---

  

## 16. 实验 I4：prefix caching

  

### 问题

  

> 多个请求共享前缀时，缓存能节省多少 prefill 计算？

  

### 实验设计

  

准备两组请求：

  

1. 随机不同 prompt；

2. 共享长系统提示词 + 不同用户问题。

  

对比开启和关闭 prefix caching。

  

### 记录表

  

| 请求类型 | prefix caching | TTFT | tokens/s | 显存 | 备注 |

|---|---|---:|---:|---:|---|

| 不同前缀 | 关 | | | | |

| 不同前缀 | 开 | | | | |

| 共享前缀 | 关 | | | | |

| 共享前缀 | 开 | | | | |

  

### 你要得出的结论

  

prefix caching 不是任何时候都有效。

它对“共享长前缀”的场景收益最大，例如：

  

- 系统提示词很长；

- RAG 检索模板固定；

- 多轮对话历史前缀重复；

- 批量任务共用格式说明。

  

---

  

## 17. 实验 I5：单卡 vs 双卡 tensor parallel

  

### 问题

  

> 双卡推理为什么可能吞吐提升，但单请求延迟变差？

  

### 实验设计

  

同一个模型，对比：

  

| 实验 | GPU 数 | TP size |

|---|---:|---:|

| I5-1 | 1 | 1 |

| I5-2 | 2 | 2 |

  

### 记录表

  

| 实验 | GPU 数 | TP size | TTFT | TPOT | tokens/s | 每卡显存 | 备注 |

|---|---:|---:|---:|---:|---:|---:|---|

| I5-1 | 1 | 1 | | | | | |

| I5-2 | 2 | 2 | | | | | |

  

### 你要得出的结论

  

- 双卡能降低单卡显存压力；

- 大模型可能必须 TP 才能放下；

- 小模型 TP 可能因通信开销导致延迟上升；

- 是否使用 TP 取决于模型大小、并发、互联带宽、延迟要求。

  

---

  

# 第四部分：代码阅读方法

  

## 18. 训练代码只读主路径

  

nanochat 第一轮只读：

  

```text

配置参数

↓

数据 batch

↓

model forward

↓

loss

↓

backward

↓

optimizer step

↓

log / eval / checkpoint

```

  

你需要建立一张“训练 step 路径图”。

  

示例：

  

```markdown

## nanochat 训练 step 路径图

  

入口文件：

核心函数：

  

1. 读取参数：

2. 构建模型：

3. 构建数据加载器：

4. 取 batch：

5. forward：

6. loss：

7. backward：

8. optimizer step：

9. logging：

10. checkpoint：

```

  

## 19. 推理代码只读请求路径

  

nano-vLLM 第一轮只读：

  

```text

请求进入

↓

调度器排队

↓

prefill

↓

写入 KV cache

↓

decode loop

↓

sampling

↓

返回 token

```

  

你需要建立一张“请求生命周期图”。

  

示例：

  

```markdown

## nano-vLLM 请求生命周期图

  

入口文件：

核心类：

  

1. 请求对象创建：

2. 调度器接收：

3. prefill：

4. KV cache 分配：

5. decode：

6. sampling：

7. 输出：

```

  

---

  

# 第五部分：最终交付物

  

## 20. 最终报告结构

  

你的最终报告不应该写成“我完成了哪些章节”。

应该写成“我提出了哪些系统问题，并通过实验回答了它们”。

  

推荐标题：

  

> 基于 nanochat 与 nano-vLLM 的小型 LLM 训练与推理系统实验报告

  

结构：

  

```markdown

# 1. 实验环境

  

# 2. 训练系统实验

## 2.1 batch size 对吞吐和显存的影响

## 2.2 sequence length 对训练成本的影响

## 2.3 单卡与双卡训练扩展效率

## 2.4 gradient accumulation 对显存和吞吐的影响

## 2.5 eval/checkpoint/logging 对有效吞吐的影响

  

# 3. 后训练实验

## 3.1 SFT 中 assistant-only loss mask 的作用

## 3.2 GRPO 中 reward 分布与学习信号分析

  

# 4. 推理系统实验

## 4.1 prompt length 对 TTFT 的影响

## 4.2 output length 对 TPOT 和总延迟的影响

## 4.3 并发请求下吞吐与延迟的权衡

## 4.4 prefix caching 的收益边界

## 4.5 tensor parallel 的收益与代价

  

# 5. 代码路径分析

## 5.1 nanochat 训练 step 路径

## 5.2 nano-vLLM 请求生命周期

  

# 6. 关键发现

  

# 7. 后续优化方向

```

  

---

  

## 21. 简历描述方向

  

不要写：

  

> 跑通了 nanochat 全流程。

  

要写：

  

```text

基于 nanochat 与 nano-vLLM 构建小型 LLM 训练与推理系统实验平台：

- 设计并执行 batch size、sequence length、GPU 数量、gradient accumulation 等训练系统变量对照实验；

- 量化分析不同配置下的 tokens/s、step time、显存峰值与 GPU 利用率；

- 基于 nano-vLLM 测试 prompt length、output length、并发数、prefix caching、tensor parallel 对 TTFT、TPOT、QPS 和 P95 延迟的影响；

- 梳理训练 step 与推理 request 生命周期，定位数据加载、attention、KV cache、调度器和采样等关键代码路径；

- 形成可复现实验报告，用系统指标解释 LLM 训练与推理中的性能瓶颈和工程权衡。

```

  

---

  

## 22. 面试讲述方式

  

当面试官问“你做过什么 AI Infra 项目”时，不要先讲模型效果。

  

你应该这样讲：

  

> 我没有把 nanochat 当成一个单纯的训练脚本来跑，而是把它作为训练系统实验平台。我围绕 batch size、sequence length、GPU 数量、gradient accumulation 设计了短实验，记录 tokens/s、step time、显存和 GPU 利用率，分析不同变量对训练吞吐和资源占用的影响。然后我用 nano-vLLM 做推理侧实验，重点测 TTFT、TPOT、QPS、P95 latency、KV cache 和 prefix caching 的影响。最后我回到代码中梳理了训练 step 和推理 request 的关键路径，把实验现象和具体代码实现对应起来。

  

这比“我跑通了一个大模型训练项目”更像 AI Infra。

  

---

  

## 23. 推荐执行顺序

  

### 第 1 天

  

- 跑 T0；

- 跑 T1 batch size 扫描；

- 看训练 step 主路径代码。

  

### 第 2 天

  

- 跑 T2 sequence length 扫描；

- 跑 T3 单卡 vs 双卡；

- 初步解释扩展效率。

  

### 第 3 天

  

- 跑 T4 gradient accumulation；

- 跑 T5 eval/checkpoint/logging；

- 写训练侧小结。

  

### 第 4 天

  

- 跑 I0 单请求；

- 跑 I1 prompt length；

- 看 nano-vLLM request 主路径代码。

  

### 第 5 天

  

- 跑 I2 output length；

- 跑 I3 并发压测；

- 记录 TTFT、TPOT、P95。

  

### 第 6 天

  

- 跑 I4 prefix caching；

- 跑 I5 tensor parallel；

- 写推理侧小结。

  

### 第 7 天

  

- 整理最终报告；

- 画两张路径图：

- nanochat 训练 step 路径图；

- nano-vLLM 请求生命周期图。

  

---

  

## 24. 最重要的判断标准

  

一个实验是否有价值，不看它是否训练出了强模型，而看它是否回答了一个系统问题。

  

有价值的问题：

  

- 为什么这组配置更快？

- 为什么显存突然爆了？

- 为什么双卡没有两倍加速？

- 为什么第一个 token 慢？

- 为什么并发提升后延迟恶化？

- 为什么 prefix caching 只在某些场景有效？

- 为什么 TP 不一定降低延迟？

  

你要成为的不是“会跑 README 的人”，而是：

  

> **能用实验和指标解释 LLM 系统行为的人。**