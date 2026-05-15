# verl FSDP + vLLM 全流程梳理

> verl 团队更推荐 FSDP + vLLM 方案:相比 Megatron,FSDP 不需要对模型结构做侵入式改动,一句 `FSDP(model)` 就够。ZeRO-3 + gradient checkpoint + offload 已经能撑到 70B 百卡量级,**覆盖 95% 以上的需求**。下面这条路径是去掉 Ray 之后能跑通的"丐版 verl",理解它再加 Ray 和 Megatron 就不是难事了。

---

## Init 阶段:搭好分布式环境

给定 prompt 数据集 + N 卡集群,构建分布式训练框架。

### 数据分片
FSDP + vLLM 方案下,除了 rollout 之外所有模块都用 DP。`DataLoader` 用 `DistributedSampler` 做分片,保证各 rank 拿到 batch 内不同分区。

### 模型初始化
1. 用 `transformers` 初始化模型。
2. 调 `_apply_liger_kernel_to_instance` 替换成 liger 高性能内核。
3. 对 actor / critic / ref / reward **四个模型分别用 FSDP 包起来**。

### HybridEngine 构建
HybridEngine = vLLM model + ShardingManager。

- **vLLM 适配**:改造 v0.7 前版本走 SPMD 并行,移除 Ray 依赖。
- **构建 rollout 模型**:把 actor 的相关参数传给 vLLM model,确保它和 actor 是同一个模型结构,然后立即把 vLLM model offload 到内存。
- **ShardingManager**:负责 rollout 时把 vLLM 参数和 actor 对齐并搬到 GPU,生成结束后再 offload 回去,并处理 generate 前后的数据 chunk/gather。
- **角色包装**:把原始 model 再 wrap 一层实现各角色功能——actor 在 stage2 算 logprobs、stage3 做 model update;critic 在 stage2 算 values、stage3 做 model update;等等。

---

## Forward 阶段:三个 stage

### Stage 1 — 生成 response

从 dataloader 拿到数据后,进入 rollout。每个 rank 上:

1. 遍历 actor model 的 `state_dict`,对每个 weight 调 `full_tensor()` 拼出完整张量。
2. 做必要的 split/concat,保证 layout 和 vLLM 实现对齐(参数名翻译 + qkv 合并这类)。
3. `weight.copy_()` 到 vLLM model,把 vLLM model 搬到对应 rank 的 GPU。
4. 释放掉刚才的 state_dict,节约显存。
5. 数据按 vLLM 的 tp_size 做 allgather(TP group 内 4 张卡接收相同输入)。
6. 调 `vllm.generate()`。
7. 结束后 offload 模型 + 清 KV cache,数据按原 DP 切分发回各 rank。

GRPO 这种一条样本要 sample 多次的算法,verl 的做法是 **stage1 把原始数据 repeat n 份**,stage2 再按 `uid` 计算同 prompt 样本之间的 mean / var。

### Stage 2 — inference-only

actor / critic / ref / reward 各自做一次 inference,产出 logprob、value、reward 等训练所需信号。

为避免 batch size 被显存卡死导致吞吐上不去,这里用 **micro_batch + `use_dynamic_bsz`** 控制显存占用。

### Stage 3 — 训练

数据已经齐了,走常规训练。注意三个超参的区别:

- `data_batch_size`:一次 rollout 拿多少条 prompt
- `mini_batch_size`:一次参数更新用多少条(决定 PPO 走几个 epoch)
- `micro_batch_size`:一次前反向算多少条(纯显存/速度调节)

---

## 这条简化版的边界

这里描述的流程和 verl 实际流程的区别主要在 **Ray**——真实 verl 用 Ray 做 WorkerGroup 调度。本文给的是不依赖 Ray 的最小版本,理解到这一步,基本可以**魔改一个去掉 Ray 和 Megatron 的丐版 verl**。

有 Megatron 基础的话,把满血 3D HybridEngine 加上去也就不难了(具体见 [rollout.md](rollout.md))。根据 HybridFlow 论文和实践经验,**FSDP + vLLM 这种方式在模型不大且 GPU < 100 卡时性能很好**,大多数场景下没必要直接上 Megatron 路径。
