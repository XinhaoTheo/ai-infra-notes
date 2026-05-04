# Kernel Verification

GPU kernel 正确性验证、数值一致性、自动生成与测试相关的资料与笔记。

## 资料清单

### 论文

- [arXiv 2507.14111](https://arxiv.org/pdf/2507.14111) — *CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning*
  - **一句话**：用 speedup 作为唯一 reward，把一个原本写不好 CUDA 的 LLM 训练成会自动优化 kernel 的 agent；250 个 kernel 上平均 3.12× 加速、中位 1.42×，超过 torch.compile 和 cuDNN。
  - **三阶段 pipeline**：
    1. **SFT via data augmentation**：先用现有的（人写 / 编译器产出 / 其它模型生成的）CUDA 代码做监督微调，关键是用数据增强把同一个 kernel 的多种写法都喂进去，让模型先学会"CUDA 长什么样"。这一步只解决"能写出来 + 能编译过"的问题。
    2. **Self-supervised learning**：模型自己生成 -> 自己跑 -> 用"能不能编译 + 能不能跑对 + 跑多快"做自监督信号继续训。这一步把"语法对"提升到"功能对 + 有点性能"。重要的是这阶段不需要任何人类专家标注，正确性由 reference 输出对比保证。
    3. **Contrastive RL**：Core - 传统 RLHF 直接 reward 一份代码的 speedup，信号噪声. Instead 他们做法是把"同一 problem 下的多份候选实现"放在 prompt 里让模型对比着改，再用相对 speedup 做 reward。这样模型学到的不是"写一份快代码"而是"在已有方案上找还能再压榨什么"，避免了 reward hacking。
  - **Contrastive RL 的具体 trick（§2.4.3 – §2.4.5）**：
    - **§2.4.3 Exemplar Selection（怎么挑放进 prompt 的对比样本）**：
      - 维护一个 performance-indexed 的代码库，按 score 离散化到 buckets `B_k`（每个 bucket 是 `[s_k, s_k+Δs)` 区间）。
      - 每次 prompt 用 N=2 个 exemplar；先用 **温度归一化的 softmax** 在 bucket 之间采样：`P(B_i) ∝ exp((s̄_i − μ_s)/τ)`。和常规温度采样的区别是减去了全局 bucket 均值 μ_s，**把分数中心化到 0**，避免绝对分数大的 bucket 永远霸占采样。
      - 强制采样 N 个**不同**的 bucket，再从每个 bucket 内 uniform 采一份代码 → 同时满足"够强（偏向高分 bucket）"和"够散（性能差异大才能对比）"两个条件。
      - 试过 island-based（FunSearch 那套）但效果没差异，所以选了简单的 bucket 法。
    - **§2.4.4 Reward（怎么把 speedup 量稳）**：
      - 单次 reward = `t_ref / t_d`，但实测 `t_d` 噪声极大，于是堆了 7 条防抖措施：
        1. **独占 GPU**：每次 eval 独占一张卡，共享卡哪怕利用率低也会让方差爆掉。
        2. **配对执行 + 顺序随机化**：reference 和 candidate 在同一轮里都跑，且**随机化先后顺序**——消除 cache warm-up 带来的系统性偏置。
        3. **长测窗**：每个 candidate 跑满 30 分钟（几万到 1M 轮），用时间预算而不是固定轮数。
        4. **Bucketized 方差控制**：把所有 single-run 分数分成 7 个 bucket 求 bucket 内均值；如果 **bucket 间方差 > 0.005 直接丢弃**这次评测。
        5. **中位数取代均值**：最终 reward 取 7 个 bucket 均值的 **median**，对离群点更稳。
        6. **保守 rounding**：speedup 截到两位小数且**向 1 偏置**（1.118 → 1.11，0.992 → 1.00），不让随机抖动被记成"正收益"。
        7. **严格复核协议**：speedup 绝对值 > 3，或超过历史最大值的 2 倍，**换一张同型号 GPU 复测**，差距 > 10% 就拒收。
    - **§2.4.5 RL Training（GRPO 上的小改）**：
      - 用 **GRPO**（Group Relative Policy Optimization）：每个 prompt 采 G 份输出 `{d_1..d_G}`，组内 reward 标准化 `r̂_i = (r_i − mean(r))/std(r)`，省掉单独训 critic。
      - 和标准 GRPO 的差别：**reward 在喂进 GRPO 之前先做了 smoothing**，专门治 reward hacking（论文 §3 单独讲——比如模型会偷开 CUDA stream 让 KernelBench 的 timer 漏测、或者返回 lazy tensor 把计算推迟到 correctness check 那一步）。
      - 保留标准 GRPO 的 ratio clip + KL penalty `βD_KL(π_θ ‖ π_ref)`，防止策略漂太远。

- [arXiv 2512.23236](https://arxiv.org/abs/2512.23236) — *待补充笔记*

### 代码仓库

- [meta-pytorch/KernelAgent](https://github.com/meta-pytorch/KernelAgent) — Meta PyTorch 出品的 kernel 生成 agent
- [thinking-machines-lab/batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) — batch 不变性算子,用于推理数值一致性
