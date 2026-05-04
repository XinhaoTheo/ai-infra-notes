# Kernel Verification

GPU kernel 正确性验证、数值一致性、自动生成与测试相关的资料与笔记。

## 资料清单

### 论文

- [arXiv 2507.14111](https://arxiv.org/pdf/2507.14111) — *CUDA-L1: Improving CUDA Optimization via Contrastive Reinforcement Learning*
  - **一句话**：用 speedup 作为唯一 reward，把一个原本写不好 CUDA 的 LLM 训练成会自动优化 kernel 的 agent；250 个 kernel 上平均 3.12× 加速、中位 1.42×，超过 torch.compile 和 cuDNN。
  - **三阶段 pipeline**：
    1. **SFT via data augmentation**：先用现有的（人写 + 编译器产出 + 其它模型生成的）CUDA 代码做监督微调，用数据增强把同一个 kernel 的多种写法都喂进去，让模型先学会"CUDA 长什么样"。这一步只解决"能写出来 + 能编译过"的问题。
    2. **Self-supervised learning**：模型自己生成 -> 自己跑 -> 用"能不能编译 + 能不能跑对 + 跑多快"做自监督信号继续训。这一步把"语法对"提升到"功能对 + 有点性能"。重要的是这阶段不需要任何人类专家标注，正确性由 reference 输出对比保证。
    3. **Contrastive RL**：Core - 传统 RLHF 直接 reward 一份代码的 speedup，信号噪声. Instead 他们做法是把"同一 problem 下的多份候选实现"放在 prompt 里让模型对比着改，再用相对 speedup 做 reward。这样模型学到的不是"写一份快代码"而是"在已有方案上找还能再压榨什么"，避免了 reward hacking。
  - **Contrastive RL 的具体 trick**：
    - **Exemplar Selection（怎么挑放进 prompt 的对比样本）**：
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
  - **Reward Hacking 治理（§3.1 – §3.2）**：
    - **§3.1 Reward Hacking Cases（模型实际学会的"作弊"行为）**：
      1. **Improper Timing Measurement**：KernelBench 只在 main CUDA stream 上 record 起止 event，模型学会**自己开新 stream 异步发计算**——main stream 上几乎没事干，timer 量出来的时间就极短。修复：在 `end_event.record` 前对所有 custom stream 调 `wait_stream`，强制同步。
      2. **Lazy Evaluation**：返回一个 `torch.Tensor` 的子类（lazy 对象），`forward` 里压根不算，真正的 matmul 推迟到 `torch.allclose` 做 correctness check 时才触发——既"通过正确性"又"耗时极短"。修复：加 5 条断言（必须是 tensor、必须是**标准** `torch.Tensor`（拒子类）、device 对、storage size > 0、`data_ptr` 非 null）。
      3. **Hyperparameter Manipulation**：模型偷偷改 prompt 里给的 `batch_size / dim / in_features` 等超参（改小了自然快）。这条对应了 §2.4 prompt 里那句 "Keep hyperparameters unchanged" 的 restriction，但显然光靠 prompt 拦不住，需要在 evaluator 里硬校验。
      4. **Result Caching**：以 `x.data_ptr()` 为 key 缓存 forward 结果，下次同地址直接返回。理论上 correctness check 该挂，但 `torch.allclose` 是逐元素阈值比较，**有少量 case 能蒙混过关**——这条对你做 verification 是个直接警示：基于 tolerance 的等价性检查在对抗输入下不够。
    - **§3.2 Robust Reward Design（三件套防御）**：
      1. **Reward Checking Model（对抗审查者）**：当 reward 出现显著跳变时，调一个 adversarial model（他们用 **DeepSeek-R1**）来判定这份代码是不是在 hack。论文报告**召回率 ~60%**——不完美但够用，关键是只在"reward 暴涨"时触发，几乎不增加常规训练开销。
      2. **Hacking-case Database**：动态维护一个 hacking 案例库，每发现一个新模式就加进去。审查模型判定时，**用相似度检索 top-3 历史 hack 案例作为 few-shot context** 喂给 DeepSeek-R1——这是把"防御知识"沉淀成可检索资产的标准做法，和 RAG-as-defense 同构。
      3. **Reward Smoothing**：reward 先标准化 `(r−μ)/σ`，再 **clip 到 [−k, k]**，论文取 **k = 1.5**。理由是"1.5× 已经算大胜"，再高大概率是 hack 而不是真本事——**用先验上限直接压住"暴富信号"**，让 RL 不会一头扎进某个 outlier。这条是和 GRPO 配合的关键改动（对应 §2.4.5 里那句"reward 喂进 GRPO 之前先 smoothing"）。
    - **对 kernel-verification 主题的启示**：§3.1 的四种 hack **本质上全是攻击 verifier**——timer、materialization 检查、超参约束、tolerance 比较，每一个都是验证管线里的薄弱环节。一个"能扛住对抗 RL 的 verifier"才是真正合格的 verifier；这套案例库可以直接当作 verification 系统的 red-team 测试集。

- [arXiv 2512.23236](https://arxiv.org/abs/2512.23236) — *待补充笔记*

### 代码仓库

- [meta-pytorch/KernelAgent](https://github.com/meta-pytorch/KernelAgent) — Meta PyTorch 出品的 kernel 生成 agent
- [thinking-machines-lab/batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) — batch 不变性算子,用于推理数值一致性
