# AI Infra 学习笔记

记录我在 AI 基础设施方向的学习笔记,主题涵盖 GPU、分布式训练、推理优化、系统与网络等。

## 目录

- [gpu/](./gpu/) — GPU 架构、CUDA、显存、通信(NCCL/NVLink)
- [training/](./training/) — 分布式训练、并行策略(DP/TP/PP/EP)、Megatron、DeepSpeed、FSDP
- [inference/](./inference/) — 推理引擎(vLLM、SGLang、TensorRT-LLM)、KV Cache、量化、投机解码
- [systems/](./systems/) — 调度、存储、网络(RDMA、IB)、容器化
- [kernel-verification/](./kernel-verification/) — GPU kernel 正确性验证、数值一致性、自动生成
- [agentic-rl/](./agentic-rl/) — Agent 化 RL、multi-agent / multi-LLM 训练框架
- [verl/](./verl/) — verl (HybridFlow) RLHF 训练框架学习笔记
- [papers/](./papers/) — 论文阅读笔记
- [daily/](./daily/) — 按日期记录的学习日志
- [assets/](./assets/) — 图片等资源

## 笔记规范

- 文件名:英文小写 + 连字符,如 `kv-cache.md`
- 图片统一放 `assets/`,用相对路径引用
- 内部链接用相对路径,方便以后迁移到静态站

## 学习路线(待完善)

1. **基础**:GPU 架构 → CUDA 编程 → 显存管理
2. **训练**:数据并行 → 模型并行 → 流水线并行 → 专家并行
3. **推理**:KV Cache → PagedAttention → 量化 → 投机解码
4. **系统**:集合通信 → RDMA → 调度与编排
