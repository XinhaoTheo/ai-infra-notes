# verl 总览

> 占位文档，待补充。

## 待写要点

- HybridFlow 的 single-controller / multi-controller 混合编程模型
- Ray-based WorkerGroup 抽象与 ResourcePool 划分
- 4 大角色（Actor / Critic / Reference / Reward）+ Rollout 引擎的关系
- 训练后端：FSDP / FSDP2 / Megatron-LM
- 推理后端：vLLM / SGLang，权重 resharding 机制
- 主循环：`RayPPOTrainer.fit()` 拆解
