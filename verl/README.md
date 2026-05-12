# verl 学习笔记

[verl (Volcano Engine Reinforcement Learning)](https://github.com/volcengine/verl) — 字节火山引擎开源的大模型 RL 训练框架，原型来自论文 *HybridFlow: A Flexible and Efficient RLHF Framework*。

## 资料

| # | 类型 | 标题 / 链接 | 笔记 |
|---|------|------|------|
| 1 | 仓库 | [volcengine/verl](https://github.com/volcengine/verl) | *待补充* |
| 2 | 论文 | [HybridFlow: A Flexible and Efficient RLHF Framework (arXiv 2409.19256)](https://arxiv.org/abs/2409.19256) | *待补充* |
| 3 | 文档 | [verl.readthedocs.io](https://verl.readthedocs.io/) | *待补充* |

## 笔记目录

- [overview.md](overview.md) — 框架总览、核心抽象（待写）
- 后续按主题拆分：
  - `hybrid-flow.md` — HybridFlow 编程模型 / single-controller + multi-controller
  - `worker-and-resource.md` — WorkerGroup / RayResourcePool / 角色调度
  - `rollout.md` — 推理后端（vLLM / SGLang）集成与 rollout 流程
  - `actor-critic.md` — Actor / Critic / Reference / Reward 各角色训练后端（FSDP / Megatron）
  - `algorithms.md` — PPO / GRPO / DAPO / RLOO 等算法实现细节
  - `recipes.md` — 官方 recipe 解读（math、tool-use、multi-turn、agent loop）

## 阅读路线（建议）

1. 先看 HybridFlow 论文，理解 **single-controller 调度 + multi-controller 计算** 这一核心设计。
2. 跑通 `examples/` 下的最小 PPO / GRPO 例子，对照 `verl/trainer/ppo/ray_trainer.py` 看主循环。
3. 顺着主循环拆 4 个角色（Actor / Critic / Ref / Reward）+ Rollout，分别落到 `verl/workers/` 下的实现。
4. 最后看算法层（advantage 估计、loss 计算）和 recipe 层（multi-turn / tool / agent）。
