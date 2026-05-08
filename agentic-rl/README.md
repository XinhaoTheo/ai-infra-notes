# Agentic RL

Agent 化的强化学习相关框架、多 agent 训练系统、以及把 RL 应用到 agent / multi-LLM 场景的资料与笔记。

## Introduction

来源：[知乎专栏 — Agentic RL 综述](https://zhuanlan.zhihu.com/p/1985054130469888615)

这里的「Agent」指的是把传统 RL 的"状态–动作–环境–回报"框架套到 LLM 上：

- **状态 $s$**：环境信息 + Agent 内部记忆（history、工具输出、数据库状态……）
- **动作 $a$**：不再只是"下一个 token"，而是——选择工具、构造 SQL / API 调用、规划子任务、决定是否继续对话、是否写入知识库等等
- **环境 $E$**：真实的数据库、Web API、用户、任务队列、文件系统……会随着动作变化
- **回报 $r$**：和任务成功率、延迟、成本、用户满意度、安全约束相关
- **策略 $\pi$**：可以由 LLM + 工具组成，但 RL 优化的是「整个决策流程」

一句话总结：**Agentic RL = 在"状态–动作–环境反馈"这个闭环上做 RL，LLM 只是这个闭环里实现策略的一部分**。

这时候 LLM 不再仅仅是"嘴巴"（生成文本），而是成了"大脑"（决策中心）——它通过操纵"四肢"（工具 / API）与"世界"（环境）交互，并根据"绩效指标"（reward）来优化自身的决策逻辑。


## 论文

| # | 标题 | 论文链接 | 笔记 |
|---|------|---------|------|

## 代码仓库

| # | 名称 | 仓库 | 笔记 |
|---|------|------|------|
| 1 | agent-lightning — Microsoft 出品的 agent RL 训练框架 | [microsoft/agent-lightning](https://github.com/microsoft/agent-lightning) | [agent-lightning.md](repos/agent-lightning.md) |
| 2 | MARTI — 清华 C3I 的 multi-agent RL 训练基础设施 | [TsinghuaC3I/MARTI](https://github.com/TsinghuaC3I/MARTI) | [marti.md](repos/marti.md) |
| 3 | PettingLLMs — multi-LLM agent 训练 / 评估框架 | [pettingllms-ai/PettingLLMs](https://github.com/pettingllms-ai/PettingLLMs) | [pettingllms.md](repos/pettingllms.md) |
| 4 | CoMLRL — Cooperative multi-LLM RL | [OpenMLRL/CoMLRL](https://github.com/OpenMLRL/CoMLRL/) | [comlrl.md](repos/comlrl.md) |
