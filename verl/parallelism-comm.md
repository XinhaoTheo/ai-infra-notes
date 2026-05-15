# 并行策略的通信模式对比(DP / TP / PP)

读 verl / Megatron 这类框架前,先把三种并行的通信发生时机和原语记牢——后面看 HybridEngine、ShardingManager 的实现会顺很多。

---

## 速查表

| 并行 | 切什么 | 前向通信 | 反向通信 | 梯度同步 | 通信原语 |
|---|---|---|---|---|---|
| **DP** | 切数据 | 无 | 无 | **梯度 all-reduce**(每步) | all-reduce |
| **TP(Column)** | 沿 dim 0 切权重 | **all-gather** 输出 | **all-reduce** 输入梯度 | 不需要(本来就只持自己那片) | all-gather / all-reduce |
| **TP(Row)** | 沿 dim 1 切权重 | **all-reduce** 输出 | **all-gather** 输入梯度 | 不需要 | all-reduce / all-gather |
| **PP** | 切 layer | **P2P send/recv**(每段边界激活) | **P2P send/recv**(每段边界梯度) | 不需要(各 rank 持不同 layer) | send / recv(点对点) |

---

## DP(数据并行)

- **切法**:每个 rank 一份完整模型副本,数据切成 N 份各自跑。
- **前向 / 反向**:本地算,**不通信**。
- **反向后**:对梯度做一次 **all-reduce**,平均出"完整 batch 的梯度",所有 rank 拿到一样的梯度再各自走 optimizer。
- **变种**:ZeRO-1/2/3 把 optimizer state / gradient / parameter 也切了,通信量更大但显存更省;FSDP 的 FULL_SHARD ≈ ZeRO-3,前向反向都要 all-gather 出完整参数再用完释放。

## TP(张量并行)

切单个矩阵乘里的权重,N 张卡协作算一份 forward。切法决定通信方向——这个对偶关系很重要,verl 的 sharding manager 处理参数切分时都要按这个走:

**Column TP**(沿 dim 0 切):每个 rank 算出输出的一段,前向需要 **all-gather** 拼完整输出;反向时梯度走相反方向,需要 **all-reduce** 把输入梯度合并。

**Row TP**(沿 dim 1 切):每个 rank 拿到部分和,前向需要 **all-reduce** 求和得到完整输出;反向 **all-gather** 拼输入梯度。

实践里 Megatron 把这两种切法**配对使用**——`ColumnParallel → RowParallel` 紧挨着放,中间不通信,只在 RowParallel 输出处 all-reduce 一次,把通信量打到最低。

**梯度同步**:每个 rank 本来就只持有自己那片参数的梯度,不需要任何同步——这点和 DP 完全不同。

## PP(流水线并行)

- **切法**:模型按 layer 切成 N 段,rank 0 持第一段,rank N-1 持最后一段。
- **前向**:每段算完把边界激活通过 **P2P send/recv** 传给下一段。
- **反向**:每段算完把边界梯度通过 **P2P send/recv** 传给上一段。
- **梯度同步**:不需要(各 rank 持有的是不同 layer)。
- **代价**:bubble——朴素 PP 下 GPU 大量时间在等。1F1B、interleaved、ZeroBubble 这些 schedule 就是在压 bubble。
- **推理为什么不喜欢 PP**:bubble 在前向 only 的场景填不掉,decoding 又是 memory bound 不能靠切小 micro-batch 来弥补——所以 verl 的 Megatron HybridEngine 推理时第一步就是把 PP 压成 DP。

---

## 通信量直觉(单步)

设模型参数 P,激活 A,batch 内一条数据的激活 a:

| 并行 | 单步通信量量级 | 说明 |
|---|---|---|
| DP | O(P) | 一次全模型梯度 all-reduce |
| ZeRO-3 / FSDP | O(P) ×(前+反两次 all-gather + 一次 reduce-scatter) | 比 DP 重,但省显存 |
| TP | O(A) | 通信量正比于激活而非参数,适合大模型(P ≫ A) |
| PP | O(a × micro_batch) | 只传边界激活,通信量最小,但有 bubble |

**经验法则**:
- 小模型多卡 → DP 简单够用
- 单层都装不下的大模型 → TP 优先(NVLink 内 8 卡)
- 整个模型装不下单机 → 上 PP 跨机
- 三者混合 = 3D 并行,Megatron / DeepSpeed 的常规配置
