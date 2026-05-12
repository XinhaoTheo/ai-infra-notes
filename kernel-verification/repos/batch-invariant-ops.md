# thinking-machines-lab/batch_invariant_ops

- 仓库：[thinking-machines-lab/batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops)

batch 不变性算子，用于推理数值一致性。

## Defeating Nondeterminism in LLM Inference
**A walkthrough of Horace He & Thinking Machines Lab's blog post**

https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/#true-on-policy-rl

### 1. The Puzzle: Why Are LLMs Nondeterministic Even at Temperature 0?
Anyone who has used ChatGPT knows that asking the same question twice often yields different answers. We typically blame this on sampling randomness, so the natural fix is to set temperature = 0 (greedy sampling — always pick the highest-probability token). In theory, this should make outputs perfectly reproducible. But it doesn't. Even with temperature = 0, even when you run inference yourself on your own GPU using vLLM or SGLang, the outputs still vary across runs. Why? This blog sets out to answer that question — and the answer turns out to be much subtler than the conventional explanation suggests.

### 2. The Conventional (and Wrong) Explanation: "Concurrency + Floating Point"
The most popular hypothesis in the community goes like this:

"GPUs are massively parallel. Floating-point addition isn't associative — (a+b)+c ≠ a+(b+c). So when concurrent threads finish in nondeterministic order and accumulate into a shared sum (via atomic adds), you get nondeterministic results."

This explanation appears in arXiv preprints, blog posts, and Stack Overflow answers everywhere. It sounds technical and authoritative.
Horace shows it's wrong — at least for LLM inference — with a single counter-example:

```python
## Atomic version which is not used in nowdays LLM kernel
__shared__ float acc;
acc = 0;
__syncthreads();
atomic_add(&acc, my_partial);   # sequence is not fixed


## Tree structure reduction
__shared__ float partial[32];           # 32 slots
partial[tid] = my_value;                # thread tid write to slot tid, the slot is fixed
__syncthreads();                         # wait for all threads finished

for (int stride = 16; stride > 0; stride /= 2) {
    if (tid < stride) {
        partial[tid] += partial[tid + stride];   # thread tid always add the number in fixed position
    }
    __syncthreads();                     # sync, every step will see fixed number
}
# final answer in partial[0]

"""
T0 v0 ──┐
          (v0+v4)──┐
  T4 v4 ──┘        │
                   ((v0+v4)+(v2+v6))──┐
  T2 v2 ──┐        │                  │
          (v2+v6)──┘                  │
  T6 v6 ──┘                           │
                                      final
  T1 v1 ──┐                           │
          (v1+v5)──┐                  │
  T5 v5 ──┘        │                  │
                   ((v1+v5)+(v3+v7))──┘
  T3 v3 ──┐        │
          (v3+v7)──┘
  T7 v7 ──┘
"""
```

```python
A = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
B = torch.randn(2048, 2048, device='cuda', dtype=torch.bfloat16)
ref = torch.mm(A, B)
for _ in range(1000):
    assert (torch.mm(A, B) - ref).abs().max() == 0  # always passes!
```
If "concurrency + floating point" were the real cause, this above test should fail. The GPU is parallel; floating-point math is involved. Yet torch.mm returns bitwise-identical results 1000 times in a row. So the conventional story is missing something.
The reason: modern GPU kernels for LLMs almost never use atomic adds. They use split reductions, tree reductions, or other deterministic strategies that achieve parallelism without sacrificing reproducibility. So GPU concurrency, by itself, is not the source of LLM nondeterminism.


### 3. The Real Culprit: Lack of Batch Invariance
If the kernels themselves are run-to-run deterministic, why is LLM inference nondeterministic? Horace's key insight:

> LLM kernels are deterministic given a fixed input — but their output depends on the batch size they're run under.

This property is called batch invariance (or rather, its absence). A simple PyTorch demonstration:

```
out1 = torch.mm(a[:1], b)        # process just row 0
out2 = torch.mm(a, b)[:1]        # process whole batch, take row 0
print((out1 - out2).abs().max()) # tensor(1669.25) huge difference!
```

Mathematically, row 0's output should be identical regardless of whether we batch it with other rows. **But GPU matmul kernels switch internal strategies based on batch size** — different tiles, different reduction orders, different tensor-core instructions — so the floating-point output drifts.

**Why this makes LLM serving nondeterministic for users:** When you query an inference server, you don't know how many other people are simultaneously querying it. The server's load determines the batch size your request gets folded into, which determines which kernel path runs, which determines your exact output. From the server's perspective it's deterministic; from your perspective it's not.
This is why the nondeterminism isn't unique to GPUs — CPU- or TPU-served LLM endpoints have the same problem.

#### Affected selection within the kernel 
1. **size of tile / block** Why this changes the reduction order: the K-dim loop advances by BLOCK_K each iteration. BLOCK_K=64 and BLOCK_K=32 produce completely different accumulation sequences for the same (m, n) output element — one does (K / 64) accumulations, the other does (K / 32). Every accumulation is a floating-point add, so a different number of steps → a different accumulation error.
2. **Whether Split-K is enabled** Large batch (M, N both big): plenty of tiles to go around so each SM owns one (m, n) this makes the entire K reduction stays serial inside a single program. Small batch (M = 1): few tiles, most SMs sit idle, kernel falls back to Split-K: chop the K dim into segments, let several programs each compute a partial sum, then atomic_add them into the same C[m, n].
3. Tensor core instruction selection `wgmma.mma_async.bf16.f32`, `mma.sync.aligned.m16n8k16`
4. Shape of Grid. in directly `grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))`



### 4. The Fix: Making Three Critical Kernels Batch-Invariant

A transformer's forward pass uses three reduction-heavy operations. To get fully reproducible inference, all three must be made batch-invariant. Horace presents them in increasing order of difficulty.

#### 4a. RMSNorm (the easy case)
The natural strategy is data-parallel: assign one row to one core, perform the reduction entirely within that core. As batch size grows, just have each core handle multiple rows sequentially — the reduction order per row never changes. Batch invariance is preserved for free.

The problem: when batch size is small, you have idle cores. A kernel engineer would normally fix this by splitting the reduction across cores — but that changes the reduction order and breaks batch invariance.

Fix: Just don't split. Small batches are inherently fast, so the performance loss from leaving cores idle is acceptable.

#### 4b. Matrix Multiplication (the medium case)
Same data-parallel principle: chunk the output into 2D tiles, give one tile to each core. Two extra wrinkles:

Tensor cores require large tiles ([128, 128] is standard) to be efficient. So even if M and N look "big" mathematically, they may not generate enough tiles to saturate the GPU.
When tiles run out, kernels fall back to Split-K (splitting the reduction dimension) — which breaks batch invariance.
At extremely small batch sizes (e.g., M = 1 in decoding), kernels may switch to smaller tensor-core instructions or even bypass tensor cores entirely. Each switch breaks invariance.

Fix: Compile a single kernel configuration and use it across all shapes, accepting ~20% performance loss vs. cuBLAS. In LLM inference this isn't disastrous, since the model dimension N is usually large enough that Split-K rarely kicks in

#### 4c. Attention (the hardest case)
Attention is the toughest because it contains two matmuls and reduces over both the feature dimension and the sequence dimension — and it interacts with inference-engine optimizations like chunked prefill, prefix caching, and paged KV cache.

**Problem 1:** Inconsistent KV layout. vLLM's old kernel processes KV cache and current K/V as two separate segments. When the same query token is processed in prefill (everything in current) versus decoding (everything in cache), the kernel splits the data into a different number of blocks (e.g. 8 vs. 9), which changes the reduction order.

**Fix:** Write current K/V into the KV cache before the attention kernel runs, so the kernel always sees a single, consistently-laid-out block of data — independent of which inference stage produced it.

**Problem 2:** Split-KV (FlashDecoding). During decoding, query length is 1, so parallelism along Q is essentially zero. With long KV caches, a single core would take forever. Engines therefore split the KV dimension across cores — **but the standard strategy ("fixed number of splits") gives different per-split sizes for different KV lengths, breaking invariance.**

**Fix:** Fixed split size, not fixed split count. Always use chunks of e.g. 256 elements. The number of chunks varies with KV length, but each chunk's internal reduction is always the same. Reduction order is preserved regardless of how many query tokens are being processed.

<table>
<tr>
<td><img src="../pics/attention-03.svg" alt="Fixed split count — before fix" width="100%"></td>
<td><img src="../pics/attention-04.svg" alt="Fixed split size — after fix" width="100%"></td>
</tr>
<tr>
<td align="center"><sub>修复前：fixed split count（4×250），KV 长度变 → 每块大小变</sub></td>
<td align="center"><sub>修复后：fixed split size（256/256/256/232），归约顺序不变</sub></td>
</tr>
</table>

### 5. Implementation and Empirical Validation
Determinism test: With Qwen3-235B at temperature 0, the prompt "Tell me about Richard Feynman" generates 80 different completions across 1000 runs with default vLLM. They all agree for the first 102 tokens, then diverge — most produce "Queens, New York", a few produce "New York City". With batch-invariant kernels enabled: all 1000 completions are bitwise identical.

Performance cost: Generating 1000 sequences on a single GPU running Qwen-3-8B takes 26s with default vLLM, 42s with the optimized batch-invariant version, 55s without further optimization. Roughly 1.6× slower — usable, not optimal. Most of the slowdown is a not-yet-optimized FlexAttention integration, not a fundamental ceiling.

### 6. The Big Payoff: True On-Policy RL
The most important application — and where this work has real research impact — is in RLHF training.
The problem RL practitioners have been quietly tolerating: Sampler (vLLM) and trainer (PyTorch) compute logprobs through different forward-pass implementations. The numerical mismatch is small (KL divergence ~0.001) but persistent. This means RL algorithms that believe they're on-policy are actually subtly off-policy. The standard workaround is importance weighting — multiplying each sample by π_trainer / π_sampler to correct for the discrepancy.
Horace's experiments on BigMath with Qwen 2.5-VL 8B compare three configurations:

- Without importance weighting: Reward trains smoothly for a while, then collapses catastrophically — accompanied by a sudden KL spike. The numerical drift accumulates until the gradient direction goes badly wrong.
- With importance weighting: Training is stable. KL hovers around 0.001 with occasional spikes. This is the standard RLHF setup today.
- True on-policy (batch-invariant kernels in both sampler and trainer): KL divergence is exactly 0 throughout training. No importance weighting needed. Training is stable.

In other words: batch invariance lets us drop the importance-weighting term entirely and recover truly on-policy RL. This simplifies algorithms (PPO's `min(r atio·A, clip(ratio)·A)` collapses back to a vanilla policy gradient), removes a hyperparameter (clip range), and makes training fundamentally more stable.

![Figure 16 — Reward & KL divergence: True on-policy vs. importance weighting vs. no correction](../pics/Snipaste_2026-05-06_13-43-59.png)



## Code review

The whole library is just 4 pieces: 3 batch-invariant kernels (matmul / log_softmax / mean) + 1 global toggle mechanism that uses `torch.library` to override ATen ops. Each kernel below is contrasted against the "typical" implementation.

### 1. `matmul_kernel_persistent` — Matrix multiplication

**Design highlights**

- **Persistent grid**: `grid = min(NUM_SMS, num_tiles)` — launches exactly one program per SM, each consuming multiple tiles via `for tile_id in tl.range(start_pid, num_tiles, NUM_SMS)`.
- **K-dim runs serially inside one program**: each tile's K reduction completes in a single program through `for ki in range(k_tiles)`. **No split-K, ever.**
- **Hard-coded BLOCK configs**: `(128, 128, 64)` for bf16, `(128, 256, 64)` for fp16, `(128, 128, 32)` for fp32. **No autotune.**
- **FP32 accumulator**: `accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)`, even with bf16/fp16 inputs. Maps directly to Tensor Core `wgmma.mma_async.bf16.f32`.
- **L2-friendly super-grouping**: `_compute_pid` remaps the linear `tile_id` into a "small-block-row × column-first" `(pid_m, pid_n)` layout, so concurrently active SMs share A row-bands and B column-bands.
- **Pipeline-friendly `tile_id_c`**: `tile_id_c = start_pid - NUM_SMS` + per-iteration `tile_id_c += NUM_SMS` decouples the write-back coordinate from the K-accumulation chain, It lets the compiler work out the destination address in advance, instead of having to wait for the K-loop to wrap up before it even knows where to put the result. This gives the compiler room to overlap "store of tile N" with "K-accumulation of tile N+1".

**vs. a typical matmul kernel**

| Aspect | Typical (cuBLAS / default Triton) | This implementation | Why the change |
|---|---|---|---|
| Grid shape | `(cdiv(M, BM), cdiv(N, BN), split_k)` | 1D persistent `min(NUM_SMS, num_tiles)` | Tile count is bound to SM count, so different batch sizes don't trigger different launch shapes |
| K-dim reduction | Split-K when shapes are large (multiple programs accumulate into the same C tile) | Single program walks the full K | Split-K's partial sums use atomics / cross-program reduces → order changes → batch invariance breaks |
| Tile size | `triton.autotune` picks the fastest config per (M, N, K) | dtype fixed → BLOCK fixed | Autotune picks different tiles at different batch sizes, so reduction order changes |
| Small-batch optimization | At M=1 / small M, fall back to smaller mma instructions or even bypass Tensor Cores | Always the same code path, no switching | "Switching → different accumulation pattern → numerical drift" |

**Snippet comparison**

A typical split-K path (breaks invariance):

```python
# Under different KV lengths / batch sizes, autotune picks different SPLIT_K
@triton.autotune(configs=[
    triton.Config({'BLOCK_M':128,'BLOCK_N':128,'BLOCK_K':32,'SPLIT_K':1}),
    triton.Config({'BLOCK_M':64, 'BLOCK_N':64, 'BLOCK_K':32,'SPLIT_K':4}),
    triton.Config({'BLOCK_M':32, 'BLOCK_N':32, 'BLOCK_K':32,'SPLIT_K':8}),
], key=['M','N','K'])
@triton.jit
def matmul_split_k(...):
    pid_m, pid_n, pid_k = ...                       # 3D grid
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for ki in range(pid_k, k_tiles, SPLIT_K):       # each program sees only part of K
        acc += tl.dot(a, b)
    tl.atomic_add(c_ptr + ..., acc)                 # ← cross-program accumulation, order is non-deterministic
```

The batch-invariant version:

```python
# Pick config purely by dtype, never autotune; grid is 1D persistent
configs = {torch.bfloat16: {'BLOCK_SIZE_M':128,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':64, ...}, ...}

def grid(META):
    return (min(NUM_SMS,
                triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N'])),)

@triton.jit
def matmul_kernel_persistent(...):
    start_pid = tl.program_id(0)
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(tile_id, ...)   # super-grouping, L2-friendly
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for ki in range(k_tiles):                   # full K reduced inside a single program
            accumulator = tl.dot(a, b, accumulator)
        tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty))
```

Core idea: **trade "do less optimization" for "the K-accumulation sequence at every (m, n) is bitwise identical regardless of batch size"**. The cost is the ~20% slowdown vs. cuBLAS noted in §4b.

### 2. `_log_softmax_kernel` — Log-softmax

**Design highlights**

- **One row per program**: `grid = (n_rows,)`, each row's program runs three passes (max → sum → write), with no cross-program communication within a row.
- **Tiled in BLOCK but reduction stays in one program**: `for col_offset in range(0, n_cols, BLOCK_SIZE)` walks a row sequentially; both `tl.max` and `tl.sum` are confined to the same program.
- **Standard three-pass numerical stability trick**: row max → `sum(exp(x - max))` → write `x - max - log(sum_exp)`. All three passes share the same program-private `max_val` / `sum_exp`, **so reduction order depends only on `n_cols` and is independent of batch (`n_rows`)**.
- **Fixed `BLOCK_SIZE = 1024`**: like matmul, no autotune.

**vs. a typical log_softmax**

| Aspect | Typical | This implementation | Why |
|---|---|---|---|
| Intra-row reduction | Large vocab (e.g. 32k+) → split into multiple programs that partial-reduce, then a second-stage merge | Single program walks the full row | Two-stage merge ordering depends on launch order, breaking invariance |
| `BLOCK_SIZE` | Autotune or vocab-adaptive | Fixed at 1024 | Same as above |
| Row scheduling | One program per row (same as here) | Same | Rows are embarrassingly parallel — already invariant |

**Snippet comparison**

A typical "two-stage reduction" approach (breaks invariance):

```python
# Stage 1: split each row into chunks; each program computes its chunk's (max, sum)
@triton.jit
def stage1(...):
    row, chunk = tl.program_id(0), tl.program_id(1)
    local_max = tl.max(tl.load(...))
    local_sum = tl.sum(tl.exp(... - local_max))
    tl.atomic_max(global_max + row, local_max)      # ← cross-program merge
    # ... another atomic pass to fix up sum
```

The batch-invariant version:

```python
@triton.jit
def _log_softmax_kernel(input_ptr, output_ptr, ..., n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0).to(tl.int64)         # one program per row

    # Pass 1: serial max within the row
    max_val = -float('inf')
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        vals = tl.load(...)
        max_val = tl.max(tl.maximum(vals, max_val))

    # Pass 2: serial sum(exp(x - max)) within the row
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        sum_exp += tl.sum(tl.where(mask, tl.exp(vals - max_val), 0.0))

    # Pass 3: write x - max - log(sum_exp)
    log_sum_exp = tl.log(sum_exp)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        tl.store(..., vals - max_val - log_sum_exp)
```

Cost: when `n_cols` is huge (LM vocab), three serial sweeps of a row are slower than a parallel two-stage reduction — but this avoids the non-deterministic ordering of cross-program merges.

### 3. `mean_kernel` — Mean along one dim

**Design highlights**

- **Unified 3D view**: any ndim tensor is reshaped to `(M, N, K)`, where **N is always the reduction dim**, M is the product of preceding dims, K is the product of trailing dims.
- **One output element per program**: `grid = (M*K,)`, each program computes one scalar output by accumulating along N inside the program.
- **Reduction is fully program-local**: `for n_start in range(0, N, BLOCK_SIZE): acc += tl.sum(tl.load(...))`. `acc` is program-private — no cross-program merging.
- **Integer inputs auto-promote to fp32**, matching PyTorch's default behavior.

**vs. a typical mean**

| Aspect | Typical | This implementation | Why |
|---|---|---|---|
| Parallelizing the reduction dim | When N is large, split across multiple programs and merge via tree reduction / atomics | N is **always** serial; parallelism lives only in the (M, K) output space | Cross-program merging = non-deterministic order |
| Multi-dim reduce | Single kernel handles multi-dim reduce | Multi-dim reduce falls back to `torch.sum` then divide by N (see `mean_batch_invariant`) | Reuse the single-dim invariant path, no need for a second multi-dim kernel |
| Block size | Autotune | Fixed `BLOCK_SIZE = 1024` | Same as matmul: decouple accumulation order from batch |

**Snippet comparison**

A typical "split N + atomic merge" approach (breaks invariance):

```python
@triton.jit
def mean_split(input_ptr, output_ptr, M, N, K, ...):
    pid_mk, pid_n = tl.program_id(0), tl.program_id(1)   # N is also parallel
    local = tl.sum(tl.load(...))
    tl.atomic_add(output_ptr + pid_mk, local / N)        # ← order-sensitive
```

The batch-invariant version:

```python
@triton.jit
def mean_kernel(input_ptr, output_ptr, ..., M, N, K, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                              # one program → one output scalar
    m_idx, k_idx = pid // K, pid % K

    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):             # N is fully serial inside the program
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        vals = tl.load(input_ptr + m_idx*s0 + n_offsets*s1 + k_idx*s2,
                       mask=n_offsets < N, other=0.0)
        acc += tl.sum(vals)

    tl.store(output_ptr + m_idx*os0 + k_idx*os1, acc / N)
```

### 4. Toggle mechanism — global override via `torch.library`

**Design highlights**

- **No user code changes, no PyTorch fork, no recompile**: `torch.library.Library("aten", "IMPL")` re-registers ATen ops under a specified dispatch key (CUDA/XPU), directly overriding the default kernels.
- **Overrides 4 ops**: `aten::mm`, `aten::addmm`, `aten::_log_softmax`, `aten::mean.dim`. Note that attention is fixed at the call site (write current K/V into the cache + fixed split size), **not hijacked here**.
- **Process-wide toggle**: `enable_batch_invariant_mode()` / `disable_batch_invariant_mode()` manage a single global `_batch_invariant_LIB` handle; `disable` calls `_LIB._destroy()` to tear down the override.
- **Scoped toggle**: `set_batch_invariant_mode` is a context manager that turns the override on at entry and restores the prior state on exit.
- **Exposes an attention block-size constant**: `get_batch_invariant_attention_block_size()` returns `(16, 16)`, used by upstream attention kernels as the fixed split size.

**Key code**

```python
def enable_batch_invariant_mode():
    global _batch_invariant_MODE, _batch_invariant_LIB
    if _batch_invariant_MODE:
        return
    dispatch_key = getattr(torch.accelerator.current_accelerator(), 'type', 'cpu').upper()
    _batch_invariant_MODE = True
    _batch_invariant_LIB = torch.library.Library('aten', 'IMPL')
    _batch_invariant_LIB.impl('aten::mm',          mm_batch_invariant,           dispatch_key)
    _batch_invariant_LIB.impl('aten::addmm',       addmm_batch_invariant,        dispatch_key)
    _batch_invariant_LIB.impl('aten::_log_softmax', _log_softmax_batch_invariant, dispatch_key)
    _batch_invariant_LIB.impl('aten::mean.dim',    mean_batch_invariant,         dispatch_key)
```

**vs. other "deterministic switch" approaches**

| Approach | Effort | Caveats |
|---|---|---|
| Patch every kernel call site inside vLLM / SGLang | Heavy — every engine must be patched separately | Each engine upgrade requires re-patching |
| Monkey-patch `torch.mm = ...` | One line, but only covers the Python layer | C++ call paths and `torch.compile` bypass the patch |
| **`torch.library` re-IMPL (this approach)** | One-line registration | Takes effect at the dispatcher layer — covers Python + C++ + post-compile paths, and is isolated by dispatch key so CPU is untouched |

This is the most elegant part of the library: **op replacement happens at the dispatcher layer, fully transparent to upstream code**, so the same user code in the same vLLM process can switch between deterministic and high-performance modes by wrapping a block in `with set_batch_invariant_mode():`.
