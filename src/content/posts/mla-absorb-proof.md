---
title: 'MLA: the absorbed form equals the naive form'
description: 'DeepSeek-V2 (arXiv 2405.04434), Sec. 2.1 and Appendix C. Companion to mla_naive_vs_absorb.py.'
pubDate: 2026-10-21
tags: [attention, deepseek, proof]
---

## 0. Notation (one layer, one head $i$)

::::def
| Symbol | Shape | Meaning | DeepSeek-V2 |
|---|---|---|---|
| $\c_j$ | $\R^{d_c}$ | cached KV latent of token $j$ ($= \c^{KV}_j$) | $d_c = 512$ |
| $\k^R_j$ | $\R^{d_h^R}$ | cached RoPE key, **shared by all heads** | $d_h^R = 64$ |
| $\q^C_{t,i}$ | $\R^{d_h}$ | non-RoPE query of token $t$, head $i$ | $d_h = 128$ |
| $\q^R_{t,i}$ | $\R^{d_h^R}$ | RoPE query of token $t$, head $i$ | 64 |
| $\UK_i$ | $\R^{d_h \times d_c}$ | key up-projection, rows of head $i$ | $128 \times 512$ |
| $\UV_i$ | $\R^{d_h \times d_c}$ | value up-projection, rows of head $i$ | $128 \times 512$ |
| $W^O_i$ | $\R^{d \times d_h}$ | column block of $W^O$ that multiplies head $i$ | $5120 \times 128$ |
| $\tau$ | scalar | softmax scale $1/\sqrt{d_h + d_h^R}$ | $1/\sqrt{192}$ |

:::why
$\UK = [\UK_1; \dots; \UK_{n_h}]$ stacked by rows, likewise $\UV$. In code these are the two halves of
`wkv_b.weight.view(H, 256, 512)`.
:::
::::

## 1. The two forms

::::box[Naive (decompress, then MHA)]
$$
\k_{j,i} = \begin{bmatrix} \UK_i \c_j \\ \k^R_j \end{bmatrix} \in \R^{d_h + d_h^R},
\qquad
\v_{j,i} = \UV_i \c_j \in \R^{d_h},
\qquad
\q_{t,i} = \begin{bmatrix} \q^C_{t,i} \\ \q^R_{t,i} \end{bmatrix}
$$

$$
s^{\text{naive}}_{t,j,i} = \tau\, \q_{t,i}\T \k_{j,i},
\qquad
p_{t,j,i} = \operatorname{softmax}_j\!\big(s_{t,j,i} + m_{t,j}\big),
\qquad
\o^{\text{naive}}_{t,i} = \sum_{j} p_{t,j,i}\, \v_{j,i}
$$

:::why
$m_{t,j} \in \{0, -\infty\}$ is the causal mask.
:::
::::

:::box[Absorbed (stay in latent space)]
$$
\tilde{\q}_{t,i} = (\UK_i)\T \q^C_{t,i} \in \R^{d_c}
$$

$$
s^{\text{absorb}}_{t,j,i} = \tau \left( \tilde{\q}_{t,i}\T \c_j + (\q^R_{t,i})\T \k^R_j \right),
\qquad
\tilde{\o}_{t,i} = \sum_j p_{t,j,i}\, \c_j \in \R^{d_c},
\qquad
\o^{\text{absorb}}_{t,i} = \UV_i\, \tilde{\o}_{t,i}
$$
:::

:::ok
**Claim.** For every $t, i$: $\;\o^{\text{absorb}}_{t,i} = \o^{\text{naive}}_{t,i}$, and therefore $\u_t$ is identical.
:::

## 2. Step 1 — the attention scores are identical

Expand the naive score using the block structure of $\q_{t,i}$ and $\k_{j,i}$
(a dot product of concatenated vectors is the sum of the blockwise dot products):

$$
\begin{aligned}
s^{\text{naive}}_{t,j,i}
&= \tau \begin{bmatrix} \q^C_{t,i} \\ \q^R_{t,i} \end{bmatrix}\T
        \begin{bmatrix} \UK_i \c_j \\ \k^R_j \end{bmatrix} \\[4pt]
&= \tau \Big( (\q^C_{t,i})\T \,\UK_i \c_j \;+\; (\q^R_{t,i})\T \k^R_j \Big)
\end{aligned}
$$

Now regroup the first term. Matrix multiplication is associative, and $(A\mathbf{x})\T = \mathbf{x}\T A\T$:

$$
(\q^C_{t,i})\T \big(\UK_i \c_j\big)
\;=\; \big((\q^C_{t,i})\T \UK_i\big)\, \c_j
\;=\; \big((\UK_i)\T \q^C_{t,i}\big)\T \c_j
\;=\; \tilde{\q}_{t,i}\T \c_j
$$

:::why
The scalar is the same; only the order of multiplication changes.
Naive computes $\UK_i \c_j$ for *every key* $j$; absorbed computes $(\UK_i)\T\q^C_{t,i}$ once per *query*.
:::

Substituting back:

$$
s^{\text{naive}}_{t,j,i} = \tau \left( \tilde{\q}_{t,i}\T \c_j + (\q^R_{t,i})\T \k^R_j \right) = s^{\text{absorb}}_{t,j,i}
$$

Same scores and the same mask $\Rightarrow$ the same softmax probabilities $p_{t,j,i}$ in both forms. $\blacksquare_1$

:::warn
**The scale stays $\tau = 1/\sqrt{d_h + d_h^R} = 1/\sqrt{192}$**, even though the absorbed dot product
runs over $d_c + d_h^R = 576$ dimensions. The scale belongs to the *original* model; using
$1/\sqrt{576}$ would change the result.
:::

### Equivalent “MQA” view

Concatenate the two absorbed terms into one dot product:

$$
s^{\text{absorb}}_{t,j,i} = \tau
\underbrace{\begin{bmatrix} \tilde{\q}_{t,i} \\ \q^R_{t,i} \end{bmatrix}\T}_{\text{per head, } 576}
\underbrace{\begin{bmatrix} \c_j \\ \k^R_j \end{bmatrix}}_{\text{shared, } 576}
$$

The key vector $[\c_j; \k^R_j]$ has no head index $i$: every head attends to the same 576-dim key.
This is exactly multi-query attention with $n_h = 128$ query heads. It is also exactly what the KV cache stores.

## 3. Step 2 — the head outputs are identical

Start from the naive output and substitute $\v_{j,i} = \UV_i \c_j$:

$$
\o^{\text{naive}}_{t,i}
= \sum_j p_{t,j,i}\, \UV_i \c_j
$$

$\UV_i$ does not depend on $j$, and $p_{t,j,i}$ is a scalar, so by linearity the matrix factors out of the sum:

$$
\sum_j p_{t,j,i}\, \UV_i \c_j
\;=\; \UV_i \sum_j p_{t,j,i}\, \c_j
\;=\; \UV_i\, \tilde{\o}_{t,i}
\;=\; \o^{\text{absorb}}_{t,i}
$$

:::why
The $p_{t,j,i}$ are equal by Step 1. Naive up-projects every *key* token, then averages;
absorbed averages the latents, then up-projects once per *query*.
:::

$\blacksquare_2$ — hence $\u_t = W^O[\o_{t,1}; \dots; \o_{t,n_h}]$ is identical too. $\blacksquare$

## 4. Folding into neighbouring weights (valid, but not done in practice)

Since $\q^C_{t,i} = W^{UQ}_i \c^Q_t$ (eq. 13), the absorbed query is a fixed linear map of $\c^Q_t$:

$$
\tilde{\q}_{t,i} = (\UK_i)\T W^{UQ}_i\, \c^Q_t
= \underbrace{\big((\UK_i)\T W^{UQ}_i\big)}_{\text{precompute: } d_c \times d_c'}\, \c^Q_t
$$

On the output side, split $W^O$ into per-head column blocks:

$$
\u_t = \sum_{i=1}^{n_h} W^O_i\, \o_{t,i}
= \sum_{i=1}^{n_h} W^O_i \UV_i\, \tilde{\o}_{t,i}
= \sum_{i=1}^{n_h} \underbrace{\big(W^O_i \UV_i\big)}_{\text{precompute: } d \times d_c}\, \tilde{\o}_{t,i}
$$

This is the paper's sentence: “$\UK$ can be absorbed into $W^Q$, and $\UV$ into $W^O$.”
Whether to precompute these products is a cost trade-off. The math is identical either way.
`mla_naive_vs_absorb.py` applies $\UK_i$ and $\UV_i$ at runtime instead.

::::warn
**Note: real implementations do *not* precompute these products (checked 2026-10-20).**
Folding is valid math, but it costs more, because it removes the low-rank bottleneck. Per token, per layer, at DeepSeek-V2 sizes:

| | Runtime: shapes | Runtime: FLOPs per token | Folded: shapes | Folded: FLOPs per token |
|---|---|---|---|---|
| Query<br>$\c^Q_t \to \tilde{\q}_t$ | ① $C^Q\,[S, d_c']$<br>$\times\ W^{UQ}\,[d_c',\ n_h{\cdot}d_h]$<br>$\to Q^C\,[S, n_h, d_h]$<br>② $Q^C\,[n_h, S, d_h]$<br>$\times\ \UK\,[n_h, d_h, d_c]$<br>$\to \tilde{Q}\,[n_h, S, d_c]$ | ① $2\,d_c'\,n_h d_h$ = 50.3M<br>② $2\,n_h d_h d_c$ = 16.8M<br>= **67.1M** | $C^Q\,[S, d_c']$<br>$\times\ W^{UQK}\,[d_c',\ n_h{\cdot}d_c]$<br>$\to \tilde{Q}\,[S, n_h, d_c]$<br><span class="why">$W^{UQK}_i = (\UK_i)\T W^{UQ}_i$</span> | $2\,d_c'\,n_h d_c$<br>= **201M** (3×) |
| Output<br>$\tilde{\o}_t \to \u_t$ | ① $\tilde{O}\,[n_h, S, d_c]$<br>$\times\ (\UV)\T\,[n_h, d_c, d_h]$<br>$\to O\,[n_h, S, d_h]$<br>② $O\,[S,\ n_h{\cdot}d_h]$<br>$\times\ (W^O)\T\,[n_h{\cdot}d_h, d]$<br>$\to U\,[S, d]$ | ① $2\,n_h d_c d_h$ = 16.8M<br>② $2\,n_h d_h\,d$ = 167.8M<br>= **184.5M** | $\tilde{O}\,[S,\ n_h{\cdot}d_c]$<br>$\times\ (W^{OV})\T\,[n_h{\cdot}d_c, d]$<br>$\to U\,[S, d]$<br><span class="why">$W^{OV}_i = W^O_i \UV_i$</span> | $2\,n_h d_c\,d$<br>= **671M** (3.6×) |

| Weight parameters | Applied at runtime | Precomputed (folded) |
|---|---|---|
| Query side | $W^{UQ}$: $d_c' \cdot n_h d_h$ = 25.2M | $W^{UQK}$: $d_c' \cdot n_h d_c$ = 100.7M |
| Output side | $W^O$: $n_h d_h \cdot d$ = 83.9M | $W^{OV}$: $n_h d_c \cdot d$ = 335.5M |
| Note | <span class="why">plus $\UK, \UV$: $d_c \cdot n_h 2d_h$ = 16.8M, already in `wkv_b`</span> | <span class="why">≈4×, ≈+20B parameters over 60 layers</span> |

:::why
$d = 5120$ (hidden size), $d_c' = 1536$ (query latent), $d_c = 512$, $n_h = d_h = 128$. A shape without $n_h$ first is a plain 2-D matmul over tokens.
:::

What current code does:

- **DeepSeek-V3 reference** (`inference/model.py`): a global `attn_impl = "naive" | "absorb"` switch, defaulting to `"absorb"`.
  Absorb mode runs `einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :128])` before attention and
  `einsum("bshc,hdc->bshd", x, wkv_b[:, -128:])` after. That's the same as section 8, with no folded weights.
- **vLLM** (`model_executor/layers/attention/mla_attention.py`): runs a "compute friendly" `forward_mha` for prefill
  and a "data-movement friendly" `forward_mqa` for decode, picked by the scheduler's prefill/decode label.
  At load time it only stores the transpose `W_UK_T`. At runtime it does `torch.bmm(q_nope, W_UK_T)` and an einsum with `W_UV`.
  Chunked prefill decompresses the cached context in fixed-size chunks so the decompressed K/V fit in a bounded workspace.
- **SGLang** (`models/deepseek_common/attention_forward_methods/`): an `AttnForwardMethod` dispatch between MHA and absorbed MLA.
  The absorbed path does `torch.bmm` with per-head `w_kc` and `w_vc` at runtime.
- **FlashMLA** (DeepSeek's decode kernel) takes the already-absorbed query ($d_{qk}=576$) and the 576-dim latent cache, and computes only the attention core.
  Weight folding isn't part of the kernel.

**Correction to the "decode is memory-bound" intuition.** FlashMLA's deep-dive (2025-04-22) puts the absorbed kernel's intensity at
$\approx 2\,h_q s_q$ FLOPs/byte. On an H800 (≈865 TFLOPs after throttling, 3.35 TB/s) it becomes compute-bound once $h_q s_q \ge 128$.
DeepSeek runs decode without tensor parallelism, so $h_q = 128$ and **absorbed decode is compute-bound** even at $s_q = 1$.
So the absorbed form's ≈3.4× higher attention FLOPs (section 6) still matter in decode. They are still far below the naive form's per-step decompression cost.

**Newer DeepSeek models build the folded shape into the architecture itself.** The FlashMLA README for DeepSeek-V4.1 (2026-09) describes
Q_b projecting $q\_lora\_rank = 1536 \to$ 64 heads × 512 dims directly. The output path is a grouped low-rank projection
(`o_lora_rank` = 512) followed by $W^O$. In other words, the model is trained in the latent-space form, with half the heads,
instead of being trained as MHA and absorbed afterwards. I've only seen this in that README, not in a paper.
::::

## 5. Why RoPE has to be decoupled

Suppose RoPE were applied directly to the compressed key and query, with rotation matrices $R_t$ for position $t$:

$$
(R_t \q^C_{t,i})\T (R_j \UK_i \c_j)
= (\q^C_{t,i})\T \, \underbrace{R_t\T R_j}_{=\,R_{j-t}} \, \UK_i \c_j
= \Big( (\UK_i)\T R_{j-t}\T \q^C_{t,i} \Big)\T \c_j
$$

The “absorbed query” now depends on $j - t$, so it is different for every key and cannot be computed once per query.
Absorption only works if no position-dependent matrix sits between $\q^C$ and $\UK$.

The decoupled design keeps RoPE on a separate channel ($\q^R$, $\k^R$) that never touches $\UK$:

$$
s_{t,j,i} = \tau\Big(
\underbrace{\tilde{\q}_{t,i}\T \c_j}_{\text{no RoPE: absorbable}}
\;+\;
\underbrace{(\q^R_{t,i})\T \k^R_j}_{\text{RoPE: already small, cached as-is}}
\Big)
$$

## 6. What changes between the two forms: cost, not math

Costs are for one layer, attention core only. $S$ = query tokens in the call, $T$ = cached (context) tokens attended to.
FLOPs = 2 × multiply-adds. The projections that produce $\c^Q$, $\q^C$, $\q^R$, $\c_j$, $\k^R_j$ and the final $W^O$
are identical in both forms and are left out. Shapes are written [batch, rows, cols], with the head dimension $n_h$ first when there is one; a shape without $n_h$ is shared by all heads. Numbers use $n_h=128,\ d_h=128,\ d_h^R=64,\ d_c=512$.

| Computation | Naive: shapes | Naive: FLOPs | Absorbed: shapes | Absorbed: FLOPs |
|---|---|---|---|---|
| Decompress K/V<br>$\UK_i\c_j,\ \UV_i\c_j$ | $C\,[T, d_c]$<br>$\times\ W^{UKV}\,[d_c,\ n_h{\cdot}2d_h]$<br>$\to [T, n_h, 2d_h]$<br><span class="why">split into $K^C, V$: $[T, n_h, d_h]$ each</span> | $2\,T\,d_c\,n_h(2d_h)$<br>= **33.6M · T** | — | — |
| Absorb into query<br>$(\UK_i)\T\q^C_{t,i}$ | — | — | $Q^C\,[n_h, S, d_h]$<br>$\times\ \UK\,[n_h, d_h, d_c]$<br>$\to \tilde{Q}\,[n_h, S, d_c]$<br><span class="why">batched over heads</span> | $2\,S\,n_h\,d_h\,d_c$<br>= **16.8M · S** |
| Scores $QK\T$ | $Q\,[n_h, S, d_h{+}d_h^R]$<br>$\times\ K\T\,[n_h, d_h{+}d_h^R, T]$<br>$\to [n_h, S, T]$<br><span class="why">a separate K per head</span> | $2\,S\,T\,n_h(d_h{+}d_h^R)$<br>= **49,152 · ST** | $[\tilde{Q}; Q^R]\,[n_h, S, d_c{+}d_h^R]$<br>$\times\ [C; K^R]\T\,[d_c{+}d_h^R, T]$<br>$\to [n_h, S, T]$<br><span class="why">one K shared by all heads (MQA)</span> | $2\,S\,T\,n_h(d_c{+}d_h^R)$<br>= **147,456 · ST** |
| Weighted sum $PV$ | $P\,[n_h, S, T]$<br>$\times\ V\,[n_h, T, d_h]$<br>$\to O\,[n_h, S, d_h]$ | $2\,S\,T\,n_h\,d_h$<br>= **32,768 · ST** | $P\,[n_h, S, T]$<br>$\times\ C\,[T, d_c]$<br>$\to \tilde{O}\,[n_h, S, d_c]$<br><span class="why">the latent itself is V, shared by all heads</span> | $2\,S\,T\,n_h\,d_c$<br>= **131,072 · ST** |
| Up-project output<br>$\UV_i\tilde{\o}_{t,i}$ | — | — | $\tilde{O}\,[n_h, S, d_c]$<br>$\times\ (\UV)\T\,[n_h, d_c, d_h]$<br>$\to O\,[n_h, S, d_h]$<br><span class="why">batched over heads</span> | $2\,S\,n_h\,d_c\,d_h$<br>= **16.8M · S** |

| | Naive | Absorbed |
|---|---|---|
| **Total** | **33.6M · T + 81,920 · ST** | **33.6M · S + 278,528 · ST** |
| **Decode**: $S=1,\ T=4096$ | 137.4G + 0.3G ≈ **137.8 GFLOPs** | 0.03G + 1.14G ≈ **1.17 GFLOPs** (≈118× less) |
| **Prefill**: $S=T=4096$ | 0.14T + 1.37T ≈ **1.51 TFLOPs** (≈3.2× less) | 0.14T + 4.67T ≈ **4.81 TFLOPs** |

:::why
The $ST$ terms ignore causal masking. A kernel that skips masked blocks roughly halves them in prefill. That doesn't change the ratio.
:::

How to read the totals: naive pays a large cost per *key* token (decompression, 33.6M · T) plus cheap attention.
Absorbed pays the same-sized cost per *query* token (33.6M · S) plus attention that is about 3.4× more expensive.
When $S \ll T$ (decode), absorbed wins by a wide margin. When $S \approx T$ (prefill), the per-token terms cancel and
the cheaper attention makes naive win.

| Memory, per cached token | Naive | Absorbed |
|---|---|---|
| Read from the KV cache | 576 values | 576 values |
| Intermediate K/V the attention kernel needs | 32,768 values (written to HBM and re-read, unless the decompression is fused into the kernel) | none (it attends to the latent directly) |

:::def
**Where the naive cost comes from.** Both forms read the same 576-value latent from the cache. The difference is what the
attention step needs as input. Naive needs per-head $\UK_i\c_j$ and $\UV_i\c_j$ for every cached token, so it has to produce
32,768 values from each 576-value latent. That work depends only on $j$, so it is spread over every query in the call.
In prefill there are thousands of queries, so it's cheap per query. In decode there is a single query, so the whole decompression
of the history is paid again on every step. Absorbed moves that work to the query side: $(\UK_i)\T\q^C$ and $\UV_i\tilde{\o}$
run once per *query*, regardless of context length.
:::

:::warn
**Numerics.** The equality holds in exact arithmetic (the script measures $\sim 10^{-15}$ in float64).
In bf16/fp8 the two forms round differently, because the products are grouped differently, so outputs match only to within normal low-precision error.
:::

## 7. Estimating FLOPs: naive prefill + absorbed decode

Per layer, per sequence, matmuls only. Softmax, norms, RoPE and the MoE/FFN are left out. FLOPs = 2 × multiply-adds.
Multiply by $l$ for the whole attention stack.

| Symbol | Meaning | DeepSeek-V2 |
|---|---|---|
| $d$ | hidden size | 5120 |
| $d_c'$ | query latent dimension | 1536 |
| $d_c$ | KV latent dimension | 512 |
| $n_h$ | attention heads | 128 |
| $d_h,\ d_h^R,\ d_v$ | non-RoPE q/k head dim, RoPE head dim, value head dim | 128, 64, 128 |
| $l$ | layers | 60 |
| $S$ | new (query) tokens in the call | |
| $T_0$ | tokens already in the KV cache before the call | |
| $T$ | context attended to in a decode step, including the new token | |

### 7.1 Projections (identical in both forms, per new token)

$$
P = 2\Big[\ \underbrace{d\,d_c'}_{W^{DQ}}
\;+\; \underbrace{d_c'\,n_h(d_h + d_h^R)}_{W^{UQ},\ W^{QR}}
\;+\; \underbrace{d\,(d_c + d_h^R)}_{W^{DKV},\ W^{KR}}
\;+\; \underbrace{n_h\,d_v\,d}_{W^O}\ \Big]
$$

### 7.2 Prefill with the naive form

$S$ new tokens attend to $T_0 + S$ keys under a causal mask, so the number of unmasked (query, key) pairs is $S\,T_0 + \tfrac{S(S+1)}{2}$.

$$
F_{\text{prefill}} = S\,P
\;+\; \underbrace{2\,(T_0 + S)\,d_c\,n_h(d_h + d_v)}_{\text{decompress K/V for every key}}
\;+\; \underbrace{2\,n_h\,(d_h + d_h^R + d_v)}_{\text{QK}^\top + PV \text{ per pair}}\Big[S\,T_0 + \tfrac{S(S+1)}{2}\Big]
$$

With no cached prefix ($T_0 = 0$):

$$
F_{\text{prefill}} \approx S\Big[P + 2\,d_c\,n_h(d_h + d_v)\Big] \;+\; n_h\,(d_h + d_h^R + d_v)\,S^2
$$

### 7.3 Decode with the absorbed form

One new token attends to $T$ tokens:

$$
F_{\text{decode}}(T) = P
\;+\; \underbrace{2\,n_h\,d_h\,d_c + 2\,n_h\,d_c\,d_v}_{\text{absorb } \UK \text{ into } q,\ \text{apply } \UV \text{ to output}}
\;+\; \underbrace{2\,n_h\,(2d_c + d_h^R)}_{\text{QK}^\top\,(d_c + d_h^R) \;+\; PV\,(d_c)}\;T
$$

Generating $N$ tokens after an $S$-token prompt sums $F_{\text{decode}}(T)$ over $T = S{+}1, \dots, S{+}N$:

$$
F_{\text{gen}} = N\Big[P + 2\,n_h\,d_c(d_h + d_v)\Big] \;+\; 2\,n_h\,(2d_c + d_h^R)\Big[N S + \tfrac{N(N+1)}{2}\Big]
$$

:::why
With speculative decoding or MTP ($s_q$ query tokens per step), multiply the per-token terms by $s_q$ and the $T$ term by $s_q$.
:::

### 7.4 DeepSeek-V2 numbers

| Term | Formula | Value |
|---|---|---|
| Projections per token | $P$ | 264.9M |
| Naive: decompression per key token | $2\,d_c\,n_h(d_h+d_v)$ | 33.6M |
| Naive: attention per (q, k) pair | $2\,n_h(d_h+d_h^R+d_v)$ | 81,920 |
| Absorbed: query + output overhead per query token | $2\,n_h\,d_c(d_h+d_v)$ | 33.6M |
| Absorbed: attention per (q, k) pair | $2\,n_h(2d_c+d_h^R)$ | 278,528 |

| Case | Per layer | Attention stack (× 60) |
|---|---|---|
| Prefill, $S = 4096,\ T_0 = 0$ | 1.08T (projections) + 0.14T (decompress) + 0.69T (attention) ≈ **1.91 TFLOPs** | ≈ **115 TFLOPs** |
| Decode one token, $T = 4096$ | 0.26G (projections) + 0.03G (overhead) + 1.14G (attention) ≈ **1.44 GFLOPs** | ≈ **86 GFLOPs / token** |

- **Prefill:** the projections dominate until $S$ is around 3–4K, then the $S^2$ attention term takes over.
- **Decode:** the $278{,}528 \cdot T$ attention term dominates once $T$ is past about 1K.
- **Full model:** add the MoE/FFN cost per token, about 2 × activated FFN parameters.

:::def
**How this relates to the section 6 table.** With $P$ removed, the decode formula with $S = 1$ is exactly the table's absorbed column sum:
$33.6\text{M} + 278{,}528\,T$. The prefill formula differs from the table's naive column in two ways. Here the keys are $T_0 + S$, and causal masking
counts $S(S+1)/2$ pairs instead of $S \cdot T$. That roughly halves the attention term: 0.69T here vs 1.37T in the table for $S = T = 4096$.
:::

## 8. Mapping to the code

| Math | `mla_naive_vs_absorb.py` |
|---|---|
| $\UK_i$, $\UV_i$ | `w = wkv_b.weight.view(H, 256, 512)`; `w_uk = w[:, :128]`, `w_uv = w[:, 128:]` |
| $\UK_i\c_j$, $\UV_i\c_j$ (naive) | `kv = self.wkv_b(c_kv).view(B, T, H, 256)` |
| $\tilde{\q}_{t,i} = (\UK_i)\T\q^C_{t,i}$ | `einsum("bshd,hdc->bshc", q_nope, w_uk)` |
| $\tilde{\q}\T\c_j + (\q^R)\T\k^R_j$ | `einsum("bshc,btc->bsht", q_lat, c_kv) + einsum("bshr,btr->bsht", q_pe, k_pe)` |
| $\tilde{\o}_{t,i} = \sum_j p\,\c_j$ | `einsum("bsht,btc->bshc", p, c_kv)` |
| $\UV_i\tilde{\o}_{t,i}$ | `einsum("bshc,hdc->bshd", o_lat, w_uv)` |
