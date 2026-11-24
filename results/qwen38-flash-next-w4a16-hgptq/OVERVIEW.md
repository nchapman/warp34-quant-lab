# Quantizing `Qwen3.8-Flash-Next`: End-to-End Engineering Overview

How we compressed `Qwen/Qwen3.8-Flash-Next` — a 180B-class, 512-expert multimodal MoE model with a `51.2B`-parameter Layer-1 N-gram table (`335.3 GiB` / `360 GB` in BF16) — down to **`92.84 GiB` (`W4A16-HGPTQ`)** and **`71.67 GiB` total / `44.84 GiB` GPU trunk (`W2.6A16-HGPTQ-RCO` V2)** so it runs on consumer and workstation GPUs (`1×`/`2×` RTX 3090, `48 GB`, or `80–96 GB` GPUs) with near-BF16 fidelity.

## Summary

- **Two checkpoints covering every hardware tier:**
  - [`Qwen3.8-Flash-Next-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ) (**`92.84 GiB`**, `3.61×` smaller): pure 4-bit across all `75,682` linear layers + 4-bit PLE + 8-bit `lm_head`. Hits **`0.0421` `probe-v4` weighted KL** (`-1.11%` PPL), **`0.0673` holdout KL**, `95.41%` MTP top-1, and `97.35%` Vision top-1 (`99.26%` UI/OCR guard).
  - [`Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO) V2 (**`71.67 GiB` total / `44.84 GiB` GPU trunk**, `4.68×` smaller): fits the entire 48-layer MoE trunk + Native MTP + Vision on a **single 48 GB GPU** (with the `26.82 GiB` PLE table `mmap`'d in system RAM), all-in on an **80 GB GPU**, or tiered across **2× RTX 3090 (24 GB)** at **`95–132 tok/s`** with `128K` context + MTP.
- **Why off-the-shelf quantizers fall short on `Flash-Next`:** It combines five unusual components in one model — `48 × 512` routed experts (`73,728` matrices), `4`-stream hyper-connections, a `51.2B`-parameter Layer-1 N-gram hash table (`95.37 GiB` BF16 alone), a `512`-expert Native MTP head, and a 27-block Vision encoder. Compressing only the text trunk leaves `>100 GiB` of unquantized bloat behind.
- **Five engineering ideas that made it work together:**
  1. **Rotating and quantizing *both* routed and shared experts** across all 48 layers + MTP (`-6.7×` less perplexity degradation than skipping shared experts/rotation).
  2. **Calibration-free 4-bit `Custom-LUT4` `W_kv`-HGPTQ + 4 KiB page-slotting** for the `51.2B` Layer-1 PLE table (`95.37 GiB → 26.82 GiB`, `29.6%` lower downstream error than `NVFP4`, `0%` split OS pages).
  3. **End-to-end FP32-master KL QAT** over the 4-stream hyper-connection injectors, norms, GDN recurrence gates, and PLE projection norms.
  4. **V2 symmetric mid-riser 2-bit (`w2_g128_sym`) + Leave-One-Out KL layer selection** (`15` sensitive MoE layers in `W4`, `33` in `W2`), cutting holdout KL by **`37.7%`** vs. V1 at the same footprint.
  5. **Pre-profiled static hot-expert rankings (`9.8M` assignments/layer)** + a **`0.52 GiB` 4-bit `DFlash2` drafter** (`99.7%` of BF16 acceptance length, `3.76×` less draft memory bandwidth).

---

## 1. Checkpoint Comparison at a Glance

| Checkpoint | Total Size | GPU Trunk + MTP + Vision (excl. `mmap` PLE) | `probe-v4` PPL (`3.6085` base) | `probe-v4` `w_kl` ↓ | `probe-guard` `mean_kl` ↓ | MTP Iso / E2E Top-1 | Vision Iso / E2E Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`W4A16-HGPTQ`** ([report](TECHNICAL.md)) | `92.84 GiB` | `66.02 GiB` | `3.5683` (`-1.11%`) | **`0.0421`** | **`0.0673`** | **`95.41%`** / **`88.39%`** | **`97.35%`** / **`93.71%`** |
| **`W2.6A16-HGPTQ-RCO` V2** ([report](../qwen38-flash-next-w26a16-hgptq-rco/TECHNICAL.md)) | **`71.67 GiB`** | **`44.84 GiB`** | `3.3704` (`-6.60%`) | `0.1091` | `0.1961` | `95.13%` / `82.90%` | `96.36%` / `92.72%` |
| *`W2.6A16-HGPTQ-RCO` V1 (superseded)* | *`71.75 GiB`* | *`44.93 GiB`* | *`3.3931` (`-5.97%`)* | *`0.1396`* | *`0.3150`* | *`95.41%` / `81.55%`* | *`97.02%` / `90.73%`* |

Switching `W2.6` from V1 (`g64-h64`, asymmetric `uint2b2`, `12/48` W4 layers) to V2 (`g128-h128`, symmetric mid-riser `code_offset=0.5`, `15/48` W4 layers) cut `probe-v4` weighted KL by **`21.9%`**, `probe-guard` mean KL by **`37.7%`** (`weighted_kl` by **`53.3%`**), and end-to-end Vision KL by **`38.0%`** while shrinking the checkpoint by another `80 MiB`.

---

## 2. Streaming Layer-by-Layer Quantization (`75,682` Linear Layers)

Because `Qwen3.8-Flash-Next` is `335.3 GiB` in BF16, loading the whole model onto GPU at once isn't feasible even on high-memory boxes. Instead, `src/warp_quant/streaming_moe.py` streams one transformer layer at a time through GPU memory:

- **Rotate and quantize *both* routed and shared experts:** Every MoE layer has `512` routed experts (`73,728` linears across 48 layers) plus a shared expert (`144` linears). Applying block Walsh–Hadamard input rotation (`H_64` on W4, `H_128` on W2.6 V2) + MSE-initialized Cholesky GPTQ + 2-pass Gauss–Seidel Hessian Coordinate Descent (`H-CD`) to **both** routed and shared experts reduced perplexity degradation by **`6.7×`** compared to leaving shared experts or expert rotations out.
- **Full multimodal + speculative coverage:**
  - **Native MTP (`1,543` linears, `1.29–1.34 GiB`):** All `512` MTP experts, shared expert, and QSA attention projections are calibrated on the target model's own post-quantization hidden states.
  - **Vision tower (`110` linears, `0.22 GiB`):** Calibrated on disjoint `data/vision-calib-v2` images (`visual.blocks.*.mlp.linear_fc2` has $K=4304 = 16 \times 269$, so it gets per-channel GPTQ + `H_16` rotation while all other ViT layers use standard group GPTQ).
  - **Untied `lm_head` (`248,320 × 2,560`):** Unrotated 8-bit `W8_g64` GPTQ+MSE (`~0.64 GiB` instead of `1.27 GiB` BF16) with zero runtime Hadamard overhead.
  - **Small sensitive scalars stay BF16/FP32:** Router gates, GDN per-head scalars (`in_proj_a/b`, `A_log`, `dt_bias`), 4-stream hyper-connection mixers/injectors, `self_attn.indexer`, and norms stay in full precision.

---

## 3. Cracking the `51.2B`-Parameter Layer-1 PLE Table (`95.37 GiB → 26.82 GiB`)

Layer 1 contains a `51.2B`-parameter N-gram hash table (`320,000,456` rows × `160` dims across `128` shards) that makes up almost a third of the BF16 model (`95.37 GiB`). Full design and ablations live in [`PLE-QUANTIZATION-DESIGN.md`](PLE-QUANTIZATION-DESIGN.md); the short version:

1. **Why 4-bit was non-negotiable:** At 8-bit (`~48 GiB`), PLE + the quantized backbone (`~114 GiB`) overflows a `96 GB` GPU and thrashes host RAM page cache when cold MoE experts live in system memory. At 4-bit (`26.82 GiB`), the entire `W4` model fits in `92.84 GiB` (and `W2.6` in `71.67 GiB`).
2. **Why normal calibration fails:** Every hash bucket is active (`0%` dead rows), singular values are flat (so SVD breaks), and any calibration corpus touches `<0.1%` of the 320M rows.
3. **Exact calibration-free `W_kv`-HGPTQ + `Custom-LUT4`:** Because the PLE table feeds *only* `layers.1.ple.{key_proj,value_proj}`, we build an exact analytical Hessian $H_h$ straight from $(W_{k,h}, W_{v,h})$ — calibrating all 320 million rows with zero data dependency — paired with a 16-entry non-uniform lookup table (`Custom-LUT4`, `g32`), 2-pass H-CD, and closed-form population $W_{kv}$ MMSE bias/cross-covariance compensation folded into `key_proj`/`value_proj`. That cuts `key_proj(E)` error by **`29.6%` vs. `NVFP4`** (`6.655%` vs. `9.449%` Rel-RMSE) and **`22.3%` vs. `INT4 g16`**.
4. **Zero split page faults (`4 KiB` page-slotted layout):** Each row takes `90 B` (`80 B` packed nibbles + `10 B` FP16 scales), so `45` complete rows fit inside a single `4,096 B` OS memory page (`98.9%` density, `0.000%` page splits vs. `29.9%` with unaligned rows). Because PLE depends only on `input_ids`, `warp34` overlaps the `16` random lookups behind Layer 0 compute.

---

## 4. Sub-3-Bit V2 (`W2.6A16-HGPTQ-RCO`) & 4-Stream Hyper-Connection QAT

To shrink the `66.02 GiB` GPU trunk down to **`44.84 GiB`** (a single `48 GB` GPU) without losing coherence:

- **Always-active backbone stays 4-bit:** 100% of attention/GDN (`156` linears), shared experts (`144` linears), Native MTP (`1,543` linears), and Vision (`110` linears) stay in `W4_g128` (`H_128`).
- **Leave-One-Out KL layer selection for routed experts:** Instead of guessing which MoE layers matter most, we measure the exact end-to-end KL jump of dropping each layer's `512` routed experts from `W4` to `W2`, then allocate the **15 most sensitive layers** (`[9, 17, 22, 23, 28, 29, 30, 31, 34, 35, 37, 38, 39, 46, 47]`) to `W4_g128` and the other **33 layers** to symmetric mid-riser `W2_g128` (`code_offset=0.5`, codes `{-1.5, -0.5, +0.5, +1.5}`). Moving from `g64` to `g128` freed enough scale-tensor bytes to upgrade **3 extra MoE layers (`12 → 15`)** from 2-bit to 4-bit at the *same* `44.84 GiB` trunk size!
- **4-stream Hyper-Connection + Norm + GDN + PLE KL QAT:** Both `W4` and `W2.6` finish with FP32-master teacher-forced KL tuning over `Flash-Next`'s 4-stream `hyper_connection` norms/injectors, QK norms, GDN gates, `shared_expert_gate`, and Layer-1 PLE norms (`ple.{norm_key,norm_query,norm_conv,conv1d}`) — realigning all 4 residual streams after quantization.

---

## 5. Memory Tiering, Hot-Expert Rankings & Real Consumer Throughput

Serving a 512-expert MoE model on `24 GB` consumer GPUs requires putting the right weights in the right memory tier:

- **3-tier memory layout (`warp34`):**
  1. **GPU VRAM (always resident):** Embed/LM head (`1.8 GiB`), Attention/Shared/Norms (`~3.8 GiB`), Native MTP (`1.3 GiB`), Vision (`0.2 GiB`), and the hottest `k_hot` routed experts per layer.
  2. **Pinned host DDR (`HostExpertStore` / UVA):** Cold routed experts streamed on cache miss.
  3. **Linux page cache (`mmap`):** The `26.82 GiB` 4-bit PLE table.
- **Zero-warmup static hot-expert rankings (`static_hot_cache_rankings.json`):** During quantization and probing, we record `9,804,460` router assignments per layer and sort experts `0..511` hottest-first (plus `"mtp.0"`). Routing skew is strong — pinning the top `128/512` (`25%`) experts catches `41%–66%` of hits, and `256/512` (`50%`) catches `70%–88%` — so `warp34` pre-loads the hottest experts before the first token arrives.
- **Measured `128K`-context (`--kv-pages 8192`) throughput on RTX 3090s** (full breakdown in [`warp34-consumer-gpu-128k/README.md`](../warp34-consumer-gpu-128k/README.md)):
  - **`1× RTX 3090 24 GB`** (`k_hot=96/512` in VRAM): **`26.1 tok/s`** decode (`278.8 tok/s` prefill) with `4.6 GiB` free VRAM.
  - **`2× RTX 3090 24 GB`** (`TP=2`, default `k_hot=160/256`, LoRA + Desktop headroom): **`95.0 tok/s`** with MTP (`502.3 tok/s` prefill).
  - **`2× RTX 3090 24 GB`** (`TP=2`, headless `k_hot=208/256`, `81.25%` experts in VRAM): **`131.6 tok/s`** with MTP (`787.5 tok/s` prefill, `7.60 ms` ITL).

---

## 6. Companion `0.52 GiB` 4-Bit `DFlash2` Block-Diffusion Drafter

For even faster multi-token drafting than Native MTP, we also quantized [`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) to 4-bit [`nchapman/Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ) ([full report](../dense-w4a16-hgptq/dflash2-flash-next-TECHNICAL.md)):

- **Size:** `1.12 GiB → 0.52 GiB` (`531 MiB`, `2.17×` smaller), cutting decoder weights read per 8-token draft step by **`3.76×`** (`844 MiB → 224 MiB`).
- **Fidelity:** Preserves **`99.7%`** of BF16 acceptance length ($\tau$: `2.760 → 2.752` tokens/step) and **`99.9%`** of top-16 candidate recall (`59.67% → 59.60%`) across 14,336 drafted tokens.

---

## Related Docs & Receipts

- **W4A16-HGPTQ (`92.84 GiB`):** [Model Card](README.md) · [Technical Report](TECHNICAL.md) · [`probe_results.json`](probe_results.json)
- **W2.6A16-HGPTQ-RCO V2 (`71.67 GiB` / `44.84 GiB` trunk):** [Model Card](../qwen38-flash-next-w26a16-hgptq-rco/README.md) · [Technical Report](../qwen38-flash-next-w26a16-hgptq-rco/TECHNICAL.md) · [`probe_flash_next_w26_v2.json`](../qwen38-flash-next-w26a16-hgptq-rco/v2_receipts/probe_flash_next_w26_v2.json)
- **Layer-1 PLE Deep Dive:** [`PLE-QUANTIZATION-DESIGN.md`](PLE-QUANTIZATION-DESIGN.md)
- **DFlash2 Drafter (`0.52 GiB`):** [Model Card](../dense-w4a16-hgptq/dflash2-flash-next-README.md) · [Technical Report](../dense-w4a16-hgptq/dflash2-flash-next-TECHNICAL.md)
- **Consumer GPU `128K` Throughput & VRAM Guide:** [`warp34-consumer-gpu-128k/README.md`](../warp34-consumer-gpu-128k/README.md)
