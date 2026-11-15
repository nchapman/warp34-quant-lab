# Consumer GPU `128K`-Context (`--kv-pages 8192`) Fit, Throughput & Safety Buffer Guide (`warp-34`)

Empirical `w34` (`ghcr.io/nchapman/warp-34:dev`) capacity planning, throughput benchmarks, and VRAM safety buffer analysis for **`8 GB`**, **`12 GB` (`RTX 3060`)**, **`16 GB` (`RTX 4060 Ti`)**, and **`1× / 2× 24 GB` (`RTX 3090`, `TP=1` & `TP=2`)** consumer GPUs at **`128K` context (`131,072` tokens = `--kv-pages 8192`)** with TurboQuant 4-bit KV cache (`--kv-format tq4` and `tq4_all`), Native `MTP` speculative decoding (`--speculative mtp --mtp-k 3`), Multi-LoRA adapters (`--lora`), and Desktop OS environments (Windows 11 DWM / WSL2 and Linux Wayland / X11).

Machine-readable receipts: [`consumer_gpu_128k_matrix.json`](consumer_gpu_128k_matrix.json).

---

## 1. Executive Summary: Why `W2.6A16-HGPTQ-RCO` (`30% W4`) Is the Exact Consumer Sweet Spot

Across `Qwen3.5-9B`, `Qwen3.8-27B`, and `Qwen3.6-35B-A3B`, the `30% W4` (`~2.60 bpw`) Riemannian `HGPTQ-RCO` tier sits at **two simultaneous optima**:
1. **Quality Pareto Elbow:** Allocating the top `25%–30%` highest-sensitivity linear groups (all Full-Attention `qkv`/`o_proj`, boundary/late GDN projections, and top-sensitivity FFN/routed-expert blocks) to `W4_g64` while keeping the remaining `70%–75%` in `W2_g64` + E2E Scale/Norm/GDN KL QAT captures the steep elbow of the KL/PPL Pareto curve.
2. **Full-Stack Consumer VRAM Envelope (`128K` `tq4` + `LoRA` + Desktop OS + `w34` Headroom):**
   When you account for all four real-world consumer VRAM consumers—
   - **Default `128K` `tq4` KV cache** (`--kv-pages 8192`, keeping the final 4 full-attention retrieval layers in unrotated 8-bit `FP8 E4M3` / `Q8_0` and preceding layers in WHT-256 `TQ4`),
   - **Attached LoRA adapters (`~660–750 MiB`)** (`lora_arena` + `w34`'s automatic `512 MiB → 1,056 MiB` `scratch_arena` expansion),
   - **Desktop OS & Browser VRAM (`~600–900 MiB`)** (Windows 11 DWM / Linux Wayland compositor + multi-monitor + hardware-accelerated browser), and
   - **`w34`'s internal `1,024 MiB` post-pool floor (`MIN_HEADROOM_BYTES`)**,
   —all three existing `W2.6A16-HGPTQ-RCO` checkpoints (`9B` on `8 GB`, `27B` on `16 GB`, `35B-A3B` on `16 GB`) and `9B-W4A16-HGPTQ` (`12 GB`) land cleanly inside their target brackets without requiring `--kv-format tq4_all` or closing desktop apps.

---

## 2. Measured `w34` Throughput on Consumer GPUs (`RTX 3060 12 GB` & `RTX 4060 Ti 16 GB`)

All benchmarks run with `w34 bench --tp 1 --kv-format tq4 --prompt-len 512 --gen-len 128` (`CUDA Graph` enabled):

| GPU (`SM`, Bus, BW) | Checkpoint (`nchapman/*`) | Mode | Prefill (`tok/s`) | Decode (`tok/s`) | Mean Tokens / Verify Round | Inter-Token Latency (`ITL`) |
|---|---|---|---:|---:|---:|---:|
| **`RTX 3060 12 GB`** (`sm_86`, `192-bit`, `360 GB/s`) | `Qwen3.5-9B-W2.6A16-HGPTQ-RCO` | Standard (`tq4`) | `1,726.7` | `69.2` | — | `14.57 ms` |
| **`RTX 3060 12 GB`** (`sm_86`, `192-bit`, `360 GB/s`) | `Qwen3.5-9B-W2.6A16-HGPTQ-RCO` | **`--speculative mtp --mtp-k 3`** | **`1,671.4`** | **`151.5`** | **`3.97`** | **`6.60 ms`** |
| **`RTX 3060 12 GB`** (`sm_86`, `192-bit`, `360 GB/s`) | `Qwen3.5-9B-W4A16-HGPTQ` | Standard (`tq4`) | `1,692.3` | `60.9` | — | `16.56 ms` |
| **`RTX 3060 12 GB`** (`sm_86`, `192-bit`, `360 GB/s`) | `Qwen3.5-9B-W4A16-HGPTQ` | **`--speculative mtp --mtp-k 3`** | **`1,646.7`** | **`140.5`** | **`3.97`** | **`7.12 ms`** |
| **`RTX 3060 12 GB`** (`sm_86`, `192-bit`, `360 GB/s`) | `Qwen3.8-27B-W2.2A16-HGPTQ-RCO` | Standard (`tq4_all`, fits up to `64K` ctx) | `505.8` | `21.95` | — | `45.57 ms` |
| **`RTX 4060 Ti 16 GB`** (`sm_89`, `128-bit`, `288 GB/s`) | `Qwen3.5-9B-W4A16-HGPTQ` | Standard (`tq4`) | `2,862.0` | `53.6` | — | `18.67 ms` |
| **`RTX 4060 Ti 16 GB`** (`sm_89`, `128-bit`, `288 GB/s`) | `Qwen3.5-9B-W4A16-HGPTQ` | **`--speculative mtp --mtp-k 3`** | **`2,854.5`** | **`120.1`** | **`3.97`** | **`8.32 ms`** |
| **`RTX 4060 Ti 16 GB`** (`sm_89`, `128-bit`, `288 GB/s`) | `Qwen3.8-27B-W2.6A16-HGPTQ-RCO` | Standard (`tq4`) | `839.9` | `24.9` | — | `40.48 ms` |
| **`RTX 4060 Ti 16 GB`** (`sm_89`, `128-bit`, `288 GB/s`) | `Qwen3.8-27B-W2.6A16-HGPTQ-RCO` | **`--speculative mtp --mtp-k 3`** | **`838.8`** | **`48.5`** | **`2.89`** | **`20.63 ms`** |
| **`RTX 4060 Ti 16 GB`** (`sm_89`, `128-bit`, `288 GB/s`) | `Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO` | Standard (`tq4`) | `4,165.5` | `123.2` | — | `8.18 ms` |
| **`RTX 4060 Ti 16 GB`** (`sm_89`, `128-bit`, `288 GB/s`) | `Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO` | **`--speculative mtp --mtp-k 3`** | **`3,851.4`** | **`190.8`** | **`3.63`** | **`5.24 ms`** |
| **`1× RTX 3090 24 GB`** (`sm_86`, `384-bit`, `936 GB/s`, `TP=1`) | `Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO` (`231B`) | Standard (`tq4`, `k_hot=96/512`) | `278.8` | `26.1` | — | `38.31 ms` |
| **`1× RTX 3090 24 GB`** (`sm_86`, `384-bit`, `936 GB/s`, `TP=1`) | `Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO` (`231B`) | `--speculative mtp --mtp-k 3` (`k_hot=96/512`) | `278.2` | `24.0` | `3.94` | `41.59 ms` |
| **`2× RTX 3090 24 GB`** (`TP=2`, default `k_hot=128/256`) | `Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO` (`231B`) | Standard (`tq4`, `50.0%` experts in VRAM) | `393.6` | `50.3` | — | `19.87 ms` |
| **`2× RTX 3090 24 GB`** (`TP=2`, default `k_hot=128/256`) | `Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO` (`231B`) | **`--speculative mtp --mtp-k 3`** (`50.0%` in VRAM) | **`403.5`** | **`73.2`** | **`3.94`** | **`13.65 ms`** |
| **`2× RTX 3090 24 GB`** (`TP=2`, `k_hot=160/256` LoRA+Desktop) | `Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO` (`231B`) | **`--speculative mtp --mtp-k 3`** (`62.5%` in VRAM) | **`502.3`** | **`94.9`** | **`3.94`** | **`10.54 ms`** |
| **`2× RTX 3090 24 GB`** (`TP=2`, `k_hot=192/256` `75%` VRAM) | `Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO` (`231B`) | **`--speculative mtp --mtp-k 3`** (`75.0%` in VRAM) | **`707.6`** | **`124.2`** | **`3.94`** | **`8.05 ms`** |
| **`2× RTX 3090 24 GB`** (`TP=2`, `k_hot=208/256` `81.25%` VRAM) | `Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO` (`231B`) | **`--speculative mtp --mtp-k 3`** (`81.25%` in VRAM) | **`787.5`** | **`131.6`** | **`3.94`** | **`7.60 ms`** |

> **Hardware note (`3060` vs `4060 Ti`):** On `Qwen3.5-9B-W4A16-HGPTQ`, the `RTX 4060 Ti` (`sm_89` Ada Tensor Cores) is **`1.69×` faster in prefill** (`2,862` vs `1,692 tok/s`), whereas the `RTX 3060` (`192-bit` bus, `360 GB/s`) is **`14%–17%` faster in single-stream decode** (`60.9` vs `53.6 tok/s`, and `140.5` vs `120.1 tok/s` with `MTP k=3`) than the `128-bit` (`288 GB/s`) `RTX 4060 Ti`.

---

## 3. How `w34` Manages Consumer VRAM Under the Hood

Inspection of `crates/w34-host/src/cuda.rs`, `rank_capacity.rs`, `rank.rs`, `cache.rs`, and `lora.rs` in `warp-34` reveals four key memory-planning rules:

### A. Automatic Zero-Copy `embed_tokens` Offload (`use_host_embed`)
When keeping untied `embed_tokens` (`1,940 MiB` FP16 on `9B`, `2,425 MiB` on `27B`, `970 MiB` on `35B-A3B`) in GPU `weight_arena` would leave `< 2.0 GiB` of post-boot VRAM slack (`HOST_EMBED_GPU_RESIDENCY_MIN_SLACK_BYTES`), `plan_host_embed_with_serving_budget` automatically places `embed_tokens` in pinned PCIe `MappedHostBuffer` (system RAM) while keeping `lm_head` (`W8_g64`) in GPU VRAM.
- Because `embed_tokens` is a sparse row lookup (`8–10 KiB` per token) rather than a GEMM, PCIe zero-copy lookup takes `< 1 µs` and has **zero impact on decode throughput**.
- In addition, `w34` lazily skips `visual.*` (`~235 MiB`) until an image request arrives and skips `mtp.*` (`240–433 MiB`) unless `--speculative mtp` is enabled.

### B. Hard `1.0 GiB` Post-Pool Floor (`MIN_HEADROOM_BYTES`) + Elastic GDN Slot Clamping
At boot, `w34` queries `cuMemGetInfo(&free_vram, &total_vram)` (which already subtracts OS desktop/compositor allocations) and enforces:
1. **`MIN_HEADROOM_BYTES = 1,024 MiB`**: `w34` never allocates the last `1.0 GiB` of `free_vram`, preserving room for CUDA graph capture (`decode_graph` / `spec_verify_graphs`), cuBLAS workspaces, and desktop compositor spikes.
2. **Elastic `StateBundlePool` Clamping (`clamp_state_slots`)**: In `w34 serve`, `w34` requests up to `64` GDN recurrent state slots (`146.81 MiB/slot` on `27B`, `61.41 MiB/slot` on `35B-A3B`, `49.13 MiB/slot` on `9B`) and automatically clamps the slot count down (to `8` slots for `--max-num-seqs 2`, or down to `MIN_ACTIVE_STATE_SLOTS = 3`) when `free_vram` is tighter. Extra GDN slots above `3–8` act as an **automatic shock absorber** when a desktop OS or LoRA adapter is active.

### C. Why Default `tq4` Beats `tq4_all` as the Primary Target
In `crates/w34-host/src/cache.rs`, `--kv-format` defaults to **`tq4`** (`KvFormatPolicy::Tq4`):
- **`tq4` (Tail-Anchored TurboQuant):** Keeps the **last 4 full-attention layers** (`N - 4 .. N - 1`, directly before `final_norm` + `lm_head`) in unrotated 8-bit (`FP8 E4M3` `264 B/row` on `sm_89+`, `Q8_0` `272 B/row` on `sm_86`) and all earlier full-attention layers in WHT-256 rotated 4-bit `TQ4` (`132 B/row`).
- **`tq4_all` (Uniform 4-bit):** Places all full-attention layers in `TQ4` (`132 B/row`).
- At **`128K` context (`131,072` tokens)**, the total VRAM difference between `tq4` and `tq4_all` is only:
  - **`+264 MiB`** on `Qwen3.6-35B-A3B` (`2` KV heads × `4` tail layers = `3.4% W4` routed-expert equivalent)
  - **`+528 MiB`** (`sm_89+`) / **`+560 MiB`** (`sm_86`) on `Qwen3.8-27B` and `Qwen3.5-9B` (`4` KV heads × `4` tail layers)
- Sizing checkpoints so **`tq4` works out of the box** ensures default `w34 --kv-pages 8192` commands succeed without extra flags while preserving `FP16`-oracle long-context retrieval accuracy on the final 4 attention layers, leaving `--kv-format tq4_all` as an instant `+264 to +560 MiB` escape hatch.

### D. LoRA Adapter VRAM Cost (`~660–750 MiB` at `128K` Context)
When `--lora` / `--lora-modules` is passed (`crates/w34-host/src/lora.rs`), `w34` reserves two GPU allocations:
1. **`lora_arena` (`LoraCapacityReserve::device_bytes_per_rank`) — `~120–220 MiB`**:
   - Pre-rotated FP16 `lora_A` + `lora_B` weights (`~40–140 MiB` per `r=16..32` adapter; $H_{64}$ is folded into `lora_A` at load time),
   - `lowrank_ws_bytes` (`6,144 tokens × max_rank_pad × 4 B` = `~1–3 MiB`),
   - Fixed `+64 MiB` arena margin (`lora.rs:869`).
2. **`scratch_arena` Expansion (`rank_capacity.rs:311–321`) — `+448 to +544 MiB`**:
   - Whenever `lora.device_bytes_per_rank > 0` and `requested_kv_pages <= 8192` (`128K` context), `w34` expands `scratch_arena` from `512 MiB` (`DEFAULT_SCRATCH_ARENA_BYTES`) to **`1,056 MiB`** (`TIERED_MOE_SCRATCH_ARENA_BYTES`, `+544 MiB`) for standard/bidirectional-prefix LoRAs or **`960 MiB`** (`MIDLOOP_LORA_SCRATCH_ARENA_BYTES`, `+448 MiB`) for multi-pass `midloop` LoRAs.

---

## 4. Complete `128K`-Context VRAM & Safety Buffer Matrix (`8 GB`, `12 GB`, `16 GB`)

| Target Bracket | Recommended Checkpoint (`nchapman/*`) | `128K` `tq4` VRAM Used (`tq4_all`) | Headless Buffer Above `128K` `tq4` Floor | Minus LoRA (`lora_arena` + `1,056 MiB` Scratch) | Remaining Buffer for Desktop OS (Win11 / Wayland + Browser) | Headless-Only Max `% W4` Ceiling (`tq4` `128K`) |
|---|---|---:|---:|---:|---:|---:|
| **`8 GB` (`RTX 4060`) — `9B`** | [`Qwen3.5-9B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.5-9B-W2.6A16-HGPTQ-RCO) (`30% W4`, `2.60 bpw`) | `6,153 MiB` (`5,593 MiB`) | **`+1,635 MiB`** | `-700 MiB` | **`+935 MiB`** (`+1,495 MiB` `tq4_all`) | `80% W4` (`W3.60`) |
| **`12 GB` (`RTX 3060`) — `9B` + `MTP k=3`** | [`Qwen3.5-9B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.5-9B-W4A16-HGPTQ) (`100% W4`, `4.00 bpw`) | `8,835 MiB` (`8,275 MiB`) | **`+3,111 MiB`** | `-700 MiB` | **`+2,411 MiB`** (`+2,971 MiB` `tq4_all`) | `100% W4` (`W4.00`) |
| **`12 GB` (`RTX 3060`) — `27B` (`64K` ctx)** | [`Qwen3.8-27B-W2.2A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-27B-W2.2A16-HGPTQ-RCO) (`10% W4`, `2.20 bpw`) | `10,883 MiB` (`64K` `tq4_all`, `gdn=4`) | **`+1,063 MiB`** *(at `32K–48K` ctx)* | — | **`+1,063 MiB`** *(at `32K–48K` ctx)* | `10% W4` (`W2.20`) |
| **`16 GB` (`4060 Ti`) — `27B`** | [`Qwen3.8-27B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO) (`30% W4`, `2.60 bpw`) | `14,976 MiB` (`14,832 MiB`) *(incl. `13–16` GDN slots)* | **`+1,276 MiB`** (`+1,804 MiB` `tq4_all`) | `-700 MiB` | **`+576 MiB`** (`+1,104 MiB` `tq4_all`) | `50%–52% W4` (`W3.00`) |
| **`16 GB` (`4060 Ti`) — `35B-A3B`** | [`Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO) (`25% W4` exp, `2.62 bpw`) | `14,934 MiB` (`14,918 MiB`) *(incl. `22–26` GDN slots)* | **`+921 MiB`** (`+1,185 MiB` `tq4_all`) | `-700 MiB` | **`+221 MiB`** (`+485 MiB` `tq4_all`) | `40%–42% W4` (`W2.80`) |
| **`1× 24 GB` (`3090`, `TP=1`) — `Flash-Next` (`231B`)** | [`Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO) (`25% W4` exp, `k_hot=96/512`) | `19,489 MiB` *(incl. `20` GDN slots + `MTP`)* | **`+3,614 MiB`** *(`4,638 MiB` free)* | `-220 MiB` *(scratch already `1,056 MiB`)* | **`+3,394 MiB`** *(can raise `k_hot` to `136–152`)* | Tiered MoE (`k_hot`) |
| **`2× 24 GB` (`2× 3090`, `TP=2`) — `Flash-Next` (`k_hot=128`)** | [`Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO) (`256/512` experts in VRAM) | `18,112 MiB`/GPU *(incl. `18` GDN slots + `MTP`)* | **`+4,976 MiB`/GPU** *(`6,000 MiB`/GPU free)* | `-220 MiB`/GPU | **`+4,756 MiB`/GPU** | Tiered MoE (`k_hot`) |
| **`2× 24 GB` (`2× 3090`, `TP=2`) — `Flash-Next` (`k_hot=160`)** | [`Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO) (`320/512` = `62.5%` in VRAM) | `21,256 MiB`/GPU *(incl. `18` GDN slots + `MTP`)* | **`+1,832 MiB`/GPU** *(`2,856 MiB`/GPU free)* | `-220 MiB`/GPU | **`+1,612 MiB`/GPU** *(LoRA + Desktop OS sweet spot)* | Tiered MoE (`k_hot`) |
| **`2× 24 GB` (`2× 3090`, `TP=2`) — `Flash-Next` (`k_hot=184–208`)** | [`Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO) (`368–416/512` in VRAM) | `23,728 MiB`/GPU *(incl. `18` GDN slots + `MTP`)* | Headless ceiling (`384 MiB`/GPU post-graph free) | — | Headless only (`131.6 tok/s` w/ `MTP k=3`) | Tiered MoE (`k_hot`) |

---

## 5. `Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO` (`231B`) on `1× 24 GB` (`TP=1`) & `2× 24 GB` (`TP=2`): Tiered-MoE `k_hot` Scaling

On `Qwen3.8-Flash-Next` (`48` layers, `512` routed experts/layer, `hidden=2560`, `moe_intermediate=512`), a single `W4_g64` expert slot is `1.992 MiB` (`2,088,960 B`) and a `W2_g64` expert slot is `1.055 MiB` (`1,105,920 B`):
- **Mixed-Precision `actual_per_hot_step` Reduction (`-35.3%`):** Across all 48 layers (`12` `W4_g64` + `36` `W2_g64` in `W2.6A16-HGPTQ-RCO`), each `+1` hot expert slot per rank costs **`61.875 MiB`** instead of `95.625 MiB` on pure `W4A16` (`15,840 MiB` vs `24,480 MiB` for all `256` local experts per rank on `TP=2`).
- **Why `w34` Leaves `6,000 MiB/GPU` Free by Default on `2× 24 GB` (`TP=2`):** In `crates/w34-host/src/model/rank_capacity.rs` (`plan_tiered_moe_hot_per_layer_with_dflash`), `w34` currently divides `available_vram - fixed_overhead` by `per_hot_step` (`48 × 1.992 MiB = 95.625 MiB`, the pure-`W4` upper bound) instead of `actual_per_hot_step` (`61.875 MiB`) and clamps `TP=2` to `TIERED_MOE_TP2_DEFAULT_HOT = 128`. As a result, `w34 serve --tp 2 --kv-pages 8192 --kv-format tq4 --speculative mtp --mtp-k 3` uses **`18,112 MiB` per GPU** (`73.23 tok/s` decode) and leaves **`6,000 MiB` (`5.86 GiB`) free on each `24 GB` card** (`4,638 MiB` free on `1× 24 GB` `TP=1` at `19,489 MiB` used, `26.10 tok/s`).
- **Unlocking the Saved `W2.6` VRAM for Hot Experts (`k_hot = 160..208` on `TP=2`):** Dividing by `actual_per_hot_step` (`61.875 MiB/slot`) and raising `TIERED_MOE_TP2_DEFAULT_HOT` converts those saved `W2.6` bytes directly into GPU-resident hot experts while keeping 100% (`256 / 256` local experts per GPU = `512 / 512` total) of the Native MTP layer in VRAM:
  - **`k_hot = 160 / 256` (`320 / 512` = `62.5%` of experts in VRAM, LoRA + Desktop OS sweet spot):** Uses **`21,256 MiB/GPU`** at `128K` `tq4` + `18` GDN slots + `MTP k=3`, leaving **`2,856 MiB/GPU` free** (`+1,612 MiB/GPU` for Desktop OS after `1,024 MiB` `MIN_HEADROOM_BYTES` and `~220 MiB` `lora_arena`; note `scratch_arena` is already `1,056 MiB` on tiered MoE) and boosting throughput to **`502.3 tok/s` prefill** and **`94.92 tok/s` decode (`10.54 ms` ITL)**.
  - **`k_hot = 192–208 / 256` (`384–416 / 512` = `75.0%–81.25%` of experts in VRAM, headless ceiling):** Reaches **`707.6–787.5 tok/s` prefill (`1.95×`)** and **`124.17–131.55 tok/s` decode (`7.60–8.05 ms` ITL, `1.80×` speedup over `k_hot=128`)** on `2× RTX 3090`.
