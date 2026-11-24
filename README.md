# warp34-quant-lab

Full-model 4-bit (`W4A16-HGPTQ`) and sub-3-bit (`W2.6` `HGPTQ-RCO`) quantization, calibration, and teacher-forced fidelity probing for the Qwen 3.5 / 3.6 / 3.8 model family.

Most low-bit checkpoints only quantize the main text layers — dropping or blindly rounding the vision encoder and built-in speculative draft head. `warp34-quant-lab` calibrates and quantizes **text, Vision, Native Multi-Token Prediction (MTP), routed and shared MoE experts, and `Qwen3.8-Flash-Next`'s Layer-1 N-gram table** together in standard `2/4/8`-bit `compressed-tensors` format for [`warp34`](https://github.com/nchapman/warp34) and `vLLM`.

## Highlights

- **Full-model 4-bit calibration (`W4A16-HGPTQ`):** Calibrates text layers, Vision (`96.4%–99.3%` top-1 agreement), and built-in Native MTP (`90.2%–96.6%` top-1 agreement) together — plus companion 4-bit `DFlash2` block-diffusion drafters (`99.7%–99.8%` of BF16 acceptance length) and salvaged MTP heads for fine-tunes that shipped without them.
- **Hadamard rotation + Cholesky GPTQ + H-CD:** Block Walsh–Hadamard input rotation (`H_64` / `H_128` / `H_512`) spreads weight and activation outliers before GPTQ + Hessian Coordinate Descent and FP32-master `RMSNorm`/GDN KL tuning, matching `BF16` and `EXL3 4.0bpw` / `UD-Q4_K_XL` on `IFEval`, `MATH-500`, and `HumanEval+` at `16%–19%` smaller text-backbone size than `UD-Q4_K_XL`.
- **Strict `2/4/8`-bit sub-3-bit RCO V2 (`HGPTQ-RCO`):** Symmetric mid-riser 2-bit (`w2_g128_sym`, `code_offset=0.5`) mixed with `w4_g128` via Riemannian Constrained Optimization fits `9B` on `8 GB`, `27B` & `35B-A3B` on `16 GB`, and `180B` `Qwen3.8-Flash-Next` on `48 GB` (`mmap` PLE) or `2× 24 GB` GPUs — beating `UD-Q2_K_XL` on `9B` (`75.74%` vs `73.12%` EvalScope mean) at lower `bpw`.
- **MoE & 51.2B-parameter PLE compression:** Rotates and quantizes both routed and shared experts, compresses `Qwen3.8-Flash-Next`'s `102.4 GB` Layer-1 PLE table to `26.82 GiB` (`4`-bit `Custom-LUT4` `W_kv`-HGPTQ), and exports pre-profiled hot-expert rankings (`static_hot_cache_rankings.json`).
- **Built-in verification harness:** Single-pass teacher-forced probing across frozen `probe-v4` (`378` prompts), `probe-guard` (`76` prompts), Native MTP, and disjoint `vision-probe-v2` / `vision-guard-v2` splits, plus an OpenAI-compatible `EvalScope` runner.
- **Multi-domain coverage:** Supports dense VLMs (`0.8B`–`27B`), MoE models (`35B-A3B`, `Qwen3.8-Flash-Next`), `Qwen-Image-2.1` diffusion pipelines, `d3` / `GLiDE` decision classifiers, and Heretic abliteration LoRA extraction.

## Prerequisites & Quick Start

**Requirements:**
- Linux (x86_64) with Python `>= 3.12`
- [`mise`](https://mise.jdx.dev/) and [`uv`](https://docs.astral.sh/uv/) (exact versions pinned in [`.mise.toml`](.mise.toml))
- NVIDIA CUDA GPU for quantization, probing, and GPU kernel compilation (CPU-only hosts can run `list-models`, `verify-data`, and `mise run check`)

```bash
git clone https://github.com/nchapman/warp34-quant-lab.git
cd warp34-quant-lab

# 1. Create and sync .venv with uv
mise run setup

# 2. (GPU hosts) Compile fast-path GatedDeltaNet kernels (causal-conv1d + flash-linear-attention)
mise run setup-kernels

# 3. Verify frozen calibration/probe checksums and run linter + unit tests
mise run verify-data
mise run check
```

Confirm the CLI is ready and inspect all registered model keys and default recipes:

```bash
mise run list-models
```

Expected output starts with:

```text
KEY                ARCH           UPSTREAM HUB                               TARGET QUANT REPO
------------------------------------------------------------------------------------------------------------------------
0.8b               qwen3_5        Qwen/Qwen3.5-0.8B                          nchapman/Qwen3.5-0.8B-W4A16-HGPTQ
  recipe: g64-h64 actorder=static calib=data/calib-cascade-v1.jsonl (999x4096) mtp=True vision=True
...
```

## Usage

### 1. Materialize Vision & Image-Generation Datasets

Before quantizing or probing Vision / `Qwen-Image-2.1` models, materialize the strictly disjoint, commit-pinned image splits under `data/`:

```bash
mise run build-vision-data   # data/vision-{calib,probe,guard}-v2 (120 / 96 / 48 images)
mise run build-image-data    # data/image-gen-{calib,probe,guard}-v1 (96 / 64 / 24 rows)
```

### 2. Quantize a Model (`W4A16-HGPTQ`)

`warp-quant quantize` automatically dispatches dense models (`0.8b`–`27b`), streaming MoE models (`35b`, `35b-ornith`, `35b-apex`, `flash-next`), diffusion pipelines (`qwen-image-2.1`, `qwen-image-2.1-turbo`), and decision classifiers (`d3*`, `glide-*`):

```bash
# Dense model (e.g. Qwen3.5-9B)
mise run quantize -- \
  --model 9b \
  --output-dir ./out/Qwen3.5-9B-W4A16-HGPTQ

# Streaming MoE model (Qwen3.6-35B-A3B: routed + shared experts + MTP + Vision + hot-expert rankings)
mise run quantize -- \
  --model 35b \
  --output-dir ./out/Qwen3.6-35B-A3B-W4A16-HGPTQ

# 180B MoE + 51.2B Layer-1 PLE (Qwen3.8-Flash-Next)
mise run quantize -- \
  --model flash-next \
  --output-dir ./out/Qwen3.8-Flash-Next-W4A16-HGPTQ

# Diffusion pipeline (Qwen-Image-2.1-Turbo: Qwen3VL text encoder + DiT)
mise run quantize -- \
  --model qwen-image-2.1-turbo \
  --output-dir ./out/Qwen-Image-2.1-Turbo-W4A16-HGPTQ \
  --quantize-vision \
  --quantize-lm-head-w8 \
  --vision-calib-dir data/vision-calib-v2
```

### 3. Probe Fidelity Against BF16

Evaluate text perplexity and KL divergence (`probe-v4` and `probe-guard`), Native MTP agreement, and Vision v2 agreement in a single pass:

```bash
mise run probe -- \
  --model 35b \
  --quant ./out/Qwen3.6-35B-A3B-W4A16-HGPTQ \
  --eval-sets data/probe-v4.jsonl data/probe-guard.jsonl \
  --eval-mtp \
  --eval-vision data/vision-probe-v2 \
  --eval-vision-guard data/vision-guard-v2 \
  --output-json ./out/probe_summary.json
```

### 4. Build Sub-3-Bit Mixed-Precision Checkpoints (`HGPTQ-RCO` V2)

Run the V2 (`w2_g128_sym` + `w4_g128` + `H_512`) Riemannian Constrained Optimization pipeline for dense models (`9b`, `9b-ornith`, `27b`):

```bash
uv run --no-sync warp-quant rco \
  --model 9b \
  --work-dir ./out/rco-9b-v2 \
  --options w2_g128_sym w4_g128 \
  --hadamard-block-size 512 \
  --budgets 0.3675 \
  --assemble 37pct_rco
```

For MoE models (`35b`, `35b-ornith`, `flash-next`), pass `--save-rco-bank-dir` to `warp-quant quantize` to capture the per-layer W2 candidate bank, then run `run_moe_rco_search_and_qat` in `src/warp_quant/streaming_moe.py`.

### 5. Run Downstream Benchmarks & Extract Abliteration LoRAs

```bash
# EvalScope benchmarks against a running warp34 or vLLM OpenAI-compatible server
mise run eval -- \
  --model nchapman/Qwen3.5-9B-W4A16-HGPTQ \
  --api-url http://127.0.0.1:8000/v1 \
  --datasets ifeval math_500 humaneval_plus \
  --no-thinking \
  --temperature 0.0 \
  --max-tokens 2048 \
  --output-dir ./out/eval-9b

# Extract an unrotated BF16 Heretic abliteration LoRA adapter (PEFT / vLLM / warp34)
mise run heretic-lora -- \
  --model 9b \
  --output-dir ./out/qwen35-9b-heretic-lora
```

## Published Models

All pre-quantized checkpoints are published on Hugging Face under [`nchapman`](https://huggingface.co/nchapman), organized into curated collections:

- **[Qwen Models (W4A16-HGPTQ)](https://huggingface.co/collections/nchapman/qwen-models-w4a16-hgptq-6b050569380158e23bc46085)** — core `0.8B`–`27B`, `35B-A3B`, and `Qwen3.8-Flash-Next` 4-bit checkpoints
- **[Qwen Sub-3-Bit Models (W2.6A16-HGPTQ-RCO)](https://huggingface.co/collections/nchapman/qwen-sub-3-bit-models-w26a16-hgptq-rco-6b05056a986f3669623474ca)** — all 7 V2 strict `2/4/8`-bit RCO checkpoints
- **[Qwen Fine-Tunes (W4A16-HGPTQ)](https://huggingface.co/collections/nchapman/qwen-fine-tunes-w4a16-hgptq-6b05056b6ec5bf4ecad11ac0)** — `Ornith-1.5`, `Apex-35B`, `Swift-1.5`, `Arete-27B`, `DaisyRose-27B`, `sabi-noema`
- **[DFlash2 Drafters (W4A16-HGPTQ)](https://huggingface.co/collections/nchapman/dflash2-drafters-w4a16-hgptq-6b05056de9d3897cbc611241)** — 4-bit block-diffusion speculative drafters for `27B` and `Flash-Next`
- **[Decision Models (W4A16-HGPTQ)](https://huggingface.co/collections/nchapman/decision-models-w4a16-hgptq-6b05056dfe6ebb9a6d419809)** — all 8 `vllm-sr/d3*` and 4 `fastino/GLiDE-*` classifiers ([receipts](results/decision-w4a16-hgptq/))
- **[LoRA Adapters](https://huggingface.co/collections/nchapman/lora-adapters-6b05056ffc921b5ab472443b)** — hot-swappable BF16 adapters for `warp34` and `vLLM`

### Core & Fine-Tuned LLMs / VLMs (`W4A16-HGPTQ`)

| Model | Artifact (`nchapman/*`) | Size | `probe-v4` PPL (BF16 → Quant) | `probe-v4` `weighted_kl` | `probe-guard` `weighted_kl` (`mean_kl`) | MTP Iso `mean_kl` (Top-1) | Vision Iso `mean_kl` (Top-1) | Vision Guard Iso Top-1 |
|---|---|---:|---|---:|---|---:|---:|---:|
| `Qwen/Qwen3.5-0.8B` | [`Qwen3.5-0.8B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.5-0.8B-W4A16-HGPTQ) ([report](results/dense-w4a16-hgptq/0.8b-g64-h64/TECHNICAL.md)) | `0.79 GiB` | `4.1695 → 4.3015` (`+3.166%`) | `0.073509` | `0.088739` (`0.070613`) | `0.028836` (`90.24%`) | `0.026580` (`94.04%`) | `97.04%` |
| `Qwen/Qwen3.5-2B` | [`Qwen3.5-2B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.5-2B-W4A16-HGPTQ) ([report](results/dense-w4a16-hgptq/g64-h64/TECHNICAL.md)) | `1.83 GiB` | `3.8302 → 3.7784` (`-1.352%`) | `0.059069` | `0.062987` (`0.050014`) | `0.020116` (`92.78%`) | `0.012995` (`97.35%`) | `99.26%` |
| `Qwen/Qwen3.5-4B` | [`Qwen3.5-4B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.5-4B-W4A16-HGPTQ) ([report](results/dense-w4a16-hgptq/4b-g64-h64/TECHNICAL.md)) | `3.19 GiB` | `4.0299 → 3.9537` (`-1.891%`) | `0.081090` | `0.074129` (`0.053346`) | `0.014983` (`94.26%`) | `0.004972` (`99.34%`) | `99.26%` |
| `Qwen/Qwen3.5-9B` | [`Qwen3.5-9B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.5-9B-W4A16-HGPTQ) ([report](results/dense-w4a16-hgptq/9b-g64-h64/TECHNICAL.md)) | `6.67 GiB` | `3.5754 → 3.4999` (`-2.112%`) | `0.063824` | `0.058614` (`0.039538`) | `0.010453` (`94.86%`) | `0.007192` (`99.34%`) | `98.52%` |
| `ornith-ai/Ornith-1.5-9B` | [`Ornith-1.5-9B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Ornith-1.5-9B-W4A16-HGPTQ) ([report](results/dense-w4a16-hgptq/9b-ornith-g64-h64/TECHNICAL.md)) | `6.67 GiB` | `6.6829 → 5.6579` (`-15.338%`) | `0.144025` | `0.133591` (`0.096463`) | `0.012652` (`94.68%`) | `0.020540` (`97.02%`) | `97.04%` |
| `Qwen/Qwen3.8-27B` | [`Qwen3.8-27B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-27B-W4A16-HGPTQ) ([report](results/dense-w4a16-hgptq/27b-g64-h64/TECHNICAL.md)) | `16.14 GiB` | `8.5332 → 8.1841` (`-4.091%`) | `0.100223` | `0.073907` (`0.052982`) | `0.008979` (`96.64%`) | `0.005603` (`97.68%`) | `99.26%` |
| `ukisai/Swift-1.5-Qwen3.8-27b` | [`Swift-1.5-Qwen3.8-27b-W4A16-HGPTQ`](https://huggingface.co/nchapman/Swift-1.5-Qwen3.8-27b-W4A16-HGPTQ) ([report](results/dense-w4a16-hgptq/27b-swift-g64-h64/TECHNICAL.md)) | `16.14 GiB` | `9.6265 → 9.0881` (`-5.593%`) | `0.107444` | `0.071995` (`0.055502`) | `0.009043` (`96.43%`) | `0.007063` (`98.01%`) | `99.26%` |
| `NousResearch/Arete-27B` | [`Arete-27B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Arete-27B-W4A16-HGPTQ) ([report](results/arete-27b-w4a16-hgptq/TECHNICAL.md)) | `16.14 GiB` | `4.9736 → 4.8222` (`-3.044%`) | `0.088445` | `0.091283` (`0.067951`) | `0.008862` (`96.47%`, salvaged) | `0.004956` (`98.34%`) | `97.78%` |
| `sabicap/sabi-noema` | [`sabi-noema-W4A16-HGPTQ`](https://huggingface.co/nchapman/sabi-noema-W4A16-HGPTQ) ([report](results/sabi-noema-w4a16-hgptq/TECHNICAL.md)) | `16.14 GiB` | `9.6074 → 9.4030` (`-2.127%`) | `0.110509` | `0.102326` (`0.067460`) | `0.008735` (`96.63%`, salvaged) | `0.007908` (`99.01%`) | `99.26%` |
| `RadixArk/DaisyRose-27B` | [`DaisyRose-27B-W4A16-HGPTQ`](https://huggingface.co/nchapman/DaisyRose-27B-W4A16-HGPTQ) ([report](results/daisyrose-27b-w4a16-hgptq/TECHNICAL.md)) | `16.14 GiB` | `8.5603 → 8.2929` (`-3.123%`) | `0.111026` | `0.054826` (`0.075052`) | `0.009181` (`96.30%`) | `0.010187` (`99.01%`) | `98.52%` |
| `Qwen/Qwen3.6-35B-A3B` | [`Qwen3.6-35B-A3B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ) ([report](results/qwen36-35b-a3b-w4a16-hgptq/TECHNICAL.md)) | `18.78 GiB` | `4.0208 → 3.7522` (`-6.682%`) | `0.084694` | `0.050908` (`0.046230`) | `0.010490` (`95.80%`) | `0.007135` (`99.34%`) | `98.52%` |
| `ornith-ai/Ornith-1.5-35B-A3B` | [`Ornith-1.5-35B-A3B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Ornith-1.5-35B-A3B-W4A16-HGPTQ) ([report](results/ornith-15-35b-a3b-w4a16-hgptq/TECHNICAL.md)) | `18.78 GiB` | `10.8033 → 10.2627` (`-5.004%`) | `0.113922` | `0.228886` (`0.164721`) | `0.021504` (`91.44%`) | `0.030482` (`96.36%`) | `97.04%` |
| `Apexintelligence-AI/Apex-35B-V1.0` | [`Apex-35B-V1.0-W4A16-HGPTQ`](https://huggingface.co/nchapman/Apex-35B-V1.0-W4A16-HGPTQ) ([report](results/apex-35b-v1-w4a16-hgptq/TECHNICAL.md)) | `18.78 GiB` | `5.1325 → 4.9373` (`-3.804%`) | `0.070100` | `0.062547` (`0.057495`) | `0.012086` (`95.06%`) | `0.016896` (`97.35%`) | `99.26%` |
| `Qwen/Qwen3.8-Flash-Next` | [`Qwen3.8-Flash-Next-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ) ([report](results/qwen38-flash-next-w4a16-hgptq/TECHNICAL.md)) | `92.84 GiB` | `3.6085 → 3.5683` (`-1.114%`) | `0.042091` | `0.088441` (`0.067320`) | `0.018417` (`95.41%`) | `0.017124` (`97.35%`) | `99.26%` |

### Sub-3-Bit Mixed-Precision Checkpoints (`HGPTQ-RCO` V2: `w2_g128_sym` + `w4_g128` + `W8` Head)

| Base Model | Artifact (`nchapman/*`) | `W4` Allocation | Full (`Pure Text` / Trunk) | Target GPU | `probe-v4` `w_kl` | `probe-guard` `mean_kl` | EvalScope 3-Bench Mean |
|---|---|---:|---:|---|---:|---:|---:|
| `Qwen/Qwen3.5-9B` | [`Qwen3.5-9B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.5-9B-W2.6A16-HGPTQ-RCO) ([report](results/rco-9b-w2w4/TECHNICAL-W2.6A16-HGPTQ-RCO.md)) | `36.75%` `W4` (`H_512`) | `5.55 GiB` (`2.31 GiB`) | `8 GB` | `0.2417` | `0.2933` | `75.74%` |
| `ornith-ai/Ornith-1.5-9B` | [`Ornith-1.5-9B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Ornith-1.5-9B-W2.6A16-HGPTQ-RCO) ([report](results/ornith-15-9b-w26a16-hgptq-rco/TECHNICAL.md)) | `36.75%` `W4` (`H_512`) | `5.55 GiB` (`2.31 GiB`) | `8 GB` | `0.3258` | `0.4067` | — |
| `Qwen/Qwen3.8-27B` | [`Qwen3.8-27B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO) ([report](results/rco-27b-w2w4/TECHNICAL-W2.6A16-HGPTQ-RCO.md)) | `36.75%` `W4` (`H_512`) | `12.20 GiB` (`8.15 GiB`) | `16 GB` | `0.2691` | `0.3168` | `82.87%` |
| `Qwen/Qwen3.6-35B-A3B` | [`Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO) ([report](results/rco-35b-w2w4/TECHNICAL-W2.6A16-HGPTQ-RCO.md)) | `12/40` exp (`H_512`) | `13.01 GiB` (`10.94 GiB`) | `16 GB` | `0.1784` | `0.1614` | `86.10%` |
| `ornith-ai/Ornith-1.5-35B-A3B` | [`Ornith-1.5-35B-A3B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Ornith-1.5-35B-A3B-W2.6A16-HGPTQ-RCO) ([report](results/ornith-15-35b-a3b-w26a16-hgptq-rco/TECHNICAL.md)) | `12/40` exp (`H_512`) | `13.01 GiB` (`10.94 GiB`) | `16 GB` | `0.2283` | `0.3934` | — |
| `Qwen/Qwen3.8-Flash-Next` | [`Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO) ([report](results/qwen38-flash-next-w26a16-hgptq-rco/TECHNICAL.md)) | `15/48` exp (`H_128`) | `71.67 GiB` (`44.84 GiB` trunk) | `48 GB` / `2× 24 GB` | `0.1091` | `0.1961` | — |

### Speculative Drafters & Diffusion Pipelines (`W4A16-HGPTQ`)

| Type | Artifact (`nchapman/*`) | Size (BF16 → Quant) | Key Fidelity Metric | Report |
|---|---|---|---|---|
| `DFlash2` drafter (`27B`) | [`Qwen3.8-27B-DFlash2-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-27B-DFlash2-W4A16-HGPTQ) | `3.58 GiB → 1.31 GiB` (`2.74×`) | $\tau$: `2.540 → 2.535` (`99.8%`), `Recall@16`: `53.88%` (`100.0%`) | [report](results/dense-w4a16-hgptq/dflash2-27b-TECHNICAL.md) |
| `DFlash2` drafter (`Flash-Next`) | [`Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-DFlash2-W4A16-HGPTQ) | `1.12 GiB → 0.52 GiB` (`2.17×`) | $\tau$: `2.760 → 2.752` (`99.7%`), `Recall@16`: `59.67% → 59.60%` | [report](results/dense-w4a16-hgptq/dflash2-flash-next-TECHNICAL.md) |
| Image-gen (`Turbo`) | [`Qwen-Image-2.1-Turbo-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen-Image-2.1-Turbo-W4A16-HGPTQ) | `32.45 GiB → 9.82 GiB` (`3.31×`) | Velocity cos `0.9994` (`30.42 dB`), PSNR `25.46 dB` (`36.53 dB` TI2I) | [report](results/qwen-image-21-turbo-w4a16-hgptq/README.md) |
| Image-gen (`Base`) | [`Qwen-Image-2.1-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen-Image-2.1-W4A16-HGPTQ) | `32.45 GiB → 9.82 GiB` (`3.31×`) | Velocity cos `0.9991` (`28.85 dB`), PSNR `26.97 dB` (`36.61 dB` TI2I) | [report](results/qwen-image-21-w4a16-hgptq/README.md) |

## Benchmarks & Technical Reports

Detailed methodology, ablations, bootstrap confidence intervals, and measured GPU receipts live in dedicated reports under `results/`:

| Topic | Report | Highlights |
|---|---|---|
| **7-Arm Cross-Format Comparison (`EXL3`, Unsloth `GGUF`, `HGPTQ-RCO` V2)** | [`results/dense-w4a16-hgptq/COMPARISON.md`](results/dense-w4a16-hgptq/COMPARISON.md) | Full `N=1,205` EvalScope (`IFEval`, `MATH-500`, `HumanEval+`) + `20,000`-draw paired bootstrap CIs. `W4A16-HGPTQ` reaches statistical parity with `BF16` (`85.99%` on `27B`, `83.35%` on `9B`); `HGPTQ-RCO` V2 reaches `75.74%` (`9B` W2.6, beating `UD-Q2_K_XL` `73.12%`), `82.87%` (`27B` W2.6), and `86.10%` (`35B` W2.6). |
| **Consumer GPU `128K` Fit & Throughput (`warp34`)** | [`results/warp34-consumer-gpu-128k/README.md`](results/warp34-consumer-gpu-128k/README.md) | Live `RTX 4060 8 GB`, `RTX 3060 12 GB`, `RTX 4060 Ti 16 GB`, and `1× / 2× RTX 3090 24 GB` VRAM + LoRA + Desktop OS headroom and MTP throughput matrices. |
| **Quantizing `Qwen3.8-Flash-Next` (End-to-End Overview)** | [`results/qwen38-flash-next-w4a16-hgptq/OVERVIEW.md`](results/qwen38-flash-next-w4a16-hgptq/OVERVIEW.md) | Full stack from `335.3 GiB → 92.84 GiB` (`W4`) / `44.84 GiB` GPU trunk (`W2.6` V2): streaming 75,682 linears, 4-bit PLE, 4-stream HC QAT, hot-expert rankings, `0.52 GiB` `DFlash2`, and `95–132 tok/s` on `2× RTX 3090`. |
| **Layer-1 PLE `Custom-LUT4` Design (`Qwen3.8-Flash-Next`)** | [`results/qwen38-flash-next-w4a16-hgptq/PLE-QUANTIZATION-DESIGN.md`](results/qwen38-flash-next-w4a16-hgptq/PLE-QUANTIZATION-DESIGN.md) | Analytical `W_kv`-HGPTQ + Coordinate Descent (`6.655%` `key_proj` Rel-RMSE vs. `9.449%` `NVFP4`) and 4 KiB page-slotted NVMe layout (`0.000%` page-straddle rate). |
| **Sub-3-Bit RCO Pareto Sweeps** | [`2B`](results/rco-2b-w2w4/README.md) · [`9B`](results/rco-9b-w2w4/README.md) · [`27B`](results/rco-27b-w2w4/README.md) · [`35B-A3B`](results/rco-35b-w2w4/README.md) | Per-layer `W2`/`W4` allocation search, Hessian Coordinate Descent (`H-CD`), and FP32-master `RMSNorm`/GDN KL QAT receipts. |

## Architecture

| Path | Role |
|---|---|
| `src/warp_quant/cli.py` | Unified `warp-quant` CLI (`list-models`, `build-vision-data`, `build-image-data`, `quantize`, `probe`, `rco`, `heretic-lora`, `eval`) |
| `src/warp_quant/models.py` | Model registry (`MODELS`), pinned Hub revisions, and default quantization recipes |
| `src/warp_quant/hadamard_gptq.py` | Dense `W4A16-HGPTQ` quantization, target regexes (`TEXT_TARGET_RE`, `MTP_TARGET_RE`, `VISION_*_RE`), Marlin alignment checks, and teacher-forced probing |
| `src/warp_quant/streaming_moe.py` | Streaming layer-by-layer MoE quantization, 4-bit `Custom-LUT4` PLE, hot-expert routing profiler, and MoE RCO/QAT |
| `src/warp_quant/rco/` | Dense candidate bank builder, Gauss–Seidel `H-CD`, Riemannian `HGPTQ-RCO` search, E2E Scale/Norm KL QAT, and checkpoint assembly |
| `src/warp_quant/vllm_plugin.py` | `vllm.general_plugins` runtime plugin (`vllm_warp_quant.py` bundled in checkpoints) for Hadamard transforms, `code_offset=0.5` W2 Marlin/Humming repack, and `Qwen4Exp` PLE |
| `data/` | Frozen SHA-256-verified text calibration/probe splits (`calib-cascade-v1.jsonl`, `probe-v4.jsonl`, `probe-guard.jsonl`, `SHA256SUMS`) |

## Scope & Limitations

- **Supported model families only:** Designed for Qwen 3.5 / 3.6 / 3.8 dense VLMs, MoE models (`35B-A3B`, `Qwen3.8-Flash-Next`), `Qwen-Image-2.1`, `DFlash2` drafters, and `d3` / `GLiDE` decision classifiers. It is not a general-purpose quantizer for arbitrary Hugging Face architectures.
- **GPU & kernel requirements:** Quantizing and probing require Linux + NVIDIA CUDA GPUs. Always run `mise run setup-kernels` before calibrating or probing; without `causal-conv1d` and `flash-linear-attention`, GatedDeltaNet layers fall back to a ~15× slower Python loop.
- **Serving runtime:** Output checkpoints use strict `2/4/8`-bit `compressed-tensors` with online Hadamard transforms and are intended for [`warp34`](https://github.com/nchapman/warp34) or `vLLM` (`--trust-remote-code` with the bundled `vllm_warp_quant.py` plugin).

## Development

```bash
mise run fmt          # format src/ and tests/ with ruff
mise run check        # ruff format --check + ruff check + pytest
mise run verify-data  # verify data/SHA256SUMS (frozen datasets must never be modified)
```

## License

This repository does not currently include a repository-level `LICENSE` file. Each published quantized checkpoint inherits the license of its upstream base model (e.g., Apache-2.0, MIT, Qwen Community 1.0, Swift Open License 1.0, or Fastino Community License), listed in each checkpoint's Hugging Face model card and companion `TECHNICAL.md`.
