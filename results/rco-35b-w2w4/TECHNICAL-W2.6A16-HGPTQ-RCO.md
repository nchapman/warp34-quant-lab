# Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO

Mixed 2-bit / 4-bit Hadamard + GPTQ (`W2.6A16`, group size 128, `H_512` Walsh–Hadamard rotation, symmetric mid-riser `code_offset=0.5` 2-bit grid) quantization of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) built with Riemannian Constrained Optimization (`HGPTQ-RCO`). It compresses the 67.0 GiB BF16 256-expert MoE model down to **13.01 GiB** (`10.94 GiB` pure text backbone; 5.15× smaller than BF16, saving 5.77 GiB vs. pure `W4A16`) so the entire 35B-A3B MoE model, Vision tower, and Native MTP drafter fit on a single **16 GB GPU**.

- **VRAM & GPU fit:** `13.01 GiB` total across all 40 MoE layers, Vision, and Native MTP (`10.94 GiB` pure text backbone), leaving ~3.0 GiB on a 16 GB card for KV cache and speculative decoding (for 24 GB GPUs, see [`Qwen3.6-35B-A3B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ) at `18.78 GiB`).
- **Quantization recipe:** Keeps 100% of Attention/GDN (130 linears), Shared Experts (120 linears), Native MTP (775 linears), and Vision (110 linears) in `W4_g128` (`lm_head` and `mtp.fc` in `W8_g128`), while RCO allocates the 12 most sensitive routed-expert layers (3,072 experts: layers `7, 11, 23, 27, 31, 32, 34..39`) to `W4_g128` and the remaining 28 routed-expert layers (7,168 experts) to `W2_g128` (`code_offset=0.5`), polished with end-to-end FP32-master Norm/GDN KL QAT.
- **Fidelity vs. BF16:** `0.1784` weighted KL (`0.3445` mean KL, `−3.66%` PPL) on `probe-v4`, `0.1614` mean KL (`0.1883` weighted KL, `+3.28%` PPL) on the `probe-guard` holdout, `95.50%` / `83.19%` isolated/E2E MTP top-1 agreement, and `98.01%` / `93.05%` isolated/E2E vision top-1 agreement (`99.26%` / `94.81%` on `vision-guard-v2`).
- **Downstream EvalScope (Full `N=1,205`, greedy `temp=0.0`, `enable_thinking=False`):** **`86.10%` 3-benchmark mean** (`84.29%` IFEval prompt-strict / `87.43%` prompt-loose, `85.60%` MATH-500, `88.41%` HumanEval+) in strict `2/4/8`-bit at `13.01 GiB`.

## Quality vs. BF16 Base Model (`Qwen/Qwen3.6-35B-A3B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model and the `18.78 GiB` pure 4-bit [`Qwen3.6-35B-A3B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ) checkpoint:

| Evaluation | BF16 Base (`Qwen/Qwen3.6-35B-A3B`) | `W4A16-HGPTQ` (`4.00 bpw`) | **`W2.6A16-HGPTQ-RCO` (`2.60 bpw`)** | Notes |
|---|---:|---:|---:|---|
| **Safetensors Size** | `66.97 GiB` (`71,903,776,776 B`) | `18.78 GiB` (`20,160,901,544 B`) | **`13.01 GiB` (`13,972,519,256 B`)** | **`5.15×` compression** (`10.94 GiB` pure text backbone; `−53.96 GiB` vs. BF16, `−5.77 GiB` vs. W4A16) |
| **Text Perplexity** (378 prompts, 840k tokens) ↓ | `4.0208` | `3.7522` (`−6.68%`) | **`3.8737` (`−3.66%`)** | Chat (`0.1114`), math (`0.0627`), science (`0.0795`), instruction (`0.0951`), coding, tool use, multilingual |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | `0.0847` / `0.2103` | **`0.1784` / `0.3445`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout Perplexity** (76 prompts, 20k tokens) ↓ | `25.7581` | `24.5254` (`−4.79%`) | **`26.6020` (`+3.28%`)** | Token mean KL `0.1614` (weighted KL `0.1883`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 2.0268`) | `95.80%` Top-1 (`KL 0.0105`) | **`95.50%` Top-1 (`NLL 2.0389`)** | Isolated MTP KL `0.0115` (end-to-end Top-1 `83.19%`, KL `0.2231`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | `0.9860` cos sim · `99.34%` Top-1 | **`0.9865` cos sim · `98.01%` Top-1** | Isolated ViT KL `0.0059` (end-to-end Top-1 `93.05%`, KL `0.0914`) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | `0.9899` cos sim · `98.52%` Top-1 | **`0.9897` cos sim · `99.26%` Top-1** | Isolated ViT KL `0.0026` (end-to-end Top-1 `94.81%`, KL `0.0662`) |

### End-to-End EvalScope Benchmarks (`N=1,205`, greedy `temp=0.0`, `enable_thinking=False`)

| Checkpoint | Size (Total / Pure Text Backbone) | IFEval (`541`, strict / loose) | MATH-500 (`500`) | HumanEval+ (`164`) | **3-Bench Mean** |
|---|---:|---:|---:|---:|---:|
| **`W2.6A16-HGPTQ-RCO` V2 (strict `2/4/8`-bit)** | **`13.01 GiB` / `10.94 GiB`** | **`84.29%` / `87.43%`** | **`85.60%` (`428/500`)** | **`88.41%` (`145/164`)** | **`86.10%`** |

## What Is Quantized (`HGPTQ-RCO` 16 GB Allocation)

- **Strictly Standard Bit-Widths (`{2, 4, 8}` bits, `group_size=128`, `H_512`)**:
  - **Always-Active Backbone (`100%` in `W4_g128`, `250` linears)**: All `130` hybrid GatedDeltaNet + full-attention projections (`self_attn`, `linear_attn`) and all `120` shared expert projections (`mlp.shared_expert.{gate_proj,up_proj,down_proj}`) across all 40 layers stay in `W4_g128` (`head_dim=512`, MSE scale search + 2-pass H-CD).
  - **Routed MoE Experts (`2.60 bpw` average across `30,720` linears)**:
    - **`W4_g128` (`12` layers = `3,072` experts = `9,216` linears)**: RCO-selected layers `7, 11, 23, 27, 31, 32, 34, 35, 36, 37, 38, 39` (`model.language_model.layers.*.mlp.experts.0..255.{gate_proj,up_proj,down_proj}`).
    - **`W2_g128` (`28` layers = `7,168` experts = `21,504` linears)**: Remaining 28 layers (`0..6, 8..10, 12..22, 24..26, 28..30, 33`) using symmetric mid-riser 2-bit quantization (`code_offset=0.5`, codes `{-1.5, -0.5, +0.5, +1.5}`) with `H_512` rotation.
  - **Untied Output & Drafter Fusion Heads (`W8_g128`, `2` linears)**: `lm_head` (`248320 × 2048`) and `mtp.fc` (`2048 × 4096`) quantized with unrotated 8-bit Cholesky GPTQ + MSE scale search (`group_size=128`).
  - **Native MTP (`775` linears in `W4_g128`)**: `4` `self_attn` projections + `3` `mlp.shared_expert` projections + `768` routed MoE expert projections (`head_dim=512`, `group_size=128`).
  - **Vision (`110` linears in `W4_g128`)**: All 27 ViT blocks + 2 spatial merger projections (`visual.blocks.*.mlp.linear_fc2` uses per-channel GPTQ + `head_dim=16` Hadamard rotation `v_vision_fc2`; `qkv`/`proj`/`linear_fc1` use `head_dim=128` `v_vision`; spatial merger uses `v_vision_merger`).
  - **End-to-End FP32-Master Norm & GDN-Gate KL QAT**: Post-allocation KL-divergence QAT over all layer `RMSNorm`s, `q_norm`/`k_norm`, GatedDeltaNet scalar projections (`in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`), and `mlp.shared_expert_gate` against exact BF16 teacher final-norm states.
- **Serving note (`vLLM`)**: Serve with `VLLM_WARP_QUANT_W2_MARLIN=1` for full-speed CUDA-graph Marlin/Humming fused MoE kernels (lossless `uint4b8` repack of the `code_offset=0.5` grid), or `--enforce-eager` for minimum true 2-bit VRAM (`13.01 GiB` on a 16 GB GPU).
- **Kept in BF16**:
  - `embed_tokens`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `mlp.gate`, `mlp.shared_expert_gate`, `visual.patch_embed.proj`, norms, and biases.
- **Static Hot-Expert Cache Rankings**:
  - `static_hot_cache_rankings.json` (and `runtime/static_hot_cache_rankings.json` + `runtime/expert_routing_profile.json`) provides pre-profiled expert rankings (`0..255` ordered hottest-first per layer `"0".."39"`, plus `"mtp.0"` in `expert_routing_profile.json`; `top_64 = 43.15%` hit rate, `top_128 = 69.58%` hit rate).
- **Default sampling** (`generation_config.json`): `temperature 0.6, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).
