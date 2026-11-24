# Qwen3.5-9B-W2.6A16-HGPTQ-RCO

Mixed 2-bit / 4-bit Hadamard + GPTQ (`W2.6A16-HGPTQ-RCO` V2: `w2_g128_sym` with symmetric half-integer `code_offset=0.5`, `w4_g128`, `H_512` Walsh–Hadamard rotation) quantization of [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) built with Riemannian Co-design (`HGPTQ-RCO`). It compresses the 17.98 GiB BF16 model down to **5.55 GiB** (3.24× smaller; **2.314 GiB** pure text backbone / `2.86 bpw`) so the full multimodal model and Native MTP drafter fit comfortably on a single **8 GB GPU** (RTX 3070 / 4060 / 4060 Ti 8GB / 5060).

- **VRAM & GPU fit:** **5.55 GiB** full VLM+MTP (`5,958,715,384 B`; **2.314 GiB** pure text backbone, **5.19 GiB** text-only incl. `embed_tokens` + `lm_head`), leaving ~2.45 GiB on an 8 GB card for KV cache, vision tokens, speculative decoding, and CUDA graphs.
- **V2 (`D1`) quantization recipe:** Switching `W2` to symmetric half-integer levels (`{-1.5, -0.5, +0.5, +1.5}`, `code_offset=0.5`), enlarging the Hadamard block from `H_64` to `H_512`, and moving backbone scale groups from `g64` to `g128` frees scale overhead to upgrade **36.75%** of backbone weights (`56/128` fused `vLLM` groups, `78/200` linears) to `W4_g128` at the same `5.55 GiB` footprint, followed by clean-GPTQ/clean-BF16 FP32-master `log_scale` + `RMSNorm`/GDN-gate KL QAT.
- **End-to-end EvalScope (`vLLM`, `temperature=0`, `max_tokens=2048`, `enable_thinking=False`):** **`75.79%`** IFEval (`541`), **`75.20%`** MATH-500 (`500`), **`76.22%`** HumanEval+ (`164`) — **`75.74%` 3-benchmark mean**, gaining **`+12.72 pp`** over V1 (`63.02%`) and beating Unsloth `UD-Q2_K_XL` (`73.12%` @ `3.13 bpw` trunk) by **`+2.62 pp`** while using `0.27 bpw` fewer trunk bits and strictly `2/4/8`-bit kernels.
- **Fidelity vs. BF16:** `0.2417` weighted KL (`0.3809` mean KL) on `probe-v4`, `0.2933` mean KL (`+1.11%` PPL) on the `probe-guard` holdout, `95.47%` / `78.36%` isolated/E2E MTP top-1 agreement, and `98.68%` / `92.38%` isolated/E2E vision top-1 agreement (`100.00%` / `88.89%` on `vision-guard-v2`).

| Variant | Text Backbone Size | Full VLM+MTP Size | 3-Bench EvalScope Mean | Target GPU Tier |
|---|---:|---:|---:|---|
| **BF16 Reference (`Qwen/Qwen3.5-9B`)** | `12.89 GiB` (`16.00 bpw`) | `17.98 GiB` | `84.41%` | `24 GB` GPU |
| [`Qwen3.5-9B-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.5-9B-W4A16-HGPTQ) | `3.43 GiB` (`4.26 bpw`) | `6.67 GiB` | `83.35%` | `10–12 GB` GPU (or `8 GB` shorter context) |
| **`Qwen3.5-9B-W2.6A16-HGPTQ-RCO` V2 (this repo)** | **`2.314 GiB` (`2.86 bpw`)** | **`5.55 GiB`** | **`75.74%`** | **`8 GB` GPU (full VLM + MTP + CUDA graphs)** |
| Unsloth `UD-Q2_K_XL` (GGUF, mixed `Q2/Q3/Q4`) | `2.52 GiB` (`3.13 bpw`) | `3.83 GiB` (text-only) | `73.12%` | `8 GB` GPU |

## Quality vs. BF16 Base Model (`Qwen/Qwen3.5-9B`)

Teacher-forced KL probing and full `vLLM` + EvalScope benchmarks comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Qwen/Qwen3.5-9B`) | **`Qwen3.5-9B-W2.6A16-HGPTQ-RCO` V2** | Notes |
|---|---:|---:|---|
| **Safetensors Size (Full / Pure Text Backbone)** | `17.98 GiB` / `12.89 GiB` | **`5.55 GiB` / `2.314 GiB`** | **`3.24×` full compression** (`5,958,715,384 B`) |
| **EvalScope: IFEval (prompt strict, `n=541`)** ↑ | `84.47%` | **`75.79%`** | `inst_strict 82.85%`, `prompt_loose 79.48%`, `inst_loose 85.61%` (`+15.72 pp` vs V1 `60.07%`) |
| **EvalScope: MATH-500 (`n=500`)** ↑ | `84.60%` | **`75.20%`** | `376 / 500` correct (`avg_output_tokens 996.4`, `+7.20 pp` vs V1 `68.00%`) |
| **EvalScope: HumanEval+ (`n=164`)** ↑ | `84.15%` | **`76.22%`** | `125 / 164` passed (`+15.24 pp` vs V1 `60.98%`) |
| **EvalScope 3-Bench Mean** ↑ | `84.41%` | **`75.74%`** | **`+12.72 pp` vs V1 (`63.02%`), `+2.62 pp` vs `UD-Q2_K_XL` (`73.12%`)** |
| **Text Perplexity (`probe-v4`)** (378 prompts, 840k tokens) ↓ | `3.5754` | **`2.8053` (`-21.54%`)** | `math` `0.0848`, `science` `0.1416`, `if` `0.1523`, `chat` `0.1842` |
| **Text Divergence vs. BF16 (`probe-v4`)** (weighted KL / mean KL) ↓ | `0.0000` | **`0.2417` / `0.3809`** | Full-vocabulary FP32 KL divergence |
| **General & Multilingual Holdout (`probe-guard`)** (76 prompts, 20k tokens) ↓ | `10.5643` | **`10.6818` (`+1.11%`)** | Token mean KL `0.2933` (weighted KL `0.3769`, `no_robots` KL `0.2352`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 | **`95.47%` Isolated Top-1 · `78.36%` E2E Top-1** | Isolated MTP KL `0.0106`, E2E KL `0.3652` |
| **Vision (`vision-probe-v2`)** (96 images, 302 answer tokens) | `100.0%` Top-1 | **`98.68%` Isolated Top-1 · `92.38%` E2E Top-1** | Isolated ViT KL `0.0039`, E2E KL `0.1368` (`vision-guard-v2` Iso `100.00%`, E2E `88.89%`) |

## What Is Quantized

- **Quantized (`319` linear layers total across `model-00001-of-00004.safetensors` .. `model-00004-of-00004.safetensors`)**:
  - **Language Backbone (`200` linears across `128` fused `vLLM` groups, `2.314 GiB` / `2.86 bpw` incl. scales & norms)**:
    - **`78` linears (`56 / 128` groups, `36.75%` of trunk weights) in `W4A16-HGPTQ` (`w4_g128`, `H_512`)**: `19` Full-Attention, `23` GatedDeltaNet Linear-Attention, and `36` MLP projections selected via non-myopic Gumbel-Softmax Riemannian Co-design, calibrated with MSE-initialized Cholesky GPTQ + 2-pass Gauss–Seidel Hessian Coordinate Descent (`H-CD`).
    - **`122` linears (`72 / 128` groups) in `W2A16-HGPTQ` (`w2_g128_sym`, `H_512`)**: Remaining backbone groups quantized with symmetric half-integer 2-bit levels (`{-1.5, -0.5, +0.5, +1.5}`, `observer_kwargs.code_offset = 0.5`) + Cholesky GPTQ + 3-pass `H-CD`, then polished end-to-end from clean GPTQ scales and clean BF16 norms via continuous `log_scale` + FP32-master `RMSNorm`/GDN-gate KL distillation on `data/calib-cascade-v1.jsonl`.
  - **Untied Output Head & MTP Fusion (`2` linears in `W8_g64`)**: `lm_head` (`[248320, 4096]`) and `mtp.fc` (`[4096, 8192]`) quantized to unrotated 8-bit group-64 Cholesky GPTQ (`w8_head`), saving `0.95 GiB` with negligible KL impact.
  - **Native MTP (`7` transformer linears in `W4A16-HGPTQ`)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj` (`head_dim=64`, `group_size=64`).
  - **Vision (`110` linears in `W4A16-HGPTQ`)**: All 27 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`). `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard rotation (`v_vision_fc2`); all other Vision linears use `group_size=64` + `head_dim=128` (`v_vision`).
- **Kept in BF16**:
  - `embed_tokens` (`1.895 GiB`), `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `visual.patch_embed.proj`, norms, and biases.
- **Default sampling** (`generation_config.json`) & **release chat template** (`chat_template.jinja`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Auto-loading `vLLM` plugin (`--trust-remote-code`)**:
  - Ships `vllm_warp_quant.py` wired via `auto_map` so `vllm serve nchapman/Qwen3.5-9B-W2.6A16-HGPTQ-RCO --trust-remote-code` automatically handles symmetric half-integer 2-bit `code_offset=0.5` (`{-1.5, -0.5, +0.5, +1.5}`) in true 2-bit GPU VRAM (`native_2bit`, or set `VLLM_WARP_QUANT_W2_MARLIN=1` for lossless `uint4b8` Marlin/Humming repacking).
  - All `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj` sibling projections share identical bit-widths within every layer so `vLLM` fused `QKVParallelLinear` and `MergedColumnParallelLinear` load directly.
- **Online Input Hadamard Transform (`transform_config`)**:
  - Quantized backbone (`v`, `head_dim=512`), MTP (`v_mtp`, `head_dim=64`), Vision (`v_vision`, `head_dim=128`), and Vision `linear_fc2` (`v_vision_fc2`, `head_dim=16`) linear layers have weights pre-rotated offline along the input dimension ($W' = W H_b / \sqrt{b}$) and apply the matching online block Walsh–Hadamard transform ($x' = x H_b / \sqrt{b}$) at runtime.
