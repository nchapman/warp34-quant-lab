# Ornith-1.5-9B-W2.6A16-HGPTQ-RCO

Mixed 2-bit / 4-bit Hadamard + GPTQ (`W2.6A16-HGPTQ-RCO` V2: `w2_g128_sym` with symmetric half-integer `code_offset=0.5`, `w4_g128`, `H_512` Walsh–Hadamard rotation) quantization of [ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) (`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, built on `Qwen/Qwen3.8-9B`) produced with Riemannian Co-design (`HGPTQ-RCO`). It compresses the 17.98 GiB BF16 model down to **5.55 GiB** (3.24× smaller; **2.314 GiB** pure text backbone / `2.86 bpw`) so the full multimodal model and Native MTP speculative drafter fit comfortably on a single **8 GB GPU** (RTX 3070 / 4060 / 4060 Ti 8GB / 5060).

- **VRAM & GPU fit:** **5.55 GiB** full VLM+MTP (`5,958,709,344 B`; **2.314 GiB** pure text backbone, **5.19 GiB** text-only incl. `embed_tokens` + `lm_head`), leaving ~2.45 GiB on an 8 GB card for KV cache, vision tokens, speculative decoding, and CUDA graphs.
- **V2 (`D1`) quantization recipe:** Switching `W2` to symmetric half-integer levels (`{-1.5, -0.5, +0.5, +1.5}`, `code_offset=0.5`), enlarging the Hadamard block from `H_64` to `H_512`, and moving backbone scale groups from `g64` to `g128` upgrades **36.75%** of backbone parameters (`57/128` fused `vLLM` groups, `79/200` linears — attention groups hold fewer parameters than MLP groups) to `W4_g128` at the same `5.55 GiB` footprint, followed by clean-GPTQ/clean-BF16 FP32-master `log_scale` + `RMSNorm`/GDN-gate KL QAT.
- **Fidelity vs. BF16:** `0.3258` weighted KL (`0.5395` mean KL; `-0.0292` vs V1 `0.3550`) on `probe-v4`, `0.4067` mean KL (`-0.0672` vs V1 `0.4739`, `no_robots` `0.2976`) on the `probe-guard` holdout, `94.59%` / `69.90%` isolated/E2E MTP top-1 agreement (`+1.33 pp` E2E vs V1), and `98.01%` / `91.39%` isolated/E2E vision top-1 agreement (`97.04%` / `91.85%` on `vision-guard-v2`, `+3.70 pp` E2E vs V1).

| Variant | Text Backbone Size | Full VLM+MTP Size | `probe-v4` wKL ↓ | `probe-guard` mean KL ↓ | Target GPU Tier |
|---|---:|---:|---:|---:|---|
| **`Ornith-1.5-9B-W2.6A16-HGPTQ-RCO` V2 (this repo)** | **`2.314 GiB` (`2.86 bpw`)** | **`5.55 GiB`** | **`0.3258`** | **`0.4067`** | **`8 GB` GPU (full VLM + MTP + CUDA graphs)** |
| `Ornith-1.5-9B-W2.6A16-HGPTQ-RCO` V1 (`w2_g64`, `H_64`) | `2.31 GiB` | `5.54 GiB` | `0.3550` | `0.4739` | `8 GB` GPU |
| [`Qwen3.5-9B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.5-9B-W2.6A16-HGPTQ-RCO) V2 | `2.314 GiB` (`2.86 bpw`) | `5.55 GiB` | `0.2417` | `0.2933` | `8 GB` GPU (`75.74%` EvalScope mean) |

## Quality vs. BF16 Base Model (`ornith-ai/Ornith-1.5-9B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`ornith-ai/Ornith-1.5-9B`) | **`Ornith-1.5-9B-W2.6A16-HGPTQ-RCO` V2** | Notes |
|---|---:|---:|---|
| **Safetensors Size (Full / Pure Text Backbone)** | `17.98 GiB` / `12.89 GiB` | **`5.55 GiB` / `2.314 GiB`** | **`3.24×` full compression** (`5,958,709,344 B`) |
| **Text Perplexity (`probe-v4`)** (378 prompts, 842,510 tokens) ↓ | `6.6829` | **`3.7717` (`−43.56%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16 (`probe-v4`)** (weighted KL / mean KL) ↓ | `0.0000` | **`0.3258` / `0.5395`** | 378 prompts (`math` `0.1244`, `science` `0.2124`, `if` `0.2268`, `chat` `0.2491`) |
| **General & Multilingual Holdout (`probe-guard`)** (76 prompts, 19,570 tokens) ↓ | `15.0115` | **`14.2715` (`−4.93%`)** | Token mean KL `0.4067` (weighted KL `0.5508`, `no_robots` KL `0.2976`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 | **`94.59%` Isolated Top-1 · `69.90%` E2E Top-1** | Isolated MTP KL `0.0126`, E2E KL `0.6520` |
| **Vision (`vision-probe-v2`)** (96 images, 302 answer tokens) | `100.0%` Top-1 | **`98.01%` Isolated Top-1 · `91.39%` E2E Top-1** | Isolated ViT KL `0.0087`, E2E KL `0.1761` (`vision-guard-v2` Iso `97.04%`, E2E `91.85%`) |

## What Is Quantized

- **Quantized (`319` linear layers total across `model-00001-of-00004.safetensors` .. `model-00004-of-00004.safetensors`)**:
  - **Language Backbone (`200` linears across `128` fused `vLLM` groups, `2.314 GiB` / `2.86 bpw` incl. scales & norms)**:
    - **`79` linears (`57 / 128` groups, `36.75%` of trunk weights) in `W4A16-HGPTQ` (`w4_g128`, `H_512`)**: `20` Full-Attention, `24` GatedDeltaNet Linear-Attention, and `35` MLP projections selected via non-myopic Gumbel-Softmax Riemannian Co-design, calibrated with MSE-initialized Cholesky GPTQ + 2-pass Gauss–Seidel Hessian Coordinate Descent (`H-CD`).
    - **`121` linears (`71 / 128` groups) in `W2A16-HGPTQ` (`w2_g128_sym`, `H_512`)**: Remaining backbone groups quantized with symmetric half-integer 2-bit levels (`{-1.5, -0.5, +0.5, +1.5}`, `observer_kwargs.code_offset = 0.5`) + Cholesky GPTQ + 3-pass `H-CD`, then polished end-to-end from clean GPTQ scales and clean BF16 norms via continuous `log_scale` + FP32-master `RMSNorm`/GDN-gate KL distillation on `data/calib-cascade-v1.jsonl`.
  - **Untied Output Head & MTP Fusion (`2` linears in `W8_g64`)**: `lm_head` (`[248320, 4096]`) and `mtp.fc` (`[4096, 8192]`) quantized to unrotated 8-bit group-64 Cholesky GPTQ (`w8_head`), saving `0.95 GiB` with zero online Hadamard overhead.
  - **Native MTP (`7` transformer linears in `W4A16-HGPTQ`)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj` (`head_dim=64`, `group_size=64`).
  - **Vision (`110` linears in `W4A16-HGPTQ`)**: All 27 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`). `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard rotation (`v_vision_fc2`); all other Vision linears use `group_size=64` + `head_dim=128` (`v_vision`).
- **Kept in BF16**:
  - `embed_tokens` (`1.895 GiB`), `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `visual.patch_embed.proj`, norms, and biases.
- **Default sampling** (`generation_config.json`) & **release chat template** (`chat_template.jinja`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Auto-loading `vLLM` plugin (`--trust-remote-code`)**:
  - Ships `vllm_warp_quant.py` wired via `auto_map` so `vllm serve nchapman/Ornith-1.5-9B-W2.6A16-HGPTQ-RCO --trust-remote-code` automatically handles symmetric half-integer 2-bit `code_offset=0.5` (`{-1.5, -0.5, +0.5, +1.5}`) in true 2-bit GPU VRAM (`native_2bit`, or set `VLLM_WARP_QUANT_W2_MARLIN=1` for lossless `uint4b8` Marlin/Humming repacking).
  - All `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj` sibling projections share identical bit-widths within every layer so `vLLM` fused `QKVParallelLinear` and `MergedColumnParallelLinear` load directly.
- **Online Input Hadamard Transform (`transform_config`)**:
  - Quantized backbone (`v`, `head_dim=512`), MTP (`v_mtp`, `head_dim=64`), Vision (`v_vision`, `head_dim=128`), and Vision `linear_fc2` (`v_vision_fc2`, `head_dim=16`) linear layers have weights pre-rotated offline along the input dimension ($W' = W H_b / \sqrt{b}$) and apply the matching online block Walsh–Hadamard transform ($x' = x H_b / \sqrt{b}$) at runtime.
