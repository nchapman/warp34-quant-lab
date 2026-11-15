---
base_model: ornith-ai/Ornith-1.5-9B
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- quip
- w2a16
- w4a16
- compressed-tensors
- vllm
---

# Ornith-1.5-9B-W2.6A16-HGPTQ-RCO

Mixed 2-bit / 4-bit Hadamard + GPTQ (`W2.6A16`, group size 64) quantization of [ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B) (`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`, built on `Qwen/Qwen3.8-9B`) produced with Riemannian Co-design (`HGPTQ-RCO`). It compresses the 17.98 GiB BF16 model down to **5.54 GiB** (3.24× smaller; 5.11 GiB text-only) so the full multimodal model and Native MTP speculative drafter fit on a single **8 GB GPU** (RTX 3070 / 4060 / 4060 Ti 8GB / 5060).

- **VRAM & GPU fit:** 5.54 GiB full VLM+MTP (`5,951,369,048 B`; 5.11 GiB text-only), leaving ~2.45 GiB on an 8 GB card for KV cache, vision tokens, speculative decoding, and CUDA graphs.
- **Quantization recipe:** Upgrades the top 30% most sensitive fused `vLLM` backbone groups (12/16 Full-Attention, 17/48 GDN Linear-Attention, 21/64 MLP — 50/128 groups, 69/200 linear matrices) to `W4_g64`, keeps Vision and Native MTP transformer blocks in `W4_g64` (`lm_head` and `mtp.fc` in `W8_g64`), and polishes the remaining 131 `W2_g64` scales + FP32-master `RMSNorm`/GDN gates end-to-end.
- **Fidelity vs. BF16:** 0.3550 weighted KL (0.5041 mean KL) on `probe-v4` (`math` KL `0.1435`, `science` KL `0.2440`, `instruction_following` KL `0.2649`, `chat` KL `0.2962`), 0.4739 mean KL (`−5.97%` PPL) on the `probe-guard` holdout, 94.68% / 68.57% isolated/E2E MTP top-1 agreement, and 98.68% / 88.41% isolated/E2E vision top-1 agreement (`97.78%` / `88.15%` on `vision-guard-v2`).

| Variant | Avg Backbone Bits | Full VLM+MTP Size | Text-Only Size | Target GPU Tier |
|---|---:|---:|---:|---|
| **`Ornith-1.5-9B-W2.6A16-HGPTQ-RCO` (this repo)** | **`2.60` (`30%` `W4`)** | **`5.54 GiB`** | **`5.11 GiB`** | **`8 GB` GPU (full VLM + MTP + CUDA graphs)** |
| [`Qwen3.8-9B-W2.6A16-HGPTQ-RCO`](https://huggingface.co/nchapman/Qwen3.8-9B-W2.6A16-HGPTQ-RCO) | `2.60` (`30%` `W4`) | `5.54 GiB` | `5.11 GiB` | `8 GB` GPU (base `Qwen3.8-9B` RCO reference) |

## Quality vs. BF16 Base Model (`ornith-ai/Ornith-1.5-9B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`ornith-ai/Ornith-1.5-9B`) | **`Ornith-1.5-9B-W2.6A16-HGPTQ-RCO`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `17.98 GiB` (`19,306,310,880 B`) | **`5.54 GiB` (`5,951,369,048 B`)** | **`3.24×` compression** (`−12.44 GiB`) |
| **Text Perplexity (`probe-v4`)** (378 prompts, 842,510 tokens) ↓ | `6.6829` | **`3.8334` (`−42.64%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16 (`probe-v4`)** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.3550` / `0.5041`** | 378 prompts, 842.5k tokens (`math` `0.1435`, `science` `0.2440`, `if` `0.2649`, `chat` `0.2962`) |
| **General & Multilingual Holdout (`probe-guard`)** (76 prompts, 19,570 tokens) ↓ | `15.0115` | **`14.1150` (`−5.97%`)** | Token mean KL `0.4739` (weighted KL `0.6147`, `no_robots` KL `0.3671`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 2.8335`) | **`94.68%` Isolated Top-1 · `68.57%` E2E Top-1** | Isolated MTP KL `0.0127`, E2E KL `0.6938` |
| **Vision (`vision-probe-v2`)** (96 images, 302 answer tokens) | `100.0%` Top-1 | **`98.68%` Isolated Top-1 · `88.41%` E2E Top-1** | Isolated ViT KL `0.0077`, E2E KL `0.2298` (`vision-guard-v2` Iso Top-1 `97.78%`, E2E Top-1 `88.15%`) |

## What Is Quantized

- **Quantized (`319` linear layers total across `model-00001-of-00004.safetensors` .. `model-00004-of-00004.safetensors`)**:
  - **Language Backbone (`200` linears across `128` fused `vLLM` groups, `2.60` bits/weight)**:
    - **`69` linears (`50 / 128` groups) in `W4A16-HGPTQ` (`w4_g64`)**: `12 / 16` (`75.0%`) Full-Attention groups, `17 / 48` (`35.4%`) GatedDeltaNet Linear-Attention groups, and `21 / 64` (`32.8%`) MLP groups selected via non-myopic Gumbel-Softmax Riemannian Co-design (`59` group swaps vs. greedy Hessian sensitivity, reducing pre-QAT validation KL from `0.4949 -> 0.4350`), calibrated with MSE-initialized Cholesky GPTQ + 2-pass Gauss–Seidel Hessian Coordinate Descent (`H-CD`).
    - **`131` linears (`78 / 128` groups) in `W2A16-HGPTQ` (`w2_g64`)**: Remaining backbone groups quantized with symmetric 2-bit (`{-2, -1, 0, 1}`) Hadamard + Cholesky GPTQ + 3-pass `H-CD` and polished end-to-end via continuous `log_scale` + FP32-master `RMSNorm`/GDN-gate KL distillation on `data/calib-cascade-v1.jsonl` (reducing post-allocation validation KL from `0.4350 -> 0.3732`).
  - **Untied Output Head & MTP Fusion (`2` linears in `W8_g64`)**: `lm_head` (`[248320, 4096]`) and `mtp.fc` (`[4096, 8192]`) quantized to unrotated 8-bit group-64 Cholesky GPTQ (`w8_head`), saving `0.95 GiB` (`970.7 MiB`) with zero online Hadamard overhead.
  - **Native MTP (`7` transformer linears in `W4A16-HGPTQ`)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj` (`head_dim=64`, `group_size=64`).
  - **Vision (`110` linears in `W4A16-HGPTQ`)**: All 27 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`). `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard rotation (`v_vision_fc2`); all other Vision linears use `group_size=64` + `head_dim=64`.
- **Kept in BF16**:
  - `embed_tokens`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `visual.patch_embed.proj`, norms, and biases.
- **Default sampling** (`generation_config.json`) & **release chat template** (`chat_template.jinja`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Zero-patch stock `vLLM` + Hugging Face `compressed-tensors` compatibility**:
  - Uses standard `pack-quantized` int32 shards with `w2_g64` (`num_bits: 2`), `group_0` (`num_bits: 4`), `vision_fc2_channel` (`num_bits: 4`), and `w8_head` (`num_bits: 8`) config groups.
  - All `q_proj`/`k_proj`/`v_proj` and `gate_proj`/`up_proj` sibling projections share identical bit-widths within every layer so `vLLM` fused `QKVParallelLinear` and `MergedColumnParallelLinear` load directly without patches.
- **Online Input Hadamard Transform (`transform_config`)**:
  - Quantized backbone, MTP, and Vision linear layers have weights pre-rotated offline along the input dimension ($W' = W H_b / \sqrt{b}$, where $H_b$ is the deterministic $b \times b$ Sylvester Walsh–Hadamard matrix; $b=64$ for `v` and $b=16$ for `v_vision_fc2`) prior to Cholesky GPTQ quantization.
  - At runtime, an online block Walsh–Hadamard transform ($x' = x H_b / \sqrt{b}$) is applied to input activations before the GEMM (`HadaCore` / `CompressedTensorsLinearTransformMethod`).
