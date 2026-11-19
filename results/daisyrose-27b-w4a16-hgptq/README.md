---
base_model: RadixArk/DaisyRose-27B
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- quip
- w4a16
- compressed-tensors
- vllm
---

# DaisyRose-27B-W4A16-HGPTQ

4-bit Hadamard + GPTQ (`W4A16`, group size 64) quantization of [RadixArk/DaisyRose-27B](https://huggingface.co/RadixArk/DaisyRose-27B) (`293a7a2ab54aaea78098ee917c546390667521dc`) with full Native Multi-Token Prediction (`MTP`) speculative decoding, Vision (`ViT` + spatial merger) quantization, End-to-End FP32-master `Scale/Norm` KL QAT, and unrotated `W8_g64` `lm_head` + `mtp.fc`. It compresses the 55.56 GB BF16 model down to **16.14 GiB** (`17.33 GB`, `3.21×` smaller; `15.70 GiB` text-only) so the full 27B multimodal model and Native MTP drafter fit on a single **24 GB GPU** (RTX 3090 / 4090 / 5090).

- **VRAM & GPU fit:** `16.14 GiB` full VLM+MTP (`17,329,725,200 B`; `15.70 GiB` text-only), leaving ~7.8 GiB on a 24 GB card for long-context KV cache, vision tokens, MTP speculative decoding, and CUDA graphs.
- **Fidelity vs. BF16 (`RadixArk/DaisyRose-27B`):** **`0.1110` weighted KL** (`0.2864` mean KL, `−3.12%` PPL) on `probe-v4` (`chat` KL `0.0230`, `instruction_following` KL `0.0412`, `science` KL `0.0751`, `math` KL `0.0802`), **`0.0751` weighted KL** (`0.0548` mean KL, `−7.19%` PPL) on the `probe-guard` holdout, **`96.30%` / `89.38%` isolated/E2E Native MTP Top-1 agreement**, and **`99.01%` / `95.03%` isolated/E2E vision Top-1 agreement** on `vision-probe-v2` (`98.52%` / `94.81%` on `vision-guard-v2`).

## Quality vs. BF16 Base Model (`RadixArk/DaisyRose-27B`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 `RadixArk/DaisyRose-27B` base model:

| Evaluation | BF16 Base (`RadixArk/DaisyRose-27B`) | **`DaisyRose-27B-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `51.75 GiB` (`55,562,855,904 B`) | **`16.14 GiB` (`17,329,725,200 B`)** | **`3.21×` compression** (`−35.61 GiB`) |
| **Text Perplexity (`probe-v4`)** (378 prompts, 852,269 tokens) ↓ | `8.5603` | **`8.2929` (`−3.12%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16 (`probe-v4`)** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.1110` / `0.2864`** | 378 prompts, 852k tokens (`chat` `0.0230`, `instruction_following` `0.0412`, `science` `0.0751`) |
| **General & Multilingual Holdout (`probe-guard`)** (76 prompts, 22,746 tokens) ↓ | `60.2295` | **`55.9007` (`−7.19%`)** | Weighted KL `0.0751` (token mean KL `0.0548`) |
| **Native MTP Speculative Head** (12,339 positions) | `100.0%` Top-1 (`NLL 2.9731`) | **`96.30%` Isolated Top-1 · `89.38%` E2E Top-1** | Isolated MTP KL `0.0092`, E2E KL `0.1720` (`NLL 2.9460`) |
| **Vision (`vision-probe-v2`)** (96 images, 302 answer tokens) | `100.0%` Top-1 | **`99.01%` Isolated Top-1 · `95.03%` E2E Top-1** | Isolated ViT KL `0.0102`, E2E KL `0.1126` (`vision-guard-v2` Iso Top-1 `98.52%`, E2E Top-1 `94.81%`) |

## What Is Quantized

- **Quantized (`519` linear layers total across `model-00001-of-00018.safetensors` .. `model-00018-of-00018.safetensors`)**:
  - **Language Backbone (`400` linears across `256` fused `vLLM` groups in `W4A16-HGPTQ`)**: All 64 decoder layers (`self_attn.{q,k,v,o}_proj` on the 16 full-attention layers, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` on the 48 GatedDeltaNet layers, and `mlp.{gate,up,down}_proj` on all 64 layers) rotated with $H_{64}$, quantized with MSE-initialized Cholesky GPTQ (`group_size=64`, `actorder="static"`) + 2-pass Gauss–Seidel Hessian Coordinate Descent (`H-CD`), and polished end-to-end via FP32-master `RMSNorm` KL distillation on `data/calib-cascade-v1.jsonl`.
  - **Untied Output Head & MTP Fusion (`2` linears in `W8_g64`)**: `lm_head` (`[248320, 5120]`) and `mtp.fc` (`[5120, 10240]`) quantized to unrotated 8-bit group-64 Cholesky GPTQ (`w8_head`), saving `1.20 GiB` (`1,230.8 MiB`) with zero online Hadamard overhead.
  - **Native MTP (`7` transformer linears in `W4A16-HGPTQ`)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj` (`head_dim=64`, `group_size=64`), preserving native `mtp` speculative decoding.
  - **Vision (`110` linears in `W4A16-HGPTQ`)**: All 27 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`). `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard rotation (`v_vision_fc2`); all other Vision linears use `group_size=64` + `head_dim=64`.
- **Kept in BF16**:
  - `embed_tokens`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `visual.patch_embed.proj`, norms, and biases.
- **Default sampling** (`generation_config.json`) & **release chat template** (`chat_template.jinja`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Zero-patch stock `vLLM` + Hugging Face `compressed-tensors` compatibility**:
  - Uses standard `pack-quantized` int32 shards with `group_0` (`num_bits: 4`), `vision_fc2_channel` (`num_bits: 4`), and `w8_head` (`num_bits: 8`) config groups.
  - Ships `vllm_warp_quant.py` (`auto_map`) with `vLLM 0.34+` `tp>=1` `SharedWeightParameter` and Vision `linear_fc2` boundary-straddling Hadamard support.
  - Supports native `qwen3_next_mtp` / `mtp` speculative decoding in `vLLM` (`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`).
- **Online Input Hadamard Transform (`transform_config`)**:
  - Quantized backbone, MTP, and Vision linear layers have weights pre-rotated offline along the input dimension ($W' = W H_b / \sqrt{b}$, where $H_b$ is the deterministic $b \times b$ Sylvester Walsh–Hadamard matrix; $b=64$ for `v` and $b=16$ for `v_vision_fc2`) prior to Cholesky GPTQ quantization.
  - At runtime, an online block Walsh–Hadamard transform ($x' = x H_b / \sqrt{b}$) is applied to input activations before the GEMM (`HadaCore` / `CompressedTensorsLinearTransformMethod`).
