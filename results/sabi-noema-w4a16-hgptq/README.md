---
base_model: sabicap/sabi-noema
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

# sabi-noema-W4A16-HGPTQ

4-bit Hadamard + GPTQ (`W4A16`, group size 64) quantization of [sabicap/sabi-noema](https://huggingface.co/sabicap/sabi-noema) (`2908c2d59081522a6cbfdb60ef0f848508295667`) with **salvaged Native Multi-Token Prediction (`MTP`) speculative decoding** grafted from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`), End-to-End FP32-master `Scale/Norm` KL QAT, and unrotated `W8_g64` `lm_head` + `mtp.fc`. It compresses the 54.71 GB BF16 model down to **16.14 GiB** (`17.33 GB`, `3.16×` smaller; `15.70 GiB` text-only) so the full 27B multimodal model and restored Native MTP drafter fit on a single **24 GB GPU** (RTX 3090 / 4090 / 5090).

- **Restored Native MTP speculative decoding (`90.48%` E2E Top-1 agreement):** `sabicap/sabi-noema` was released without `mtp.*` weights (`mtp_num_hidden_layers: 0`). Because `sabi-noema` stays tightly in `Qwen/Qwen3.8-27B`'s representation subspace (`embed_tokens.weight` relative Frobenius diff `0.0857%`, `lm_head.weight` `0.0946%`, `norm.weight` `0.0001%`), this checkpoint grafts `mtp.*` from `Qwen/Qwen3.8-27B`, calibrates `mtp.layers.0.*` (`W4_g64` + $H_{64}$) and `mtp.fc` (`W8_g64`) directly on `sabi-noema` Layer-63 hidden states, and sets `"mtp_num_hidden_layers": 1` — achieving **96.63% isolated Top-1 agreement** (`mean_kl = 0.0087`) and **90.48% E2E Top-1 agreement** (`mean_kl = 0.0912`, `NLL 2.8937 → 2.8951`) against `sabi-noema`'s main `lm_head`.
- **VRAM & GPU fit:** `16.14 GiB` full VLM+MTP (`17,329,742,096 B`; `15.70 GiB` text-only), leaving ~7.8 GiB on a 24 GB card for long-context KV cache, vision tokens, MTP speculative decoding, and CUDA graphs.
- **Fidelity vs. BF16 (`sabicap/sabi-noema`):** **`0.1105` weighted KL** (`0.2710` mean KL, `−2.13%` PPL) on `probe-v4` (`chat` KL `0.0236`, `science` KL `0.0691`, `math` KL `0.0830`, `instruction_following` KL `0.0963`), **`0.0675` weighted KL** (`0.1023` mean KL, `−12.42%` PPL) on the `probe-guard` holdout, and **`99.01%` / `92.05%` isolated/E2E vision Top-1 agreement** on `vision-probe-v2` (`99.26%` / `94.07%` on `vision-guard-v2`).

## Quality vs. BF16 Base Model (`sabicap/sabi-noema`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 `sabicap/sabi-noema` base model:

| Evaluation | BF16 Base (`sabicap/sabi-noema`) | **`sabi-noema-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `50.96 GiB` (`54,713,474,016 B`, no MTP) | **`16.14 GiB` (`17,329,742,096 B`, with MTP)** | **`3.16×` compression** (`−34.82 GiB`) |
| **Text Perplexity (`probe-v4`)** (378 prompts, 851,151 tokens) ↓ | `9.6074` | **`9.4030` (`−2.13%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16 (`probe-v4`)** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.1105` / `0.2710`** | 378 prompts, 851k tokens (`chat` `0.0236`, `science` `0.0691`, `math` `0.0830`) |
| **General & Multilingual Holdout (`probe-guard`)** (76 prompts, 22,406 tokens) ↓ | `329.8565` | **`288.8898` (`−12.42%`)** | Weighted KL `0.0675` (token mean KL `0.1023`) |
| **Salvaged Native MTP (`Qwen3.8-27B` donor)** (12,267 positions) | `100.0%` Top-1 (`NLL 2.8937`) | **`96.63%` Isolated Top-1 · `90.48%` E2E Top-1** | Isolated MTP KL `0.0087`, E2E KL `0.0912` (`NLL 2.8951`) |
| **Vision (`vision-probe-v2`)** (96 images, 302 answer tokens) | `100.0%` Top-1 | **`99.01%` Isolated Top-1 · `92.05%` E2E Top-1** | Isolated ViT KL `0.0079`, E2E KL `0.1609` (`vision-guard-v2` Iso Top-1 `99.26%`, E2E Top-1 `94.07%`) |

## What Is Quantized

- **Quantized (`519` linear layers total across `model-00001-of-00012.safetensors` .. `model-00012-of-00012.safetensors`)**:
  - **Language Backbone (`400` linears across `256` fused `vLLM` groups in `W4A16-HGPTQ`)**: All 64 decoder layers (`self_attn.{q,k,v,o}_proj` on the 16 full-attention layers, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` on the 48 GatedDeltaNet layers, and `mlp.{gate,up,down}_proj` on all 64 layers) rotated with $H_{64}$, quantized with MSE-initialized Cholesky GPTQ (`group_size=64`, `actorder="static"`) + 2-pass Gauss–Seidel Hessian Coordinate Descent (`H-CD`), and polished end-to-end via FP32-master `RMSNorm` KL distillation on `data/calib-cascade-v1.jsonl`.
  - **Untied Output Head & MTP Fusion (`2` linears in `W8_g64`)**: `lm_head` (`[248320, 5120]`) and salvaged `mtp.fc` (`[5120, 10240]`) quantized to unrotated 8-bit group-64 Cholesky GPTQ (`w8_head`), saving `1.20 GiB` (`1,230.8 MiB`) with zero online Hadamard overhead.
  - **Salvaged Native MTP (`7` transformer linears in `W4A16-HGPTQ`)**: `mtp.layers.0.self_attn.{q,k,v,o}_proj` and `mtp.layers.0.mlp.{gate,up,down}_proj` (`head_dim=64`, `group_size=64`), grafted from `Qwen/Qwen3.8-27B` and calibrated on `sabi-noema` Layer-63 hidden states.
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
