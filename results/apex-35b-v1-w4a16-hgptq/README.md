---
base_model: Apexintelligence-AI/Apex-35B-V1.0
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- quip
- w4a16
- moe
- compressed-tensors
- vllm
---

# Apex-35B-V1.0-W4A16-HGPTQ

4-bit Hadamard + GPTQ (`W4A16`, group size 64) quantization of [Apexintelligence-AI/Apex-35B-V1.0](https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0) (`5105f8f9ed8fd8212a72da546bcbb4f01d2cfd68`) across all 256 routed experts, shared experts, Attention/GDN layers, Vision, and the Native MTP speculative drafter. It compresses the 66.97 GiB BF16 MoE model down to **18.78 GiB** (`3.57×` smaller) so it fits within a single **24 GB GPU**.

- **VRAM & GPU fit:** `18.78 GiB` total (`20,160,901,544 B`, saving `48.19 GiB` vs. BF16) across `30,970` text linears, `775` MTP linears, and `110` vision linears, with `lm_head` and `mtp.fc` in unrotated 8-bit (`W8_g64`).
- **Quantization recipe:** 64×64 Walsh–Hadamard input rotation + MSE-initialized Cholesky GPTQ + 2-pass Hessian Coordinate Descent (`H-CD`) + End-to-End FP32-master `RMSNorm`/GDN KL tuning.
- **Fidelity vs. BF16:** **`0.0701` weighted KL** (`0.1109` mean KL, `−3.80%` PPL) on `probe-v4` (`math` `0.0436`, `instruction_following` `0.0507`, `science` `0.0517`, `chat` `0.0541`), **`0.0625` weighted KL** (`0.0575` mean KL, `−1.65%` PPL) on `probe-guard`, **`95.06%` / `88.30%` isolated/E2E Native MTP Top-1 agreement** (`NLL 2.4931 → 2.4870`), and **`97.35%` / `94.37%` isolated/E2E vision Top-1 agreement** on `vision-probe-v2` (**`99.26%` / `96.30%`** on `vision-guard-v2`).

## Quality vs. BF16 Base Model (`Apexintelligence-AI/Apex-35B-V1.0`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model:

| Evaluation | BF16 Base (`Apexintelligence-AI/Apex-35B-V1.0`) | **`Apex-35B-V1.0-W4A16-HGPTQ`** | Notes |
|---|---:|---:|---|
| **Safetensors Size** | `66.97 GiB` (`71,903,645,408 B`) | **`18.78 GiB` (`20,160,901,544 B`)** | **`3.57×` compression** (`−48.19 GiB`) |
| **Text Perplexity (`probe-v4`)** (378 prompts, 840,187 tokens) ↓ | `5.1325` | **`4.9373` (`−3.80%`)** | Chat, math, science, coding, tool use, multilingual |
| **Text Divergence vs. BF16 (`probe-v4`)** (weighted KL / token mean KL) ↓ | `0.0000` | **`0.0701` / `0.1109`** | Full-vocabulary FP32 KL (`math` `0.0436`, `IF` `0.0507`, `science` `0.0517`) |
| **General & Multilingual Holdout (`probe-guard`)** (76 prompts, 19,534 tokens) ↓ | `13.8270` | **`13.5988` (`−1.65%`)** | Weighted KL `0.0625` (token mean KL `0.0575`) |
| **Native MTP (Speculative Drafter)** (11,583 positions) | `100.0%` Top-1 (`NLL 2.4931`) | **`95.06%` Isolated Top-1 · `88.30%` E2E Top-1** | Isolated MTP KL `0.0121`, E2E KL `0.0758` (`NLL 2.4870`) |
| **Vision (`vision-probe-v2`)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9881` cos sim · `97.35%` Iso / `94.37%` E2E Top-1** | Isolated ViT KL `0.0169`, E2E KL `0.0528` |
| **Vision Guard (`vision-guard-v2`)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | **`0.9912` cos sim · `99.26%` Iso / `96.30%` E2E Top-1** | Isolated ViT KL `0.0035`, E2E KL `0.0221` |

## What Is Quantized

- **Quantized (`31,857` linear layers total)**:
  - **Language (`30,970` W4A16 linears + `1` `W8_g64` linear)**: `130` attention projections (`self_attn`, `linear_attn`) + `120` shared expert projections (`mlp.shared_expert.{gate_proj,up_proj,down_proj}`) + `30,720` routed MoE expert projections (`40` layers × `256` experts × `3` projections) with online block-Hadamard rotation (`head_dim=64`) + Cholesky GPTQ (`group_size=64`, MSE scale search, 2-pass H-CD) + End-to-End FP32-master Norm/GDN KL QAT, plus `lm_head` in unrotated `W8_g64`.
  - **Native MTP (`775` W4A16 linears + `1` `W8_g64` linear)**: `4` `self_attn` projections + `3` `mlp.shared_expert` projections + `768` routed MoE expert projections (`256` experts × `3` projections) (`head_dim=64`, `group_size=64`), plus `mtp.fc` in unrotated `W8_g64`.
  - **Vision (`110` linears)**: All 27 ViT blocks (`visual.blocks.*.{attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`) plus the 2 spatial merger projections (`visual.merger.{linear_fc1,linear_fc2}`). `visual.blocks.*.mlp.linear_fc2` (`K=4304`) uses per-channel Cholesky GPTQ + `head_dim=16` online Hadamard rotation (`v_vision_fc2`); all other Vision linears use `group_size=64` + `head_dim=64`.
- **Kept in BF16**:
  - `embed_tokens`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`, `mlp.gate`, `mlp.shared_expert_gate`, `visual.patch_embed.proj`, norms, and biases.
- **Static Hot-Expert Cache Rankings**:
  - `static_hot_cache_rankings.json` (and `runtime/static_hot_cache_rankings.json` + `runtime/expert_routing_profile.json`) provides pre-profiled expert rankings (`0..255` ordered hottest-first per layer `"0".."39"`).
- **Default sampling** (`generation_config.json`): `temperature 1.0, top_p 0.95, top_k 20` (`eos_token_id: [248046, 248044]`, `pad_token_id: 248044`).

## Runtime & vLLM Notes

- **Zero-patch stock `vLLM` + Hugging Face `compressed-tensors` compatibility**:
  - Ships `vllm_warp_quant.py` (`auto_map`) with `vLLM 0.34+` `tp>=1` `SharedWeightParameter` and Vision `linear_fc2` boundary-straddling Hadamard support.
  - Supports native `qwen3_next_mtp` / `mtp` speculative decoding in `vLLM` (`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`).
- **Online Input Hadamard Rotation Across Attention, Shared Experts, and Routed MoE Experts**:
  - Every quantized linear in Text (`30,970` linears) and Native MTP (`775` linears) — including `self_attn`, `linear_attn`, `mlp.shared_expert.{gate_proj,up_proj,down_proj}`, and `mlp.experts.0..255.{gate_proj,up_proj,down_proj}` — has its weights pre-rotated offline along the input dimension ($W' = W H_{64} / 8$) prior to Cholesky GPTQ quantization (`group_size=64`, `actorder="static"`).
  - At runtime, apply block-64 Walsh–Hadamard transform ($x \mapsto x H_{64} / 8$) to the input activations before `gate_proj`/`up_proj` (`K=2048`) and to the post-SwiGLU intermediate activations before `down_proj` (`K=512`), matching `quantization_config.transform_config.config_groups.v`.
  - Only `mlp.gate` and `mlp.shared_expert_gate` remain in BF16 (listed in `quantization_config.ignore`).
