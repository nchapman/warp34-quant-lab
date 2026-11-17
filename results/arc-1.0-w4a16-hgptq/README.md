---
base_model:
- rayline-ai/ARC-1.0
- Qwen/Qwen3.5-0.8B
library_name: compressed-tensors
license: other
pipeline_tag: feature-extraction
tags:
- qwen3_5
- arc-1.0
- router
- decision-model
- compressed-tensors
- w4a16
- hadamard
- gptq
- warp34
---

# ARC-1.0-W4A16-HGPTQ

`W4A16-HGPTQ` (`g64-h64` online Walsh–Hadamard input rotation + MSE-initialized Cholesky GPTQ + 2-pass coordinate descent) quantization of [`rayline-ai/ARC-1.0`](https://huggingface.co/rayline-ai/ARC-1.0) (`f16e93f2581b3690938e47206881c7926e1ff341`) for `warp34` and `rayline-arc-decide`.

## Highlights

- **Direct `yarn-f4` RoPE calibration (`layers.0..11` + `question_head/qh4in-full-e3` `layers.12..23`)**: Calibrated directly under `ARC-1.0`'s YaRN (`factor=4.0`, `original_max_position_embeddings=262144`) RoPE on 256 canonical `<|im_start|>{role}\n...<|im_end|>\n` multi-turn role-block trajectories (`data/calib-cascade-v1.jsonl` + `data/traj-v1.jsonl`), improving `rayline-arc-decide` routing parity by **+11.1 percentage points** (`92.6%` vs `81.5%`) and cutting action-score MAE by **34%** (`0.0435` vs `0.0662`) compared to reusing `default`-RoPE `Qwen3.5-0.8B-W4A16-HGPTQ` weights.
- **Unmasked local Qwen routing slots (`[53..55]`)**: Unmasks the three trained Qwen model slots in `runtime/arc-runtime.json`, `config.json`, and `provenance.json` so `rayline-arc-decide` and `warp34` can route across both cloud frontier models and local Qwen deployments with dynamic thinking-effort controls:
  - `models[53]` (`withheld-model-42`) → `qwen3.8-flash-next` (`wire: "qwen/qwen3.8-flash-next"`, 8 served arms across `default` and `high` effort controls)
  - `models[54]` (`withheld-model-43`) → `qwen3.5-9b` (`wire: "qwen/qwen3.5-9b"`, 16 served arms across `none`, `down`, `default`, and `up` steering controls)
  - `models[55]` (`withheld-model-44`) → `qwen3.8-27b` (`wire: "qwen/qwen3.8-27b"`, 16 served arms across `none`, `down`, `default`, and `up` steering controls)
- **Full component coverage**:
  - **Base 12-layer encoder (`model.language_model.layers.0..11`)**: `W4A16-HGPTQ` (`g64-h64`), with `embed_tokens`, RMSNorms, and GatedDeltaNet scalars (`A_log`, `dt_bias`, `conv1d`, `in_proj_a`, `in_proj_b`) preserved in exact `BF16`/`FP32`.
  - **Vision tower (`model.visual.*`)**: `W4A16-HGPTQ` (`g64-h64` on `patch_merger`/`attn`/`linear_fc1`, per-channel `h16` on `linear_fc2`).
  - **Stage-1 / Stage-2 & Signal heads (`runtime/heads.safetensors`, `signals.*`)**: Preserved bit-for-bit in `FP32`/`FP64` inside both `model.safetensors` and `runtime/heads.safetensors`.
  - **Question head (`question_head/qh4in-full-e3`)**: Upper 12 layers (`model.language_model.layers.12..23` in `upper.safetensors`) quantized to `W4A16-HGPTQ` (`g64-h64`), with `pointer.safetensors` preserved in `FP32`.

## Fidelity & Parity Receipts

| Evaluation Suite | Metric | `ARC-1.0-W4A16-HGPTQ` (vs BF16 `ARC-1.0`) | Reused `0.8B` (`default` RoPE) Baseline |
| :--- | :--- | :---: | :---: |
| **`goldens.jsonl` (54 cases, text + vision)** | Layer-11 Readout Cosine (`mean` / `min`) | **`0.9822` / `0.9693`** | `0.9685` / `0.9163` |
| **`rayline-arc-decide` (`runtime/prices.json`)** | Top-1 Action Agreement (`54` goldens) | **`50 / 54` (`92.6%`)** | `44 / 54` (`81.5%`) |
| **`rayline-arc-decide` (`runtime/prices.json`)** | Action Utility Score MAE | **`0.0435`** | `0.0662` |
| **`goldens.signals.jsonl` (54 cases)** | All-Signal Probability MAE (`mean`) | **`0.0113`** | `0.0197` |
| **`goldens.signals.jsonl` (`pii.payload`)** | Binary Classification Accuracy (`15` classes) | **`808 / 810` (`99.75%`)** | `807 / 810` (`99.63%`) |
| **`question_head/qh4in-full-e3` (`layers.12..23`)** | Layer-24 Query Readout Cosine (`mean`) | **`0.9950`** | — |
| **`question_head/qh4in-full-e3` (`pointer`)** | Pointer Option Probability MAE | **`0.0286`** | — |

## Reproduction

```bash
uv run --no-sync warp-quant quantize --model arc-1.0 --output-dir ./out/ARC-1.0-W4A16-HGPTQ
uv run --no-sync warp-quant probe --model arc-1.0 --quant ./out/ARC-1.0-W4A16-HGPTQ
```
