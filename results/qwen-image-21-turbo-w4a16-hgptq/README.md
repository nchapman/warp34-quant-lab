---
license: apache-2.0
base_model:
  - Qwen/Qwen-Image-2.1-Turbo
base_model_relation: quantized
library_name: diffusers
pipeline_tag: text-to-image
tags:
  - diffusers
  - qwen-image-2.1
  - text-to-image
  - image-to-image
  - compressed-tensors
  - gptq
  - hadamard
  - w4a16
---

# Qwen-Image-2.1-Turbo — W4A16-HGPTQ (`g64-h64` + Multimodal DeepStack/Trajectory HGPTQ + Norm/Velocity QAT)

## Summary

- **What it is:** A 4-bit quantization of [`Qwen/Qwen-Image-2.1-Turbo`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo), the fast **8-step** distilled variant of Qwen's 16B bilingual (English + Chinese) text-to-image and image-editing diffusion pipeline (`Qwen3VL` text/vision encoder + 32-block DiT + BF16 VAE).
- **Fits on 12–16 GB GPUs:** Shrinks the full pipeline from `32.45 GiB` in BF16 down to **`9.82 GiB`** (`3.31×` smaller: `5.44 GiB` text encoder + `3.75 GiB` DiT + `0.63 GiB` BF16 VAE).
- **Near-identical 8-step image quality:** Calibrated on real multi-aspect-ratio T2I and image-editing trajectories, retaining `0.9994` teacher-forced velocity cosine (`30.42 dB` SQNR) and **`25.46 dB`** end-to-end 8-step decoded-image PSNR (**`36.53 dB`** on image-to-image editing) against the 16-bit pipeline.

This repository contains a **9.82 GiB** (`3.31×` compressed from `32.45 GiB` BF16) **`W4A16-HGPTQ`** quantization of [`Qwen/Qwen-Image-2.1-Turbo`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (`fc13ba1986fcb905854dc8de2f52eee395f49b5b`) produced with [`warp-quant`](https://github.com/nchapman/warp34-quant-lab).

## Recipe Summary

| Submodule | Architecture | Quantization & Calibration Scheme | Size |
| :--- | :--- | :--- | ---: |
| `text_encoder/` | `Qwen3VLForConditionalGeneration` (36 decoder layers + 27 ViT blocks + `merger` + `deepstack_merger_list.0..2`) | `W4_g64` + $H_{64}$ online Hadamard (`group_0`), per-channel $H_{16}$ on `visual.blocks.*.mlp.linear_fc2` ($K=4304$), unrotated `W8_g64` `lm_head`; Vision calibrated on `60` images (`24` `image-gen-calib-v1` TI2I + `36` `vision-calib-v2`), then post-quantized Vision + DeepStack (`_deepstack_process`) + 3D M-RoPE propagated through all 36 `Qwen3VL` decoder layers (`256` EN+ZH T2I/TI2I/text sequences) + 160-step Layer-35 Norm QAT (`-7.08%` val MSE) | `5.44 GiB` |
| `transformer/` | `QwenImage21Transformer2DModel` (32 single-stream 3D-RoPE DiT blocks = 224 linears; `img_in`, `txt_in`, `proj_out`, `time_text_embed`, `modulation.1`, `norm_out.linear` kept in BF16) | Sequential block-by-block `W4_g64` + $H_{64}$ online Hadamard (`group_0`) calibrated on `32` multi-aspect-ratio T2I+TI2I trajectories (`128` states across 5 aspect ratios & VAE condition-image prefixes) + 160-step Block-31 Velocity QK-Norm QAT (`-2.26%` val MSE) | `3.75 GiB` |
| `vae/` | `AutoencoderKLQwenImage21` | Unmodified `BF16` | `0.63 GiB` |
| **Total** | Full 16B T2I + TI2I Pipeline | **`compressed-tensors` `pack-quantized`** | **`9.82 GiB`** |

## Dataset-Backed Calibration & Evaluation Suite (`image-gen-v1`)

Calibrated and evaluated on strictly disjoint, commit-pinned splits (`OneIG-Bench` `f80e6317` EN+ZH, `DPG-Bench` `f7f1626d`, `PartiPrompts` `944b156a`, `GEdit-Bench` `50766778` EN+ZH TI2I, and `MagicBrush` `6c254560` OOD guard) across 5 canonical 32-aligned aspect ratios (`1024×1024`, `1344×768`, `768×1344`, `1152×896`, `896×1152`):

- **Calibration (`data/image-gen-calib-v1`)**: `96` rows (`16` domains × `6`: `72` T2I + `24` TI2I, `54` EN + `42` ZH)
- **Held-Out Probe (`data/image-gen-probe-v1`)**: `64` rows (`16` domains × `4`: `48` T2I + `16` TI2I, `36` EN + `28` ZH)
- **Held-Out OOD Guard (`data/image-gen-guard-v1`)**: `24` rows (`6` unseen domains × `4`: `16` T2I + `8` TI2I, `22` EN + `2` ZH)

## Teacher-Forced & Trajectory Fidelity Receipt

Evaluated against the unmodified `BF16` `Qwen/Qwen-Image-2.1-Turbo` pipeline (`probe_results.json`):

| Metric | Held-Out Probe (`image-gen-probe-v1`, 64 rows) | Held-Out OOD Guard (`image-gen-guard-v1`, 24 rows) | Notes |
| :--- | ---: | ---: | :--- |
| **`transformer.txt_in` Conditioning Cosine** | **`0.999603`** (`31.95 dB` SQNR) | **`0.999424`** (`32.55 dB` SQNR) | T2I: `0.999834`, TI2I: `0.998910` (post-`QwenImage21ZeroCenterRMSNorm` DiT text conditioning) |
| **`text_encoder` Prompt Embeds Cosine (T2I / All)** | **`0.992175`** / `0.957850` | **`0.991102`** / `0.939555` | Raw Layer-35 `outputs.hidden_states[-1]` (`18.21 dB` T2I SQNR) |
| **Teacher-Forced DiT Velocity Cosine** | **`0.999425`** (`99.94%`) | **`0.999214`** (`99.92%`) | `48` probe / `18` guard states across early/mid/late FlowMatch steps & 5 aspect ratios |
| **Teacher-Forced DiT Velocity SQNR** | **`30.42 dB`** (`Rel-RMSE 3.21%`) | **`29.29 dB`** (`Rel-RMSE 3.68%`) | T2I: `29.74 dB` (`0.99935`), TI2I: **`32.44 dB`** (`0.99966`); EN: `30.63 dB`, ZH: `30.14 dB` |
| **End-to-End 8-Step Latent Cosine** | **`0.978239`** (`16.09 dB` SQNR) | **`0.988577`** (`18.84 dB` SQNR) | Full 8-step Turbo trajectory (`16` probe / `6` guard trajectories, T2I + TI2I) |
| **End-to-End Decoded Image Mean PSNR** | **`25.46 dB`** | **`27.25 dB`** | T2I mean PSNR: `21.77 dB` (`23.93 dB` guard); TI2I mean PSNR: **`36.53 dB`** (`33.87 dB` guard) |

## Usage (`diffusers` + `warp-quant`)

```python
import torch
from warp_quant.qwen_image import load_qwen_image_pipeline

pipe = load_qwen_image_pipeline(
    "nchapman/Qwen-Image-2.1-Turbo-W4A16-HGPTQ",
    device="cuda",
    dtype=torch.bfloat16,
    keep_text_encoder_compressed=True,
    keep_transformer_compressed=False,
)

image = pipe(
    prompt='A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement',
    height=1024,
    width=1024,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("out.png")
```

Set `keep_transformer_compressed=True` as well to keep both the 8.77B `Qwen3VL` encoder and the 7.12B DiT packed in 4-bit `int32` inside VRAM (~10.2 GiB total GPU memory).

## License

Apache-2.0 (inherited from [`Qwen/Qwen-Image-2.1-Turbo`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)).
