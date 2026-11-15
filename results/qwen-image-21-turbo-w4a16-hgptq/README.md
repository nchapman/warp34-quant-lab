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

# Qwen-Image-2.1-Turbo — W4A16-HGPTQ (`g64-h64` + Trajectory DiT HGPTQ + Norm/Velocity QAT)

This repository contains a **9.82 GiB** (`3.31×` compressed from `32.45 GiB` BF16) **`W4A16-HGPTQ`** quantization of [`Qwen/Qwen-Image-2.1-Turbo`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) (`fc13ba1986fcb905854dc8de2f52eee395f49b5b`) produced with [`warp-quant`](https://github.com/nchapman/warp34-quant-lab).

## Recipe Summary

| Submodule | Architecture | Quantization Scheme | Size |
| :--- | :--- | :--- | ---: |
| `text_encoder/` | `Qwen3VLForConditionalGeneration` (36 decoder layers + 27 ViT blocks + `merger` + `deepstack_merger_list.0..2`) | `W4_g64` + $H_{64}$ online Hadamard (`g0_text`, `g2_vision`), per-channel $H_{16}$ on `visual.blocks.*.mlp.linear_fc2` ($K=4304$), unrotated `W8_g64` `lm_head`, + 120-step Layer-35 unnormalized hidden-state Norm QAT (`-24.7%` val MSE) | `5.44 GiB` |
| `transformer/` | `QwenImage21Transformer2DModel` (32 single-stream 3D-RoPE DiT blocks = 224 linears; `img_in`, `txt_in`, `proj_out`, `time_text_embed`, `modulation.1`, `norm_out.linear` kept in BF16) | Trajectory-calibrated sequential block-by-block `W4_g64` + $H_{64}$ online Hadamard (`g0_dit`) + 120-step Block-31 Velocity QK-Norm QAT | `3.75 GiB` |
| `vae/` | `AutoencoderKLQwenImage21` | Unmodified `BF16` | `0.63 GiB` |
| **Total** | Full 16B T2I + TI2I Pipeline | **`compressed-tensors` `pack-quantized`** | **`9.82 GiB`** |

## Teacher-Forced & Trajectory Fidelity Receipt

Evaluated against the unmodified `BF16` `Qwen/Qwen-Image-2.1-Turbo` pipeline across held-out text-to-image (`T2I`) and reference-image editing (`TI2I`, `data/vision-probe-v2`) prompts (`probe_results.json`):

| Metric | Value | Notes |
| :--- | ---: | :--- |
| **Teacher-Forced DiT Velocity Cosine** | **`0.998846`** (`99.88%`) | Mean step-by-step velocity prediction cosine similarity across early/mid/late FlowMatch steps |
| **Teacher-Forced DiT Velocity SQNR** | **`26.99 dB`** | Relative RMSE = `0.04638` (`4.64%`) |
| **`text_encoder` Prompt Embeds Cosine** | **`0.951356`** | Layer-35 unnormalized hidden states (`outputs.hidden_states[-1]`), SQNR = `13.58 dB` |
| **End-to-End 8-Step Latent Cosine** | **`0.964757`** (`96.48%`) | Full 8-step Turbo trajectory with identical initial noise seed (`SQNR = 16.50 dB`) |
| **End-to-End Decoded Image Mean PSNR** | **`23.15 dB`** | Decoded RGB pixel PSNR against BF16 teacher trajectory |

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
