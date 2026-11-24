---
base_model: ornith-ai/Ornith-1.5-9B
base_model_relation: quantized
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-9B/blob/main/LICENSE
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- w2a16
- w4a16
- compressed-tensors
- vllm
- warp34
---

# Ornith-1.5-9B-W2.6A16-HGPTQ-RCO

A compact sub-3-bit quantized version of [ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B), Ornith AI's lightweight 9B dense self-improvement model built on the Qwen hybrid architecture. It fits the full multimodal model and built-in speculative draft head onto a single **8 GB GPU**.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (8 GB RTX 30/40-series GPUs):

```bash
warp34 serve nchapman/Ornith-1.5-9B-W2.6A16-HGPTQ-RCO
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Ornith-1.5-9B-W2.6A16-HGPTQ-RCO --trust-remote-code
```

## About this quant

- **Fits on a single 8 GB GPU** (`5.55 GiB` on disk vs. `18.0 GiB` in BF16, `3.2×` smaller), leaving room for KV cache, vision inputs, and speculative decoding.
- **Smart mixed 2-bit / 4-bit precision (V2):** Keeps the most sensitive `36.75%` of backbone weights—plus vision and Native MTP—in 4-bit (with an 8-bit output head), while compressing the rest with symmetric 2-bit quantization.
- **Multimodal & speculative features preserved:** Retains `98.0%` isolated vision top-1 agreement (`97.0%` on UI/OCR guard) and `94.6%` isolated MTP draft agreement.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Uses strictly `2 / 4 / 8`-bit GPU-friendly formats for fast local serving with zero extra flags.

**Have a 12 GB+ GPU?** See [Ornith-1.5-9B-W4A16-HGPTQ](https://huggingface.co/nchapman/Ornith-1.5-9B-W4A16-HGPTQ) (`6.67 GiB`) for full 4-bit quality.

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
