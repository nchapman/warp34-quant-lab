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
- w4a16
- compressed-tensors
- vllm
- warp34
---

# Ornith-1.5-9B-W4A16-HGPTQ

A 4-bit quantized version of [ornith-ai/Ornith-1.5-9B](https://huggingface.co/ornith-ai/Ornith-1.5-9B), Ornith AI's lightweight 9B dense model built through end-to-end self-improvement on top of the Qwen hybrid architecture, with vision and built-in Multi-Token Prediction (MTP) intact.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (12 GB+ RTX 30/40-series GPUs):

```bash
warp34 serve nchapman/Ornith-1.5-9B-W4A16-HGPTQ
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Ornith-1.5-9B-W4A16-HGPTQ --trust-remote-code
```

## About this quant

- **Fits easily on a 12 GB GPU** (`6.67 GiB` on disk vs. `17.98 GiB` in BF16, `2.7×` smaller), leaving ample VRAM for long context, vision inputs, and speculative decoding.
- **Full-model 4-bit calibration:** Text backbone, vision tower, and built-in MTP speculative drafter are all calibrated and quantized together (with an 8-bit output head).
- **Near-BF16 fidelity:** Retains `97.0%` isolated vision top-1 agreement and `94.7%` isolated MTP draft agreement against the 16-bit base model.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Uses standard `4-bit` / `8-bit` packed weights with Hadamard rotation for full-speed CUDA serving with zero extra flags.

**On an 8 GB GPU?** Try [Ornith-1.5-9B-W2.6A16-HGPTQ-RCO](https://huggingface.co/nchapman/Ornith-1.5-9B-W2.6A16-HGPTQ-RCO) (`5.55 GiB`).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
