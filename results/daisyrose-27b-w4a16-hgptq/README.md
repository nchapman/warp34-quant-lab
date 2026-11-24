---
base_model: RadixArk/DaisyRose-27B
base_model_relation: quantized
license: apache-2.0
license_link: https://huggingface.co/RadixArk/DaisyRose-27B/blob/main/LICENSE
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

# DaisyRose-27B-W4A16-HGPTQ

A 4-bit quantized version of [RadixArk/DaisyRose-27B](https://huggingface.co/RadixArk/DaisyRose-27B), RadixArk's multimodal reasoning model trained with pure reinforcement learning on top of `Qwen3.8-27B`, preserving full image understanding and built-in Multi-Token Prediction (MTP).

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (24 GB RTX 30/40-series GPUs (e.g., RTX 3090 / 4090)):

```bash
warp34 serve nchapman/DaisyRose-27B-W4A16-HGPTQ
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/DaisyRose-27B-W4A16-HGPTQ --trust-remote-code
```

## About this quant

- **Fits on a single 24 GB GPU** (`16.14 GiB` on disk vs. `51.75 GiB` in BF16, `3.2×` smaller), leaving room for long context, vision inputs, and speculative decoding.
- **Full-model 4-bit calibration:** Text backbone, vision encoder, and built-in MTP speculative drafter are all calibrated and quantized together (with an 8-bit output head).
- **Near-BF16 fidelity:** Retains `99.0%` isolated vision top-1 (`98.5%` on UI/OCR guard) and `96.3%` isolated MTP draft agreement against the 16-bit base model.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Uses standard `4-bit` / `8-bit` packed weights with Hadamard rotation so `warp34` runs full-speed CUDA kernels with zero extra flags.

**Looking for the base 27B model?** See [Qwen3.8-27B-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.8-27B-W4A16-HGPTQ).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
