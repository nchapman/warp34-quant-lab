---
base_model: Qwen/Qwen3.5-4B
base_model_relation: quantized
license: apache-2.0
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

# Qwen3.5-4B-W4A16-HGPTQ

A 4-bit quantized version of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), Qwen's balanced 4B multimodal model for fast local chat, coding, document/chart reading, and instruction following. It uses Qwen's hybrid attention architecture and includes built-in Multi-Token Prediction (MTP) for speculative decoding.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (6–8 GB+ RTX 30/40-series GPUs):

```bash
warp34 serve nchapman/Qwen3.5-4B-W4A16-HGPTQ
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Qwen3.5-4B-W4A16-HGPTQ --trust-remote-code
```

## About this quant

- **Fits easily on 6–8 GB GPUs** (`3.19 GiB` on disk vs. `8.68 GiB` in BF16, `2.7×` smaller), leaving ample VRAM for long context and high batch concurrency.
- **Full-model 4-bit calibration:** Text layers, vision encoder, and MTP speculative drafter are all calibrated and quantized together so multimodal and speculative features work out of the box.
- **Near-BF16 quality:** Retains `99.3%` isolated vision top-1 agreement and `94.3%` MTP draft agreement compared to the unquantized 16-bit model.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Uses standard `4-bit` packed weights with Hadamard rotation so `warp34` runs full-speed CUDA kernels with zero extra flags.

**Have an 8 GB or 12 GB GPU and want a stronger model?** Try [Qwen3.5-9B-W2.6A16-HGPTQ-RCO](https://huggingface.co/nchapman/Qwen3.5-9B-W2.6A16-HGPTQ-RCO) (`5.55 GiB`, fits on 8 GB) or [Qwen3.5-9B-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.5-9B-W4A16-HGPTQ) (`6.67 GiB`, fits on 12 GB).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
