---
base_model: Qwen/Qwen3.5-0.8B
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

# Qwen3.5-0.8B-W4A16-HGPTQ

A 4-bit quantized version of [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), Qwen's ultra-lightweight 0.8B multimodal model for fast local chat, lightweight tasks, and vision. It uses Qwen's hybrid attention architecture and includes built-in Multi-Token Prediction (MTP) for speculative decoding.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (any 2–4 GB+ RTX 30/40-series GPU):

```bash
warp34 serve nchapman/Qwen3.5-0.8B-W4A16-HGPTQ
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Qwen3.5-0.8B-W4A16-HGPTQ --trust-remote-code
```

## About this quant

- **Tiny footprint (`0.79 GiB` vs. `1.63 GiB` in BF16):** Runs comfortably on any 2–4 GB+ GPU with plenty of room for long context and high concurrency.
- **Full-model 4-bit calibration:** Quantizes the text backbone, vision encoder, and built-in MTP speculative draft head together so vision and speculative decoding work out of the box.
- **Near-BF16 fidelity:** Preserves strong agreement with the unquantized 16-bit model (`94.0%` isolated vision top-1, `90.2%` MTP draft agreement) at half the size.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Uses standard `4-bit` packed weights with Hadamard rotation so `warp34` runs full-speed CUDA kernels with zero extra flags.

**Want more capability?** Step up to [Qwen3.5-2B-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.5-2B-W4A16-HGPTQ) (`1.83 GiB`) or [Qwen3.5-4B-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.5-4B-W4A16-HGPTQ) (`3.19 GiB`).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
