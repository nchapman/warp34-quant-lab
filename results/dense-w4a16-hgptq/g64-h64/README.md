---
base_model: Qwen/Qwen3.5-2B
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

# Qwen3.5-2B-W4A16-HGPTQ

A 4-bit quantized version of [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B), Qwen's compact 2B multimodal model for everyday chat, extraction, lightweight coding, and vision. It uses Qwen's hybrid attention architecture and includes built-in Multi-Token Prediction (MTP) for speculative decoding.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (4–6 GB+ RTX 30/40-series GPUs):

```bash
warp34 serve nchapman/Qwen3.5-2B-W4A16-HGPTQ
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Qwen3.5-2B-W4A16-HGPTQ --trust-remote-code
```

## About this quant

- **Compact `1.83 GiB` footprint** (down from `4.24 GiB` in BF16, `2.3×` smaller), fitting easily on 4–6 GB GPUs with generous room for long context.
- **Full-model 4-bit calibration:** Calibrates the text layers, vision encoder, and built-in MTP speculative draft head together so vision and fast speculative decoding both work out of the box.
- **Near-BF16 fidelity:** Retains `97.4%` isolated vision top-1 agreement (`99.3%` on UI/OCR guard) and `92.8%` MTP draft agreement against the 16-bit base model.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Uses standard `4-bit` packed weights with Hadamard rotation for full-speed CUDA serving with zero extra flags.

**Need a different size?** Try [Qwen3.5-0.8B-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.5-0.8B-W4A16-HGPTQ) (`0.79 GiB`) or [Qwen3.5-4B-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.5-4B-W4A16-HGPTQ) (`3.19 GiB`).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
