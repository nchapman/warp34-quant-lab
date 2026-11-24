---
base_model: Qwen/Qwen3.5-9B
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

# Qwen3.5-9B-W4A16-HGPTQ

A 4-bit quantized version of [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), Qwen's 9B multimodal model for everyday coding, reasoning, instruction following, and vision tasks. It uses Qwen's hybrid attention architecture for fast long-context serving and includes built-in Multi-Token Prediction (MTP) for speculative decoding.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (12 GB+ RTX 30/40-series GPUs (or 8 GB with shorter context)):

```bash
warp34 serve nchapman/Qwen3.5-9B-W4A16-HGPTQ
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Qwen3.5-9B-W4A16-HGPTQ --trust-remote-code
```

## About this quant

- **Great fit for 12 GB GPUs** (`6.67 GiB` on disk vs. `17.98 GiB` in BF16, `2.7×` smaller), leaving room for `128K` context, vision inputs, and speculative decoding.
- **Everything is calibrated, not just text.** The text backbone, vision encoder, and MTP draft head are all calibrated and quantized to 4-bit (with an 8-bit output head).
- **Near-lossless quality:** Scores `83.35%` across IFEval, MATH-500, and HumanEval+ (vs. `84.41%` for the full 16-bit model), with `99.3%` isolated vision agreement.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Uses standard `4-bit` / `8-bit` packed weights with Hadamard rotation so `warp34` runs full-speed CUDA kernels with zero extra flags.

**On an 8 GB GPU?** Use [Qwen3.5-9B-W2.6A16-HGPTQ-RCO](https://huggingface.co/nchapman/Qwen3.5-9B-W2.6A16-HGPTQ-RCO) (`5.55 GiB`) for extra KV-cache headroom on 8 GB cards.

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
