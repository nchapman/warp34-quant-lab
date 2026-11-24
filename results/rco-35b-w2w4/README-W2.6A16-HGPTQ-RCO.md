---
base_model: Qwen/Qwen3.6-35B-A3B
base_model_relation: quantized
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- w2a16
- w4a16
- moe
- compressed-tensors
- vllm
- warp34
---

# Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO

A compact sub-3-bit quantized version of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), Qwen's fast 35B Mixture-of-Experts multimodal model (3B active parameters per token). It shrinks the full 256-expert MoE model—plus vision and built-in speculative decoding—onto a single **16 GB GPU**.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (16 GB RTX 30/40-series GPUs (e.g., RTX 4060 Ti 16 GB / 4070 Ti Super / 4080)):

```bash
warp34 serve nchapman/Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO --trust-remote-code
```

## About this quant

- **Fits on a single 16 GB GPU** (`13.01 GiB` on disk vs. `67.0 GiB` in BF16, `5.15×` smaller), leaving ~`3 GiB` for KV cache, vision inputs, and speculative decoding.
- **Smart mixed 2-bit / 4-bit MoE precision (V2):** Keeps 100% of attention, shared experts, vision, and MTP in 4-bit, upgrades the 12 most sensitive routed-expert layers to 4-bit, and compresses the remaining 28 routed-expert layers with symmetric 2-bit quantization.
- **Standout accuracy on 16 GB:** Scores **`86.10%`** across IFEval (`84.3%`), MATH-500 (`85.6%`), and HumanEval+ (`88.4%`), with `98.0%` isolated vision agreement.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Bundles pre-profiled hot-expert rankings (`static_hot_cache_rankings.json`) and strict `2 / 4 / 8`-bit layouts for fast single-GPU MoE serving.

**Have a 24 GB GPU?** See [Qwen3.6-35B-A3B-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ) (`18.78 GiB`) for full 4-bit quality.

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
