---
base_model: Qwen/Qwen3.6-35B-A3B
base_model_relation: quantized
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- w4a16
- moe
- compressed-tensors
- vllm
- warp34
---

# Qwen3.6-35B-A3B-W4A16-HGPTQ

A 4-bit quantized version of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), Qwen's 35B Mixture-of-Experts multimodal model with only 3B active parameters per token. It pairs very fast decode throughput with strong coding, reasoning, vision, and built-in Multi-Token Prediction (MTP) speculative decoding.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (24 GB RTX 30/40-series GPUs (e.g., RTX 3090 / 4090)):

```bash
warp34 serve nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ --trust-remote-code
```

## About this quant

- **Fits on a single 24 GB GPU** (`18.78 GiB` on disk vs. `67.0 GiB` in BF16, `3.6×` smaller), leaving headroom for long context, vision inputs, and speculative decoding.
- **Full MoE + Vision + MTP coverage:** All 256 routed experts, shared experts, attention/GDN layers, vision encoder, and the MTP speculative drafter are calibrated and quantized to 4-bit (with an 8-bit output head).
- **Near-BF16 fidelity:** Retains `99.3%` isolated vision top-1 agreement (`98.5%` on UI/OCR guard) and `95.8%` MTP draft agreement with very low divergence from BF16.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Bundles pre-profiled hot-expert rankings (`static_hot_cache_rankings.json`) and standard `4-bit` / `8-bit` layouts for fast single-GPU MoE serving.

**Have a 16 GB GPU?** Use [Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO](https://huggingface.co/nchapman/Qwen3.6-35B-A3B-W2.6A16-HGPTQ-RCO) (`13.01 GiB`, fits comfortably on 16 GB).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
