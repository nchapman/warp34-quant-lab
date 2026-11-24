---
base_model: Apexintelligence-AI/Apex-35B-V1.0
base_model_relation: quantized
license: apache-2.0
license_link: https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0/blob/main/LICENSE
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

# Apex-35B-V1.0-W4A16-HGPTQ

A 4-bit quantized version of [Apexintelligence-AI/Apex-35B-V1.0](https://huggingface.co/Apexintelligence-AI/Apex-35B-V1.0), Apex Intelligence's general-purpose and research-focused 35B-A3B Mixture-of-Experts fine-tune built on `Qwen3.6-35B-A3B` for coding, search, agentic workflows, and scientific/engineering tasks.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (24 GB RTX 30/40-series GPUs (e.g., RTX 3090 / 4090)):

```bash
warp34 serve nchapman/Apex-35B-V1.0-W4A16-HGPTQ
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Apex-35B-V1.0-W4A16-HGPTQ --trust-remote-code
```

## About this quant

- **Fits on a single 24 GB GPU** (`18.78 GiB` on disk vs. `66.97 GiB` in BF16, `3.6×` smaller), leaving room for long context, vision inputs, and speculative decoding.
- **Full MoE + Vision + MTP coverage:** All 256 routed experts, shared experts, attention/GDN layers, vision encoder, and built-in MTP drafter are calibrated and quantized to 4-bit (with an 8-bit output head).
- **Near-BF16 fidelity:** Retains `97.4%` isolated vision top-1 (`99.3%` on UI/OCR guard) and `95.1%` MTP draft agreement with very low divergence from BF16.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Bundles pre-profiled hot-expert rankings (`static_hot_cache_rankings.json`) and standard `4-bit` / `8-bit` layouts for fast single-GPU MoE serving.

**Looking for the base 35B MoE model?** See [Qwen3.6-35B-A3B-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.6-35B-A3B-W4A16-HGPTQ).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
