---
base_model: ukisai/Swift-1.5-Qwen3.8-27b
base_model_relation: quantized
license: other
license_name: swift-open-license-1.0
license_link: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/blob/main/LICENSE
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

# Swift-1.5-Qwen3.8-27b-W4A16-HGPTQ

A 4-bit quantized version of [ukisai/Swift-1.5-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b), UkisAI's token-efficient reasoning fine-tune of `Qwen3.8-27B`, trained to use substantially fewer thinking tokens while maintaining strong reasoning, coding, and multimodal capability.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (24 GB RTX 30/40-series GPUs (e.g., RTX 3090 / 4090)):

```bash
warp34 serve nchapman/Swift-1.5-Qwen3.8-27b-W4A16-HGPTQ
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Swift-1.5-Qwen3.8-27b-W4A16-HGPTQ --trust-remote-code
```

## About this quant

- **Fits on a single 24 GB GPU** (`16.14 GiB` on disk vs. `51.75 GiB` in BF16, `3.2×` smaller), leaving ample headroom for long context, vision inputs, and speculative decoding.
- **Full-model 4-bit calibration:** Text layers, vision encoder, and built-in MTP speculative drafter are calibrated and quantized together (with an 8-bit output head).
- **Near-BF16 fidelity:** Retains `98.0%` isolated vision top-1 (`99.3%` on UI/OCR guard) and `96.4%` isolated MTP draft agreement against the 16-bit base model.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Uses standard `4-bit` / `8-bit` packed weights with Hadamard rotation so `warp34` runs full-speed CUDA kernels with zero extra flags.

**Already serving `Qwen3.8-27B` with multi-LoRA?** You can also load the hot-swappable adapter: [Qwen3.8-27B-Swift-1.5-LoRA](https://huggingface.co/nchapman/Qwen3.8-27B-Swift-1.5-LoRA).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
