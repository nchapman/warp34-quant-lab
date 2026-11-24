---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- w2a16
- w4a16
- compressed-tensors
- vllm
- warp34
---

# Qwen3.8-27B-W2.2A16-HGPTQ-RCO

An ultra-compact ~2.2-bit quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), Qwen's 27B multimodal model for coding, reasoning, and vision. It is sized to squeeze a 27B model onto a **12 GB GPU** (or unlock extra-long context on a 16 GB card).

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (12 GB GPUs (e.g., RTX 3060 12 GB / 4070) or 16 GB GPUs with long context):

```bash
warp34 serve nchapman/Qwen3.8-27B-W2.2A16-HGPTQ-RCO
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Qwen3.8-27B-W2.2A16-HGPTQ-RCO --trust-remote-code
```

## About this quant

- **Fits on a 12 GB GPU** (`11.06 GiB` total / `10.59 GiB` text-only vs. `51.8 GiB` in BF16, `4.7×` smaller); `warp34` loads vision and MTP lazily so a headless 12 GB card still has KV-cache headroom.
- **Smart mixed 2-bit / 4-bit precision (V2):** Keeps the most critical `16.5%` of backbone weights—plus vision and MTP—in 4-bit (with an 8-bit output head), while compressing the rest with symmetric 2-bit quantization.
- **Solid 27B capability at 11 GiB:** Scores `78.00%` across IFEval (`72.1%`), MATH-500 (`80.2%`), and HumanEval+ (`81.7%`).
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Uses strictly `2 / 4 / 8`-bit GPU-friendly formats for fast local serving with zero extra flags.

**Have more VRAM?**
- **16 GB GPU:** [Qwen3.8-27B-W2.6A16-HGPTQ-RCO](https://huggingface.co/nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO) (`12.20 GiB`, `82.87%` EvalScope mean)
- **24 GB GPU:** [Qwen3.8-27B-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.8-27B-W4A16-HGPTQ) (`16.14 GiB`, `85.99%` EvalScope mean)

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
