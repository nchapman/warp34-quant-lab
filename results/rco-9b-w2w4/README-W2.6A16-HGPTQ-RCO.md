---
base_model: Qwen/Qwen3.5-9B
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

# Qwen3.5-9B-W2.6A16-HGPTQ-RCO

A compact sub-3-bit quantized version of [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), Qwen's 9B multimodal model for everyday coding, reasoning, instruction following, and vision. It brings the full 9B model—with vision and built-in speculative decoding intact—down to a single **8 GB GPU**.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (8 GB RTX 30/40-series GPUs (e.g., RTX 3070 / 4060 / 4060 Ti 8 GB)):

```bash
warp34 serve nchapman/Qwen3.5-9B-W2.6A16-HGPTQ-RCO
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Qwen3.5-9B-W2.6A16-HGPTQ-RCO --trust-remote-code
```

## About this quant

- **Fits on a single 8 GB GPU** (`5.55 GiB` on disk vs. `18.0 GiB` in BF16, `3.2×` smaller), leaving ~`2.45 GiB` for long context, vision inputs, and speculative decoding.
- **Smart mixed 2-bit / 4-bit precision (V2):** Keeps the most sensitive `36.75%` of backbone weights—plus the vision encoder and MTP speculative drafter—in 4-bit (with an 8-bit output head), while compressing the remaining layers with symmetric 2-bit quantization.
- **Strong accuracy for 8 GB VRAM:** Scores `75.74%` across IFEval (`75.8%`), MATH-500 (`75.2%`), and HumanEval+ (`76.2%`)—outperforming Unsloth `UD-Q2_K_XL` (`73.12%`) while using fewer trunk bits.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Uses strictly `2 / 4 / 8`-bit GPU-friendly layouts (no slow 3-bit unpacking) so it runs fast out of the box.

**Have a 12 GB+ GPU?** See [Qwen3.5-9B-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.5-9B-W4A16-HGPTQ) (`6.67 GiB`) for full 4-bit quality.

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
