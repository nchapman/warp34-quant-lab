---
base_model: vllm-sr/d3
base_model_relation: quantized
license: apache-2.0
library_name: transformers
pipeline_tag: zero-shot-classification
quantized_by: nchapman
tags:
- gptq
- hadamard
- w4a16
- compressed-tensors
- decision-model
- code-readout-v1
- warp34
---

# d3-W4A16-HGPTQ

A 4-bit quantized version of [vllm-sr/d3](https://huggingface.co/vllm-sr/d3), vLLM Semantic Router's flagship 27B multimodal Decision 3.0 model for zero-decode classification, routing, and structured multi-option decisions in a single forward pass.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (24 GB GPUs (e.g., RTX 3090 / 4090) — serves zero-decode `/v1/decisions` and `/v1/systemone` endpoints):

```bash
warp34 serve nchapman/d3-W4A16-HGPTQ
```

## About this quant

- **`3.31×` smaller in VRAM** (`14.69 GiB` on disk vs. `48.59 GiB` in BF16), fitting comfortably on 24 GB GPUs (e.g., RTX 3090 / 4090).
- **Exact BF16 decision head preserved:** Quantizes the backbone and vision encoder to calibrated 4-bit while keeping `readout.safetensors` in exact BF16 for clean calibrated probabilities.
- **Matches BF16 decisions:** Retains `100.00%` top-1 decision agreement and `0.9999` readout logit cosine similarity against the 16-bit base model.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Loads cleanly out of the box for zero-decode `/v1/decisions` and `/v1/systemone` inference.

Browse all quantized `GLiDE` & `d3` sizes in the [Decision Models (W4A16-HGPTQ) collection](https://huggingface.co/collections/nchapman/decision-models-w4a16-hgptq-6b05056dfe6ebb9a6d419809).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
