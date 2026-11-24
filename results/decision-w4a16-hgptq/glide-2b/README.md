---
base_model: fastino/GLiDE-2B
base_model_relation: quantized
license: other
license_name: fastino-community
license_link: https://huggingface.co/fastino/GLiDE-2B/blob/main/LICENSE
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

# GLiDE-2B-W4A16-HGPTQ

A 4-bit quantized version of [fastino/GLiDE-2B](https://huggingface.co/fastino/GLiDE-2B), Fastino Labs' 2B text GLiDE decision model for low-latency, calibrated classification and routing in a single forward pass.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (4 GB+ GPUs — serves zero-decode `/v1/decisions` and `/v1/systemone` endpoints):

```bash
warp34 serve nchapman/GLiDE-2B-W4A16-HGPTQ
```

## About this quant

- **`2.15×` smaller in VRAM** (`1.63 GiB` on disk vs. `3.51 GiB` in BF16), fitting comfortably on 4 GB+ GPUs.
- **Exact BF16 decision head preserved:** Quantizes the backbone to calibrated 4-bit while keeping `readout.safetensors` in exact BF16 for clean calibrated probabilities.
- **Matches BF16 decisions:** Retains `100.00%` top-1 decision agreement and `0.9991` readout logit cosine similarity against the 16-bit base model.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Loads cleanly out of the box for zero-decode `/v1/decisions` and `/v1/systemone` inference.

Browse all quantized `GLiDE` & `d3` sizes in the [Decision Models (W4A16-HGPTQ) collection](https://huggingface.co/collections/nchapman/decision-models-w4a16-hgptq-6b05056dfe6ebb9a6d419809).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
