---
base_model: fastino/GLiDE-0.8B
base_model_relation: quantized
license: other
license_name: fastino-community
license_link: https://huggingface.co/fastino/GLiDE-0.8B/blob/main/LICENSE
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

# GLiDE-0.8B-W4A16-HGPTQ

A 4-bit quantized version of [fastino/GLiDE-0.8B](https://huggingface.co/fastino/GLiDE-0.8B), Fastino Labs' ultra-lightweight 0.8B text GLiDE decision model for millisecond-scale classification and routing in a single forward pass.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (2 GB+ GPUs — serves zero-decode `/v1/decisions` and `/v1/systemone` endpoints):

```bash
warp34 serve nchapman/GLiDE-0.8B-W4A16-HGPTQ
```

## About this quant

- **`1.94×` smaller in VRAM** (`0.72 GiB` on disk vs. `1.40 GiB` in BF16), fitting comfortably on 2 GB+ GPUs.
- **Exact BF16 decision head preserved:** Quantizes the backbone to calibrated 4-bit while keeping `readout.safetensors` in exact BF16 for clean calibrated probabilities.
- **Matches BF16 decisions:** Retains `100.00%` top-1 decision agreement and `0.9982` readout logit cosine similarity against the 16-bit base model.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Loads cleanly out of the box for zero-decode `/v1/decisions` and `/v1/systemone` inference.

Browse all quantized `GLiDE` & `d3` sizes in the [Decision Models (W4A16-HGPTQ) collection](https://huggingface.co/collections/nchapman/decision-models-w4a16-hgptq-6b05056dfe6ebb9a6d419809).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
