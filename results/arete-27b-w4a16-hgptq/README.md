---
base_model: NousResearch/Arete-27B
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

# Arete-27B-W4A16-HGPTQ

A 4-bit quantized version of [NousResearch/Arete-27B](https://huggingface.co/NousResearch/Arete-27B), Nous Research's post-trained `Qwen3.8-27B` model focused on factual accuracy and value alignment. In addition to 4-bit quantization, this checkpoint **restores built-in Multi-Token Prediction (MTP) speculative decoding** that was omitted from the upstream release.

## Quickstart

Built and optimized for [**warp34**](https://github.com/nchapman/warp34) (24 GB RTX 30/40-series GPUs (e.g., RTX 3090 / 4090)):

```bash
warp34 serve nchapman/Arete-27B-W4A16-HGPTQ
```

Also runs in **vLLM** (`0.34+`):

```bash
vllm serve nchapman/Arete-27B-W4A16-HGPTQ --trust-remote-code
```

## About this quant

- **Fits on a single 24 GB GPU** (`16.14 GiB` on disk vs. `52.80 GiB` in BF16, `3.3×` smaller), leaving room for long context, vision inputs, and speculative decoding.
- **Restores Native MTP speculative decoding:** The upstream `Arete-27B` release shipped without `mtp.*` weights. This checkpoint grafts the compatible `Qwen3.8-27B` MTP head and calibrates it directly on `Arete-27B` hidden states, recovering `96.5%` isolated (`88.7%` end-to-end) draft agreement.
- **Near-BF16 fidelity:** Calibrates the text backbone and vision tower together (`98.3%` isolated vision top-1 agreement) with an 8-bit output head.
- **Optimized for [`warp34`](https://github.com/nchapman/warp34):** Runs full-speed CUDA kernels and built-in MTP speculative decoding out of the box with zero extra flags.

**Looking for the base 27B model?** See [Qwen3.8-27B-W4A16-HGPTQ](https://huggingface.co/nchapman/Qwen3.8-27B-W4A16-HGPTQ).

*Benchmarks, evaluation receipts, and quantization details: [Technical Report](TECHNICAL.md) · [`warp34-quant-lab`](https://github.com/nchapman/warp34-quant-lab).*
