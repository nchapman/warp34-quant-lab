# 7-Arm Cross-Format Comparison (`Qwen3.8-27B` & `Qwen3.5-9B`)

Controlled evaluation comparing `W4A16-HGPTQ` and `W2.6A16-HGPTQ-RCO` (`nchapman/*`) against `ExLlamaV3` (`EXL3` trellis `4.00 bpw` / `3.00 bpw`), Unsloth Dynamic `GGUF` (`UD-Q4_K_XL` / `UD-Q2_K_XL`), and the unquantized `BF16` reference across both **`Qwen/Qwen3.8-27B`** and **`Qwen/Qwen3.5-9B`**.

- **Machine-readable receipts (including 20,000-draw paired bootstrap CIs & per-domain KL)**:
  - `27B`: [`27b-g64-h64/probe_comparison.json`](27b-g64-h64/probe_comparison.json) · [`27b-g64-h64/evalscope_comparison.json`](27b-g64-h64/evalscope_comparison.json)
  - `9B`: [`9b-g64-h64/probe_comparison.json`](9b-g64-h64/probe_comparison.json) · [`9b-g64-h64/evalscope_comparison.json`](9b-g64-h64/evalscope_comparison.json)
- **Implementation**: [`src/warp_quant/comparison_runner.py`](../../src/warp_quant/comparison_runner.py) · [`scripts/run_comparison_suite.py`](../../scripts/run_comparison_suite.py) · [`tests/test_comparison_runner.py`](../../tests/test_comparison_runner.py)

---

## 1. Methodology & Apples-to-Apples Controls

1. **Primary size metric — Pure Text Backbone (`layers 0..L-1`) GiB & bpw:** Total file size mixes incomparable modules across ecosystems (`nchapman/*` bundles Vision + Native MTP + `BF16` `embed_tokens` + `W8_g64` `lm_head`; Unsloth GGUF splits Vision into a separate `mmproj` and quantizes `embed_tokens` to `Q4_K`; EXL3 checkpoints vary in Vision/MTP precision). Every arm is broken down into exact stored bytes for `trunk` (`layers 0..L-1` + final RMSNorm), `embed_tokens`, `lm_head`, `vision`, `mtp`, `text_total`, and `file_total`.
2. **Exact BF16 weight reconstruction for weight-only (`A16`) parity:**
   - `EXL3` weights are reconstructed on GPU via `exllamav3==1.6.7` (`LinearEXL3.get_weight_tensor()`) and checked against live `LinearEXL3.forward` on `layers.0.linear_attn.in_proj_qkv` (`cosine >= 0.9999998`, `max_abs_diff <= 0.0015` FP16).
   - `GGUF` weights are dequantized with `gguf>=0.19` and saved in `bfloat16` (avoiding the FP32-vs-BF16 activation mismatch of HF's native `gguf_file` loader).
   - Both are exported as clean `bfloat16` `Qwen3_5ForCausalLM` (`qwen3_5_text`) directories with `tie_word_embeddings=False` inherited from the base config.
   - `W4A16-HGPTQ` and `W2.6A16-HGPTQ-RCO` are evaluated natively through their real quantized `compressed-tensors` / Marlin kernels.
3. **Teacher-forced probe (`warp-quant probe`):** Cached-teacher `BF16` logits (`causal-conv1d` + `fla` GDN kernels) scored across all `378` rows of `data/probe-v4.jsonl` (`852k` tokens) and `76` rows of `data/probe-guard.jsonl` (`23k` tokens).
4. **Generative `EvalScope` (`vLLM 0.34.0`):** Full datasets with zero subsampling (`541` `IFEval`, `500` `MATH-500`, `164` `HumanEval+` = `1,205` prompts per arm), `max_tokens=2048`, greedy `temperature=0.0`, `enable_thinking=False`, `seed=42`, served identically via `vLLM` (`bfloat16`, `--gdn-prefill-backend triton`, `--max-model-len 8192`).

---

## 2. `Qwen/Qwen3.8-27B` (`24,353,201,664` Text-Backbone Parameters)

### Provenance & Exact Byte Breakdown

| Arm | Hugging Face Source (`rev` @ `sha`) | Trunk (`bpw`) | Embed | `lm_head` | Vision | MTP | Text Total | File Total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bf16` | `Qwen/Qwen3.8-27B` (`main` @ `37561816e66b`) | `45.36 GiB` (`16.00`) | `2.37` | `2.37` | `0.86` | `0.79` | `50.10 GiB` | `51.75 GiB` |
| `w4_hgptq` | `nchapman/Qwen3.8-27B-W4A16-HGPTQ` (`main` @ `9ecfad02750a`) | **`12.09 GiB`** (`4.26`) | `2.37` | `1.22` | `0.23` | `0.23` | `15.67 GiB` | `16.14 GiB` |
| `exl3_4bpw` | `turboderp/Qwen3.8-27B-exl3` (`SCP_4.00bpw` @ `387b84455685`) | `11.39 GiB` (`4.02`) | `2.37` | `0.74` | `0.53` | `0.20` | `14.50 GiB` | `15.23 GiB` |
| `gguf_q4_k_xl` | `unsloth/Qwen3.8-27B-GGUF` (`UD-Q4_K_XL` @ `4af51b3542f7`) | `14.47 GiB` (`5.11`) | `0.67` | `0.97` | `0.00` | `0.33` | `16.11 GiB` | `16.44 GiB` |
| `w26_rco` | `nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO` (`main` @ `c663b36cee2f`) | `8.12 GiB` (`2.86`) | `2.37` | `1.22` | `0.23` | `0.23` | `11.71 GiB` | `12.18 GiB` |
| `exl3_3bpw` | `turboderp/Qwen3.8-27B-exl3` (`SCP_3.00bpw` @ `550c12cf3cad`) | `8.56 GiB` (`3.02`) | `2.37` | `0.59` | `0.53` | `0.20` | `11.52 GiB` | `12.25 GiB` |
| `gguf_q2_k_xl` | `unsloth/Qwen3.8-27B-GGUF` (`UD-Q2_K_XL` @ `4af51b3542f7`) | `7.87 GiB` (`2.78`) | `0.39` | `0.67` | `0.00` | `0.33` | `8.93 GiB` | `9.26 GiB` |

*(EXL3 GPU forward verification on `layers.0.linear_attn.in_proj_qkv` `[10240, 5120]`: `4.00bpw` cosine `0.9999998`, max abs diff `0.001465`; `3.00bpw` cosine `0.9999998`, max abs diff `0.000977`.)*

### Full Probe & EvalScope Scores (`27B`)

| Arm | `probe-v4` `w_kl` (`mean_kl`) | `probe-guard` `mean_kl` | `IFEval` Prompt Strict / Loose | `IFEval` Inst Strict / Loose | `MATH-500` Acc (`correct/500`, `trunc`) | `HumanEval+` Pass@1 (`passed/164`) | 3-Task Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bf16` | — | — | `82.62%` / `86.32%` | `87.77%` / `90.77%` | `84.80%` (`424`, `79`) | `93.29%` (`153`) | `86.90%` |
| `w4_hgptq` | `0.1002` (`0.2583`) | `0.0530` | `81.89%` / `85.21%` | `87.41%` / `90.05%` | `82.80%` (`414`, `84`) | **`93.29%`** (`153`) | `85.99%` |
| `exl3_4bpw` | `0.0929` (`0.2486`) | **`0.0343`** | **`82.26%`** / `85.40%` | `87.41%` / `89.93%` | **`83.80%`** (`419`, `78`) | `92.68%` (`152`) | **`86.25%`** |
| `gguf_q4_k_xl` | **`0.0714`** (`0.1819`) | `0.0418` | `80.41%` / `83.73%` | `86.33%` / `89.09%` | `83.20%` (`416`, `87`) | **`93.90%`** (`154`) | `85.84%` |
| `w26_rco` | `0.2717` (`0.4265`) | `0.3272` | `75.23%` / `80.04%` | `82.73%` / `86.09%` | `82.40%` (`412`, `72`) | `81.10%` (`133`) | `79.58%` |
| `exl3_3bpw` | **`0.1924`** (`0.4871`) | **`0.0918`** | **`82.07%`** / `85.40%` | `87.41%` / `90.17%` | **`82.80%`** (`414`, `85`) | **`92.07%`** (`151`) | **`85.65%`** |
| `gguf_q2_k_xl` | `0.3385` (`0.6760`) | `0.3732` | `81.52%` / `86.14%` | `87.17%` / `90.65%` | **`82.80%`** (`414`, `84`) | **`92.07%`** (`151`) | `85.46%` |

### Paired Bootstrap 95% CIs (`27B`, 20,000 draws, `B − A`)

| Comparison (`A vs B`) | `IFEval` Prompt Strict | `MATH-500` Acc | `HumanEval+` Pass@1 |
|---|---:|---:|---:|
| `bf16` vs `w4_hgptq` | `-0.74 pp [-3.33, +1.85]` | `-2.00 pp [-4.20, +0.20]` | `+0.00 pp [-1.83, +1.83]` |
| `bf16` vs `exl3_4bpw` | `-0.37 pp [-2.77, +2.03]` | `-1.00 pp [-3.20, +1.20]` | `-0.61 pp [-3.05, +1.22]` |
| `bf16` vs `gguf_q4_k_xl` | `-2.22 pp [-4.62, +0.18]` | `-1.60 pp [-3.80, +0.60]` | `+0.61 pp [-1.22, +3.05]` |
| `w4_hgptq` vs `exl3_4bpw` | `+0.37 pp [-2.22, +2.96]` | `+1.00 pp [-1.40, +3.40]` | `-0.61 pp [-3.05, +1.83]` |
| `w4_hgptq` vs `gguf_q4_k_xl` | `-1.48 pp [-4.44, +1.29]` | `+0.40 pp [-2.00, +2.80]` | `+0.61 pp [-1.22, +3.05]` |
| `w26_rco` vs `exl3_3bpw` | **`+6.84 pp [+3.51, +10.17]`** | `+0.40 pp [-2.40, +3.00]` | **`+10.98 pp [+6.10, +16.46]`** |
| `w26_rco` vs `gguf_q2_k_xl` | **`+6.28 pp [+2.77, +9.98]`** | `+0.40 pp [-2.40, +3.20]` | **`+10.98 pp [+6.10, +16.46]`** |

---

## 3. `Qwen/Qwen3.5-9B` (`6,919,565,824` Text-Backbone Parameters)

### Provenance & Exact Byte Breakdown

| Arm | Hugging Face Source (`rev` @ `sha`) | Trunk (`bpw`) | Embed | `lm_head` | Vision | MTP | Text Total | File Total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `bf16` | `Qwen/Qwen3.5-9B` (`main` @ `c20223623576`) | `12.89 GiB` (`16.00`) | `1.90` | `1.90` | `0.85` | `0.45` | `16.68 GiB` | `17.98 GiB` |
| `w4_hgptq` | `nchapman/Qwen3.5-9B-W4A16-HGPTQ` (`main` @ `68a367fdb019`) | **`3.43 GiB`** (`4.26`) | `1.90` | `0.98` | `0.23` | `0.14` | `6.30 GiB` | `6.67 GiB` |
| `exl3_4bpw` | `TheMelonGod/Qwen3.5-9B-exl3` (`6hb-4.0bpw` @ `e98453aa143a`) | `3.24 GiB` (`4.02`) | `1.90` | `0.71` | `0.85` | `0.11` | `5.84 GiB` | `6.80 GiB` |
| `gguf_q4_k_xl` | `unsloth/Qwen3.5-9B-GGUF` (`UD-Q4_K_XL` @ `3885219b6810`) | `4.24 GiB` (`5.26`) | `0.53` | `0.78` | `0.00` | `0.00` | `5.55 GiB` | `5.55 GiB` |
| `w26_rco` | `nchapman/Qwen3.5-9B-W2.6A16-HGPTQ-RCO` (`main` @ `e8edfab266ba`) | **`2.31 GiB`** (`2.86`) | `1.90` | `0.98` | `0.23` | `0.14` | `5.18 GiB` | `5.54 GiB` |
| `exl3_3bpw` | `TheMelonGod/Qwen3.5-9B-exl3` (`6hb-3.0bpw` @ `22ef1303062e`) | `2.43 GiB` (`3.02`) | `1.90` | `0.71` | `0.85` | `0.11` | `5.04 GiB` | `6.00 GiB` |
| `gguf_q2_k_xl` | `unsloth/Qwen3.5-9B-GGUF` (`UD-Q2_K_XL` @ `3885219b6810`) | `2.52 GiB` (`3.13`) | `0.53` | `0.78` | `0.00` | `0.00` | `3.83 GiB` | `3.83 GiB` |

*(EXL3 GPU forward verification on `layers.0.linear_attn.in_proj_qkv` `[8192, 4096]`: `4.0bpw` cosine `0.9999999`, max abs diff `0.000977`; `3.0bpw` cosine `0.9999998`, max abs diff `0.000977`.)*

### Full Probe & EvalScope Scores (`9B`)

| Arm | `probe-v4` `w_kl` (`mean_kl`) | `probe-guard` `mean_kl` | `IFEval` Prompt Strict / Loose | `IFEval` Inst Strict / Loose | `MATH-500` Acc (`correct/500`, `trunc`) | `HumanEval+` Pass@1 (`passed/164`) | 3-Task Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bf16` | — | — | `84.47%` / `88.54%` | `89.33%` / `92.33%` | `84.60%` (`423`, `81`) | `84.15%` (`138`) | `84.41%` |
| `w4_hgptq` | `0.0638` (`0.1897`) | `0.0399` | `83.92%` / `87.25%` | `88.61%` / `91.25%` | `82.60%` (`413`, `83`) | `83.54%` (`137`) | `83.35%` |
| `exl3_4bpw` | `0.0610` (`0.1914`) | `0.0206` | `83.55%` / `87.62%` | `88.85%` / `91.61%` | **`83.60%`** (`418`, `81`) | `82.32%` (`135`) | `83.16%` |
| `gguf_q4_k_xl` | **`0.0537`** (`0.1736`) | **`0.0191`** | **`84.66%`** / `87.99%` | `89.57%` / `91.97%` | **`83.60%`** (`418`, `76`) | **`85.37%`** (`140`) | **`84.54%`** |
| `w26_rco` | `0.2661` (`0.3431`) | `0.3642` | `60.07%` / `62.85%` | `69.54%` / `72.54%` | `68.00%` (`340`, `139`) | `60.98%` (`100`) | `63.02%` |
| `exl3_3bpw` | **`0.1395`** (`0.3708`) | **`0.0685`** | **`84.10%`** / `86.14%` | `89.21%` / `90.77%` | **`82.80%`** (`414`, `86`) | **`78.66%`** (`129`) | **`81.85%`** |
| `gguf_q2_k_xl` | `0.2810` (`0.6090`) | `0.2011` | `77.45%` / `81.33%` | `83.93%` / `86.81%` | `73.00%` (`365`, `133`) | `68.90%` (`113`) | `73.12%` |

### Paired Bootstrap 95% CIs (`9B`, 20,000 draws, `B − A`)

| Comparison (`A vs B`) | `IFEval` Prompt Strict | `MATH-500` Acc | `HumanEval+` Pass@1 |
|---|---:|---:|---:|
| `bf16` vs `w4_hgptq` | `-0.55 pp [-2.96, +1.85]` | `-2.00 pp [-4.60, +0.60]` | `-0.61 pp [-5.49, +4.27]` |
| `bf16` vs `exl3_4bpw` | `-0.92 pp [-3.14, +1.29]` | `-1.00 pp [-3.40, +1.40]` | `-1.83 pp [-5.49, +1.83]` |
| `bf16` vs `gguf_q4_k_xl` | `+0.18 pp [-2.22, +2.59]` | `-1.00 pp [-3.40, +1.40]` | `+1.22 pp [-3.05, +5.49]` |
| `w4_hgptq` vs `exl3_4bpw` | `-0.37 pp [-2.77, +2.03]` | `+1.00 pp [-1.40, +3.40]` | `-1.22 pp [-6.10, +3.66]` |
| `w4_hgptq` vs `gguf_q4_k_xl` | `+0.74 pp [-1.66, +3.14]` | `+1.00 pp [-1.20, +3.20]` | `+1.83 pp [-3.66, +7.32]` |
| `w26_rco` vs `exl3_3bpw` | **`+24.03 pp [+19.78, +28.28]`** | **`+14.80 pp [+11.40, +18.40]`** | **`+17.68 pp [+9.76, +25.61]`** |
| `w26_rco` vs `gguf_q2_k_xl` | **`+17.38 pp [+13.12, +21.63]`** | **`+5.00 pp [+1.40, +8.80]`** | `+7.93 pp [+0.00, +15.85]` |
| `exl3_3bpw` vs `gguf_q2_k_xl` | **`-6.65 pp [-9.80, -3.51]`** | **`-9.80 pp [-12.80, -7.00]`** | **`-9.76 pp [-17.07, -2.44]`** |

---

## 4. Reproduction

Requires a multi-GPU host with `>= 600 GB` disk (e.g., `2× H200`) and three isolated virtualenvs to avoid PyTorch/ABI clashes:

```bash
# 1. Main .venv (probe + GGUF dequant)
mise run setup && mise run setup-kernels
uv pip install --python .venv/bin/python "gguf>=0.19.0"

# 2. EXL3 reconstruction venv
uv venv --python 3.12 /tmp/.venv-exl3
uv pip install --python /tmp/.venv-exl3/bin/python "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /tmp/.venv-exl3/bin/python \
  "https://github.com/turboderp-org/exllamav3/releases/download/v1.6.7/exllamav3-1.6.7+cu128.torch2.11.0-cp312-cp312-linux_x86_64.whl" \
  safetensors huggingface-hub numpy pyyaml pillow tokenizers transformers pyarrow datasets

# 3. vLLM + EvalScope serving venv
uv venv --python 3.12 /tmp/.venv-vllm
uv pip install --python /tmp/.venv-vllm/bin/python --prerelease=allow \
  vllm "evalscope[app]" math_verify immutabledict langdetect nltk
uv pip install --python /tmp/.venv-vllm/bin/python --no-deps -e .
/tmp/.venv-vllm/bin/python -c "import nltk; nltk.download('punkt', download_dir='/root/nltk_data'); nltk.download('punkt_tab', download_dir='/root/nltk_data')"

# 4. Run all 4 phases (prepare -> probe -> evalscope -> aggregate)
# Note: pass --allow-local-code-exec only inside an ephemeral container without Docker for HumanEval+.
.venv/bin/python scripts/run_comparison_suite.py \
  --models 9b 27b \
  --gpus 0 1 \
  --work-dir /root/comp-work \
  --exl3-venv /tmp/.venv-exl3 \
  --vllm-venv /tmp/.venv-vllm \
  --batch-size 64 \
  --max-tokens 2048 \
  --max-model-len 8192 \
  --gpu-mem-util 0.88 \
  --allow-local-code-exec
```
