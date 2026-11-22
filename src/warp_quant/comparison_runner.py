"""Reproducible multi-format (BF16 / W4A16-HGPTQ / W2.6A16-HGPTQ-RCO / Unsloth GGUF / EXL3) comparison utilities."""

from __future__ import annotations

import json
import shutil
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from warp_quant.data import WEIGHTED_KL_PROTOCOL, aggregate_weighted_kl
from warp_quant.evalscope_runner import paired_binary_bootstrap

EXL3_TENSOR_SUFFIXES: tuple[str, ...] = (
    ".trellis",
    ".suh",
    ".svh",
    ".su",
    ".sv",
    ".mcg",
    ".mul1",
)

TOKENIZER_ASSET_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
)


@dataclass(frozen=True)
class ComparisonArm:
    """Specification for one evaluated checkpoint arm."""

    arm_id: str
    label: str
    family: str  # "bf16", "warp_quant", "unsloth_gguf", "exl3"
    tier: str  # "bf16", "4bit", "lowbit"
    format_label: str
    repo: str
    revision: str = "main"
    gguf_filename: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SUITE_27B: tuple[ComparisonArm, ...] = (
    ComparisonArm(
        arm_id="bf16",
        label="Qwen/Qwen3.8-27B",
        family="bf16",
        tier="bf16",
        format_label="BF16 (Teacher)",
        repo="Qwen/Qwen3.8-27B",
    ),
    ComparisonArm(
        arm_id="w4_hgptq",
        label="nchapman/Qwen3.8-27B-W4A16-HGPTQ",
        family="warp_quant",
        tier="4bit",
        format_label="W4_g64 + H64 + H-CD + QAT",
        repo="nchapman/Qwen3.8-27B-W4A16-HGPTQ",
    ),
    ComparisonArm(
        arm_id="exl3_4bpw",
        label="turboderp/Qwen3.8-27B-exl3@SCP_4.00bpw",
        family="exl3",
        tier="4bit",
        format_label="EXL3 v1.6.7 SCP 4.00bpw (YAQA + frac-K)",
        repo="turboderp/Qwen3.8-27B-exl3",
        revision="SCP_4.00bpw",
    ),
    ComparisonArm(
        arm_id="gguf_q4_k_xl",
        label="unsloth/Qwen3.8-27B-GGUF (UD-Q4_K_XL)",
        family="unsloth_gguf",
        tier="4bit",
        format_label="Unsloth Dynamic V3.1 UD-Q4_K_XL",
        repo="unsloth/Qwen3.8-27B-GGUF",
        gguf_filename="Qwen3.8-27B-UD-Q4_K_XL.gguf",
    ),
    ComparisonArm(
        arm_id="w26_rco",
        label="nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO",
        family="warp_quant",
        tier="lowbit",
        format_label="W2/W4 HGPTQ-RCO (30% W4)",
        repo="nchapman/Qwen3.8-27B-W2.6A16-HGPTQ-RCO",
    ),
    ComparisonArm(
        arm_id="exl3_3bpw",
        label="turboderp/Qwen3.8-27B-exl3@SCP_3.00bpw",
        family="exl3",
        tier="lowbit",
        format_label="EXL3 v1.6.7 SCP 3.00bpw (YAQA + frac-K)",
        repo="turboderp/Qwen3.8-27B-exl3",
        revision="SCP_3.00bpw",
    ),
    ComparisonArm(
        arm_id="gguf_q2_k_xl",
        label="unsloth/Qwen3.8-27B-GGUF (UD-Q2_K_XL)",
        family="unsloth_gguf",
        tier="lowbit",
        format_label="Unsloth Dynamic V3.1 UD-Q2_K_XL",
        repo="unsloth/Qwen3.8-27B-GGUF",
        gguf_filename="Qwen3.8-27B-UD-Q2_K_XL.gguf",
    ),
)

SUITE_9B: tuple[ComparisonArm, ...] = (
    ComparisonArm(
        arm_id="bf16",
        label="Qwen/Qwen3.5-9B",
        family="bf16",
        tier="bf16",
        format_label="BF16 (Teacher)",
        repo="Qwen/Qwen3.5-9B",
    ),
    ComparisonArm(
        arm_id="w4_hgptq",
        label="nchapman/Qwen3.5-9B-W4A16-HGPTQ",
        family="warp_quant",
        tier="4bit",
        format_label="W4_g64 + H64 + H-CD + QAT",
        repo="nchapman/Qwen3.5-9B-W4A16-HGPTQ",
    ),
    ComparisonArm(
        arm_id="exl3_4bpw",
        label="TheMelonGod/Qwen3.5-9B-exl3@6hb-4.0bpw",
        family="exl3",
        tier="4bit",
        format_label="EXL3 v1.4.1 mul1 4.00bpw (6hb)",
        repo="TheMelonGod/Qwen3.5-9B-exl3",
        revision="6hb-4.0bpw",
    ),
    ComparisonArm(
        arm_id="gguf_q4_k_xl",
        label="unsloth/Qwen3.5-9B-GGUF (UD-Q4_K_XL)",
        family="unsloth_gguf",
        tier="4bit",
        format_label="Unsloth Dynamic V3.1 UD-Q4_K_XL",
        repo="unsloth/Qwen3.5-9B-GGUF",
        gguf_filename="Qwen3.5-9B-UD-Q4_K_XL.gguf",
    ),
    ComparisonArm(
        arm_id="w26_rco",
        label="nchapman/Qwen3.5-9B-W2.6A16-HGPTQ-RCO",
        family="warp_quant",
        tier="lowbit",
        format_label="W2/W4 HGPTQ-RCO (30% W4)",
        repo="nchapman/Qwen3.5-9B-W2.6A16-HGPTQ-RCO",
    ),
    ComparisonArm(
        arm_id="exl3_3bpw",
        label="TheMelonGod/Qwen3.5-9B-exl3@6hb-3.0bpw",
        family="exl3",
        tier="lowbit",
        format_label="EXL3 v1.4.1 mul1 3.00bpw (6hb)",
        repo="TheMelonGod/Qwen3.5-9B-exl3",
        revision="6hb-3.0bpw",
    ),
    ComparisonArm(
        arm_id="gguf_q2_k_xl",
        label="unsloth/Qwen3.5-9B-GGUF (UD-Q2_K_XL)",
        family="unsloth_gguf",
        tier="lowbit",
        format_label="Unsloth Dynamic V3.1 UD-Q2_K_XL",
        repo="unsloth/Qwen3.5-9B-GGUF",
        gguf_filename="Qwen3.5-9B-UD-Q2_K_XL.gguf",
    ),
)

SUITES: dict[str, tuple[ComparisonArm, ...]] = {
    "27b": SUITE_27B,
    "9b": SUITE_9B,
}


def list_model_safetensors(model_dir: Path) -> list[Path]:
    """Return only model weight safetensors files, ignoring auxiliary files like cal_trace.safetensors."""
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.is_file():
        idx = json.loads(idx_path.read_text())
        names = sorted(set(idx.get("weight_map", {}).values()))
        root = model_dir.resolve()
        paths: list[Path] = []
        for name in names:
            rel = Path(name)
            if rel.is_absolute() or ".." in rel.parts or rel.suffix != ".safetensors":
                raise ValueError(f"Unsafe shard path in {idx_path}: {name!r}")
            candidate = model_dir / rel
            if not candidate.parent.resolve().is_relative_to(root):
                raise ValueError(f"Indexed shard escapes model_dir in {idx_path}: {name!r}")
            paths.append(candidate)
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing indexed safetensors shards in {model_dir}: {missing[:5]}")
        return paths

    single = model_dir / "model.safetensors"
    if single.is_file():
        return [single]

    sharded = sorted(model_dir.glob("model-*.safetensors"))
    if sharded:
        return sharded

    raise FileNotFoundError(f"No model safetensors files found in {model_dir}")


def _read_local_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(header_len)
    return json.loads(raw)


def _format_byte_budget(cats: dict[str, int], trunk_params: int) -> dict[str, Any]:
    gib = float(1024**3)
    text_total = cats["trunk"] + cats["embed"] + cats["head"]
    total_file = sum(cats.values())
    bpw = (cats["trunk"] * 8.0 / trunk_params) if trunk_params > 0 else 0.0
    return {
        "trunk_params": int(trunk_params),
        "trunk_bytes": int(cats["trunk"]),
        "trunk_gib": round(cats["trunk"] / gib, 3),
        "trunk_bpw": round(bpw, 3),
        "embed_bytes": int(cats["embed"]),
        "embed_gib": round(cats["embed"] / gib, 3),
        "head_bytes": int(cats["head"]),
        "head_gib": round(cats["head"] / gib, 3),
        "text_total_bytes": int(text_total),
        "text_total_gib": round(text_total / gib, 3),
        "vision_bytes": int(cats["vision"]),
        "vision_gib": round(cats["vision"] / gib, 3),
        "mtp_bytes": int(cats["mtp"]),
        "mtp_gib": round(cats["mtp"] / gib, 3),
        "other_bytes": int(cats["other"]),
        "total_file_bytes": int(total_file),
        "total_file_gib": round(total_file / gib, 3),
    }


def count_bf16_trunk_params(base_dir: Path) -> int:
    """Count exact parameter elements in the BF16 language backbone (layers 0..L-1 + final norm)."""
    params = 0
    for sf in list_model_safetensors(base_dir):
        hdr = _read_local_safetensors_header(sf)
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            if "visual." in k or "mtp." in k or "lm_head" in k or "embed_tokens" in k:
                continue
            if ".layers." in k or k.endswith(".norm.weight"):
                n = 1
                for dim in v["shape"]:
                    n *= int(dim)
                params += n
    return params


def inspect_safetensors_byte_budget(model_dir: Path, trunk_params: int) -> dict[str, Any]:
    """Compute exact component byte breakdown (text backbone vs embed/head/vision/MTP) from safetensors headers."""
    cats = {"trunk": 0, "embed": 0, "head": 0, "vision": 0, "mtp": 0, "other": 0}
    for sf in list_model_safetensors(model_dir):
        hdr = _read_local_safetensors_header(sf)
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            nbytes = int(v["data_offsets"][1]) - int(v["data_offsets"][0])
            if "visual." in k:
                cats["vision"] += nbytes
            elif "mtp." in k:
                cats["mtp"] += nbytes
            elif "lm_head" in k:
                cats["head"] += nbytes
            elif "embed_tokens" in k:
                cats["embed"] += nbytes
            elif ".layers." in k or k.endswith(".norm.weight"):
                cats["trunk"] += nbytes
            else:
                cats["other"] += nbytes
    return _format_byte_budget(cats, trunk_params)


def inspect_gguf_byte_budget(gguf_path: Path, n_layers: int, trunk_params: int) -> dict[str, Any]:
    """Compute exact component byte breakdown from a local GGUF file header."""
    scalar_sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    total_file_size = gguf_path.stat().st_size
    with gguf_path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file: {gguf_path}")
        _ver, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

        def read_str() -> str:
            ln = struct.unpack("<Q", f.read(8))[0]
            return f.read(ln).decode("utf-8", errors="replace")

        def skip_val(vtype: int) -> None:
            if vtype == 8:
                ln = struct.unpack("<Q", f.read(8))[0]
                f.seek(ln, 1)
            elif vtype == 9:
                atype, alen = struct.unpack("<IQ", f.read(12))
                if atype in scalar_sizes:
                    f.seek(scalar_sizes[atype] * alen, 1)
                else:
                    for _ in range(alen):
                        skip_val(atype)
            else:
                f.seek(scalar_sizes[vtype], 1)

        alignment = 32
        for _ in range(n_kv):
            key = read_str()
            vtype = struct.unpack("<I", f.read(4))[0]
            if key == "general.alignment" and vtype == 4:
                alignment = struct.unpack("<I", f.read(4))[0]
            else:
                skip_val(vtype)

        tensors: list[tuple[str, int]] = []
        for _ in range(n_tensors):
            name = read_str()
            ndim = struct.unpack("<I", f.read(4))[0]
            f.seek(8 * ndim, 1)
            _ttype, toff = struct.unpack("<IQ", f.read(12))
            tensors.append((name, int(toff)))

        data_start = ((f.tell() + alignment - 1) // alignment) * alignment

    tensors.sort(key=lambda x: x[1])
    cats = {"trunk": 0, "embed": 0, "head": 0, "vision": 0, "mtp": 0, "other": 0}
    for i, (name, toff) in enumerate(tensors):
        nxt = tensors[i + 1][1] if i + 1 < len(tensors) else (total_file_size - data_start)
        nbytes = max(0, nxt - toff)
        if name == "token_embd.weight":
            cats["embed"] += nbytes
        elif name == "output.weight":
            cats["head"] += nbytes
        elif name.startswith("blk."):
            layer_idx = int(name.split(".")[1])
            if layer_idx >= n_layers:
                cats["mtp"] += nbytes
            else:
                cats["trunk"] += nbytes
        elif name == "output_norm.weight":
            cats["trunk"] += nbytes
        elif "mtp" in name or "nextn" in name:
            cats["mtp"] += nbytes
        else:
            cats["other"] += nbytes
    return _format_byte_budget(cats, trunk_params)


def build_text_only_config(base_config_path: Path) -> dict[str, Any]:
    """Derive a clean Qwen3_5ForCausalLM (qwen3_5_text) BF16 config from a VLM or text base config.json."""
    raw = json.loads(base_config_path.read_text())
    if "text_config" in raw and isinstance(raw["text_config"], dict):
        text_cfg = dict(raw["text_config"])
        tie_default = raw.get("tie_word_embeddings", False)
    else:
        text_cfg = dict(raw)
        tie_default = False
    tie_val = text_cfg.get("tie_word_embeddings")
    text_cfg["tie_word_embeddings"] = bool(tie_default if tie_val is None else tie_val)
    text_cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    text_cfg["model_type"] = "qwen3_5_text"
    text_cfg["dtype"] = "bfloat16"
    text_cfg["mtp_num_hidden_layers"] = 0
    text_cfg.pop("quantization_config", None)
    return text_cfg


def copy_tokenizer_assets(base_dir: Path, out_dir: Path) -> None:
    """Copy tokenizer, chat template, and generation config assets from base_dir to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_ASSET_FILES:
        src = base_dir / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)


def normalize_text_weight_key(key: str) -> str | None:
    """Map a VLM/text safetensors key to Qwen3_5ForCausalLM key space, returning None for vision/MTP keys."""
    if "visual." in key or "mtp." in key:
        return None
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model.") :]
    return key


def dequantize_gguf_to_hf_dir(gguf_path: Path, base_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Dequantize a GGUF checkpoint to a clean BF16 Qwen3_5ForCausalLM safetensors directory."""
    import torch
    from transformers import AutoTokenizer, GgufConfig

    from warp_quant.models import gguf_model_class

    base_tok = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True)
    gguf_tok = AutoTokenizer.from_pretrained(str(base_dir), gguf_file=str(gguf_path), local_files_only=True)
    for token_id in range(len(base_tok)):
        b_t = base_tok.convert_ids_to_tokens(token_id)
        g_t = gguf_tok.convert_ids_to_tokens(token_id)
        if b_t != g_t:
            raise ValueError(f"GGUF token mismatch at {token_id}: {g_t!r} != {b_t!r}")

    cls = gguf_model_class(str(base_dir))
    model = cls.from_pretrained(
        str(base_dir),
        gguf_file=str(gguf_path),
        quantization_config=GgufConfig(gguf_file=str(gguf_path), dequantize=True),
        dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
    )
    model.eval()
    if hasattr(model.config, "quantization_config"):
        delattr(model.config, "quantization_config")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="5GB")
    text_cfg = build_text_only_config(base_dir / "config.json")
    (out_dir / "config.json").write_text(json.dumps(text_cfg, indent=2) + "\n")
    copy_tokenizer_assets(base_dir, out_dir)
    n_params = sum(p.numel() for p in model.parameters())
    del model
    return {"out_dir": str(out_dir), "params": int(n_params), "source": str(gguf_path)}


def dequantize_exl3_to_hf_dir(
    exl3_dir: Path,
    base_dir: Path,
    out_dir: Path,
    *,
    device: str = "cuda:0",
    verify_first_linear: bool = True,
    max_shard_bytes: int = 5 * (1024**3),
) -> dict[str, Any]:
    """Reconstruct an EXL3 checkpoint into a clean BF16 Qwen3_5ForCausalLM safetensors directory."""
    import torch
    import torch.nn.functional as F
    from exllamav3.modules.quant.exl3 import LinearEXL3
    from safetensors import safe_open
    from safetensors.torch import save_file

    shard_files = list_model_safetensors(exl3_dir)
    key_to_shard: dict[str, Path] = {}
    for sf in shard_files:
        hdr = _read_local_safetensors_header(sf)
        for k in hdr:
            if k != "__metadata__":
                key_to_shard[k] = sf

    # Group EXL3 linear prefixes and pass-through tensors (excluding vision & MTP)
    exl3_groups: dict[str, dict[str, str]] = {}
    passthrough_keys: list[str] = []
    for k in sorted(key_to_shard):
        norm = normalize_text_weight_key(k)
        if norm is None:
            continue
        matched = False
        for suf in EXL3_TENSOR_SUFFIXES:
            if k.endswith(suf):
                prefix = k[: -len(suf)]
                exl3_groups.setdefault(prefix, {})[suf[1:]] = k
                matched = True
                break
        if not matched:
            passthrough_keys.append(k)

    # Check if any EXL3 prefix also has a bias in passthrough_keys
    bias_keys_used: set[str] = set()
    for prefix in exl3_groups:
        bk = f"{prefix}.bias"
        if bk in key_to_shard:
            exl3_groups[prefix]["bias"] = bk
            bias_keys_used.add(bk)
    passthrough_keys = [k for k in passthrough_keys if k not in bias_keys_used]

    open_shards = {sf: safe_open(str(sf), framework="pt", device="cpu") for sf in shard_files}
    out_dir.mkdir(parents=True, exist_ok=True)

    weight_map: dict[str, str] = {}
    shard_tensors: dict[str, torch.Tensor] = {}
    shard_bytes = 0
    shard_idx = 0
    written_shards: list[Path] = []
    total_size = 0
    verification: dict[str, Any] | None = None

    def flush_shard() -> None:
        nonlocal shard_tensors, shard_bytes, shard_idx
        if not shard_tensors:
            return
        shard_idx += 1
        tmp_name = f"shard_{shard_idx:05d}.safetensors"
        tmp_path = out_dir / tmp_name
        save_file(shard_tensors, str(tmp_path))
        for tk in shard_tensors:
            weight_map[tk] = tmp_name
        written_shards.append(tmp_path)
        shard_tensors = {}
        shard_bytes = 0

    def add_tensor(name: str, tensor: torch.Tensor) -> None:
        nonlocal shard_bytes, total_size
        t = tensor.detach().cpu().contiguous()
        nbytes = t.numel() * t.element_size()
        if shard_tensors and shard_bytes + nbytes > max_shard_bytes:
            flush_shard()
        shard_tensors[name] = t
        shard_bytes += nbytes
        total_size += nbytes

    def load_t(key: str, target_dev: str) -> torch.Tensor:
        return open_shards[key_to_shard[key]].get_tensor(key).to(target_dev)

    # 1. Reconstruct every EXL3 linear layer on GPU and transpose (in, out) -> (out, in)
    for prefix in sorted(exl3_groups):
        parts = exl3_groups[prefix]
        trellis = load_t(parts["trellis"], device)
        in_features = int(trellis.shape[0]) * 16
        out_features = int(trellis.shape[1]) * 16
        su = load_t(parts["su"], device) if "su" in parts else None
        sv = load_t(parts["sv"], device) if "sv" in parts else None
        suh = load_t(parts["suh"], device) if "suh" in parts else None
        svh = load_t(parts["svh"], device) if "svh" in parts else None
        mcg = load_t(parts["mcg"], device) if "mcg" in parts else None
        mul1 = load_t(parts["mul1"], device) if "mul1" in parts else None
        bias = load_t(parts["bias"], device) if "bias" in parts else None

        lin = LinearEXL3(
            config=None,
            in_features=in_features,
            out_features=out_features,
            su=su,
            sv=sv,
            suh=suh,
            svh=svh,
            trellis=trellis,
            mcg=mcg,
            mul1=mul1,
            bias=bias,
            key=prefix,
        )
        w_in_out = lin.get_weight_tensor()  # (in_features, out_features) fp16
        w_out_in = w_in_out.t().contiguous()

        if verify_first_linear and verification is None and in_features <= 8192 and out_features <= 32768:
            gen = torch.Generator(device=device).manual_seed(42)
            x = torch.randn((4, in_features), generator=gen, device=device, dtype=torch.half) * 0.1
            y_exl3 = lin.forward(x, params={})
            y_dense = F.linear(x, w_out_in, bias.to(torch.half) if bias is not None else None)
            max_abs = float((y_exl3.float() - y_dense.float()).abs().max().item())
            cos = float(
                F.cosine_similarity(y_exl3.float().flatten(), y_dense.float().flatten(), dim=0).item()
            )
            if cos < 0.999 or max_abs > 0.1:
                raise RuntimeError(
                    f"EXL3 reconstruction verification failed on {prefix}: cos={cos:.6f}, max_abs={max_abs:.6f}"
                )
            verification = {
                "checked_layer": prefix,
                "shape": [out_features, in_features],
                "bitrate_K": lin.K,
                "cosine_similarity": round(cos, 7),
                "max_abs_diff_fp16": round(max_abs, 6),
            }

        out_prefix = normalize_text_weight_key(prefix)
        assert out_prefix is not None
        add_tensor(f"{out_prefix}.weight", w_out_in.to(torch.bfloat16))
        if bias is not None:
            add_tensor(f"{out_prefix}.bias", bias.to(torch.bfloat16))

        del lin, trellis, su, sv, suh, svh, mcg, mul1, bias, w_in_out, w_out_in
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    # 2. Copy pass-through tensors (preserve fp32 for A_log/dt_bias/linear_attn.norm; cast fp16 -> bf16)
    for k in passthrough_keys:
        norm_k = normalize_text_weight_key(k)
        assert norm_k is not None
        t = load_t(k, "cpu")
        if t.dtype == torch.float16:
            t = t.to(torch.bfloat16)
        add_tensor(norm_k, t)

    flush_shard()

    # Rename shards to standard HF model-00001-of-NNNNN.safetensors naming
    n_shards = len(written_shards)
    rename_map: dict[str, str] = {}
    for idx, old_path in enumerate(written_shards, start=1):
        new_name = "model.safetensors" if n_shards == 1 else f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
        old_path.rename(out_dir / new_name)
        rename_map[old_path.name] = new_name

    if n_shards > 1:
        final_map = {k: rename_map[v] for k, v in weight_map.items()}
        index_payload = {"metadata": {"total_size": total_size}, "weight_map": final_map}
        (out_dir / "model.safetensors.index.json").write_text(json.dumps(index_payload, indent=2) + "\n")

    text_cfg = build_text_only_config(base_dir / "config.json")
    (out_dir / "config.json").write_text(json.dumps(text_cfg, indent=2) + "\n")
    copy_tokenizer_assets(base_dir, out_dir)
    return {
        "out_dir": str(out_dir),
        "reconstructed_linears": len(exl3_groups),
        "passthrough_tensors": len(passthrough_keys),
        "shards": n_shards,
        "verification": verification,
    }


def load_probe_rows_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def paired_probe_bootstrap(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    n_boot: int = 20000,
    seed: int = 20261114,
) -> dict[str, Any]:
    """Paired row-level bootstrap comparing two probe runs on token-weighted mean_kl and weighted_kl."""
    if len(rows_a) != len(rows_b) or not rows_a:
        raise ValueError("rows_a and rows_b must be non-empty lists of identical length")

    n = len(rows_a)
    tokens = np.empty(n, dtype=np.float64)
    kl_sum_a = np.empty(n, dtype=np.float64)
    kl_sum_b = np.empty(n, dtype=np.float64)
    domains: list[str] = []

    for i, (ra, rb) in enumerate(zip(rows_a, rows_b, strict=True)):
        if ra.get("domain") != rb.get("domain") or int(ra["tokens"]) != int(rb["tokens"]):
            raise ValueError(f"Row mismatch at index {i}: {ra} vs {rb}")
        tok = float(ra["tokens"])
        tokens[i] = tok
        kl_sum_a[i] = float(ra["mean_kl"]) * tok
        kl_sum_b[i] = float(rb["mean_kl"]) * tok
        domains.append(str(ra["domain"]))

    tot_tokens = float(tokens.sum())
    mean_kl_a = float(kl_sum_a.sum() / tot_tokens)
    mean_kl_b = float(kl_sum_b.sum() / tot_tokens)
    mean_diff = mean_kl_b - mean_kl_a

    # Map domains to integer indices for fast vectorized weighted_kl across bootstrap draws
    unique_doms = sorted(set(domains))
    dom_to_idx = {d: idx for idx, d in enumerate(unique_doms)}
    dom_idx = np.array([dom_to_idx[d] for d in domains], dtype=np.int64)

    def _compute_weighted(k_sum: np.ndarray, tok_arr: np.ndarray) -> float | None:
        dom_dict: dict[str, dict[str, float]] = {}
        for di, dname in enumerate(unique_doms):
            mask = dom_idx == di
            t_s = float(tok_arr[mask].sum())
            if t_s > 0:
                dom_dict[dname] = {"kl": float(k_sum[mask].sum()), "kl_n": t_s}
        try:
            return float(aggregate_weighted_kl(dom_dict))
        except ValueError:
            return None

    w_a = _compute_weighted(kl_sum_a, tokens)
    w_b = _compute_weighted(kl_sum_b, tokens)

    rng = np.random.default_rng(seed)
    boot_mean_diffs = np.empty(n_boot, dtype=np.float64)
    boot_w_diffs: list[float] = []

    # Process in chunks to keep memory modest
    chunk = 2000
    for start in range(0, n_boot, chunk):
        sz = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(sz, n))
        tok_s = tokens[idx]
        ka_s = kl_sum_a[idx]
        kb_s = kl_sum_b[idx]
        tot_s = tok_s.sum(axis=1)
        boot_mean_diffs[start : start + sz] = (kb_s.sum(axis=1) - ka_s.sum(axis=1)) / tot_s

        if w_a is not None and w_b is not None:
            for b_i in range(sz):
                row_idx = idx[b_i]
                # Accumulate domain totals with bincount
                d_tok = np.bincount(dom_idx[row_idx], weights=tokens[row_idx], minlength=len(unique_doms))
                d_ka = np.bincount(dom_idx[row_idx], weights=kl_sum_a[row_idx], minlength=len(unique_doms))
                d_kb = np.bincount(dom_idx[row_idx], weights=kl_sum_b[row_idx], minlength=len(unique_doms))
                da = {
                    unique_doms[di]: {"kl": float(d_ka[di]), "kl_n": float(d_tok[di])}
                    for di in range(len(unique_doms))
                    if d_tok[di] > 0
                }
                db = {
                    unique_doms[di]: {"kl": float(d_kb[di]), "kl_n": float(d_tok[di])}
                    for di in range(len(unique_doms))
                    if d_tok[di] > 0
                }
                try:
                    boot_w_diffs.append(float(aggregate_weighted_kl(db) - aggregate_weighted_kl(da)))
                except ValueError:
                    pass

    row_kl_a = kl_sum_a / np.maximum(tokens, 1.0)
    row_kl_b = kl_sum_b / np.maximum(tokens, 1.0)
    out: dict[str, Any] = {
        "rows": n,
        "tokens": int(tot_tokens),
        "mean_kl_a": round(mean_kl_a, 6),
        "mean_kl_b": round(mean_kl_b, 6),
        "mean_kl_diff_b_minus_a": round(mean_diff, 6),
        "mean_kl_ratio_b_over_a": round(mean_kl_b / mean_kl_a, 4) if mean_kl_a > 0 else None,
        "ci95_mean_kl_diff": [
            round(float(np.percentile(boot_mean_diffs, 2.5)), 6),
            round(float(np.percentile(boot_mean_diffs, 97.5)), 6),
        ],
        "p_a_beats_b_mean_kl": round(float(np.mean(boot_mean_diffs > 0)), 5),
        "row_wins_a": int(np.sum(row_kl_a < row_kl_b)),
        "row_wins_b": int(np.sum(row_kl_b < row_kl_a)),
        "row_ties": int(np.sum(row_kl_a == row_kl_b)),
    }
    if w_a is not None and w_b is not None and boot_w_diffs:
        arr_w = np.asarray(boot_w_diffs, dtype=np.float64)
        out.update(
            {
                "weighted_kl_protocol": WEIGHTED_KL_PROTOCOL,
                "weighted_kl_a": round(w_a, 6),
                "weighted_kl_b": round(w_b, 6),
                "weighted_kl_diff_b_minus_a": round(w_b - w_a, 6),
                "weighted_kl_ratio_b_over_a": round(w_b / w_a, 4) if w_a > 0 else None,
                "ci95_weighted_kl_diff": [
                    round(float(np.percentile(arr_w, 2.5)), 6),
                    round(float(np.percentile(arr_w, 97.5)), 6),
                ],
                "p_a_beats_b_weighted_kl": round(float(np.mean(arr_w > 0)), 5),
            }
        )
    return out


def _find_latest_evalscope_subdir(work_dir: Path, subdir_name: str) -> Path | None:
    direct = work_dir / subdir_name
    if direct.is_dir():
        return direct
    candidates = sorted(p for p in work_dir.glob(f"*/{subdir_name}") if p.is_dir())
    return candidates[-1] if candidates else None


def extract_evalscope_metrics_and_samples(
    work_dir: Path,
    *,
    max_tokens: int = 2048,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Extract structured summary metrics and per-sample binary scores from an EvalScope run directory."""
    reports_dir = _find_latest_evalscope_subdir(work_dir, "reports")
    reviews_dir = _find_latest_evalscope_subdir(work_dir, "reviews")
    summary: dict[str, Any] = {}
    sample_scores: dict[str, dict[str, float]] = {}

    if reports_dir is not None:
        for rep_file in sorted(reports_dir.rglob("*.json")):
            data = json.loads(rep_file.read_text())
            ds_name = data.get("dataset_name") or rep_file.stem
            metric_map: dict[str, float] = {}
            metric_num: dict[str, int] = {}
            for m in data.get("metrics", []):
                ident = m.get("identity") or {}
                mname = ident.get("name") or m.get("name") or "score"
                agg = ident.get("aggregation")
                key = f"{mname}:{agg}" if (agg and mname in metric_map) else mname
                metric_map[key] = float(m["score"])
                if m.get("num") is not None:
                    metric_num[key] = int(m["num"])

            if ds_name == "ifeval":
                summary["ifeval"] = {
                    "prompt_strict": round(metric_map.get("prompt_level_strict", 0.0), 4),
                    "inst_strict": round(metric_map.get("inst_level_strict", 0.0), 4),
                    "prompt_loose": round(metric_map.get("prompt_level_loose", 0.0), 4),
                    "inst_loose": round(metric_map.get("inst_level_loose", 0.0), 4),
                    "num": int(data.get("num", metric_num.get("prompt_level_strict", 0))),
                    "num_instructions": int(metric_num.get("inst_level_strict", 0)),
                }
            elif ds_name == "math_500":
                summary["math_500"] = {
                    "accuracy": round(metric_map.get("accuracy", 0.0), 4),
                    "num": int(data.get("num", metric_num.get("accuracy", 0))),
                }
            elif ds_name == "humaneval_plus":
                acc = metric_map.get("accuracy:pass_at_k", metric_map.get("accuracy", 0.0))
                n_he = int(data.get("num", metric_num.get("accuracy", 0)))
                summary["humaneval_plus"] = {
                    "pass_at_1": round(acc, 4),
                    "num": n_he,
                }

    if reviews_dir is not None:
        ifeval_scores: dict[str, float] = {}
        math_scores: dict[str, float] = {}
        math_out_tokens: list[int] = []
        he_scores: dict[str, float] = {}

        for rev_file in sorted(reviews_dir.rglob("*.jsonl")):
            fname = rev_file.name
            with rev_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    s_obj = rec.get("sample_score", {})
                    val = s_obj.get("score", {}).get("value", {})
                    meta = s_obj.get("sample_metadata") or {}
                    idx_fallback = f"{fname}:{rec.get('index')}"

                    if fname.startswith("ifeval"):
                        sid = str(meta.get("key", idx_fallback))
                        if "prompt_level_strict" in val:
                            ifeval_scores[sid] = float(val["prompt_level_strict"])
                    elif fname.startswith("math_500"):
                        sid = str(meta.get("question_id", idx_fallback))
                        if "accuracy" in val:
                            math_scores[sid] = float(val["accuracy"])
                        msgs = rec.get("messages") or []
                        if msgs:
                            pm = msgs[-1].get("perf_metrics") or {}
                            if pm.get("output_tokens") is not None:
                                math_out_tokens.append(int(pm["output_tokens"]))
                    elif fname.startswith("humaneval_plus"):
                        sid = str(meta.get("task_id", idx_fallback))
                        raw_acc = val.get("acc", val.get("accuracy"))
                        if raw_acc is not None:
                            he_scores[sid] = (
                                float(bool(raw_acc)) if isinstance(raw_acc, bool) else float(raw_acc)
                            )

        if ifeval_scores:
            sample_scores["ifeval"] = ifeval_scores
            summary.setdefault("ifeval", {})["num_reviewed"] = len(ifeval_scores)
        if math_scores:
            sample_scores["math_500"] = math_scores
            m_sum = summary.setdefault("math_500", {})
            m_sum["num_reviewed"] = len(math_scores)
            m_sum["num_correct"] = int(round(sum(math_scores.values())))
            if math_out_tokens:
                m_sum["avg_output_tokens"] = round(float(np.mean(math_out_tokens)), 1)
                m_sum["truncated_at_max_tokens"] = int(sum(1 for t in math_out_tokens if t >= max_tokens))
        if he_scores:
            sample_scores["humaneval_plus"] = he_scores
            h_sum = summary.setdefault("humaneval_plus", {})
            h_sum["num_reviewed"] = len(he_scores)
            h_sum["passed"] = int(round(sum(he_scores.values())))

    return summary, sample_scores


def paired_evalscope_bootstrap_map(
    samples_a: dict[str, dict[str, float]],
    samples_b: dict[str, dict[str, float]],
    *,
    n_boot: int = 20000,
    seed: int = 20261114,
) -> dict[str, Any]:
    """Compute paired binary bootstrap CIs across all overlapping benchmarks between arm A and arm B."""
    out: dict[str, Any] = {}
    for ds in ("ifeval", "math_500", "humaneval_plus"):
        ma = samples_a.get(ds)
        mb = samples_b.get(ds)
        if not ma or not mb:
            continue
        common = sorted(set(ma) & set(mb))
        if not common:
            continue
        out[ds] = paired_binary_bootstrap(
            [ma[k] for k in common],
            [mb[k] for k in common],
            n_boot=n_boot,
            seed=seed,
        )
    return out
