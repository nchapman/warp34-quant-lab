"""ARC-1.0 (`rayline-ai/ARC-1.0`) W4A16-HGPTQ quantization, Qwen slot unmasking, and fidelity probing.

Supports:
- 12-layer `Qwen3_5ForConditionalGeneration` base router (`model.language_model.layers.0..11` + `model.visual.*`)
  with `yarn` (`factor=4.0`) RoPE and last-token readout (`runtime/heads.safetensors`, `signals.*`).
- 12-layer upper question-head continuation (`question_head/qh4in-full-e3`: `upper.safetensors` + `pointer.safetensors`).
- Unmasking the three trained local Qwen routing slots in `runtime/arc-runtime.json`, `config.json`, and `provenance.json`:
  - `models[53]` (`withheld-model-42`) -> `qwen3.8-flash-next` (`wire: "qwen/qwen3.8-flash-next"`, 8 arms across 2 effort controls)
  - `models[54]` (`withheld-model-43`) -> `qwen3.5-9b` (`wire: "qwen/qwen3.5-9b"`, 16 arms across 4 steering controls)
  - `models[55]` (`withheld-model-44`) -> `qwen3.8-27b` (`wire: "qwen/qwen3.8-27b"`, 16 arms across 4 steering controls)
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

from warp_quant.cachepath import resolve_snapshot
from warp_quant.decision import (
    TEXT_QUANT_SUBMODULES,
    _quantize_single_text_layer,
    _restore_rotary,
    _ShardReader,
)
from warp_quant.hadamard_gptq import (
    ROOT,
    _assert_packed_alignment,
    _decompress_packed_2d,
    _hadamard_rotate_last_dim,
    _update_config_for_components,
)
from warp_quant.lablog import log_event, run_done
from warp_quant.models import quant_recipe, resolve
from warp_quant.stamp import ensure_license_files, ensure_processor_configs

# Canonical thinking-control SHA-256 IDs used by rayline-arc-decide runtime catalogs
CONTROL_ID_DEFAULT = "628d285537ca7cdb52c4bd3433cea7a1ec36005a14dbc67ffa1c40487d7b8ad0"
CONTROL_ID_HIGH = "8c4faa90467d7151b03810424e673723a62bdface213d36659862037aaf4dc2d"

# Trained Qwen slots in ARC-1.0's 56-model vocabulary
UNMASKED_QWEN_SLOTS: dict[int, dict[str, Any]] = {
    53: {
        "withheld_name": "withheld-model-42",
        "name": "qwen3.8-flash-next",
        "wire": "qwen/qwen3.8-flash-next",
        "served_arms": [741, 861, 1268, 1296, 2563, 2767, 4294, 5113],
        "actions": [
            {
                "arms": [741, 1268, 1296, 5113],
                "control_id": CONTROL_ID_DEFAULT,
                "effort": "default",
                "model": 53,
            },
            {
                "arms": [861, 2563, 2767, 4294],
                "control_id": CONTROL_ID_HIGH,
                "effort": "high",
                "model": 53,
            },
        ],
    },
    54: {
        "withheld_name": "withheld-model-43",
        "name": "qwen3.5-9b",
        "wire": "qwen/qwen3.5-9b",
        "served_arms": [
            195,
            234,
            516,
            1067,
            1085,
            1595,
            1652,
            2768,
            2900,
            2944,
            3450,
            3517,
            3645,
            3783,
            4432,
            4595,
        ],
        "actions": [],
    },
    55: {
        "withheld_name": "withheld-model-44",
        "name": "qwen3.8-27b",
        "wire": "qwen/qwen3.8-27b",
        "served_arms": [
            1079,
            1093,
            1163,
            1354,
            1889,
            2103,
            2454,
            2808,
            3047,
            3129,
            3551,
            3781,
            3993,
            4763,
            4916,
            5338,
        ],
        "actions": [],
    },
}


def _resolve_local_or_hub(repo_id: str, revision: str | None = None) -> Path:
    if Path(repo_id).exists():
        return Path(repo_id)
    import os

    offline_env = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    try:
        return Path(resolve_snapshot(repo_id, revision=revision, offline=True))
    except (FileNotFoundError, OSError):
        if offline_env:
            raise
        return Path(resolve_snapshot(repo_id, revision=revision, offline=False))


def is_arc_checkpoint(path: str | Path) -> bool:
    p = Path(path)
    return (p / "runtime" / "arc-runtime.json").is_file() and (
        (p / "runtime" / "heads.safetensors").is_file() or (p / "arc_release.json").is_file()
    )


def unmask_arc_qwen_slots(runtime: dict[str, Any]) -> dict[str, Any]:
    """Unmask trained Qwen routing slots [53..55] in an arc-runtime.json dictionary."""
    rt = copy.deepcopy(runtime)
    models = rt["models"]
    candidates = rt["stage_one"]["candidates"]
    served = rt["arms"]["served"]
    actions = rt["actions"]

    for slot_idx, spec in UNMASKED_QWEN_SLOTS.items():
        models[slot_idx] = {"name": spec["name"], "wire": spec["wire"]}
        cand = dict(candidates[slot_idx])
        cand["model"] = slot_idx
        cand.pop("withheld", None)
        candidates[slot_idx] = cand
        for arm_row in spec["served_arms"]:
            served[arm_row] = True
        if spec["actions"] and not any(a.get("model") == slot_idx for a in actions):
            actions.extend(copy.deepcopy(spec["actions"]))

    return rt


def update_arc_config_slots(cfg: dict[str, Any]) -> dict[str, Any]:
    """Replace withheld-model-42..44 with unmasked Qwen model names in config.json['arc']."""
    out = copy.deepcopy(cfg)
    arc = out.get("arc")
    if not isinstance(arc, dict):
        return out
    name_map = {spec["withheld_name"]: spec["name"] for spec in UNMASKED_QWEN_SLOTS.values()}
    s1_cands = arc.get("stage_one", {}).get("candidates", [])
    for entry in s1_cands:
        if isinstance(entry, dict) and entry.get("model") in name_map:
            entry["model"] = name_map[entry["model"]]
    s2_models = arc.get("stage_two", {}).get("arms", {}).get("models", [])
    for idx, m in enumerate(s2_models):
        if m in name_map:
            s2_models[idx] = name_map[m]
    return out


def unmask_arc_goldens_text(raw_jsonl: str) -> str:
    """Translate withheld-model-42..44 action IDs and wire keys in goldens.jsonl text."""
    controls = (
        "628d285537ca7cdb52c4bd3433cea7a1ec36005a14dbc67ffa1c40487d7b8ad0",
        "17ce2fd14e762777a95db605228cc8f731d2d3e200c4c5d07a735f85fca2eae1",
        "77f4674fea6c3b4f3f53ab0337b61fc725ef66e045f317767efdbf94b6979d48",
        "605a61ada934f704248cb99cc84179bf15747302120d11e67ba723db0d8d955b",
        "8c4faa90467d7151b03810424e673723a62bdface213d36659862037aaf4dc2d",
    )

    def _aid(model: str, cid: str) -> str:
        payload = json.dumps(
            {"control_id": cid, "model": model}, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    out = raw_jsonl
    for spec in UNMASKED_QWEN_SLOTS.values():
        old_name = spec["withheld_name"]
        new_name = spec["name"]
        for c in controls:
            out = out.replace(_aid(old_name, c), _aid(new_name, c))
        out = out.replace(f"withheld/{old_name}", spec["wire"])
        out = out.replace(old_name, new_name)
    return out


def write_unmasked_arc_runtime(src_runtime_dir: Path, dst_runtime_dir: Path) -> dict[str, Any]:
    """Copy runtime/ bundle, unmask Qwen slots [53..55], and refresh runtime/manifest.json."""
    if dst_runtime_dir.exists():
        shutil.rmtree(dst_runtime_dir)
    shutil.copytree(src_runtime_dir, dst_runtime_dir)

    rt_path = dst_runtime_dir / "arc-runtime.json"
    rt = json.loads(rt_path.read_text())
    unmasked_rt = unmask_arc_qwen_slots(rt)
    rt_bytes = (json.dumps(unmasked_rt, indent=1, sort_keys=True) + "\n").encode("utf-8")
    rt_path.write_bytes(rt_bytes)

    manifest_path = dst_runtime_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for fname in list(manifest.get("files", {}).keys()):
        fpath = dst_runtime_dir / fname
        data = fpath.read_bytes()
        manifest["files"][fname] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def _build_arc_calib_batches(
    tokenizer: Any,
    *,
    num_samples: int = 256,
    max_seq_len: int = 1024,
) -> list[torch.Tensor]:
    """Build canonical <|im_start|>role\\n...<|im_end|>\\n sequences for ARC-1.0 calibration."""
    im_start = 248045
    im_end = 248046
    nl = tokenizer.encode("\n", add_special_tokens=False)[0]

    def _role_block(role: str, text: str) -> list[int]:
        return [
            im_start,
            *tokenizer.encode(f"{role}\n{text}", add_special_tokens=False),
            im_end,
            nl,
        ]

    seqs: list[torch.Tensor] = []
    for jsonl_name, target_count in (
        ("calib-cascade-v1.jsonl", (num_samples * 3) // 4),
        ("traj-v1.jsonl", num_samples),
    ):
        jsonl_path = ROOT / "data" / jsonl_name
        if not jsonl_path.is_file():
            continue
        with jsonl_path.open() as f:
            for line in f:
                if len(seqs) >= target_count:
                    break
                row = json.loads(line)
                msgs = row.get("conversations") or row.get("messages") or []
                ids: list[int] = []
                for m in msgs:
                    role = m.get("role", "user")
                    if role not in ("system", "user", "assistant", "tool"):
                        role = "user"
                    content = m.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
                    ids.extend(_role_block(role, str(content)[:1600]))
                    if len(ids) >= max_seq_len:
                        break
                ids = ids[:max_seq_len]
                if len(ids) >= 16:
                    seqs.append(torch.tensor([ids], dtype=torch.long))

    if not seqs:
        raise RuntimeError("No calibration sequences constructed for ARC-1.0")
    return seqs[:num_samples]


def _build_arc_tcfg(cfg_raw: dict[str, Any]) -> Qwen3_5TextConfig:
    tcfg_dict = dict(cfg_raw.get("text_config", cfg_raw))
    tcfg_dict["use_cache"] = False
    tcfg_dict["_attn_implementation"] = "sdpa"
    return Qwen3_5TextConfig(**tcfg_dict)


@torch.no_grad()
def _load_bf16_arc_text_model(base_dir: Path, device: str | torch.device = "cuda") -> Qwen3_5TextModel:
    """Load the BF16/FP32-mixed 12-layer Qwen3_5TextModel from ARC-1.0 preserving exact dtypes."""
    cfg_raw = json.loads((base_dir / "config.json").read_text())
    tcfg = _build_arc_tcfg(cfg_raw)
    with torch.device("meta"):
        model = Qwen3_5TextModel(tcfg)
    sd: dict[str, torch.Tensor] = {}
    with safe_open(str(base_dir / "model.safetensors"), framework="pt", device="cpu") as h:
        for k in h.keys():
            if k.startswith("model.language_model."):
                sd[k[len("model.language_model.") :]] = h.get_tensor(k)
    model.load_state_dict(sd, strict=True, assign=True)
    _restore_rotary(model, device)
    model.to(device)
    model.eval()
    return model


def _attach_hadamard_hooks(model: Qwen3_5TextModel, hadamard_block_size: int = 64) -> None:
    if hadamard_block_size <= 0:
        return
    for layer in model.layers:
        for sub_name in TEXT_QUANT_SUBMODULES:
            parts = sub_name.split(".")
            mod: Any = layer
            for p in parts:
                mod = getattr(mod, p, None)
            if isinstance(mod, torch.nn.Linear):
                k_in = mod.weight.shape[1]
                bs = min(hadamard_block_size, k_in & -k_in)
                if bs > 1:
                    mod.register_forward_pre_hook(
                        lambda _m, hook_args, b=bs: (
                            _hadamard_rotate_last_dim(hook_args[0], b, compute_dtype=torch.float32),
                            *hook_args[1:],
                        )
                    )


@torch.no_grad()
def _load_decompressed_arc_text_model(quant_dir: Path, device: str = "cuda") -> Qwen3_5TextModel:
    """Reconstruct a decompressed + online-Hadamard Qwen3_5TextModel from a quantized ARC dir."""
    cfg_raw = json.loads((quant_dir / "config.json").read_text())
    tcfg = _build_arc_tcfg(cfg_raw)
    with torch.device("meta"):
        model = Qwen3_5TextModel(tcfg)

    sd: dict[str, torch.Tensor] = {}
    with safe_open(str(quant_dir / "model.safetensors"), framework="pt", device="cpu") as h:
        keys = set(h.keys())
        for k in keys:
            if not k.startswith("model.language_model."):
                continue
            short_k = k[len("model.language_model.") :]
            if short_k.endswith(".weight_packed"):
                base = short_k[: -len(".weight_packed")]
                packed = h.get_tensor(k)
                scale = h.get_tensor(f"model.language_model.{base}.weight_scale")
                shape_t = h.get_tensor(f"model.language_model.{base}.weight_shape")
                dq = _decompress_packed_2d(packed, scale, shape_t).to(torch.bfloat16)
                sd[f"{base}.weight"] = dq
            elif short_k.endswith((".weight_scale", ".weight_shape")):
                continue
            else:
                sd[short_k] = h.get_tensor(k)

    model.load_state_dict(sd, strict=True, assign=True)
    _restore_rotary(model, device)
    model.to(device)
    _attach_hadamard_hooks(model, 64)
    model.eval()
    return model


@torch.no_grad()
def _load_decompressed_arc_24l_model(quant_dir: Path, device: str = "cuda") -> Qwen3_5TextModel:
    """Reconstruct the full 24-layer decompressed model (layers 0..11 + question_head 12..23)."""
    cfg_raw = json.loads((quant_dir / "config.json").read_text())
    tcfg_dict = dict(cfg_raw.get("text_config", cfg_raw))
    tcfg_dict["num_hidden_layers"] = 24
    tcfg_dict["layer_types"] = list(tcfg_dict["layer_types"]) * 2
    tcfg_dict["use_cache"] = False
    tcfg_dict["_attn_implementation"] = "sdpa"
    tcfg = Qwen3_5TextConfig(**tcfg_dict)
    with torch.device("meta"):
        model = Qwen3_5TextModel(tcfg)

    sd: dict[str, torch.Tensor] = {}
    for st_path in (
        quant_dir / "model.safetensors",
        quant_dir / "question_head" / "qh4in-full-e3" / "upper.safetensors",
    ):
        with safe_open(str(st_path), framework="pt", device="cpu") as h:
            keys = set(h.keys())
            for k in keys:
                if k.startswith("model.language_model."):
                    short_k = k[len("model.language_model.") :]
                    prefix = "model.language_model."
                elif k.startswith("layers.") or k == "norm.weight":
                    short_k = k
                    prefix = ""
                else:
                    continue
                if short_k.endswith(".weight_packed"):
                    base = short_k[: -len(".weight_packed")]
                    packed = h.get_tensor(k)
                    scale = h.get_tensor(f"{prefix}{base}.weight_scale")
                    shape_t = h.get_tensor(f"{prefix}{base}.weight_shape")
                    dq = _decompress_packed_2d(packed, scale, shape_t).to(torch.bfloat16)
                    sd[f"{base}.weight"] = dq
                elif short_k.endswith((".weight_scale", ".weight_shape")):
                    continue
                else:
                    sd[short_k] = h.get_tensor(k)

    model.load_state_dict(sd, strict=True, assign=True)
    _restore_rotary(model, device)
    model.to(device)
    _attach_hadamard_hooks(model, 64)
    model.eval()
    return model


def run_quantize_arc(args: argparse.Namespace) -> Path:
    """Quantize rayline-ai/ARC-1.0 (base 12L + question_head 12L + visual) to W4A16-HGPTQ."""
    t0 = time.time()
    hub_id, info = resolve(args.model)
    rec = quant_recipe(info)
    base_dir = _resolve_local_or_hub(hub_id, revision=info.get("hub_revision"))
    out_dir = Path(args.output_dir)
    if out_dir.exists() or out_dir.is_symlink():
        raise ValueError(f"--output-dir already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    group_size = int(getattr(args, "group_size", None) or rec["group_size"])
    hadamard_block_size = int(
        getattr(args, "hadamard_block_size", None)
        if getattr(args, "hadamard_block_size", None) is not None
        else rec["hadamard_block_size"]
    )
    actorder = getattr(args, "actorder", None) or rec["actorder"]
    dampening_frac = float(
        getattr(args, "dampening_frac", None)
        if getattr(args, "dampening_frac", None) is not None
        else rec["dampening_frac"]
    )
    num_samples = int(getattr(args, "num_samples", None) or rec.get("num_samples", 256))
    max_seq_len = int(getattr(args, "max_seq_length", None) or rec.get("max_seq_length", 1024))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_event(
        "QUANTIZE_ARC_START",
        model=args.model,
        base_dir=str(base_dir),
        output_dir=str(out_dir),
        group_size=group_size,
        hadamard_block_size=hadamard_block_size,
        num_samples=num_samples,
        max_seq_len=max_seq_len,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(base_dir / "runtime"), trust_remote_code=True)
    calib_seqs = _build_arc_calib_batches(tokenizer, num_samples=num_samples, max_seq_len=max_seq_len)

    # 1. Load & calibrate base 12-layer text model under ARC-1.0's yarn-f4 RoPE
    cfg_raw = json.loads((base_dir / "config.json").read_text())
    reader = _ShardReader(base_dir)
    text_model = _load_bf16_arc_text_model(base_dir, device=device)

    state_batches: list[list[torch.Tensor]] = [[]]
    pos_cache: dict[tuple[int, int], tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]] = {}
    with torch.no_grad():
        for ids in calib_seqs:
            ids_d = ids.to(device)
            h = text_model.embed_tokens(ids_d)
            bsz, seq_len = int(ids_d.shape[0]), int(ids_d.shape[1])
            if (bsz, seq_len) not in pos_cache:
                pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
                pe = text_model.rotary_emb(h, pos)
                pos_cache[(bsz, seq_len)] = ((pe[0].detach(), pe[1].detach()), pos)
            state_batches[0].append(h.detach().cpu())

    packed_sd: dict[str, torch.Tensor] = {
        "model.language_model.embed_tokens.weight": reader.get("model.language_model.embed_tokens.weight"),
        "model.language_model.norm.weight": reader.get("model.language_model.norm.weight"),
    }

    for layer_idx, layer in enumerate(text_model.layers):
        t_layer = time.time()
        layer_prefix = f"model.language_model.layers.{layer_idx}"
        for pname, _param in layer.named_parameters():
            if not any(pname == f"{s}.weight" for s in TEXT_QUANT_SUBMODULES):
                full_key = f"{layer_prefix}.{pname}"
                packed_sd[full_key] = reader.get(full_key)

        layer_packed, state_batches = _quantize_single_text_layer(
            layer_idx,
            layer,
            state_batches,
            pos_cache,
            device=device,
            group_size=group_size,
            hadamard_block_size=hadamard_block_size,
            actorder=actorder,
            dampening_frac=dampening_frac,
        )
        for prefix, sub_dict in layer_packed.items():
            for suffix, tensor in sub_dict.items():
                packed_sd[f"{prefix}.{suffix}"] = tensor.detach().cpu()
        print(f"  [ARC base L{layer_idx:02d}/11] quantized in {time.time() - t_layer:.1f}s")

    # 2. Reuse W4A16-HGPTQ visual tower from Qwen3.5-0.8B-W4A16-HGPTQ donor
    vision_donor_repo = info.get("vision_donor_repo", "nchapman/Qwen3.5-0.8B-W4A16-HGPTQ")
    vision_donor_dir = _resolve_local_or_hub(vision_donor_repo, revision=info.get("vision_donor_revision"))
    donor_reader = _ShardReader(vision_donor_dir)
    for k in donor_reader.where:
        if k.startswith("model.visual."):
            packed_sd[k] = donor_reader.get(k)

    # 3. Preserve all non-model.* ARC head tensors (signals.*, stage_one.*, stage_two.*) in model.safetensors
    with safe_open(str(base_dir / "model.safetensors"), framework="pt", device="cpu") as bh:
        for k in bh.keys():
            if not k.startswith("model."):
                packed_sd[k] = bh.get_tensor(k)

    save_file(packed_sd, str(out_dir / "model.safetensors"))
    _assert_packed_alignment(out_dir)

    # 4. Write config.json with quantization_config and unmasked Qwen slots
    out_cfg = update_arc_config_slots(cfg_raw)
    donor_cfg = json.loads((vision_donor_dir / "config.json").read_text())
    out_cfg["quantization_config"] = copy.deepcopy(donor_cfg["quantization_config"])
    (out_dir / "config.json").write_text(json.dumps(out_cfg, indent=2, sort_keys=True) + "\n")
    _update_config_for_components(
        out_dir,
        group_size=group_size,
        actorder=actorder,
        hadamard_block_size=hadamard_block_size,
        quantize_mtp=False,
        quantize_vision=True,
        vision_fc2_channel=True,
        quantize_lm_head_w8=False,
    )

    # 5. Copy ARC metadata, goldens, tokenizer/processor, licenses, and unmasked runtime
    for fname in (
        "goldens.jsonl",
        "goldens.signals.jsonl",
        "chat_template.jinja",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
        "merges.txt",
        "presets.json",
        "CHANGELOG.md",
    ):
        src_f = base_dir / fname
        if src_f.is_file():
            shutil.copy2(src_f, out_dir / fname)

    if (out_dir / "goldens.jsonl").is_file():
        (out_dir / "goldens.jsonl").write_text(
            unmask_arc_goldens_text((out_dir / "goldens.jsonl").read_text())
        )

    if (base_dir / "provenance.json").is_file():
        prov = json.loads((base_dir / "provenance.json").read_text())
        offered = list(prov.get("offered_models", []))
        for spec in UNMASKED_QWEN_SLOTS.values():
            if spec["wire"] not in offered:
                offered.append(spec["wire"])
        prov["offered_models"] = sorted(offered)
        (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n")

    ensure_processor_configs(
        out_dir,
        info.get("processor_hub_fallback"),
        info.get("processor_hub_fallback_revision"),
    )
    ensure_license_files(out_dir, base_dir, info.get("license_files"))
    write_unmasked_arc_runtime(base_dir / "runtime", out_dir / "runtime")

    # 6. Quantize question_head/qh4in-full-e3 (layers 12..23 -> upper.safetensors + pointer.safetensors)
    qh_src = base_dir / "question_head" / "qh4in-full-e3"
    if qh_src.is_dir():
        qh_out = out_dir / "question_head" / "qh4in-full-e3"
        qh_out.mkdir(parents=True, exist_ok=True)
        qh_manifest = json.loads((qh_src / "manifest.json").read_text())

        tcfg_dict = dict(cfg_raw["text_config"])
        tcfg_dict["num_hidden_layers"] = 24
        tcfg_dict["layer_types"] = list(cfg_raw["text_config"]["layer_types"]) * 2
        tcfg_dict["use_cache"] = False
        tcfg_dict["_attn_implementation"] = "sdpa"
        tcfg24 = Qwen3_5TextConfig(**tcfg_dict)
        with torch.device("meta"):
            upper_model = Qwen3_5TextModel(tcfg24)

        upper_sd: dict[str, torch.Tensor] = {}
        with safe_open(str(qh_src / "upper.safetensors"), framework="pt", device="cpu") as qh_h:
            for k in qh_h.keys():
                short_k = k[len("model.language_model.") :] if k.startswith("model.language_model.") else k
                upper_sd[short_k] = qh_h.get_tensor(k).to(torch.bfloat16)
        upper_model.load_state_dict(upper_sd, strict=False, assign=True)

        qh_packed: dict[str, torch.Tensor] = {}
        if "norm.weight" in upper_sd:
            qh_packed["model.language_model.norm.weight"] = (
                upper_sd["norm.weight"].detach().cpu().to(torch.bfloat16)
            )
        for layer_idx in range(12, 24):
            t_layer = time.time()
            layer = upper_model.layers[layer_idx].to(device)
            _restore_rotary(layer, device)
            layer.eval()
            layer_prefix = f"model.language_model.layers.{layer_idx}"
            for pname, param in layer.named_parameters():
                if not any(pname == f"{s}.weight" for s in TEXT_QUANT_SUBMODULES):
                    qh_packed[f"{layer_prefix}.{pname}"] = param.detach().cpu()

            lp, state_batches = _quantize_single_text_layer(
                layer_idx,
                layer,
                state_batches,
                pos_cache,
                device=device,
                group_size=group_size,
                hadamard_block_size=hadamard_block_size,
                actorder=actorder,
                dampening_frac=dampening_frac,
            )
            for prefix, sub_dict in lp.items():
                for suffix, tensor in sub_dict.items():
                    qh_packed[f"{prefix}.{suffix}"] = tensor.detach().cpu()
            print(f"  [ARC question_head L{layer_idx:02d}/23] quantized in {time.time() - t_layer:.1f}s")

        save_file(qh_packed, str(qh_out / "upper.safetensors"))
        shutil.copy2(qh_src / "pointer.safetensors", qh_out / "pointer.safetensors")
        _assert_packed_alignment(qh_out)

        # Update question_head manifest with packed upper_tensors metadata and quantization_config
        upper_tensors_meta: dict[str, Any] = {}
        for k in sorted(qh_packed.keys()):
            t = qh_packed[k]
            raw_b = t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            upper_tensors_meta[k] = {
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "sha256": hashlib.sha256(raw_b).hexdigest(),
            }
        qh_manifest["upper_tensors"] = upper_tensors_meta
        qh_manifest["quantization_config"] = copy.deepcopy(
            json.loads((out_dir / "config.json").read_text())["quantization_config"]
        )
        (qh_out / "manifest.json").write_text(json.dumps(qh_manifest, indent=1, sort_keys=True) + "\n")

    if (base_dir / "arc_release.json").is_file():
        rel = json.loads((base_dir / "arc_release.json").read_text())
        if (out_dir / "question_head" / "qh4in-full-e3" / "manifest.json").is_file():
            rel["question_head"]["manifest_sha256"] = hashlib.sha256(
                (out_dir / "question_head" / "qh4in-full-e3" / "manifest.json").read_bytes()
            ).hexdigest()
        (out_dir / "arc_release.json").write_text(json.dumps(rel, indent=1, sort_keys=True) + "\n")

    elapsed = round(time.time() - t0, 2)
    run_done(output_dir=str(out_dir), elapsed_s=elapsed)
    return out_dir


def _eval_arc_signals_from_sd(
    r: torch.Tensor,
    sig_tensors: dict[str, torch.Tensor],
    sig_meta: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Evaluate ARC-1.0's 3 MLP+Platt signal heads (pii.payload, topic.domain, topic.operation)."""
    r_f = r.float().cpu()
    out: dict[str, dict[str, float]] = {}
    for hname in ("pii.payload", "topic.domain", "topic.operation"):
        mean = sig_tensors[f"signals.{hname}.mean"].float()
        scale = sig_tensors[f"signals.{hname}.scale"].float()
        w0 = sig_tensors[f"signals.{hname}.net.0.weight"].float()
        b0 = sig_tensors[f"signals.{hname}.net.0.bias"].float()
        w2 = sig_tensors[f"signals.{hname}.net.2.weight"].float()
        b2 = sig_tensors[f"signals.{hname}.net.2.bias"].float()
        platt = sig_tensors[f"signals.{hname}.platt"].double()

        x = (r_f - mean) / scale
        h = F.silu(F.linear(x, w0, b0))
        logits = F.linear(h, w2, b2).double()
        probs = torch.sigmoid(platt[:, 0] * logits + platt[:, 1])
        classes = sig_meta[hname]["classes"]
        out[hname] = {c: float(probs[i].item()) for i, c in enumerate(classes)}
    return out


@torch.no_grad()
def run_probe_arc(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate readout cosine, signal heads, and question-head pointer parity for ARC-1.0."""
    hub_id, info = resolve(args.model)
    base_dir = _resolve_local_or_hub(hub_id, revision=info.get("hub_revision"))
    quant_dir = Path(args.quant[0] if isinstance(args.quant, list) else args.quant)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(base_dir / "runtime"), trust_remote_code=True)
    cfg_raw = json.loads((base_dir / "config.json").read_text())
    bf16_model = _load_bf16_arc_text_model(base_dir, device=device)

    quant_model = _load_decompressed_arc_text_model(quant_dir, device=device)

    sig_tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(quant_dir / "model.safetensors"), framework="pt", device="cpu") as h:
        for k in h.keys():
            if k.startswith("signals."):
                sig_tensors[k] = h.get_tensor(k)
    sig_meta = json.loads((quant_dir / "config.json").read_text())["arc"]["signals"]["heads"]

    im_start, im_end = 248045, 248046
    nl = tokenizer.encode("\n", add_special_tokens=False)[0]
    cosines: list[float] = []
    abs_diffs: list[float] = []
    dom_match = 0
    op_match = 0
    dom_total = 0
    pii_match = 0
    pii_total = 0

    signals_jsonl = quant_dir / "goldens.signals.jsonl"
    if not signals_jsonl.is_file():
        signals_jsonl = base_dir / "goldens.signals.jsonl"
    if not signals_jsonl.is_file():
        raise FileNotFoundError(f"Missing goldens.signals.jsonl in {quant_dir} and {base_dir}")

    with signals_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            msgs = row["request"]["request"]["messages"]
            ids: list[int] = []
            for m in msgs:
                role = m.get("role", "user")
                c = m.get("content", "")
                if isinstance(c, list):
                    c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
                ids.extend(
                    [
                        im_start,
                        *tokenizer.encode(f"{role}\n{c}", add_special_tokens=False),
                        im_end,
                        nl,
                    ]
                )
            ids_t = torch.tensor([ids[:8192]], dtype=torch.long, device=device)
            r_bf16 = bf16_model(input_ids=ids_t).last_hidden_state[0, -1].float().cpu()
            r_q = quant_model(input_ids=ids_t).last_hidden_state[0, -1].float().cpu()
            cosines.append(float(F.cosine_similarity(r_bf16.unsqueeze(0), r_q.unsqueeze(0)).item()))

            pred = _eval_arc_signals_from_sd(r_q, sig_tensors, sig_meta)
            exp_bf16 = _eval_arc_signals_from_sd(r_bf16, sig_tensors, sig_meta)
            dom_total += 1
            if max(exp_bf16["topic.domain"], key=exp_bf16["topic.domain"].get) == max(
                pred["topic.domain"], key=pred["topic.domain"].get
            ):
                dom_match += 1
            if max(exp_bf16["topic.operation"], key=exp_bf16["topic.operation"].get) == max(
                pred["topic.operation"], key=pred["topic.operation"].get
            ):
                op_match += 1
            for hname in ("pii.payload", "topic.domain", "topic.operation"):
                for k, v in exp_bf16[hname].items():
                    pv = pred[hname][k]
                    abs_diffs.append(abs(v - pv))
                    if hname == "pii.payload":
                        pii_total += 1
                        if (v >= 0.5) == (pv >= 0.5):
                            pii_match += 1

    # Question-head 24-layer pointer probe
    qh_cosines: list[float] = []
    qh_prob_mae: list[float] = []
    if (quant_dir / "question_head" / "qh4in-full-e3" / "upper.safetensors").is_file():
        tcfg24_dict = dict(cfg_raw["text_config"])
        tcfg24_dict["num_hidden_layers"] = 24
        tcfg24_dict["layer_types"] = list(tcfg24_dict["layer_types"]) * 2
        tcfg24_dict["use_cache"] = False
        tcfg24_dict["_attn_implementation"] = "sdpa"
        with torch.device("meta"):
            bf16_24l = Qwen3_5TextModel(Qwen3_5TextConfig(**tcfg24_dict))
        sd24 = {k: v.detach().cpu() for k, v in bf16_model.state_dict().items()}
        with safe_open(
            str(base_dir / "question_head" / "qh4in-full-e3" / "upper.safetensors"),
            framework="pt",
            device="cpu",
        ) as u_h:
            for k in u_h.keys():
                short_k = k[len("model.language_model.") :] if k.startswith("model.language_model.") else k
                sd24[short_k] = u_h.get_tensor(k).to(torch.bfloat16)
        bf16_24l.load_state_dict(sd24, strict=True, assign=True)
        _restore_rotary(bf16_24l, device)
        bf16_24l.to(device)
        bf16_24l.eval()

        quant_24l = _load_decompressed_arc_24l_model(quant_dir, device=device)
        ptr_sd: dict[str, torch.Tensor] = {}
        with safe_open(
            str(quant_dir / "question_head" / "qh4in-full-e3" / "pointer.safetensors"),
            framework="pt",
            device="cpu",
        ) as p_h:
            for k in p_h.keys():
                ptr_sd[k] = p_h.get_tensor(k).float()

        probe_prompts = [
            (
                "Implement a binary search tree in Rust with insert and lookup.",
                "Does this request ask for Rust code?",
                ["true", "false"],
                1.20193,
            ),
            (
                "Fix the race condition in worker_pool.py when shutting down threads.",
                "What primary task is requested?",
                ["A: Debugging", "B: Translation", "C: Summarization", "D: SQL query"],
                1.03859,
            ),
            (
                "Prove that the square root of 2 is irrational.",
                "Rate the reasoning difficulty:",
                ["1", "2", "3", "4", "5"],
                1.8292,
            ),
        ]
        for conv, qtext, opts, temp in probe_prompts:
            prefix_ids = [
                im_start,
                *tokenizer.encode(f"user\n{conv}", add_special_tokens=False),
                im_end,
                nl,
            ]
            open_ids = [
                im_start,
                *tokenizer.encode(
                    f"user\nQuestion about the text above: {qtext}\nChoose one option:\n",
                    add_special_tokens=False,
                ),
            ]
            cur = prefix_ids + open_ids
            opt_pos: list[int] = []
            for o in opts:
                piece = tokenizer.encode(f"{o}\n", add_special_tokens=False)
                cur.extend(piece)
                opt_pos.append(len(cur) - 1)
            tail = tokenizer.encode(
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                add_special_tokens=False,
            )
            cur.extend(tail)
            ids_t = torch.tensor([cur], dtype=torch.long, device=device)
            h_b = bf16_24l(input_ids=ids_t).last_hidden_state[0].float().cpu()
            h_q = quant_24l(input_ids=ids_t).last_hidden_state[0].float().cpu()
            qh_cosines.append(float(F.cosine_similarity(h_b[-1:], h_q[-1:]).item()))
            scale = math.exp(float(ptr_sd["log_scale"].item())) / math.sqrt(256.0)
            q_b = F.linear(h_b[-1], ptr_sd["q.weight"])
            k_b = F.linear(h_b[opt_pos], ptr_sd["k.weight"])
            p_b = torch.softmax((k_b @ q_b) * scale / temp, dim=-1)
            q_q = F.linear(h_q[-1], ptr_sd["q.weight"])
            k_q = F.linear(h_q[opt_pos], ptr_sd["k.weight"])
            p_q = torch.softmax((k_q @ q_q) * scale / temp, dim=-1)
            qh_prob_mae.append(float((p_b - p_q).abs().mean().item()))

    if not cosines or not abs_diffs:
        raise RuntimeError(f"No valid signal evaluation rows found in {signals_jsonl}")

    report = {
        "model": args.model,
        "quant_dir": str(quant_dir),
        "readout_cosine_mean": sum(cosines) / len(cosines),
        "readout_cosine_min": min(cosines),
        "signals_mae_mean": sum(abs_diffs) / len(abs_diffs),
        "signals_mae_max": max(abs_diffs),
        "topic_domain_top1": f"{dom_match}/{dom_total}",
        "topic_operation_top1": f"{op_match}/{dom_total}",
        "pii_payload_binary_acc": f"{pii_match}/{pii_total}",
        "question_head_cosine_mean": (sum(qh_cosines) / len(qh_cosines) if qh_cosines else None),
        "question_head_prob_mae": (sum(qh_prob_mae) / len(qh_prob_mae) if qh_prob_mae else None),
    }
    if getattr(args, "output_json", None):
        Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report
