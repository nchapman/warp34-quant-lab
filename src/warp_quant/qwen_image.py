"""W4A16-HGPTQ quantization, Diffusers integration, and teacher-forced fidelity probing
for Qwen/Qwen-Image-2.1 and Qwen/Qwen-Image-2.1-Turbo.

Architecture summary:
- `text_encoder/`: `Qwen3VLForConditionalGeneration` (8.77B params, 36 Qwen3 decoder layers
  with hidden_size=4096, intermediate_size=12288, plus 27-block ViT with K=4304 `linear_fc2`,
  `merger.linear_fc[12]`, and `deepstack_merger_list.{0..2}.linear_fc[12]`).
  Note: `QwenImage21Pipeline._get_qwen_prompt_embeds` hooks `text_model.norm` to extract the
  unnormalized Layer-35 hidden states (`outputs.hidden_states[-1]`).
- `transformer/`: `QwenImage21Transformer2DModel` (7.12B params, 32 Single-Stream DiT blocks
  `transformer_blocks.0..31`, each with 7 bias-free `nn.Linear` layers:
  `attn.{to_q,to_k,to_v,to_out.0}` [4096, 4096] and `img_mlp.{gate_layer,proj}` [12288, 4096],
  `img_mlp.out` [4096, 12288]). Entry/exit projections (`img_in`, `txt_in`, `proj_out`) and
  timestep/AdaLN conditioning (`time_text_embed`, `modulation.1`, `norm_out.linear`) represent
  only 1.9% of DiT parameters (271 MB) and remain in BF16.
- `vae/`: `AutoencoderKLQwenImage21` (stored in BF16, 677 MB).
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from warp_quant.cachepath import resolve_snapshot
from warp_quant.data import (
    _calculate_32_aligned_dimensions,
    device_inputs,
    encode_image_row,
    prepare_image_gen_split,
    sha256,
)
from warp_quant.hadamard_gptq import (
    LM_HEAD_TARGET_RE,
    TEXT_HAD_TARGET_RE,
    TEXT_TARGET_RE,
    VISION_FC2_TARGET_RE,
    _assert_packed_alignment,
    _decompress_packed_2d,
    _gptq_pack_tensor,
    _hadamard_rotate_last_dim,
    _hadamard_rotate_weight,
    log_event,
)
from warp_quant.models import resolve
from warp_quant.stamp import ensure_license_files

QWEN_VL_VISION_GROUP_TARGET_RE = (
    r"re:.*visual\.(blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc1)|"
    r"(?:deepstack_merger_list\.\d+\.|merger\.)linear_fc[12])$"
)

DIT_BLOCK_TARGET_RE = (
    r"re:^transformer_blocks\.\d+\."
    r"(attn\.(to_q|to_k|to_v|to_out\.0)|img_mlp\.(gate_layer|proj|out))$"
)

DIT_IGNORE = [
    "img_in",
    "proj_out",
    "txt_in.in_layer",
    "txt_in.out_layer",
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "modulation.1",
    "norm_out.linear",
]

SYS_PROMPT = "Comprehend and analyze the provided prompt."
PROMPT_TEMPLATE_T2I = (
    f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TEMPLATE_TI2I = (
    f"<|im_start|>system\n{SYS_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{{}}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)


def load_image_gen_rows(
    fixture_dir: str | Path,
    *,
    require_calibration: bool,
    require_guard: bool | None = None,
    auto_materialize: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load and validate an `image-gen-{calib,probe,guard}-v1` fixture directory."""
    fdir = Path(fixture_dir)
    if (not (fdir / "manifest.json").is_file() or not (fdir / "rows.jsonl").is_file()) and auto_materialize:
        name = fdir.name
        if name in ("image-gen-calib-v1", "image-gen-probe-v1", "image-gen-guard-v1"):
            if fdir.exists():
                shutil.rmtree(fdir)
            split_map = {
                "image-gen-calib-v1": "calibration",
                "image-gen-probe-v1": "probe",
                "image-gen-guard-v1": "guard",
            }
            prepare_image_gen_split(fdir, split=split_map[name])

    manifest_path = fdir / "manifest.json"
    rows_path = fdir / "rows.jsonl"
    if not manifest_path.is_file() or not rows_path.is_file():
        raise ValueError(
            f"Image-gen fixture manifest/rows not found at {fdir} "
            "(materialize with 'mise run build-image-data')"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    is_cal = bool(manifest.get("calibration", False))
    if is_cal != require_calibration:
        role = "calibration" if require_calibration else "evaluation/probing"
        raise ValueError(
            f"Refusing to use fixture {fdir} (calibration={is_cal}) for {role} "
            f"(expected calibration={require_calibration})"
        )
    if require_guard is not None and bool(manifest.get("guard", False)) != require_guard:
        raise ValueError(
            f"Fixture {fdir} guard={manifest.get('guard')} does not match expected guard={require_guard}"
        )
    if sha256(rows_path) != manifest.get("rows_sha256"):
        raise ValueError(f"rows.jsonl checksum mismatch in {fdir}")

    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        w = row.get("width")
        h = row.get("height")
        if not isinstance(w, int | float) or not isinstance(h, int | float) or w <= 0 or h <= 0:
            raise ValueError(
                f"Invalid width/height ({w!r}, {h!r}) in {fdir} row {row.get('domain')}:{row.get('source_row')}"
            )
        img_rel = row.get("image")
        if img_rel is not None:
            img_p = fdir / img_rel
            if not img_p.is_file() or sha256(img_p) != row.get("image_sha256"):
                raise ValueError(f"Image checksum mismatch for {img_p}")
    return manifest, rows


def select_stratified_image_gen_rows(
    rows: list[dict[str, Any]],
    per_domain: int,
) -> list[dict[str, Any]]:
    """Select the first `per_domain` rows from each domain in domain appearance order."""
    by_dom: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        dom = row["domain"]
        bucket = by_dom.setdefault(dom, [])
        if len(bucket) < per_domain:
            bucket.append(row)
    selected: list[dict[str, Any]] = []
    for k in range(per_domain):
        for dom_rows in by_dom.values():
            if k < len(dom_rows):
                selected.append(dom_rows[k])
    return selected


def load_condition_pil_image(
    fixture_dir: Path,
    row: dict[str, Any],
    *,
    target_resolution: int | None = None,
):
    """Load a TI2I condition image as RGB (compositing RGBA over white as QwenImage21Pipeline does)."""
    from PIL import Image

    img_rel = row.get("image")
    if img_rel is None:
        return None
    with Image.open(fixture_dir / img_rel) as raw_img:
        if raw_img.mode == "RGBA":
            white = Image.new("RGB", raw_img.size, (255, 255, 255))
            white.paste(raw_img, mask=raw_img.getchannel("A"))
            img = white
        else:
            img = raw_img.convert("RGB")
    if target_resolution is not None:
        w, h = _calculate_32_aligned_dimensions(
            target_resolution * target_resolution, img.size[0] / img.size[1]
        )
        img = img.resize((w, h), Image.Resampling.BICUBIC)
    return img


def encode_qwen_image_row(
    processor: Any,
    fixture_dir: Path,
    row: dict[str, Any],
    *,
    target_resolution: int = 512,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Format a T2I or TI2I row using `QwenImage21Pipeline`'s exact raw system+user template."""
    prompt = row["prompt"] or " "
    if row.get("mode") == "ti2i" and row.get("image") is not None:
        cond_img = load_condition_pil_image(fixture_dir, row, target_resolution=target_resolution)
        text = PROMPT_TEMPLATE_TI2I.format(prompt)
        model_inputs = processor(
            text=[text],
            images=[cond_img],
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
    else:
        text = PROMPT_TEMPLATE_T2I.format(prompt)
        model_inputs = processor(
            text=[text],
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
    forward_kwargs: dict[str, torch.Tensor] = {
        "input_ids": model_inputs.input_ids,
        "attention_mask": model_inputs.attention_mask,
    }
    if hasattr(model_inputs, "pixel_values") and model_inputs.pixel_values is not None:
        forward_kwargs["pixel_values"] = model_inputs.pixel_values
        forward_kwargs["image_grid_thw"] = model_inputs.image_grid_thw
    if hasattr(model_inputs, "mm_token_type_ids") and model_inputs.mm_token_type_ids is not None:
        forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids
    if device is not None:
        forward_kwargs = {
            k: v.to(device=device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
            for k, v in forward_kwargs.items()
        }
    return forward_kwargs


def _online_hadamard_1d(x: torch.Tensor, block_size: int) -> torch.Tensor:
    from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix

    orig_shape = x.shape
    orig_dtype = x.dtype
    k = orig_shape[-1]
    if k % block_size != 0:
        raise ValueError(f"Last dimension {k} is not divisible by Hadamard block_size={block_size}")
    had = deterministic_hadamard_matrix(block_size, torch.float32, x.device) / math.sqrt(float(block_size))
    return (x.float().reshape(-1, k // block_size, block_size) @ had).reshape(orig_shape).to(orig_dtype)


class OnTheFlyW4Linear(nn.Module):
    """Low-VRAM W4/W8 pack-quantized Linear that unpacks on-the-fly during forward
    and optionally applies an online Hadamard input rotation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        weight_packed: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_shape: torch.Tensor,
        bias: torch.Tensor | None = None,
        hadamard_block_size: int | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hadamard_block_size = hadamard_block_size
        self.register_buffer("weight_packed", weight_packed, persistent=True)
        self.register_buffer("weight_scale", weight_scale, persistent=True)
        self.register_buffer("weight_shape", weight_shape, persistent=True)
        if bias is not None:
            self.register_buffer("bias", bias, persistent=True)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hadamard_block_size is not None:
            x = _online_hadamard_1d(x, self.hadamard_block_size)
        w = _decompress_packed_2d(self.weight_packed, self.weight_scale, self.weight_shape).to(
            device=x.device, dtype=x.dtype
        )
        b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


def build_text_encoder_quantization_config(
    *,
    group_size: int = 64,
    hadamard_block_size: int = 64,
    quantize_vision: bool = True,
    quantize_lm_head_w8: bool = True,
) -> dict[str, Any]:
    import compressed_tensors

    targets = [TEXT_TARGET_RE]
    had_targets = [TEXT_HAD_TARGET_RE]
    if quantize_vision:
        targets.append(QWEN_VL_VISION_GROUP_TARGET_RE)
        had_targets.append(QWEN_VL_VISION_GROUP_TARGET_RE)

    ignore = ["lm_head"] if not quantize_lm_head_w8 else []
    config_groups: dict[str, Any] = {
        "group_0": {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": targets,
            "weights": {
                "actorder": "static",
                "block_structure": None,
                "dynamic": False,
                "group_size": group_size,
                "num_bits": 4,
                "observer": "memoryless_minmax",
                "observer_kwargs": {},
                "scale_dtype": None,
                "strategy": "group",
                "symmetric": True,
                "type": "int",
                "zp_dtype": None,
            },
        }
    }
    if quantize_vision:
        config_groups["vision_fc2_channel"] = {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": [VISION_FC2_TARGET_RE],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": None,
                "num_bits": 4,
                "observer": "memoryless_minmax",
                "observer_kwargs": {},
                "scale_dtype": None,
                "strategy": "channel",
                "symmetric": True,
                "type": "int",
                "zp_dtype": None,
            },
        }
    if quantize_lm_head_w8:
        config_groups["w8_head"] = {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": [LM_HEAD_TARGET_RE],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": group_size,
                "num_bits": 8,
                "observer": "memoryless_minmax",
                "observer_kwargs": {},
                "scale_dtype": None,
                "strategy": "group",
                "symmetric": True,
                "type": "int",
                "zp_dtype": None,
            },
        }

    t_groups: dict[str, Any] = {
        "v": {
            "apply": [
                {
                    "ignore": ignore,
                    "inverse": True,
                    "location": "weight_input",
                    "targets": had_targets,
                },
                {
                    "ignore": ignore,
                    "inverse": False,
                    "location": "input",
                    "targets": had_targets,
                },
            ],
            "head_dim": hadamard_block_size,
            "precision": "torch.float32",
            "randomize": False,
            "requires_grad": False,
            "type": "hadamard",
        }
    }
    if quantize_vision:
        t_groups["v_vision_fc2"] = {
            "apply": [
                {
                    "ignore": [],
                    "inverse": True,
                    "location": "weight_input",
                    "targets": [VISION_FC2_TARGET_RE],
                },
                {
                    "ignore": [],
                    "inverse": False,
                    "location": "input",
                    "targets": [VISION_FC2_TARGET_RE],
                },
            ],
            "head_dim": 16,
            "precision": "torch.float32",
            "randomize": False,
            "requires_grad": False,
            "type": "hadamard",
        }

    return {
        "config_groups": config_groups,
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "ignore": ignore,
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {"config_groups": t_groups},
        "version": getattr(compressed_tensors, "__version__", "0.14.0"),
    }


def build_dit_quantization_config(
    *,
    group_size: int = 64,
    hadamard_block_size: int = 64,
) -> dict[str, Any]:
    import compressed_tensors

    return {
        "config_groups": {
            "group_0": {
                "format": "pack-quantized",
                "input_activations": None,
                "output_activations": None,
                "targets": [DIT_BLOCK_TARGET_RE],
                "weights": {
                    "actorder": "static",
                    "block_structure": None,
                    "dynamic": False,
                    "group_size": group_size,
                    "num_bits": 4,
                    "observer": "memoryless_minmax",
                    "observer_kwargs": {},
                    "scale_dtype": None,
                    "strategy": "group",
                    "symmetric": True,
                    "type": "int",
                    "zp_dtype": None,
                },
            }
        },
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "ignore": list(DIT_IGNORE),
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {
            "config_groups": {
                "v": {
                    "apply": [
                        {
                            "ignore": list(DIT_IGNORE),
                            "inverse": True,
                            "location": "weight_input",
                            "targets": [DIT_BLOCK_TARGET_RE],
                        },
                        {
                            "ignore": list(DIT_IGNORE),
                            "inverse": False,
                            "location": "input",
                            "targets": [DIT_BLOCK_TARGET_RE],
                        },
                    ],
                    "head_dim": hadamard_block_size,
                    "precision": "torch.float32",
                    "randomize": False,
                    "requires_grad": False,
                    "type": "hadamard",
                }
            }
        },
        "version": getattr(compressed_tensors, "__version__", "0.14.0"),
    }


def _apply_online_hadamard_hooks_from_qcfg(model: nn.Module, qcfg: dict[str, Any]) -> None:
    from compressed_tensors.transform import (
        TransformArgs,
        TransformConfig,
        TransformScheme,
        apply_transform_config,
    )

    tcfg_dict = qcfg.get("transform_config", {})
    if not tcfg_dict or not tcfg_dict.get("config_groups"):
        return
    input_groups = {}
    for gname, gspec in tcfg_dict["config_groups"].items():
        in_applies = [a for a in gspec["apply"] if a["location"] == "input"]
        if in_applies:
            input_groups[gname] = TransformScheme(
                type=gspec["type"],
                apply=[TransformArgs(**a) for a in in_applies],
                randomize=gspec.get("randomize", False),
                requires_grad=gspec.get("requires_grad", False),
                head_dim=gspec.get("head_dim"),
                precision=gspec.get("precision", "torch.float32"),
            )
    if input_groups:
        apply_transform_config(model, TransformConfig(config_groups=input_groups))


def load_quantized_submodule(
    module: nn.Module,
    ckpt_dir: Path,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    keep_compressed: bool = False,
) -> nn.Module:
    """Populate `module` (either `Qwen3VLForConditionalGeneration` or `QwenImage21Transformer2DModel`)
    from a `compressed-tensors` W4A16-HGPTQ directory.

    When `keep_compressed=True`, packed linear layers are replaced with `OnTheFlyW4Linear` so the
    packed int32 weights stay 4-bit in VRAM (~9.6 GiB total for the full 16B pipeline).
    When `keep_compressed=False`, packed weights are decompressed into `dtype` on GPU and online
    Hadamard input rotation hooks are attached via `apply_transform_config`.
    """
    import re

    cfg_path = ckpt_dir / "config.json"
    raw_cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    qcfg = raw_cfg.get("quantization_config", {})

    index_names = (
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
    )
    shard_paths: list[Path] = []
    for idx_name in index_names:
        idx_p = ckpt_dir / idx_name
        if idx_p.is_file():
            wmap = json.loads(idx_p.read_text())["weight_map"]
            shard_paths = [ckpt_dir / s for s in sorted(set(wmap.values()))]
            break
    if not shard_paths:
        shard_paths = sorted(ckpt_dir.glob("*.safetensors"))

    raw_tensors: dict[str, torch.Tensor] = {}
    for sp in shard_paths:
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for k in f.keys():
                raw_tensors[k] = f.get_tensor(k)

    packed_prefixes = sorted(
        k.removesuffix(".weight_packed") for k in raw_tensors if k.endswith(".weight_packed")
    )
    if not packed_prefixes:
        module.load_state_dict(raw_tensors, strict=True, assign=True)
        return module.to(device=device, dtype=dtype).eval()

    had_rules: list[tuple[re.Pattern[str], set[str], int]] = []
    tcfg = qcfg.get("transform_config", {}).get("config_groups", {})
    for gspec in tcfg.values():
        hdim = int(gspec.get("head_dim", 64))
        for app in gspec.get("apply", []):
            if app.get("location") == "input":
                ignore_set = set(app.get("ignore", []))
                for pat_str in app.get("targets", []):
                    rx = re.compile(pat_str.removeprefix("re:"))
                    had_rules.append((rx, ignore_set, hdim))

    def _had_block_for(prefix: str) -> int | None:
        for rx, ign, hdim in had_rules:
            if prefix not in ign and rx.match(prefix):
                return hdim
        return None

    if keep_compressed:
        plain_state: dict[str, torch.Tensor] = {}
        packed_set = set(packed_prefixes)
        for k, v in raw_tensors.items():
            if not any(k.startswith(p + ".") for p in packed_set):
                plain_state[k] = v.to(dtype=dtype)
        for prefix in packed_prefixes:
            parent_path, _, attr = prefix.rpartition(".")
            parent = module.get_submodule(parent_path) if parent_path else module
            old_mod = getattr(parent, attr)
            bias = raw_tensors.get(f"{prefix}.bias")
            old_bias = getattr(old_mod, "bias", None)
            if isinstance(old_bias, torch.Tensor) and old_bias.is_meta:
                old_bias = None
            w_shape = raw_tensors[f"{prefix}.weight_shape"]
            out_f, in_f = int(w_shape[0].item()), int(w_shape[1].item())
            new_mod = OnTheFlyW4Linear(
                in_features=in_f,
                out_features=out_f,
                weight_packed=raw_tensors[f"{prefix}.weight_packed"].to(device=device),
                weight_scale=raw_tensors[f"{prefix}.weight_scale"].to(device=device, dtype=dtype),
                weight_shape=w_shape,
                bias=bias.to(device=device, dtype=dtype) if bias is not None else old_bias,
                hadamard_block_size=_had_block_for(prefix),
            )
            setattr(parent, attr, new_mod)
        module.load_state_dict(plain_state, strict=False, assign=True)
        module.to(device=device, dtype=dtype)
        return module.eval()

    state_dict: dict[str, torch.Tensor] = {}
    packed_set = set(packed_prefixes)
    for k, v in raw_tensors.items():
        if not any(k.startswith(p + ".") for p in packed_set):
            state_dict[k] = v.to(dtype=dtype)
    for prefix in packed_prefixes:
        deq = _decompress_packed_2d(
            raw_tensors[f"{prefix}.weight_packed"],
            raw_tensors[f"{prefix}.weight_scale"],
            raw_tensors[f"{prefix}.weight_shape"],
        ).to(dtype=dtype)
        state_dict[f"{prefix}.weight"] = deq
        if f"{prefix}.bias" in raw_tensors:
            state_dict[f"{prefix}.bias"] = raw_tensors[f"{prefix}.bias"].to(dtype=dtype)

    module.load_state_dict(state_dict, strict=True, assign=True)
    module = module.to(device=device, dtype=dtype).eval()
    _apply_online_hadamard_hooks_from_qcfg(module, qcfg)
    return module


def load_qwen_image_pipeline(
    model_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    keep_text_encoder_compressed: bool = True,
    keep_transformer_compressed: bool = False,
):
    """Load a BF16 or W4A16-HGPTQ `QwenImage21Pipeline` from a local directory or HF Hub repo ID."""
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
    from diffusers.models import QwenImage21Transformer2DModel
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    root = Path(model_path) if Path(model_path).is_dir() else Path(resolve_snapshot(str(model_path)))
    model_index = json.loads((root / "model_index.json").read_text())

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(root / "scheduler"))
    processor = Qwen3VLProcessor.from_pretrained(str(root / "processor"))
    vae = AutoencoderKLQwenImage21.from_pretrained(str(root / "vae"), torch_dtype=dtype).to(device).eval()

    te_dir = root / "text_encoder"
    te_cfg_dict = json.loads((te_dir / "config.json").read_text())
    te_cfg_clean = {k: v for k, v in te_cfg_dict.items() if k != "quantization_config"}
    te_config = Qwen3VLConfig.from_dict(te_cfg_clean)
    with init_empty_weights():
        text_encoder = Qwen3VLForConditionalGeneration(te_config)
    text_encoder = load_quantized_submodule(
        text_encoder,
        te_dir,
        device=device,
        dtype=dtype,
        keep_compressed=keep_text_encoder_compressed and ("quantization_config" in te_cfg_dict),
    )

    tr_dir = root / "transformer"
    tr_cfg_dict = json.loads((tr_dir / "config.json").read_text())
    tr_cfg_clean = {k: v for k, v in tr_cfg_dict.items() if k != "quantization_config"}
    with init_empty_weights():
        transformer = QwenImage21Transformer2DModel.from_config(tr_cfg_clean)
    transformer = load_quantized_submodule(
        transformer,
        tr_dir,
        device=device,
        dtype=dtype,
        keep_compressed=keep_transformer_compressed and ("quantization_config" in tr_cfg_dict),
    )

    pipe = QwenImage21Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        processor=processor,
        transformer=transformer,
        sample_sigmas=model_index.get("sample_sigmas"),
    )
    return pipe


def _quantize_qwen_vl_vision_tower(
    text_encoder: nn.Module,
    processor: Any,
    vision_calib_dir: Path,
    image_gen_calib_dir: Path | None = None,
    *,
    group_size: int,
    hadamard_block_size: int,
    dampening_frac: float,
    device: torch.device,
    max_samples: int = 36,
    max_pixels: int = 1024 * 1024,
) -> dict[str, dict[str, torch.Tensor]]:
    """Calibrate and GPTQ-quantize `text_encoder.model.visual` (`blocks.0..26`, `merger`,
    and `deepstack_merger_list.0..2`) on `data/image-gen-calib-v1` TI2I reference images
    plus `data/vision-calib-v2` images, then install dequantized W4A16-HGPTQ weights and
    online Hadamard input hooks onto `text_encoder.model.visual` in-place."""
    from llmcompressor.modifiers.gptq.helpers import accumulate_hessian, make_empty_hessian

    visual = text_encoder.model.visual
    target_mods: dict[str, nn.Linear] = {}
    had_blocks: dict[str, int] = {}
    hessians: dict[str, torch.Tensor] = {}
    num_samples: dict[str, torch.Tensor] = {}
    hooks = []

    for name, mod in visual.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        target_mods[name] = mod
        k_in = mod.weight.shape[1]
        had_b = min(hadamard_block_size, k_in & -k_in) if hadamard_block_size > 0 else 0
        had_blocks[name] = had_b
        if had_b > 0:
            mod.weight.data = _hadamard_rotate_weight(mod.weight.data, had_b)
        hessians[name] = make_empty_hessian(mod, device)
        num_samples[name] = torch.zeros(1, device=device, dtype=torch.int64)

        def _make_hook(m_name: str, m_ref: nn.Linear, b_size: int):
            def _pre_hook(_module, hook_args):
                with torch.amp.autocast("cuda", enabled=False):
                    x = hook_args[0]
                    if b_size > 0:
                        x = _hadamard_rotate_last_dim(x, b_size, compute_dtype=torch.float32)
                    x_3d = x.unsqueeze(0) if x.dim() == 2 else x
                    scale = (1024.0 / max(x_3d.shape[1], 1)) ** 0.25
                    accumulate_hessian(x_3d * scale, m_ref, hessians[m_name], num_samples[m_name])
                return (x, *hook_args[1:])

            return _pre_hook

        hooks.append(mod.register_forward_pre_hook(_make_hook(name, mod, had_b)))

    n_ti2i = 0
    n_v2 = 0
    try:
        with torch.no_grad():
            if image_gen_calib_dir is not None and image_gen_calib_dir.exists():
                _, ig_rows = load_image_gen_rows(image_gen_calib_dir, require_calibration=True)
                for row in ig_rows:
                    if row.get("mode") != "ti2i" or row.get("image") is None:
                        continue
                    fw_kw = encode_qwen_image_row(
                        processor, image_gen_calib_dir, row, target_resolution=512, device=device
                    )
                    if "pixel_values" in fw_kw and "image_grid_thw" in fw_kw:
                        _ = visual(fw_kw["pixel_values"], grid_thw=fw_kw["image_grid_thw"])
                        n_ti2i += 1

            manifest_p = vision_calib_dir / "manifest.json"
            if manifest_p.is_file():
                manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
                if not manifest.get("calibration", False):
                    raise ValueError(
                        f"Refusing to calibrate Vision on non-calibration fixture: {vision_calib_dir}"
                    )
                rows_p = vision_calib_dir / "rows.jsonl"
                v_rows = [
                    json.loads(line)
                    for line in rows_p.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                for row in v_rows[:max_samples]:
                    encoded, _, _ = encode_image_row(processor, vision_calib_dir, row, max_pixels)
                    inputs = device_inputs(encoded, device)
                    _ = visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
                    n_v2 += 1
            elif n_ti2i == 0:
                raise ValueError(
                    f"No vision calibration images found in {image_gen_calib_dir} or {vision_calib_dir}"
                )
    finally:
        for h in hooks:
            h.remove()

    packed_by_prefix: dict[str, dict[str, torch.Tensor]] = {}
    for name, mod in target_mods.items():
        k_in = mod.weight.shape[1]
        strat = "group" if k_in % group_size == 0 else "channel"
        g_size = group_size if strat == "group" else None
        packed = _gptq_pack_tensor(
            mod.weight.data,
            hessians.pop(name),
            num_samples.pop(name),
            strategy=strat,
            group_size=g_size,
            actorder="static" if strat == "group" else None,
            dampening_frac=dampening_frac,
        )
        packed_by_prefix[f"model.visual.{name}"] = packed
        deq = _decompress_packed_2d(
            packed["weight_packed"], packed["weight_scale"], packed["weight_shape"]
        ).to(device=device, dtype=mod.weight.dtype)
        mod.weight.data.copy_(deq)
        hb = had_blocks[name]
        if hb > 0:

            def _vis_rot_hook(_m: nn.Module, args: tuple[Any, ...], _hb: int = hb):
                return (_online_hadamard_1d(args[0], _hb).to(dtype=_m.weight.dtype), *args[1:])

            mod.register_forward_pre_hook(_vis_rot_hook)

    torch.cuda.empty_cache()
    log_event(
        "QWEN_IMAGE_VISION_CALIB_DONE",
        matrices=len(packed_by_prefix),
        ti2i_images=n_ti2i,
        v2_images=n_v2,
    )
    return packed_by_prefix


class _StopDecoderEntry(Exception):
    """Internal control-flow signal used to capture exact inputs entering `Qwen3VLTextModel`."""


def _capture_lang_model_entry(
    model: nn.Module,
    forward_kwargs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Run `model` up to the entry of `model.model.language_model` (including Vision + DeepStack
    feature extraction and 3D M-RoPE position calculation) and return the exact decoder inputs."""
    lang_model = model.model.language_model
    captured: dict[str, Any] = {}

    def _lang_pre_hook(_m: nn.Module, _args: tuple[Any, ...], kwargs: dict[str, Any]):
        inputs_embeds = kwargs.get("inputs_embeds")
        input_ids = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = lang_model.embed_tokens(input_ids)
        pos_ids = kwargs.get("position_ids")
        if pos_ids is None:
            seq_l = inputs_embeds.shape[1]
            pos_ids = (
                torch.arange(seq_l, device=inputs_embeds.device)
                .view(1, 1, -1)
                .expand(4, inputs_embeds.shape[0], -1)
            )
        elif pos_ids.ndim == 2:
            pos_ids = pos_ids[None, ...].expand(4, pos_ids.shape[0], -1)
        if pos_ids.ndim == 3 and pos_ids.shape[0] == 4:
            text_pos = pos_ids[0]
            rope_pos = pos_ids[1:]
        else:
            text_pos = pos_ids[0]
            rope_pos = pos_ids
        pe = lang_model.rotary_emb(inputs_embeds, rope_pos)
        vis_mask = kwargs.get("visual_pos_masks")
        ds_embeds = kwargs.get("deepstack_visual_embeds")
        captured["hidden_states"] = inputs_embeds.detach().cpu()
        captured["pos_emb"] = (pe[0].detach().cpu(), pe[1].detach().cpu())
        captured["text_pos"] = text_pos.detach().cpu()
        captured["visual_pos_masks"] = vis_mask.detach().cpu() if vis_mask is not None else None
        captured["deepstack_embeds"] = (
            [d.detach().cpu() for d in ds_embeds] if ds_embeds is not None else None
        )
        raise _StopDecoderEntry()

    hk = lang_model.register_forward_pre_hook(_lang_pre_hook, with_kwargs=True)
    try:
        try:
            model(**forward_kwargs, use_cache=False)
        except _StopDecoderEntry:
            pass
    finally:
        hk.remove()
    return captured


def _quantize_qwen_vl_text_encoder(
    base_dir: Path,
    output_te_dir: Path,
    *,
    image_calib_dir: Path,
    calib_path: Path,
    vision_calib_dir: Path,
    calib_samples: int = 256,
    calib_seq_len: int = 2048,
    group_size: int = 64,
    hadamard_block_size: int = 64,
    dampening_frac: float = 0.01,
    quantize_vision: bool = True,
    quantize_lm_head_w8: bool = True,
    norm_qat_steps: int = 160,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Calibrate and quantize `text_encoder/` (`Qwen3VLForConditionalGeneration`) to W4A16-HGPTQ
    using `data/image-gen-calib-v1` (96 stratified T2I + multimodal DeepStack TI2I rows across
    16 EN+ZH domains) plus `data/calib-cascade-v1.jsonl`, with sequential layer-by-layer Hessian
    propagation and End-to-End Layer-35 unnormalized hidden-state Norm QAT."""
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    device = device or torch.device("cuda:0")

    output_te_dir.mkdir(parents=True, exist_ok=True)
    te_src_dir = base_dir / "text_encoder"
    proc_dir = base_dir / "processor"
    processor = Qwen3VLProcessor.from_pretrained(str(proc_dir))
    tokenizer = processor.tokenizer

    _, ig_rows = load_image_gen_rows(image_calib_dir, require_calibration=True)
    # Order ig_rows as 6 stratified passes over the 16 domains so any prefix is domain-balanced:
    # - indices 0..15 (1 per domain: 12 T2I + 4 TI2I, EN+ZH) -> Norm QAT validation split
    # - indices 16..63 (3 per domain: 36 T2I + 12 TI2I, EN+ZH) -> Norm QAT training split
    # - indices 0..95 (all 96 rows: 72 T2I + 24 TI2I, EN+ZH) -> sequential GPTQ calibration
    strat_ig_rows = select_stratified_image_gen_rows(ig_rows, per_domain=6)
    qat_ig_rows = strat_ig_rows[: min(64, len(strat_ig_rows))]

    log_event("QWEN_IMAGE_TE_LOAD", path=str(te_src_dir), ig_calib_rows=len(strat_ig_rows))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(te_src_dir), torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()
    lang_model = model.model.language_model
    layers = lang_model.layers
    num_layers = len(layers)

    # 1. Capture full BF16 teacher unnormalized Layer-35 output (`hidden_states[-1]`) BEFORE quantizing Vision
    teacher_pre_norm: list[torch.Tensor] = []
    with torch.no_grad():
        for row in qat_ig_rows:
            fw_kw = encode_qwen_image_row(
                processor, image_calib_dir, row, target_resolution=512, device=device
            )
            captured: list[torch.Tensor] = []
            h = lang_model.norm.register_forward_hook(
                lambda _m, a, _o, _cap=captured: _cap.append(a[0].detach().cpu()) or a[0]
            )
            try:
                model(**fw_kw, use_cache=False)
            finally:
                h.remove()
            teacher_pre_norm.append(captured[0])

    # 2. Quantize Vision tower (`blocks.0..26`, `merger`, `deepstack_merger_list.0..2`) and update live weights
    packed_tensors: dict[str, dict[str, torch.Tensor]] = {}
    if quantize_vision:
        vis_packed = _quantize_qwen_vl_vision_tower(
            model,
            processor,
            vision_calib_dir,
            image_calib_dir,
            group_size=group_size,
            hadamard_block_size=hadamard_block_size,
            dampening_frac=dampening_frac,
            device=device,
        )
        packed_tensors.update(vis_packed)

    # 3. Capture post-quantized-Vision entry states to `lang_model.layers[0]` across all 96 T2I/TI2I rows
    #    plus general text rows from `calib_path` up to `calib_samples`.
    entry_states: list[dict[str, Any]] = []
    with torch.no_grad():
        for row in strat_ig_rows:
            if len(entry_states) >= calib_samples:
                break
            fw_kw = encode_qwen_image_row(
                processor, image_calib_dir, row, target_resolution=512, device=device
            )
            entry_states.append(_capture_lang_model_entry(model, fw_kw))

        if calib_path.is_file() and len(entry_states) < calib_samples:
            with calib_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if len(entry_states) >= calib_samples:
                        break
                    row = json.loads(line)
                    text = row.get("text") or ""
                    conv = row.get("conversations") or row.get("messages")
                    if not text and conv:
                        text = "\n".join(
                            m.get("content", "") for m in conv if isinstance(m.get("content"), str)
                        )
                    if not text:
                        continue
                    ids = tokenizer(
                        text, return_tensors="pt", truncation=True, max_length=calib_seq_len
                    ).input_ids
                    if ids.shape[1] >= 32:
                        entry_states.append(_capture_lang_model_entry(model, {"input_ids": ids.to(device)}))

    n_seqs = len(entry_states)
    n_ti2i_seqs = sum(1 for e in entry_states if e["deepstack_embeds"] is not None)
    log_event("QWEN_IMAGE_TE_CALIB_SEQS", count=n_seqs, ti2i_multimodal=n_ti2i_seqs)

    # Save initial layer-0 inputs for the 64 QAT sequences before `cur_h` is mutated across layers
    qat_init_entries = [
        {
            "hidden_states": e["hidden_states"].clone(),
            "pos_emb": e["pos_emb"],
            "text_pos": e["text_pos"],
            "visual_pos_masks": e["visual_pos_masks"],
            "deepstack_embeds": e["deepstack_embeds"],
        }
        for e in entry_states[: len(teacher_pre_norm)]
    ]
    cur_h: list[torch.Tensor] = [e["hidden_states"] for e in entry_states]

    # 4. Sequential layer-by-layer Hessian GPTQ across all 36 decoder layers (with DeepStack injection)
    for l_idx in range(num_layers):
        t0 = time.perf_counter()
        layer = layers[l_idx]
        sub_batches = [
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
            ("self_attn.o_proj",),
            ("mlp.gate_proj", "mlp.up_proj"),
            ("mlp.down_proj",),
        ]
        for group_projs in sub_batches:
            first_mod: nn.Linear = layer.get_submodule(group_projs[0])
            k_dim = first_mod.in_features
            h_mat = torch.zeros((k_dim, k_dim), dtype=torch.float32, device=device)
            n_tok = [0]

            def _acc_hook(
                _m: nn.Module,
                args: tuple[Any, ...],
                _k: int = k_dim,
                _h: torch.Tensor = h_mat,
                _n: list[int] = n_tok,
            ) -> None:
                x = args[0].detach()
                x_rot = _online_hadamard_1d(x, hadamard_block_size).reshape(-1, _k).float()
                _h.addmm_(x_rot.T, x_rot)
                _n[0] += x_rot.shape[0]

            hk = first_mod.register_forward_pre_hook(_acc_hook)
            try:
                with torch.no_grad():
                    for s_idx in range(n_seqs):
                        pe = entry_states[s_idx]["pos_emb"]
                        pe_d = (
                            pe[0].to(device=device, dtype=torch.bfloat16),
                            pe[1].to(device=device, dtype=torch.bfloat16),
                        )
                        layer(
                            cur_h[s_idx].to(device=device, dtype=torch.bfloat16),
                            attention_mask=None,
                            position_ids=entry_states[s_idx]["text_pos"].to(device),
                            past_key_values=None,
                            position_embeddings=pe_d,
                        )
            finally:
                hk.remove()

            h_scaled = (2.0 / float(max(n_tok[0], 1))) * h_mat
            one_sample = torch.ones(1, device=device, dtype=torch.int64)
            for pname in group_projs:
                mod: nn.Linear = layer.get_submodule(pname)
                w_rot = _hadamard_rotate_weight(mod.weight.detach().to(device), hadamard_block_size)
                packed = _gptq_pack_tensor(
                    w_rot,
                    h_scaled,
                    one_sample,
                    strategy="group",
                    group_size=group_size,
                    actorder="static",
                    dampening_frac=dampening_frac,
                )
                full_p = f"model.language_model.layers.{l_idx}.{pname}"
                packed_tensors[full_p] = packed
                deq = _decompress_packed_2d(
                    packed["weight_packed"], packed["weight_scale"], packed["weight_shape"]
                ).to(device=device, dtype=torch.bfloat16)
                mod.weight.data.copy_(deq)

                def _rot_pre_hook(_m: nn.Module, args: tuple[Any, ...], hb: int = hadamard_block_size):
                    return (_online_hadamard_1d(args[0], hb).to(dtype=_m.weight.dtype), *args[1:])

                mod.register_forward_pre_hook(_rot_pre_hook)

        with torch.no_grad():
            for s_idx in range(n_seqs):
                pe = entry_states[s_idx]["pos_emb"]
                pe_d = (
                    pe[0].to(device=device, dtype=torch.bfloat16),
                    pe[1].to(device=device, dtype=torch.bfloat16),
                )
                out_h = layer(
                    cur_h[s_idx].to(device=device, dtype=torch.bfloat16),
                    attention_mask=None,
                    position_ids=entry_states[s_idx]["text_pos"].to(device),
                    past_key_values=None,
                    position_embeddings=pe_d,
                )
                ds_list = entry_states[s_idx]["deepstack_embeds"]
                v_mask = entry_states[s_idx]["visual_pos_masks"]
                if ds_list is not None and v_mask is not None and l_idx < len(ds_list):
                    out_h = lang_model._deepstack_process(
                        out_h,
                        v_mask.to(device=device),
                        ds_list[l_idx].to(device=device, dtype=out_h.dtype),
                    )
                cur_h[s_idx] = out_h.detach().cpu()

        log_event(
            "QWEN_IMAGE_TE_LAYER",
            layer=l_idx,
            total_layers=num_layers,
            elapsed_s=round(time.perf_counter() - t0, 2),
        )

    # 5. Quantize untied lm_head to unrotated W8_g64 GPTQ+MSE
    if quantize_lm_head_w8 and hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Linear):
        k_lm = model.lm_head.in_features
        h_lm = torch.zeros((k_lm, k_lm), dtype=torch.float32, device=device)
        n_lm = 0
        with torch.no_grad():
            for pre_h in teacher_pre_norm:
                normed = (
                    lang_model.norm(pre_h.to(device=device, dtype=torch.bfloat16)).reshape(-1, k_lm).float()
                )
                h_lm.addmm_(normed.T, normed)
                n_lm += normed.shape[0]
        h_lm.mul_(2.0 / float(max(n_lm, 1)))
        packed_tensors["lm_head"] = _gptq_pack_tensor(
            model.lm_head.weight.detach().to(device),
            h_lm,
            torch.ones(1, device=device, dtype=torch.int64),
            strategy="group",
            group_size=group_size,
            actorder=None,
            dampening_frac=dampening_frac,
            num_bits=8,
        )
        del h_lm

    # 6. End-to-End FP32-master RMSNorm QAT targeting unnormalized Layer-35 output (`hidden_states[-1]`)
    #    across stratified T2I + multimodal DeepStack TI2I sequences.
    qat_summary: dict[str, Any] = {"enabled": norm_qat_steps > 0}
    if norm_qat_steps > 0 and len(qat_init_entries) >= 16:
        val_entries = qat_init_entries[:16]
        val_targets = teacher_pre_norm[:16]
        tr_entries = qat_init_entries[16:]
        tr_targets = teacher_pre_norm[16:]

        for p in model.parameters():
            p.requires_grad_(False)
        norm_params: list[nn.Parameter] = []
        norm_targets: list[tuple[nn.Module, nn.Parameter]] = []
        qat_hooks = []
        for layer in layers:
            for n_attr in (
                "input_layernorm",
                "post_attention_layernorm",
                "self_attn.q_norm",
                "self_attn.k_norm",
            ):
                nm = layer.get_submodule(n_attr)
                sp = nn.Parameter(torch.ones_like(nm.weight, dtype=torch.float32, device=device))
                norm_params.append(sp)
                norm_targets.append((nm, sp))
                qat_hooks.append(
                    nm.register_forward_hook(
                        lambda _m, _a, out, _s=sp: (out.float() * _s).to(dtype=out.dtype)
                    )
                )

        def _forward_lang_entry(entry: dict[str, Any]) -> torch.Tensor:
            h = entry["hidden_states"].to(device=device, dtype=torch.bfloat16)
            pe = entry["pos_emb"]
            pe_d = (
                pe[0].to(device=device, dtype=torch.bfloat16),
                pe[1].to(device=device, dtype=torch.bfloat16),
            )
            tpos = entry["text_pos"].to(device)
            ds_list = entry["deepstack_embeds"]
            v_mask = entry["visual_pos_masks"]
            v_mask_d = v_mask.to(device=device) if v_mask is not None else None
            for l_i, lyr in enumerate(layers):
                h = lyr(
                    h,
                    attention_mask=None,
                    position_ids=tpos,
                    past_key_values=None,
                    position_embeddings=pe_d,
                )
                if ds_list is not None and v_mask_d is not None and l_i < len(ds_list):
                    h = lang_model._deepstack_process(
                        h,
                        v_mask_d,
                        ds_list[l_i].to(device=device, dtype=h.dtype),
                    )
            return h

        def _eval_pre_norm_mse() -> float:
            errs = []
            with torch.no_grad():
                for ent, tgt in zip(val_entries, val_targets, strict=True):
                    pred = _forward_lang_entry(ent).float()
                    ref = tgt.to(device=device, dtype=torch.float32)
                    errs.append(float(((pred - ref).pow(2).sum() / ref.pow(2).sum().clamp_min(1e-12)).item()))
            return sum(errs) / max(len(errs), 1)

        val_before = _eval_pre_norm_mse()
        opt = torch.optim.Adam(norm_params, lr=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=norm_qat_steps, eta_min=1e-5)
        for step in range(norm_qat_steps):
            idx = step % len(tr_entries)
            ent = tr_entries[idx]
            tgt = tr_targets[idx].to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            pred = _forward_lang_entry(ent).float()
            loss = (pred - tgt).pow(2).mean() / tgt.pow(2).mean().clamp_min(1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(norm_params, 1.0)
            opt.step()
            sched.step()

        val_after = _eval_pre_norm_mse()
        for qh in qat_hooks:
            qh.remove()
        accepted = val_after < val_before
        if accepted:
            for nm, sp in norm_targets:
                nm.weight.data.copy_((nm.weight.float() * sp.detach()).to(dtype=nm.weight.dtype))
        qat_summary = {
            "enabled": True,
            "steps": norm_qat_steps,
            "val_sequences": len(val_entries),
            "train_sequences": len(tr_entries),
            "val_rel_mse_before": val_before,
            "val_rel_mse_after": val_after,
            "accepted": accepted,
        }
        log_event("QWEN_IMAGE_TE_NORM_QAT", **qat_summary)

    # Save packed state_dict to text_encoder/model.safetensors
    out_state: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k.endswith(".weight") and k[:-7] in packed_tensors:
            prefix = k[:-7]
            for sfx, pt in packed_tensors[prefix].items():
                out_state[f"{prefix}.{sfx}"] = pt.cpu()
        else:
            out_state[k] = v.detach().cpu().to(torch.bfloat16)

    save_file(out_state, str(output_te_dir / "model.safetensors"))
    te_cfg = json.loads((te_src_dir / "config.json").read_text())
    te_cfg["quantization_config"] = build_text_encoder_quantization_config(
        group_size=group_size,
        hadamard_block_size=hadamard_block_size,
        quantize_vision=quantize_vision,
        quantize_lm_head_w8=quantize_lm_head_w8,
    )
    (output_te_dir / "config.json").write_text(json.dumps(te_cfg, indent=2) + "\n")
    if (te_src_dir / "generation_config.json").is_file():
        shutil.copy2(te_src_dir / "generation_config.json", output_te_dir / "generation_config.json")
    _assert_packed_alignment(output_te_dir)

    gc.collect()
    torch.cuda.empty_cache()
    return {
        "packed_matrices": len(packed_tensors),
        "calib_sequences": n_seqs,
        "calib_ti2i_sequences": n_ti2i_seqs,
        "norm_qat": qat_summary,
    }


def _collect_dit_calib_states(
    base_dir: Path,
    image_calib_dir: Path,
    *,
    prompts_per_domain: int = 2,
    steps_per_prompt: int = 4,
    calib_resolution: int = 512,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Collect real FlowMatch denoising trajectory inputs at the entry to `transformer_blocks[0]`
    plus teacher block-31 output states across stratified T2I (multi-aspect-ratio, EN+ZH) and
    TI2I (real reference-image editing, EN+ZH) rows from `data/image-gen-calib-v1`."""
    device = device or torch.device("cuda:0")

    _, all_rows = load_image_gen_rows(image_calib_dir, require_calibration=True)
    selected_rows = select_stratified_image_gen_rows(all_rows, per_domain=prompts_per_domain)

    pipe = load_qwen_image_pipeline(
        base_dir,
        device=device,
        dtype=torch.bfloat16,
        keep_text_encoder_compressed=False,
        keep_transformer_compressed=False,
    )
    tr = pipe.transformer

    records: list[dict[str, Any]] = []
    for p_idx, row in enumerate(selected_rows):
        cond_pil = load_condition_pil_image(image_calib_dir, row, target_resolution=calib_resolution)
        cond_img = [cond_pil] if cond_pil is not None else None
        w_cal, h_cal = _calculate_32_aligned_dimensions(
            calib_resolution * calib_resolution, float(row["width"]) / float(row["height"])
        )
        total_steps = 8 if pipe.config.sample_sigmas is not None else 12
        capture_indices = set(
            int(round(i * (total_steps - 1) / max(steps_per_prompt - 1, 1))) for i in range(steps_per_prompt)
        )
        step_counter = [0]
        cur_entry: dict[str, Any] = {}

        def _pre_block0(
            _m: nn.Module,
            _args: tuple[Any, ...],
            kwargs: dict[str, Any],
            _sc: list[int] = step_counter,
            _ci: set[int] = capture_indices,
            _ce: dict[str, Any] = cur_entry,
            _row: dict[str, Any] = row,
        ):
            if _sc[0] in _ci:
                _ce["domain"] = _row["domain"]
                _ce["mode"] = _row["mode"]
                _ce["lang"] = _row["lang"]
                _ce["hidden_states"] = kwargs["hidden_states"].detach().cpu()
                _ce["modulation"] = kwargs["modulation"].detach().cpu()
                _ce["rotary_emb"] = kwargs["rotary_emb"].detach().cpu()
                _ce["target_token_mask"] = (
                    kwargs["target_token_mask"].detach().cpu()
                    if kwargs.get("target_token_mask") is not None
                    else None
                )
                _ce["segments"] = kwargs.get("segments")
                _ce["key_valid"] = (
                    kwargs["key_valid"].detach().cpu() if kwargs.get("key_valid") is not None else None
                )

        def _post_block_last(
            _m: nn.Module,
            _args: tuple[Any, ...],
            output: torch.Tensor,
            _sc: list[int] = step_counter,
            _ci: set[int] = capture_indices,
            _ce: dict[str, Any] = cur_entry,
        ):
            if _sc[0] in _ci and "hidden_states" in _ce:
                _ce["teacher_last_hidden"] = output.detach().cpu()
                records.append(dict(_ce))
                _ce.clear()
            _sc[0] += 1

        h0 = tr.transformer_blocks[0].register_forward_pre_hook(_pre_block0, with_kwargs=True)
        h1 = tr.transformer_blocks[-1].register_forward_hook(_post_block_last)
        try:
            gen = torch.Generator(device=device).manual_seed(1000 + p_idx)
            call_kwargs: dict[str, Any] = {
                "prompt": row["prompt"],
                "height": h_cal,
                "width": w_cal,
                "output_resolution": calib_resolution,
                "generator": gen,
                "use_kv_cache": False,
                "output_type": "latent",
            }
            if pipe.config.sample_sigmas is None:
                call_kwargs["num_inference_steps"] = total_steps
            if cond_img is not None:
                call_kwargs["image"] = cond_img
            pipe(**call_kwargs)
        finally:
            h0.remove()
            h1.remove()

    log_event(
        "QWEN_IMAGE_DIT_CALIB_STATES",
        trajectories=len(selected_rows),
        samples=len(records),
        t2i_samples=sum(1 for r in records if r.get("mode") == "t2i"),
        ti2i_samples=sum(1 for r in records if r.get("mode") == "ti2i"),
    )
    del pipe, tr
    gc.collect()
    torch.cuda.empty_cache()
    return records


def _quantize_qwen_image_dit(
    base_dir: Path,
    output_tr_dir: Path,
    calib_records: list[dict[str, Any]],
    *,
    group_size: int = 64,
    hadamard_block_size: int = 64,
    dampening_frac: float = 0.01,
    dit_qat_steps: int = 160,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Sequential block-by-block HGPTQ + End-to-End Velocity QAT on `QwenImage21Transformer2DModel`."""
    from diffusers.models import QwenImage21Transformer2DModel

    device = device or torch.device("cuda:0")

    output_tr_dir.mkdir(parents=True, exist_ok=True)
    tr_src_dir = base_dir / "transformer"
    model = (
        QwenImage21Transformer2DModel.from_pretrained(str(tr_src_dir), torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    blocks = model.transformer_blocks
    num_blocks = len(blocks)
    packed_tensors: dict[str, dict[str, torch.Tensor]] = {}

    # Maintain running `hidden_states` across sequential blocks
    cur_hidden = [r["hidden_states"].clone() for r in calib_records]

    for b_idx in range(num_blocks):
        t0 = time.perf_counter()
        block = blocks[b_idx]
        sub_batches = [
            ("attn.to_q", "attn.to_k", "attn.to_v"),
            ("attn.to_out.0",),
            ("img_mlp.gate_layer", "img_mlp.proj"),
            ("img_mlp.out",),
        ]
        for group_projs in sub_batches:
            first_mod: nn.Linear = block.get_submodule(group_projs[0])
            k_dim = first_mod.in_features
            h_mat = torch.zeros((k_dim, k_dim), dtype=torch.float32, device=device)
            n_tok = [0]

            def _acc_hook(
                _m: nn.Module,
                args: tuple[Any, ...],
                _k: int = k_dim,
                _h: torch.Tensor = h_mat,
                _n: list[int] = n_tok,
            ) -> None:
                x = args[0].detach()
                x_rot = _online_hadamard_1d(x, hadamard_block_size).reshape(-1, _k).float()
                _h.addmm_(x_rot.T, x_rot)
                _n[0] += x_rot.shape[0]

            hk = first_mod.register_forward_pre_hook(_acc_hook)
            try:
                with torch.no_grad():
                    for s_idx, rec in enumerate(calib_records):
                        block(
                            hidden_states=cur_hidden[s_idx].to(device=device, dtype=torch.bfloat16),
                            modulation=rec["modulation"].to(device=device, dtype=torch.bfloat16),
                            rotary_emb=rec["rotary_emb"].to(device=device),
                            attention_mask=None,
                            target_token_mask=(
                                rec["target_token_mask"].to(device=device)
                                if rec["target_token_mask"] is not None
                                else None
                            ),
                            layer_cache=None,
                            kv_cache_mode=None,
                            cache_write_slice=None,
                            segments=rec["segments"],
                            key_valid=(
                                rec["key_valid"].to(device=device) if rec["key_valid"] is not None else None
                            ),
                        )
            finally:
                hk.remove()

            h_scaled = (2.0 / float(max(n_tok[0], 1))) * h_mat
            one_sample = torch.ones(1, device=device, dtype=torch.int64)
            for pname in group_projs:
                mod: nn.Linear = block.get_submodule(pname)
                w_rot = _hadamard_rotate_weight(mod.weight.detach().to(device), hadamard_block_size)
                packed = _gptq_pack_tensor(
                    w_rot,
                    h_scaled,
                    one_sample,
                    strategy="group",
                    group_size=group_size,
                    actorder="static",
                    dampening_frac=dampening_frac,
                )
                full_p = f"transformer_blocks.{b_idx}.{pname}"
                packed_tensors[full_p] = packed
                deq = _decompress_packed_2d(
                    packed["weight_packed"], packed["weight_scale"], packed["weight_shape"]
                ).to(device=device, dtype=torch.bfloat16)
                mod.weight.data.copy_(deq)

                def _rot_pre_hook(_m: nn.Module, args: tuple[Any, ...], hb: int = hadamard_block_size):
                    return (_online_hadamard_1d(args[0], hb), *args[1:])

                mod.register_forward_pre_hook(_rot_pre_hook)

        # Advance cur_hidden through the quantized block
        with torch.no_grad():
            for s_idx, rec in enumerate(calib_records):
                out_h = block(
                    hidden_states=cur_hidden[s_idx].to(device=device, dtype=torch.bfloat16),
                    modulation=rec["modulation"].to(device=device, dtype=torch.bfloat16),
                    rotary_emb=rec["rotary_emb"].to(device=device),
                    attention_mask=None,
                    target_token_mask=(
                        rec["target_token_mask"].to(device=device)
                        if rec["target_token_mask"] is not None
                        else None
                    ),
                    layer_cache=None,
                    kv_cache_mode=None,
                    cache_write_slice=None,
                    segments=rec["segments"],
                    key_valid=rec["key_valid"].to(device=device) if rec["key_valid"] is not None else None,
                )
                cur_hidden[s_idx] = out_h.detach().cpu()

        log_event(
            "QWEN_IMAGE_DIT_BLOCK",
            block=b_idx,
            total_blocks=num_blocks,
            elapsed_s=round(time.perf_counter() - t0, 2),
        )

    # End-to-End QK-Norm QAT on transformer_blocks.*.attn.{norm_q,norm_k} against teacher block-31 output
    qat_summary: dict[str, Any] = {"enabled": dit_qat_steps > 0}
    if dit_qat_steps > 0 and len(calib_records) >= 16:
        n_val = min(max(len(calib_records) // 4, 8), 32)
        val_recs = calib_records[:n_val]
        tr_recs = calib_records[n_val:]
        for p in model.parameters():
            p.requires_grad_(False)
        qk_params: list[nn.Parameter] = []
        qk_targets: list[tuple[nn.Module, nn.Parameter]] = []
        dit_qat_hooks = []
        for block in blocks:
            for n_attr in ("attn.norm_q", "attn.norm_k"):
                nm = block.get_submodule(n_attr)
                sp = nn.Parameter(torch.ones_like(nm.weight, dtype=torch.float32, device=device))
                qk_params.append(sp)
                qk_targets.append((nm, sp))
                dit_qat_hooks.append(
                    nm.register_forward_hook(
                        lambda _m, _a, out, _s=sp: (out.float() * _s).to(dtype=out.dtype)
                    )
                )

        def _forward_all_blocks(rec: dict[str, Any]) -> torch.Tensor:
            h = rec["hidden_states"].to(device=device, dtype=torch.bfloat16)
            mod_t = rec["modulation"].to(device=device, dtype=torch.bfloat16)
            rot = rec["rotary_emb"].to(device=device)
            tmask = (
                rec["target_token_mask"].to(device=device) if rec["target_token_mask"] is not None else None
            )
            kval = rec["key_valid"].to(device=device) if rec["key_valid"] is not None else None
            for blk in blocks:
                h = blk(
                    hidden_states=h,
                    modulation=mod_t,
                    rotary_emb=rot,
                    attention_mask=None,
                    target_token_mask=tmask,
                    layer_cache=None,
                    kv_cache_mode=None,
                    cache_write_slice=None,
                    segments=rec["segments"],
                    key_valid=kval,
                )
            return h

        def _eval_val_mse() -> float:
            errs = []
            with torch.no_grad():
                for rec in val_recs:
                    pred = _forward_all_blocks(rec).float()
                    ref = rec["teacher_last_hidden"].to(device=device, dtype=torch.float32)
                    tmask = rec["target_token_mask"]
                    if tmask is not None:
                        m = tmask.to(device=device)
                        pred = pred[:, m]
                        ref = ref[:, m]
                    errs.append(float(((pred - ref).pow(2).sum() / ref.pow(2).sum().clamp_min(1e-12)).item()))
            return sum(errs) / max(len(errs), 1)

        val_before = _eval_val_mse()
        opt = torch.optim.Adam(qk_params, lr=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=dit_qat_steps, eta_min=1e-5)
        for step in range(dit_qat_steps):
            rec = tr_recs[step % len(tr_recs)]
            opt.zero_grad(set_to_none=True)
            pred = _forward_all_blocks(rec).float()
            ref = rec["teacher_last_hidden"].to(device=device, dtype=torch.float32)
            tmask = rec["target_token_mask"]
            if tmask is not None:
                m = tmask.to(device=device)
                pred = pred[:, m]
                ref = ref[:, m]
            loss = (pred - ref).pow(2).mean() / ref.pow(2).mean().clamp_min(1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qk_params, 1.0)
            opt.step()
            sched.step()

        val_after = _eval_val_mse()
        for qh in dit_qat_hooks:
            qh.remove()
        accepted = val_after < val_before
        if accepted:
            for nm, sp in qk_targets:
                nm.weight.data.copy_((nm.weight.float() * sp.detach()).to(dtype=nm.weight.dtype))
        qat_summary = {
            "enabled": True,
            "steps": dit_qat_steps,
            "val_states": len(val_recs),
            "train_states": len(tr_recs),
            "val_rel_mse_before": val_before,
            "val_rel_mse_after": val_after,
            "accepted": accepted,
        }
        log_event("QWEN_IMAGE_DIT_QAT", **qat_summary)

    # Save packed state_dict to transformer/diffusion_pytorch_model.safetensors
    out_state: dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if k.endswith(".weight") and k[:-7] in packed_tensors:
            prefix = k[:-7]
            for sfx, pt in packed_tensors[prefix].items():
                out_state[f"{prefix}.{sfx}"] = pt.cpu()
        else:
            out_state[k] = v.detach().cpu().to(torch.bfloat16)

    save_file(out_state, str(output_tr_dir / "diffusion_pytorch_model.safetensors"))
    tr_cfg = json.loads((tr_src_dir / "config.json").read_text())
    tr_cfg["quantization_config"] = build_dit_quantization_config(
        group_size=group_size,
        hadamard_block_size=hadamard_block_size,
    )
    (output_tr_dir / "config.json").write_text(json.dumps(tr_cfg, indent=2) + "\n")
    _assert_packed_alignment(output_tr_dir)

    gc.collect()
    torch.cuda.empty_cache()
    return {
        "packed_matrices": len(packed_tensors),
        "calib_states": len(calib_records),
        "dit_qat": qat_summary,
    }


def _copy_pipeline_metadata_and_bf16_vae(base_dir: Path, output_dir: Path, info: dict[str, Any]) -> int:
    """Copy `model_index.json`, `scheduler/`, `processor/`, `LICENSE`, and save `vae/` in BF16."""
    shutil.copy2(base_dir / "model_index.json", output_dir / "model_index.json")
    for sub in ("scheduler", "processor"):
        src_sub = base_dir / sub
        dst_sub = output_dir / sub
        if dst_sub.exists():
            shutil.rmtree(dst_sub)
        shutil.copytree(src_sub, dst_sub)

    vae_src = base_dir / "vae"
    vae_dst = output_dir / "vae"
    vae_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vae_src / "config.json", vae_dst / "config.json")
    vae_tensors = load_file(str(vae_src / "diffusion_pytorch_model.safetensors"))
    vae_bf16 = {k: v.to(torch.bfloat16) for k, v in vae_tensors.items()}
    save_file(vae_bf16, str(vae_dst / "diffusion_pytorch_model.safetensors"))
    ensure_license_files(output_dir, base_dir, info.get("license_files"))
    return sum(p.stat().st_size for p in output_dir.rglob("*.safetensors"))


def run_quantize_qwen_image(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    hub_id, info = resolve(args.model)
    base_dir = (
        Path(hub_id)
        if Path(hub_id).is_dir()
        else Path(resolve_snapshot(hub_id, revision=info.get("hub_revision")))
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_calib = Path(args.calib) if getattr(args, "calib", None) else Path("data/image-gen-calib-v1")
    if raw_calib.is_dir() or raw_calib.name.startswith("image-gen-"):
        image_calib_dir = raw_calib
        calib_path = Path("data/calib-cascade-v1.jsonl")
    else:
        image_calib_dir = Path("data/image-gen-calib-v1")
        calib_path = raw_calib

    vision_calib_dir = Path(args.vision_calib_dir or "data/vision-calib-v2")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Ensure `image-gen-calib-v1` is materialized and passes manifest/checksum verification
    ig_manifest, _ = load_image_gen_rows(image_calib_dir, require_calibration=True)

    # 1. Quantize or reuse text_encoder/
    reuse_te = getattr(args, "reuse_ple_vision_from", None)
    te_out_dir = output_dir / "text_encoder"
    if reuse_te and (Path(reuse_te) / "text_encoder" / "model.safetensors").is_file():
        src_te = (Path(reuse_te) / "text_encoder").resolve()
        log_event("QWEN_IMAGE_TE_REUSE", from_dir=str(src_te))
        if src_te != te_out_dir.resolve():
            if te_out_dir.exists():
                shutil.rmtree(te_out_dir)
            shutil.copytree(src_te, te_out_dir)
        te_summary = {"reused_from": str(src_te)}
    elif (te_out_dir / "model.safetensors").is_file():
        log_event("QWEN_IMAGE_TE_REUSE_EXISTING", dir=str(te_out_dir))
        te_summary = {"reused_from": str(te_out_dir)}
    else:
        te_summary = _quantize_qwen_vl_text_encoder(
            base_dir,
            te_out_dir,
            image_calib_dir=image_calib_dir,
            calib_path=calib_path,
            vision_calib_dir=vision_calib_dir,
            calib_samples=args.num_samples or 256,
            calib_seq_len=args.max_seq_length or 2048,
            group_size=args.group_size,
            hadamard_block_size=args.hadamard_block_size,
            dampening_frac=args.dampening_frac,
            quantize_vision=args.quantize_vision,
            quantize_lm_head_w8=args.quantize_lm_head_w8,
            device=dev,
        )

    # 2. Collect DiT trajectory calibration states across all 16 domains (2 per domain = 32 trajectories, 128 states)
    calib_records = _collect_dit_calib_states(
        base_dir,
        image_calib_dir,
        prompts_per_domain=2,
        steps_per_prompt=4,
        calib_resolution=512,
        device=dev,
    )
    dit_summary = _quantize_qwen_image_dit(
        base_dir,
        output_dir / "transformer",
        calib_records,
        group_size=args.group_size,
        hadamard_block_size=args.hadamard_block_size,
        dampening_frac=args.dampening_frac,
        device=dev,
    )

    total_bytes = _copy_pipeline_metadata_and_bf16_vae(base_dir, output_dir, info)
    wall_s = round(time.perf_counter() - t0, 2)
    receipt = {
        "model": args.model,
        "base_snapshot": base_dir.name,
        "group_size": args.group_size,
        "hadamard_block_size": args.hadamard_block_size,
        "actorder": "static",
        "dampening_frac": args.dampening_frac,
        "quantize_vision": args.quantize_vision,
        "quantize_lm_head_w8": args.quantize_lm_head_w8,
        "calibration_suite": {
            "dir": str(image_calib_dir),
            "suite": ig_manifest.get("suite"),
            "rows": ig_manifest.get("rows"),
            "rows_sha256": ig_manifest.get("rows_sha256"),
        },
        "text_encoder": te_summary,
        "transformer": dit_summary,
        "total_safetensors_bytes": total_bytes,
        "quant_wall_s": wall_s,
    }
    (output_dir / "rot_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    log_event("QWEN_IMAGE_QUANT_DONE", output_dir=str(output_dir), total_bytes=total_bytes, wall_s=wall_s)


def _compute_psnr_and_cosine(pred: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    p = pred.float()
    r = ref.float()
    diff = p - r
    mse = float(diff.pow(2).mean().item())
    ref_pow = float(r.pow(2).mean().item())
    rel_rmse = math.sqrt(mse / max(ref_pow, 1e-12))
    sqnr_db = 10.0 * math.log10(max(ref_pow, 1e-12) / max(mse, 1e-12))
    cos = float(F.cosine_similarity(p.flatten().unsqueeze(0), r.flatten().unsqueeze(0)).item())
    return {
        "cosine": cos,
        "rel_rmse": rel_rmse,
        "sqnr_db": sqnr_db,
    }


def _compute_image_psnr(img_pred, img_ref) -> float:
    import numpy as np

    a = np.asarray(img_pred, dtype=np.float64) / 255.0
    b = np.asarray(img_ref, dtype=np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def _summarize_metric_list(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"count": 0, "mean_cosine": 0.0, "mean_rel_rmse": 0.0, "mean_sqnr_db": 0.0}
    n = len(items)
    out: dict[str, Any] = {
        "count": n,
        "mean_cosine": sum(d["cosine"] for d in items) / n,
        "mean_rel_rmse": sum(d["rel_rmse"] for d in items) / n,
        "mean_sqnr_db": sum(d["sqnr_db"] for d in items) / n,
    }
    if "image_psnr_db" in items[0]:
        psnrs = [d["image_psnr_db"] for d in items]
        out["image_mean_psnr_db"] = sum(psnrs) / n
        out["image_min_psnr_db"] = min(psnrs)
    return out


def _summarize_stratified(items: list[dict[str, Any]]) -> dict[str, Any]:
    base = _summarize_metric_list(items)
    t2i_items = [d for d in items if d.get("mode") == "t2i"]
    ti2i_items = [d for d in items if d.get("mode") == "ti2i"]
    en_items = [d for d in items if d.get("lang") == "en"]
    zh_items = [d for d in items if d.get("lang") == "zh"]
    by_dom: dict[str, list[dict[str, Any]]] = {}
    for d in items:
        by_dom.setdefault(str(d.get("domain", "all")), []).append(d)
    base["by_mode"] = {
        "t2i": _summarize_metric_list(t2i_items),
        "ti2i": _summarize_metric_list(ti2i_items),
    }
    base["by_lang"] = {
        "en": _summarize_metric_list(en_items),
        "zh": _summarize_metric_list(zh_items),
    }
    base["per_domain"] = {dom: _summarize_metric_list(d_items) for dom, d_items in by_dom.items()}
    return base


def _capture_suite_teacher_references(
    base_pipe: Any,
    fixture_dir: Path,
    rows: list[dict[str, Any]],
    *,
    traj_per_domain: int = 1,
    probe_res: int = 512,
    total_steps: int = 8,
    seed_base: int = 42,
    device: torch.device,
) -> dict[str, Any]:
    """Capture BF16 teacher prompt embeddings across all `rows`, plus single-step teacher
    velocities and full end-to-end trajectories on `traj_per_domain` rows per domain."""
    traj_rows = select_stratified_image_gen_rows(rows, per_domain=traj_per_domain)
    traj_keys = {(r["domain"], r["source_row"], r["prompt_sha256"]) for r in traj_rows}

    teacher_embeds: list[tuple[dict[str, Any], torch.Tensor]] = []
    teacher_velocities: list[tuple[dict[str, Any], dict[str, Any], torch.Tensor]] = []
    teacher_trajectories: list[tuple[dict[str, Any], int, int, int, torch.Tensor, Any]] = []

    with torch.no_grad():
        for r_idx, row in enumerate(rows):
            cond_pil = load_condition_pil_image(fixture_dir, row, target_resolution=probe_res)
            cond_img = [cond_pil] if cond_pil is not None else None
            pe, _, _ = base_pipe.encode_prompt(prompt=row["prompt"], image=cond_img, device=device)
            teacher_embeds.append((row, pe.detach().cpu()))

            r_key = (row["domain"], row["source_row"], row["prompt_sha256"])
            if r_key not in traj_keys:
                continue

            w_p, h_p = _calculate_32_aligned_dimensions(
                probe_res * probe_res, float(row["width"]) / float(row["height"])
            )
            target_latents_len = (h_p // base_pipe.vae_scale_factor) * (w_p // base_pipe.vae_scale_factor)
            step_idx = [0]
            capture_steps = (0, total_steps // 2, total_steps - 1)

            def _tr_hook(
                _m: nn.Module,
                _a: tuple[Any, ...],
                kw: dict[str, Any],
                out: Any,
                _si: list[int] = step_idx,
                _row: dict[str, Any] = row,
                _tlen: int = target_latents_len,
                _cs: tuple[int, ...] = capture_steps,
            ):
                if _si[0] in _cs:
                    v_out = out[0] if isinstance(out, tuple) else out.sample
                    v_target = v_out[:, -_tlen:]
                    teacher_velocities.append(
                        (
                            _row,
                            {
                                "hidden_states": kw["hidden_states"].detach().cpu(),
                                "timestep": kw["timestep"].detach().cpu(),
                                "encoder_hidden_states": kw["encoder_hidden_states"].detach().cpu(),
                                "encoder_hidden_states_mask": (
                                    kw["encoder_hidden_states_mask"].detach().cpu()
                                    if kw.get("encoder_hidden_states_mask") is not None
                                    else None
                                ),
                                "img_shapes": copy.deepcopy(kw["img_shapes"]),
                                "img_mask": kw["img_mask"].detach().cpu(),
                                "target_latents_len": _tlen,
                            },
                            v_target.detach().cpu(),
                        )
                    )
                _si[0] += 1

            hk = base_pipe.transformer.register_forward_hook(_tr_hook, with_kwargs=True)
            try:
                seed = seed_base + r_idx
                gen = torch.Generator(device=device).manual_seed(seed)
                call_kw: dict[str, Any] = {
                    "prompt": row["prompt"],
                    "height": h_p,
                    "width": w_p,
                    "output_resolution": probe_res,
                    "generator": gen,
                    "use_kv_cache": False,
                    "output_type": "latent",
                }
                if base_pipe.config.sample_sigmas is None:
                    call_kw["num_inference_steps"] = total_steps
                if cond_img is not None:
                    call_kw["image"] = cond_img
                lat_out = base_pipe(**call_kw).images
                unpacked = base_pipe._unpack_latents(lat_out, h_p, w_p, base_pipe.vae_scale_factor).to(
                    base_pipe.vae.dtype
                )
                l_mean = (
                    torch.tensor(base_pipe.vae.config.latents_mean)
                    .view(1, base_pipe.vae.config.z_dim, 1, 1, 1)
                    .to(unpacked.device, unpacked.dtype)
                )
                l_std = (
                    torch.tensor(base_pipe.vae.config.latents_std)
                    .view(1, base_pipe.vae.config.z_dim, 1, 1, 1)
                    .to(unpacked.device, unpacked.dtype)
                )
                dec = base_pipe.vae.decode(unpacked * l_std + l_mean, return_dict=False)[0][:, :, 0]
                pil_img = base_pipe.image_processor.postprocess(dec, output_type="pil")[0]
                teacher_trajectories.append((row, seed, w_p, h_p, lat_out.detach().cpu(), pil_img))
            finally:
                hk.remove()

    return {
        "embeds": teacher_embeds,
        "velocities": teacher_velocities,
        "trajectories": teacher_trajectories,
    }


def _evaluate_suite_against_teacher(
    q_pipe: Any,
    fixture_dir: Path,
    teacher_refs: dict[str, Any],
    *,
    probe_res: int = 512,
    total_steps: int = 8,
    device: torch.device,
) -> dict[str, Any]:
    embed_metrics: list[dict[str, Any]] = []
    txt_in_metrics: list[dict[str, Any]] = []
    vel_metrics: list[dict[str, Any]] = []
    traj_metrics: list[dict[str, Any]] = []

    with torch.no_grad():
        for row, ref_pe in teacher_refs["embeds"]:
            cond_pil = load_condition_pil_image(fixture_dir, row, target_resolution=probe_res)
            cond_img = [cond_pil] if cond_pil is not None else None
            q_pe, _, _ = q_pipe.encode_prompt(prompt=row["prompt"], image=cond_img, device=device)
            m = _compute_psnr_and_cosine(q_pe.cpu(), ref_pe)
            m.update(domain=row["domain"], mode=row["mode"], lang=row["lang"])
            embed_metrics.append(m)
            if hasattr(q_pipe, "transformer") and hasattr(q_pipe.transformer, "txt_in"):
                ref_ti = q_pipe.transformer.txt_in(ref_pe.to(device=device, dtype=q_pe.dtype)).cpu()
                q_ti = q_pipe.transformer.txt_in(q_pe).cpu()
                m_ti = _compute_psnr_and_cosine(q_ti, ref_ti)
                m_ti.update(domain=row["domain"], mode=row["mode"], lang=row["lang"])
                txt_in_metrics.append(m_ti)

        for row, kw_in, ref_v in teacher_refs["velocities"]:
            q_v = q_pipe.transformer(
                hidden_states=kw_in["hidden_states"].to(device, dtype=torch.bfloat16),
                timestep=kw_in["timestep"].to(device, dtype=torch.bfloat16),
                encoder_hidden_states=kw_in["encoder_hidden_states"].to(device, dtype=torch.bfloat16),
                encoder_hidden_states_mask=(
                    kw_in["encoder_hidden_states_mask"].to(device)
                    if kw_in["encoder_hidden_states_mask"] is not None
                    else None
                ),
                img_shapes=kw_in["img_shapes"],
                img_mask=kw_in["img_mask"].to(device),
                return_dict=False,
            )[0]
            tlen = kw_in["target_latents_len"]
            m = _compute_psnr_and_cosine(q_v[:, -tlen:].cpu(), ref_v)
            m.update(domain=row["domain"], mode=row["mode"], lang=row["lang"])
            vel_metrics.append(m)

        for row, seed, w_p, h_p, ref_lat, ref_pil in teacher_refs["trajectories"]:
            cond_pil = load_condition_pil_image(fixture_dir, row, target_resolution=probe_res)
            cond_img = [cond_pil] if cond_pil is not None else None
            gen = torch.Generator(device=device).manual_seed(seed)
            call_kw: dict[str, Any] = {
                "prompt": row["prompt"],
                "height": h_p,
                "width": w_p,
                "output_resolution": probe_res,
                "generator": gen,
                "use_kv_cache": False,
                "output_type": "latent",
            }
            if q_pipe.config.sample_sigmas is None:
                call_kw["num_inference_steps"] = total_steps
            if cond_img is not None:
                call_kw["image"] = cond_img
            q_lat = q_pipe(**call_kw).images
            m = _compute_psnr_and_cosine(q_lat.cpu(), ref_lat)
            unpacked = q_pipe._unpack_latents(q_lat, h_p, w_p, q_pipe.vae_scale_factor).to(q_pipe.vae.dtype)
            l_mean = (
                torch.tensor(q_pipe.vae.config.latents_mean)
                .view(1, q_pipe.vae.config.z_dim, 1, 1, 1)
                .to(unpacked.device, unpacked.dtype)
            )
            l_std = (
                torch.tensor(q_pipe.vae.config.latents_std)
                .view(1, q_pipe.vae.config.z_dim, 1, 1, 1)
                .to(unpacked.device, unpacked.dtype)
            )
            dec = q_pipe.vae.decode(unpacked * l_std + l_mean, return_dict=False)[0][:, :, 0]
            q_pil = q_pipe.image_processor.postprocess(dec, output_type="pil")[0]
            m["image_psnr_db"] = _compute_image_psnr(q_pil, ref_pil)
            m.update(domain=row["domain"], mode=row["mode"], lang=row["lang"], width=w_p, height=h_p)
            traj_metrics.append(m)

    e2e_summary = _summarize_stratified(traj_metrics)
    out_summary: dict[str, Any] = {
        "text_encoder_prompt_embeds": _summarize_stratified(embed_metrics),
        "teacher_forced_velocity": {
            "samples": len(vel_metrics),
            **_summarize_stratified(vel_metrics),
        },
        "end_to_end_trajectory": {
            "trajectories": len(traj_metrics),
            "latent_mean_cosine": e2e_summary["mean_cosine"],
            "latent_mean_rel_rmse": e2e_summary["mean_rel_rmse"],
            "latent_mean_sqnr_db": e2e_summary["mean_sqnr_db"],
            "image_mean_psnr_db": e2e_summary.get("image_mean_psnr_db", 0.0),
            "image_min_psnr_db": e2e_summary.get("image_min_psnr_db", 0.0),
            "by_mode": e2e_summary["by_mode"],
            "by_lang": e2e_summary["by_lang"],
            "per_domain": e2e_summary["per_domain"],
        },
    }
    if txt_in_metrics:
        out_summary["txt_in_conditioning_embeds"] = _summarize_stratified(txt_in_metrics)
    return out_summary


def run_probe_qwen_image(args: argparse.Namespace) -> None:
    """Evaluate a quantized Qwen-Image-2.1 / Qwen-Image-2.1-Turbo checkpoint against its BF16 teacher
    on `data/image-gen-probe-v1` (64 rows across 16 T2I/TI2I EN+ZH domains) and `data/image-gen-guard-v1`
    (24 OOD rows across 6 T2I/TI2I domains):
    1. Held-out `text_encoder` prompt_embeds fidelity (all 64 probe + 24 guard rows),
    2. Single-step teacher-forced DiT velocity prediction fidelity across denoising timesteps
       (48 probe + 18 guard velocity evaluations across 5 aspect ratios and TI2I condition latents),
    3. Full deterministic trajectory latent & decoded RGB image PSNR / cosine / SQNR
       (16 probe + 6 guard trajectories)."""
    hub_id, info = resolve(args.model)
    base_dir = (
        Path(hub_id)
        if Path(hub_id).is_dir()
        else Path(resolve_snapshot(hub_id, revision=info.get("hub_revision")))
    )
    quant_dirs = (
        [Path(args.quant).resolve()]
        if isinstance(args.quant, str | Path)
        else [Path(q).resolve() for q in args.quant]
    )
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    raw_probe = getattr(args, "eval_vision", None)
    probe_dir = (
        Path(raw_probe)
        if raw_probe and Path(raw_probe).name != "vision-probe-v2"
        else Path("data/image-gen-probe-v1")
    )
    probe_manifest, probe_rows = load_image_gen_rows(probe_dir, require_calibration=False)

    raw_guard = getattr(args, "eval_vision_guard", None)
    guard_dir = (
        Path(raw_guard)
        if raw_guard and Path(raw_guard).name != "vision-guard-v2"
        else Path("data/image-gen-guard-v1")
    )
    guard_manifest, guard_rows = load_image_gen_rows(guard_dir, require_calibration=False, require_guard=True)

    if getattr(args, "limit", None):
        probe_rows = probe_rows[: int(args.limit)]
        guard_rows = guard_rows[: min(int(args.limit), len(guard_rows))]

    log_event(
        "QWEN_IMAGE_PROBE_BASE_LOAD",
        base=str(base_dir),
        probe_rows=len(probe_rows),
        guard_rows=len(guard_rows),
    )
    base_pipe = load_qwen_image_pipeline(
        base_dir,
        device=dev,
        dtype=torch.bfloat16,
        keep_text_encoder_compressed=False,
        keep_transformer_compressed=False,
    )

    total_steps = 8 if base_pipe.config.sample_sigmas is not None else 16
    probe_res = 512

    teacher_probe = _capture_suite_teacher_references(
        base_pipe,
        probe_dir,
        probe_rows,
        traj_per_domain=1,
        probe_res=probe_res,
        total_steps=total_steps,
        seed_base=42,
        device=dev,
    )
    teacher_guard = _capture_suite_teacher_references(
        base_pipe,
        guard_dir,
        guard_rows,
        traj_per_domain=1,
        probe_res=probe_res,
        total_steps=total_steps,
        seed_base=2042,
        device=dev,
    )

    del base_pipe
    gc.collect()
    torch.cuda.empty_cache()

    for quant_dir in quant_dirs:
        log_event("QWEN_IMAGE_PROBE_QUANT_LOAD", quant=str(quant_dir))
        q_pipe = load_qwen_image_pipeline(
            quant_dir,
            device=dev,
            dtype=torch.bfloat16,
            keep_text_encoder_compressed=False,
            keep_transformer_compressed=False,
        )
        probe_eval = _evaluate_suite_against_teacher(
            q_pipe,
            probe_dir,
            teacher_probe,
            probe_res=probe_res,
            total_steps=total_steps,
            device=dev,
        )
        guard_eval = _evaluate_suite_against_teacher(
            q_pipe,
            guard_dir,
            teacher_guard,
            probe_res=probe_res,
            total_steps=total_steps,
            device=dev,
        )

        total_bytes = sum(p.stat().st_size for p in quant_dir.rglob("*.safetensors"))
        result = {
            "model": args.model,
            "base": str(base_dir),
            "quant": str(quant_dir),
            "total_safetensors_bytes": total_bytes,
            "total_safetensors_gib": round(total_bytes / (1024**3), 3),
            "probe_suite": {
                "dir": str(probe_dir),
                "rows": len(probe_rows),
                "rows_sha256": probe_manifest.get("rows_sha256"),
            },
            "guard_suite": {
                "dir": str(guard_dir),
                "rows": len(guard_rows),
                "rows_sha256": guard_manifest.get("rows_sha256"),
            },
            "probe_cases": len(probe_rows),
            "guard_cases": len(guard_rows),
            "probe_resolution": probe_res,
            "num_inference_steps": total_steps,
            "text_encoder_prompt_embeds": probe_eval["text_encoder_prompt_embeds"],
            "teacher_forced_velocity": probe_eval["teacher_forced_velocity"],
            "end_to_end_trajectory": probe_eval["end_to_end_trajectory"],
            "image_gen_guard_v1": guard_eval,
        }
        out_json = (
            Path(args.output_json)
            if getattr(args, "output_json", None)
            else (quant_dir / "probe_results.json")
        )
        out_json.write_text(json.dumps(result, indent=2) + "\n")
        log_event(
            "QWEN_IMAGE_PROBE_DONE",
            vel_cosine=result["teacher_forced_velocity"]["mean_cosine"],
            vel_sqnr_db=result["teacher_forced_velocity"]["mean_sqnr_db"],
            img_psnr_db=result["end_to_end_trajectory"]["image_mean_psnr_db"],
            guard_vel_cosine=guard_eval["teacher_forced_velocity"]["mean_cosine"],
            guard_img_psnr_db=guard_eval["end_to_end_trajectory"]["image_mean_psnr_db"],
        )
        del q_pipe
        gc.collect()
        torch.cuda.empty_cache()
