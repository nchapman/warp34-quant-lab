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
from warp_quant.hadamard_gptq import (
    LM_HEAD_TARGET_RE,
    TEXT_HAD_TARGET_RE,
    TEXT_TARGET_RE,
    VISION_FC2_TARGET_RE,
    _assert_packed_alignment,
    _decompress_packed_2d,
    _gptq_calibrate_vision,
    _gptq_pack_tensor,
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

# Disjoint calibration vs probe prompts for DiT trajectory collection
CALIB_PROMPTS_T2I: tuple[str, ...] = (
    'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement, 35mm photo',
    'A hand-lettered classroom chemistry poster titled "States of Matter" with clean diagrams and bullet notes',
    "A studio portrait of an elderly clockmaker examining a brass tourbillon under warm tungsten light",
    "An isometric cutaway architectural diagram of a subterranean hydroponic farm with labeled levels",
    "A transparent RGBA sticker of a mechanical owl with copper wings and glowing amber eyes, isolated subject",
    "A vintage railway travel poster for Kyoto in autumn, bold serif typography at top and bottom",
    "A macro photograph of morning dew on a dragonfly wing, iridescent diffraction patterns, shallow depth of field",
    'A chalkboard menu outside a Parisian bakery reading "Croissants Chauds & Cafe Crème - 4.50 EUR"',
    "A watercolor botanical illustration of a pomegranate split open with Latin binomial calligraphy",
    "A cyberpunk noodle stall at dusk with bilingual kanji and English holo-signs, volumetric steam",
    "A flat-lay knolling photograph of vintage camera gear, brass lenses, film rolls, and handwritten notes",
    'A retro 1970s science fiction book cover titled "ORBITAL DRIFT" in chrome geometric lettering',
    "A high-contrast ink woodblock print of Mount Rainier above storm clouds in the style of Hokusai",
    "A modern mobile banking UI dashboard screen showing expense charts, clean sans-serif typography",
    "A transparent PNG sprite of a crystal potion bottle filled with swirling bioluminescent nebula liquid",
    "An oil painting of a sunlit conservatory filled with monsteras, terracotta pots, and a sleeping tabby cat",
)

CALIB_PROMPTS_EDIT: tuple[str, ...] = (
    'Add a small brass plaque at the bottom reading "ARCHIVE SPECIMEN NO. 14" while preserving the subject',
    "Transform the lighting to golden hour sunset while keeping all text and structural layout intact",
    "Convert the scene into a clean watercolor and ink architectural sketch while preserving proportions",
    "Extract the main subject on a clean transparent background with crisp edges",
)

PROBE_PROMPTS_T2I: tuple[str, ...] = (
    'A museum exhibition banner reading "SYMMETRY & LIGHT: 1925-2025" in crisp bauhaus typography, architectural hall',
    "A close-up studio portrait of a violinist mid-performance, dramatic rim lighting, fine wood grain texture",
    'An illustrated infographic poster titled "The Water Cycle" with arrows, clouds, mountains, and legible labels',
    "A transparent RGBA cutout of an antique brass astrolabe with engraved celestial coordinates",
    "A bustling night market in Taipei after rain, neon reflections on wet asphalt, cinematic wide shot",
    'A minimalist coffee bag packaging design reading "SINGLE ORIGIN ETHIOPIA - YIRGACHEFFE" on matte kraft paper',
)

PROBE_PROMPTS_EDIT: tuple[str, ...] = (
    'Relight the scene with cool moonlight and add a subtle caption "NIGHT STUDY" in the lower corner',
    "Render the image in a crisp ligne-claire comic book style while preserving all text and geometry",
)


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
    processor,
    vision_calib_dir: Path,
    *,
    group_size: int,
    hadamard_block_size: int,
    dampening_frac: float,
    device: torch.device,
    max_samples: int = 48,
) -> dict[str, dict[str, torch.Tensor]]:
    """Calibrate and GPTQ-quantize `text_encoder.model.visual` (`blocks.0..26`, `merger`,
    and `deepstack_merger_list.0..2`) on `data/vision-calib-v2`."""
    packed_out = _gptq_calibrate_vision(
        text_encoder,
        processor,
        vision_calib_dir,
        group_size=group_size,
        hadamard_block_size=hadamard_block_size,
        actorder="static",
        dampening_frac=dampening_frac,
        max_calib_samples=max_samples,
    )
    log_event("QWEN_IMAGE_VISION_CALIB_DONE", matrices=len(packed_out), dir=str(vision_calib_dir))
    return packed_out


def _quantize_qwen_vl_text_encoder(
    base_dir: Path,
    output_te_dir: Path,
    *,
    calib_path: Path,
    vision_calib_dir: Path,
    calib_samples: int = 256,
    calib_seq_len: int = 2048,
    group_size: int = 64,
    hadamard_block_size: int = 64,
    dampening_frac: float = 0.01,
    quantize_vision: bool = True,
    quantize_lm_head_w8: bool = True,
    norm_qat_steps: int = 120,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Calibrate and quantize `text_encoder/` (`Qwen3VLForConditionalGeneration`) to W4A16-HGPTQ
    with sequential layer-by-layer Hessian propagation and Layer-35 unnormalized hidden-state Norm QAT."""
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    device = device or torch.device("cuda:0")

    output_te_dir.mkdir(parents=True, exist_ok=True)
    te_src_dir = base_dir / "text_encoder"
    proc_dir = base_dir / "processor"
    processor = Qwen3VLProcessor.from_pretrained(str(proc_dir))
    tokenizer = processor.tokenizer

    log_event("QWEN_IMAGE_TE_LOAD", path=str(te_src_dir))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(te_src_dir), torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()

    packed_tensors: dict[str, dict[str, torch.Tensor]] = {}
    if quantize_vision:
        vis_packed = _quantize_qwen_vl_vision_tower(
            model,
            processor,
            vision_calib_dir,
            group_size=group_size,
            hadamard_block_size=hadamard_block_size,
            dampening_frac=dampening_frac,
            device=device,
        )
        packed_tensors.update(vis_packed)

    # Build calibration token sequences combining calib-cascade-v1.jsonl and QwenImage21 T2I prompt templates
    seqs: list[torch.Tensor] = []
    sys_tpl = (
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    for p in CALIB_PROMPTS_T2I + CALIB_PROMPTS_EDIT:
        ids = tokenizer(
            sys_tpl.format(p), return_tensors="pt", truncation=True, max_length=calib_seq_len
        ).input_ids
        if ids.shape[1] >= 16:
            seqs.append(ids)

    with calib_path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(seqs) >= calib_samples:
                break
            row = json.loads(line)
            text = row.get("text") or ""
            conv = row.get("conversations") or row.get("messages")
            if not text and conv:
                text = "\n".join(m.get("content", "") for m in conv if isinstance(m.get("content"), str))
            if not text:
                continue
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=calib_seq_len).input_ids
            if ids.shape[1] >= 32:
                seqs.append(ids)

    log_event("QWEN_IMAGE_TE_CALIB_SEQS", count=len(seqs))
    lang_model = model.model.language_model
    layers = lang_model.layers
    num_layers = len(layers)

    # Capture teacher unnormalized Layer-(num_layers-1) output on the first 64 sequences for Norm QAT & W8 lm_head
    qat_seqs = seqs[: min(64, len(seqs))]
    teacher_pre_norm: list[torch.Tensor] = []
    with torch.no_grad():
        for ids in qat_seqs:
            captured: list[torch.Tensor] = []
            h = lang_model.norm.register_forward_hook(
                lambda _m, a, _o, _cap=captured: _cap.append(a[0].detach().cpu()) or a[0]
            )
            try:
                model(input_ids=ids.to(device), use_cache=False)
            finally:
                h.remove()
            teacher_pre_norm.append(captured[0])

    # Sequential layer-by-layer Hessian GPTQ across all 36 decoder layers (streaming hidden states)
    cur_h: list[torch.Tensor] = []
    pos_embs: list[tuple[torch.Tensor, torch.Tensor]] = []
    text_pos_list: list[torch.Tensor] = []
    with torch.no_grad():
        for ids in seqs:
            ids_d = ids.to(device)
            emb = lang_model.embed_tokens(ids_d)
            seq_l = emb.shape[1]
            p3d = torch.arange(seq_l, device=device).view(1, 1, -1).expand(3, emb.shape[0], -1)
            tpos = torch.arange(seq_l, device=device).view(1, -1).expand(emb.shape[0], -1)
            pe = lang_model.rotary_emb(emb, p3d)
            cur_h.append(emb.detach().cpu())
            pos_embs.append((pe[0].detach().cpu(), pe[1].detach().cpu()))
            text_pos_list.append(tpos.detach().cpu())

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
                    for s_idx in range(len(seqs)):
                        pe_d = (
                            pos_embs[s_idx][0].to(device=device, dtype=torch.bfloat16),
                            pos_embs[s_idx][1].to(device=device, dtype=torch.bfloat16),
                        )
                        layer(
                            cur_h[s_idx].to(device=device, dtype=torch.bfloat16),
                            attention_mask=None,
                            position_ids=text_pos_list[s_idx].to(device),
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
            for s_idx in range(len(seqs)):
                pe_d = (
                    pos_embs[s_idx][0].to(device=device, dtype=torch.bfloat16),
                    pos_embs[s_idx][1].to(device=device, dtype=torch.bfloat16),
                )
                out_h = layer(
                    cur_h[s_idx].to(device=device, dtype=torch.bfloat16),
                    attention_mask=None,
                    position_ids=text_pos_list[s_idx].to(device),
                    past_key_values=None,
                    position_embeddings=pe_d,
                )
                cur_h[s_idx] = out_h.detach().cpu()

        log_event(
            "QWEN_IMAGE_TE_LAYER",
            layer=l_idx,
            total_layers=num_layers,
            elapsed_s=round(time.perf_counter() - t0, 2),
        )

    # Quantize untied lm_head to unrotated W8_g64 GPTQ+MSE
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

    # End-to-End FP32-master RMSNorm QAT targeting unnormalized Layer-35 output (`hidden_states[-1]`)
    qat_summary: dict[str, Any] = {"enabled": norm_qat_steps > 0}
    if norm_qat_steps > 0 and len(qat_seqs) >= 8:
        val_seqs = qat_seqs[:8]
        val_targets = teacher_pre_norm[:8]
        tr_seqs = qat_seqs[8:]
        tr_targets = teacher_pre_norm[8:]

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

        def _eval_pre_norm_mse() -> float:
            errs = []
            with torch.no_grad():
                for ids, tgt in zip(val_seqs, val_targets, strict=True):
                    cap: list[torch.Tensor] = []
                    hk = lang_model.norm.register_forward_hook(
                        lambda _m, a, _o, _c=cap: _c.append(a[0]) or a[0]
                    )
                    try:
                        model(input_ids=ids.to(device), use_cache=False)
                    finally:
                        hk.remove()
                    pred = cap[0].float()
                    ref = tgt.to(device=device, dtype=torch.float32)
                    errs.append(float(((pred - ref).pow(2).sum() / ref.pow(2).sum().clamp_min(1e-12)).item()))
            return sum(errs) / max(len(errs), 1)

        val_before = _eval_pre_norm_mse()
        opt = torch.optim.Adam(norm_params, lr=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=norm_qat_steps, eta_min=1e-5)
        for step in range(norm_qat_steps):
            idx = step % len(tr_seqs)
            ids = tr_seqs[idx].to(device)
            tgt = tr_targets[idx].to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            cap_step: list[torch.Tensor] = []
            hk = lang_model.norm.register_forward_hook(lambda _m, a, _o, _c=cap_step: _c.append(a[0]) or a[0])
            try:
                model(input_ids=ids, use_cache=False)
            finally:
                hk.remove()
            pred = cap_step[0].float()
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
        "norm_qat": qat_summary,
    }


def _collect_dit_calib_states(
    base_dir: Path,
    vision_calib_dir: Path,
    *,
    num_prompts: int = 16,
    steps_per_prompt: int = 4,
    calib_resolution: int = 512,
    device: torch.device | None = None,
) -> list[dict[str, Any]]:
    """Collect real FlowMatch denoising trajectory inputs at the entry to `transformer_blocks[0]`
    plus teacher block-31 output states across T2I and TI2I (reference-image editing) prompts."""
    from PIL import Image

    device = device or torch.device("cuda:0")

    pipe = load_qwen_image_pipeline(
        base_dir,
        device=device,
        dtype=torch.bfloat16,
        keep_text_encoder_compressed=False,
        keep_transformer_compressed=False,
    )
    tr = pipe.transformer

    # Load a few calibration images from data/vision-calib-v2 (enforcing calibration=true)
    manifest_p = vision_calib_dir / "manifest.json"
    if manifest_p.is_file():
        m = json.loads(manifest_p.read_text())
        if not m.get("calibration", False):
            raise ValueError(f"Refusing to calibrate DiT on non-calibration fixture {vision_calib_dir}")
    calib_imgs: list[Image.Image] = []
    img_dir = vision_calib_dir / "images"
    if img_dir.is_dir():
        for p in sorted(img_dir.glob("*.image"))[:4]:
            calib_imgs.append(Image.open(p).convert("RGB"))

    records: list[dict[str, Any]] = []
    prompts = list(CALIB_PROMPTS_T2I[: max(num_prompts - len(CALIB_PROMPTS_EDIT), 4)])

    # Hook transformer_blocks[0] input and transformer_blocks[-1] output on selected steps
    for p_idx, prompt in enumerate(prompts + list(CALIB_PROMPTS_EDIT)):
        is_edit = p_idx >= len(prompts) and len(calib_imgs) > 0
        cond_img = [calib_imgs[p_idx % len(calib_imgs)]] if is_edit else None
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
        ):
            if _sc[0] in _ci:
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
                "prompt": prompt,
                "height": calib_resolution,
                "width": calib_resolution,
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

    log_event("QWEN_IMAGE_DIT_CALIB_STATES", samples=len(records))
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
    dit_qat_steps: int = 120,
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
    if dit_qat_steps > 0 and len(calib_records) >= 8:
        val_recs = calib_records[:8]
        tr_recs = calib_records[8:]
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

    calib_path = Path(args.calib or "data/calib-cascade-v1.jsonl")
    vision_calib_dir = Path(args.vision_calib_dir or "data/vision-calib-v2")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # 2. Collect DiT trajectory calibration states and quantize transformer/
    calib_records = _collect_dit_calib_states(
        base_dir,
        vision_calib_dir,
        num_prompts=16,
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


def run_probe_qwen_image(args: argparse.Namespace) -> None:
    """Evaluate a quantized Qwen-Image-2.1 / Qwen-Image-2.1-Turbo checkpoint against its BF16 teacher:
    1. Held-out `text_encoder` prompt_embeds fidelity (T2I + TI2I with `data/vision-probe-v2`),
    2. Single-step teacher-forced DiT velocity prediction fidelity across denoising timesteps,
    3. Full deterministic trajectory latent & decoded RGB image PSNR / cosine / SQNR."""
    from PIL import Image

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

    vision_probe_dir = Path(args.eval_vision or "data/vision-probe-v2")
    if (vision_probe_dir / "manifest.json").is_file():
        m = json.loads((vision_probe_dir / "manifest.json").read_text())
        if m.get("calibration", False):
            raise ValueError(f"Refusing to evaluate Qwen-Image on calibration fixture {vision_probe_dir}")
    probe_imgs: list[Image.Image] = []
    if (vision_probe_dir / "images").is_dir():
        for p in sorted((vision_probe_dir / "images").glob("*.image"))[:2]:
            probe_imgs.append(Image.open(p).convert("RGB"))

    log_event("QWEN_IMAGE_PROBE_BASE_LOAD", base=str(base_dir))
    base_pipe = load_qwen_image_pipeline(
        base_dir,
        device=dev,
        dtype=torch.bfloat16,
        keep_text_encoder_compressed=False,
        keep_transformer_compressed=False,
    )

    # 1. Capture teacher prompt_embeds, teacher single-step velocities, and teacher full trajectories
    teacher_embeds: list[torch.Tensor] = []
    teacher_velocities: list[tuple[dict[str, Any], torch.Tensor]] = []
    teacher_trajectories: list[tuple[torch.Tensor, Image.Image]] = []

    all_probe_cases: list[tuple[str, list[Image.Image] | None]] = [(p, None) for p in PROBE_PROMPTS_T2I]
    for i, ep in enumerate(PROBE_PROMPTS_EDIT):
        if probe_imgs:
            all_probe_cases.append((ep, [probe_imgs[i % len(probe_imgs)]]))

    total_steps = 8 if base_pipe.config.sample_sigmas is not None else 16
    probe_res = 512

    with torch.no_grad():
        for c_idx, (prompt, cond_img) in enumerate(all_probe_cases):
            pe, _, _ = base_pipe.encode_prompt(prompt=prompt, image=cond_img, device=dev)
            teacher_embeds.append(pe.detach().cpu())

            step_idx = [0]

            def _tr_hook(
                _m: nn.Module,
                _a: tuple[Any, ...],
                kw: dict[str, Any],
                out: Any,
                _si: list[int] = step_idx,
                _ci: int = c_idx,
            ):
                if _si[0] in (0, total_steps // 2, total_steps - 1) and _ci < 4:
                    v_out = out[0] if isinstance(out, tuple) else out.sample
                    teacher_velocities.append(
                        (
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
                            },
                            v_out.detach().cpu(),
                        )
                    )
                _si[0] += 1

            hk = base_pipe.transformer.register_forward_hook(_tr_hook, with_kwargs=True)
            try:
                gen = torch.Generator(device=dev).manual_seed(42 + c_idx)
                call_kw: dict[str, Any] = {
                    "prompt": prompt,
                    "height": probe_res,
                    "width": probe_res,
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
                # Decode latents with VAE for PSNR comparison
                unpacked = base_pipe._unpack_latents(
                    lat_out, probe_res, probe_res, base_pipe.vae_scale_factor
                )
                unpacked = unpacked.to(base_pipe.vae.dtype)
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
                teacher_trajectories.append((lat_out.detach().cpu(), pil_img))
            finally:
                hk.remove()

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
        embed_metrics = []
        vel_metrics = []
        lat_metrics = []
        img_psnrs = []

        with torch.no_grad():
            for c_idx, (prompt, cond_img) in enumerate(all_probe_cases):
                q_pe, _, _ = q_pipe.encode_prompt(prompt=prompt, image=cond_img, device=dev)
                embed_metrics.append(_compute_psnr_and_cosine(q_pe.cpu(), teacher_embeds[c_idx]))

                gen = torch.Generator(device=dev).manual_seed(42 + c_idx)
                call_kw = {
                    "prompt": prompt,
                    "height": probe_res,
                    "width": probe_res,
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
                ref_lat, ref_pil = teacher_trajectories[c_idx]
                lat_metrics.append(_compute_psnr_and_cosine(q_lat.cpu(), ref_lat))

                unpacked = q_pipe._unpack_latents(q_lat, probe_res, probe_res, q_pipe.vae_scale_factor).to(
                    q_pipe.vae.dtype
                )
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
                img_psnrs.append(_compute_image_psnr(q_pil, ref_pil))

            for kw_in, ref_v in teacher_velocities:
                q_v = q_pipe.transformer(
                    hidden_states=kw_in["hidden_states"].to(dev, dtype=torch.bfloat16),
                    timestep=kw_in["timestep"].to(dev, dtype=torch.bfloat16),
                    encoder_hidden_states=kw_in["encoder_hidden_states"].to(dev, dtype=torch.bfloat16),
                    encoder_hidden_states_mask=(
                        kw_in["encoder_hidden_states_mask"].to(dev)
                        if kw_in["encoder_hidden_states_mask"] is not None
                        else None
                    ),
                    img_shapes=kw_in["img_shapes"],
                    img_mask=kw_in["img_mask"].to(dev),
                    return_dict=False,
                )[0]
                vel_metrics.append(_compute_psnr_and_cosine(q_v.cpu(), ref_v))

        def _avg(items: list[dict[str, float]], key: str) -> float:
            return sum(d[key] for d in items) / max(len(items), 1)

        total_bytes = sum(p.stat().st_size for p in quant_dir.rglob("*.safetensors"))
        result = {
            "model": args.model,
            "base": str(base_dir),
            "quant": str(quant_dir),
            "total_safetensors_bytes": total_bytes,
            "total_safetensors_gib": round(total_bytes / (1024**3), 3),
            "probe_cases": len(all_probe_cases),
            "probe_resolution": probe_res,
            "num_inference_steps": total_steps,
            "text_encoder_prompt_embeds": {
                "mean_cosine": _avg(embed_metrics, "cosine"),
                "mean_rel_rmse": _avg(embed_metrics, "rel_rmse"),
                "mean_sqnr_db": _avg(embed_metrics, "sqnr_db"),
            },
            "teacher_forced_velocity": {
                "samples": len(vel_metrics),
                "mean_cosine": _avg(vel_metrics, "cosine"),
                "mean_rel_rmse": _avg(vel_metrics, "rel_rmse"),
                "mean_sqnr_db": _avg(vel_metrics, "sqnr_db"),
            },
            "end_to_end_trajectory": {
                "latent_mean_cosine": _avg(lat_metrics, "cosine"),
                "latent_mean_rel_rmse": _avg(lat_metrics, "rel_rmse"),
                "latent_mean_sqnr_db": _avg(lat_metrics, "sqnr_db"),
                "image_mean_psnr_db": sum(img_psnrs) / max(len(img_psnrs), 1),
                "image_min_psnr_db": min(img_psnrs) if img_psnrs else 0.0,
            },
        }
        out_json = (
            Path(args.output_json)
            if getattr(args, "output_json", None)
            else (quant_dir / "probe_results.json")
        )
        out_json.write_text(json.dumps(result, indent=2) + "\n")
        log_event(
            "QWEN_IMAGE_PROBE_DONE", **result["teacher_forced_velocity"], **result["end_to_end_trajectory"]
        )
        del q_pipe
        gc.collect()
        torch.cuda.empty_cache()
