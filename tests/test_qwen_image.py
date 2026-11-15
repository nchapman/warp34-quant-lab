"""CPU unit tests for Qwen-Image-2.1 / Qwen-Image-2.1-Turbo W4A16-HGPTQ support."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from warp_quant.hadamard_gptq import (
    _assert_packed_alignment,
    _hadamard_rotate_weight,
    _pack_w4_tensor,
)
from warp_quant.models import resolve
from warp_quant.qwen_image import (
    DIT_BLOCK_TARGET_RE,
    DIT_IGNORE,
    QWEN_VL_VISION_GROUP_TARGET_RE,
    build_dit_quantization_config,
    build_text_encoder_quantization_config,
    load_quantized_submodule,
)


def test_qwen_image_models_and_target_regexes():
    h1, i1 = resolve("qwen-image-2.1")
    h2, i2 = resolve("qwen-image-2.1-turbo")
    assert h1 == "Qwen/Qwen-Image-2.1"
    assert h2 == "Qwen/Qwen-Image-2.1-Turbo"
    assert i1["arch"] == "qwen_image_21"
    assert i2["arch"] == "qwen_image_21"

    dit_rx = re.compile(DIT_BLOCK_TARGET_RE.removeprefix("re:"))
    for valid_dit in (
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
        "transformer_blocks.0.attn.to_out.0",
        "transformer_blocks.31.img_mlp.gate_layer",
        "transformer_blocks.31.img_mlp.proj",
        "transformer_blocks.31.img_mlp.out",
    ):
        assert dit_rx.match(valid_dit), f"Expected DIT_BLOCK_TARGET_RE to match {valid_dit}"

    for ignored in DIT_IGNORE:
        assert not dit_rx.match(ignored), f"Expected DIT_BLOCK_TARGET_RE to reject {ignored}"

    vis_rx = re.compile(QWEN_VL_VISION_GROUP_TARGET_RE.removeprefix("re:"))
    for valid_vis in (
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.26.attn.proj",
        "model.visual.blocks.10.mlp.linear_fc1",
        "model.visual.merger.linear_fc1",
        "model.visual.merger.linear_fc2",
        "model.visual.deepstack_merger_list.0.linear_fc1",
        "model.visual.deepstack_merger_list.2.linear_fc2",
    ):
        assert vis_rx.match(valid_vis), f"Expected QWEN_VL_VISION_GROUP_TARGET_RE to match {valid_vis}"
    assert not vis_rx.match("model.visual.blocks.0.mlp.linear_fc2")


def test_load_quantized_submodule_compressed_and_dequantized_match():
    class TinyDiT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.img_in = nn.Linear(64, 64, bias=False)
            self.transformer_blocks = nn.ModuleList([nn.Module()])
            self.transformer_blocks[0].attn = nn.Module()
            self.transformer_blocks[0].attn.to_q = nn.Linear(64, 64, bias=False)
            self.transformer_blocks[0].attn.to_out = nn.ModuleList(
                [nn.Linear(64, 64, bias=False), nn.Dropout(0.0)]
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.img_in(x)
            h = self.transformer_blocks[0].attn.to_q(h)
            return self.transformer_blocks[0].attn.to_out[0](h)

    torch.manual_seed(0)
    ref_mod = TinyDiT().to(torch.bfloat16).eval()
    w_q_rot = _hadamard_rotate_weight(ref_mod.transformer_blocks[0].attn.to_q.weight.float(), 64)
    w_o_rot = _hadamard_rotate_weight(ref_mod.transformer_blocks[0].attn.to_out[0].weight.float(), 64)
    p_q = _pack_w4_tensor(w_q_rot, strategy="group", group_size=64)
    p_o = _pack_w4_tensor(w_o_rot, strategy="group", group_size=64)

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        tensors = {
            "img_in.weight": ref_mod.img_in.weight.detach(),
            **{f"transformer_blocks.0.attn.to_q.{k}": v for k, v in p_q.items()},
            **{f"transformer_blocks.0.attn.to_out.0.{k}": v for k, v in p_o.items()},
        }
        save_file(tensors, str(d / "diffusion_pytorch_model.safetensors"))
        (d / "config.json").write_text(
            json.dumps(
                {
                    "_class_name": "QwenImage21Transformer2DModel",
                    "quantization_config": build_dit_quantization_config(
                        group_size=64, hadamard_block_size=64
                    ),
                }
            )
        )
        _assert_packed_alignment(d)

        mod_deq = load_quantized_submodule(
            TinyDiT(), d, device="cpu", dtype=torch.bfloat16, keep_compressed=False
        )
        mod_cmp = load_quantized_submodule(
            TinyDiT(), d, device="cpu", dtype=torch.bfloat16, keep_compressed=True
        )

        x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
        with torch.no_grad():
            y_deq = mod_deq(x)
            y_cmp = mod_cmp(x)
        assert torch.allclose(y_deq, y_cmp, rtol=1e-3, atol=1e-3)

    te_qcfg = build_text_encoder_quantization_config()
    assert "w8_head" in te_qcfg["config_groups"]
    assert "vision_fc2_channel" in te_qcfg["config_groups"]
