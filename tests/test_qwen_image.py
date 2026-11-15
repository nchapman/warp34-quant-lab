"""CPU unit tests for Qwen-Image-2.1 / Qwen-Image-2.1-Turbo W4A16-HGPTQ support."""

from __future__ import annotations

import io
import json
import re
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import save_file

from warp_quant.data import (
    IMAGE_GEN_V1_ASPECT_RATIOS,
    IMAGE_GEN_V1_IN_DIST_DOMAINS,
    _select_image_gen_split,
    sha256,
)
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
    _summarize_stratified,
    build_dit_quantization_config,
    build_text_encoder_quantization_config,
    load_image_gen_rows,
    load_quantized_submodule,
    select_stratified_image_gen_rows,
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


def test_image_gen_split_selection_and_manifest_guards():
    # Build synthetic T2I and bilingual TI2I candidates across 2 domains
    doms = ("oneig_en_text", "gedit_text_change")
    cands = []
    for i in range(12):
        cands.append(
            ("oneig_en_text", "oneig_en:Text_Rendering", i, None, f"English prompt number {i}", "en")
        )
        buf = io.BytesIO()
        Image.new("RGB", (256 + i * 16, 192 + i * 8), (i * 10, 50, 100)).save(buf, format="PNG")
        lang = "en" if i % 2 == 0 else "zh"
        cands.append(
            (
                "gedit_text_change",
                f"gedit:text_change:{lang}",
                i,
                buf.getvalue(),
                f"Edit instruction {i}",
                lang,
            )
        )

    seen: set[str] = set()
    probe = _select_image_gen_split(
        cands, "probe-seed", {"oneig_en_text": 2, "gedit_text_change": 2}, doms, seen
    )
    calib = _select_image_gen_split(
        cands, "calib-seed", {"oneig_en_text": 4, "gedit_text_change": 4}, doms, seen
    )

    assert len(probe) == 4
    assert len(calib) == 8
    # Disjoint prompts and images
    probe_prompts = {r["prompt_sha256"] for r in probe}
    calib_prompts = {r["prompt_sha256"] for r in calib}
    assert probe_prompts.isdisjoint(calib_prompts)

    # Bilingual TI2I domain alternates en and zh
    gedit_langs = [r["lang"] for r in calib if r["domain"] == "gedit_text_change"]
    assert gedit_langs.count("en") == 2 and gedit_langs.count("zh") == 2
    assert gedit_langs[0] != gedit_langs[1]

    # All dimensions are multiples of 32
    for r in probe + calib:
        assert r["width"] % 32 == 0 and r["height"] % 32 == 0

    # Round-robin stratified selection
    strat = select_stratified_image_gen_rows(calib, per_domain=2)
    assert [r["domain"] for r in strat] == [
        "oneig_en_text",
        "gedit_text_change",
        "oneig_en_text",
        "gedit_text_change",
    ]

    # Manifest & SHA-256 guards in load_image_gen_rows
    with tempfile.TemporaryDirectory() as tmp:
        fdir = Path(tmp) / "custom-calib"
        (fdir / "images").mkdir(parents=True)
        rows_out = []
        for r in calib:
            item = {k: v for k, v in r.items() if k != "_blob"}
            if r["_blob"] is not None and r["image"] is not None:
                (fdir / r["image"]).write_bytes(r["_blob"])
            rows_out.append(item)
        rows_p = fdir / "rows.jsonl"
        rows_p.write_text("".join(json.dumps(x) + "\n" for x in rows_out), encoding="utf-8")
        manifest = {
            "suite": "image-gen-v1",
            "split": "calibration",
            "rows": len(rows_out),
            "rows_sha256": sha256(rows_p),
            "calibration": True,
            "guard": False,
        }
        (fdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        m_loaded, r_loaded = load_image_gen_rows(fdir, require_calibration=True, auto_materialize=False)
        assert m_loaded["rows"] == 8 and len(r_loaded) == 8

        with pytest.raises(ValueError, match="Refusing to use fixture"):
            load_image_gen_rows(fdir, require_calibration=False, auto_materialize=False)
        with pytest.raises(ValueError, match="guard="):
            load_image_gen_rows(fdir, require_calibration=True, require_guard=True, auto_materialize=False)

    # Stratified metric aggregation
    summary = _summarize_stratified(
        [
            {
                "cosine": 0.999,
                "rel_rmse": 0.04,
                "sqnr_db": 28.0,
                "image_psnr_db": 26.0,
                "domain": "d1",
                "mode": "t2i",
                "lang": "en",
            },
            {
                "cosine": 0.997,
                "rel_rmse": 0.06,
                "sqnr_db": 24.0,
                "image_psnr_db": 22.0,
                "domain": "d2",
                "mode": "ti2i",
                "lang": "zh",
            },
        ]
    )
    assert summary["count"] == 2
    assert summary["by_mode"]["t2i"]["count"] == 1
    assert summary["by_mode"]["ti2i"]["count"] == 1
    assert summary["by_lang"]["en"]["count"] == 1
    assert summary["by_lang"]["zh"]["count"] == 1
    assert len(IMAGE_GEN_V1_IN_DIST_DOMAINS) == 16
    assert len(IMAGE_GEN_V1_ASPECT_RATIOS) == 5


def test_capture_lang_model_entry_handles_3d_mrope_positions() -> None:
    from warp_quant.qwen_image import _capture_lang_model_entry

    class _DummyRotary(torch.nn.Module):
        def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Qwen3VLTextRotaryEmbedding.forward calls position_ids.expand(3, -1, -1)
            pos3 = position_ids.expand(3, -1, -1)
            assert pos3.shape[0] == 3
            s = x.shape[1]
            return torch.ones(1, s, 8), torch.zeros(1, s, 8)

    class _DummyLangModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(32, 16)
            self.rotary_emb = _DummyRotary()

        def forward(self, **kwargs: object) -> None:
            raise AssertionError("pre-hook should stop before language_model.forward")

    class _DummyWrapper(torch.nn.Module):
        def __init__(self, pos_dim0: int | None) -> None:
            super().__init__()
            self.pos_dim0 = pos_dim0
            self.model = torch.nn.Module()
            self.model.language_model = _DummyLangModel()

        def forward(self, input_ids: torch.Tensor, **kwargs: object) -> None:
            embeds = self.model.language_model.embed_tokens(input_ids)
            s = input_ids.shape[1]
            pos = torch.zeros((self.pos_dim0, 1, s), dtype=torch.long) if self.pos_dim0 is not None else None
            self.model.language_model(
                input_ids=None,
                inputs_embeds=embeds,
                position_ids=pos,
                visual_pos_masks=torch.zeros((1, s), dtype=torch.bool),
                deepstack_visual_embeds=[torch.zeros((0, 16))],
            )

    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    for dim0 in (3, 4, None):
        cap = _capture_lang_model_entry(_DummyWrapper(dim0), {"input_ids": ids})
        assert cap["hidden_states"].shape == (1, 4, 16)
        assert cap["text_pos"].shape == (1, 4)
        assert cap["pos_emb"][0].shape == (1, 4, 8)
