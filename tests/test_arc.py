"""Unit tests for ARC-1.0 W4A16-HGPTQ packaging and Qwen slot unmasking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from warp_quant.arc import (
    CONTROL_ID_DEFAULT,
    CONTROL_ID_HIGH,
    UNMASKED_QWEN_SLOTS,
    is_arc_checkpoint,
    unmask_arc_goldens_text,
    unmask_arc_qwen_slots,
    update_arc_config_slots,
    write_unmasked_arc_runtime,
)
from warp_quant.models import quant_recipe, resolve


def test_arc_model_registry_entry() -> None:
    hub_id, info = resolve("arc-1.0")
    assert hub_id == "rayline-ai/ARC-1.0"
    assert info["quant_repo"] == "nchapman/ARC-1.0-W4A16-HGPTQ"
    assert info["decision_family"] == "arc"
    assert info["num_hidden_layers"] == 12
    assert info["question_head_layers"] == 12
    rec = quant_recipe(info)
    assert rec["group_size"] == 64
    assert rec["hadamard_block_size"] == 64
    assert rec["quantize_mtp"] is False
    assert rec["quantize_vision"] is True


def test_is_arc_checkpoint_detection(tmp_path: Path) -> None:
    assert not is_arc_checkpoint(tmp_path)
    rt_dir = tmp_path / "runtime"
    rt_dir.mkdir()
    (rt_dir / "arc-runtime.json").write_text("{}")
    assert not is_arc_checkpoint(tmp_path)
    (rt_dir / "heads.safetensors").write_bytes(b"stub")
    assert is_arc_checkpoint(tmp_path)


def test_unmask_arc_qwen_slots_and_manifest_refresh(tmp_path: Path) -> None:
    src_rt = tmp_path / "src_runtime"
    src_rt.mkdir()
    models = [{"name": f"withheld-model-{i:02d}"} for i in range(56)]
    candidates = [{"model": i, "withheld": "excluded"} for i in range(56)]
    candidates[54] = {"model": 54}
    candidates[55] = {"model": 55}
    served = [False] * 5732
    for idx in UNMASKED_QWEN_SLOTS[54]["served_arms"] + UNMASKED_QWEN_SLOTS[55]["served_arms"]:
        served[idx] = True

    rt_payload = {
        "models": models,
        "stage_one": {"candidates": candidates},
        "arms": {"served": served},
        "actions": [
            {
                "arms": UNMASKED_QWEN_SLOTS[54]["served_arms"][:4],
                "control_id": CONTROL_ID_DEFAULT,
                "effort": "default",
                "model": 54,
            },
            {
                "arms": UNMASKED_QWEN_SLOTS[55]["served_arms"][:4],
                "control_id": CONTROL_ID_DEFAULT,
                "effort": "default",
                "model": 55,
            },
        ],
    }
    (src_rt / "arc-runtime.json").write_text(json.dumps(rt_payload))
    (src_rt / "prices.json").write_text('{"rates":{}}\n')
    (src_rt / "manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "arc-runtime.json": {"bytes": 0, "sha256": ""},
                    "prices.json": {"bytes": 0, "sha256": ""},
                }
            }
        )
    )

    dst_rt = tmp_path / "dst_runtime"
    manifest = write_unmasked_arc_runtime(src_rt, dst_rt)

    unmasked = json.loads((dst_rt / "arc-runtime.json").read_text())
    assert unmasked["models"][53] == {
        "name": "qwen3.8-flash-next",
        "wire": "qwen/qwen3.8-flash-next",
    }
    assert unmasked["models"][54] == {"name": "qwen3.5-9b", "wire": "qwen/qwen3.5-9b"}
    assert unmasked["models"][55] == {"name": "qwen3.8-27b", "wire": "qwen/qwen3.8-27b"}
    assert unmasked["stage_one"]["candidates"][53] == {"model": 53}
    for row in UNMASKED_QWEN_SLOTS[53]["served_arms"]:
        assert unmasked["arms"]["served"][row] is True

    m53_actions = [a for a in unmasked["actions"] if a["model"] == 53]
    assert len(m53_actions) == 2
    assert {a["control_id"] for a in m53_actions} == {CONTROL_ID_DEFAULT, CONTROL_ID_HIGH}

    # Idempotence check
    unmasked_twice = unmask_arc_qwen_slots(unmasked)
    assert len([a for a in unmasked_twice["actions"] if a["model"] == 53]) == 2

    # Manifest sha256 and byte length verification
    rt_bytes = (dst_rt / "arc-runtime.json").read_bytes()
    assert manifest["files"]["arc-runtime.json"] == {
        "bytes": len(rt_bytes),
        "sha256": hashlib.sha256(rt_bytes).hexdigest(),
    }


def test_update_arc_config_and_goldens_translation() -> None:
    cfg = {
        "arc": {
            "stage_one": {
                "candidates": [
                    {"effort": None, "model": "withheld-model-42"},
                    {"effort": None, "model": "withheld-model-43"},
                    {"effort": None, "model": "withheld-model-44"},
                ]
            },
            "stage_two": {
                "arms": {
                    "models": [
                        "claude-opus-5.5",
                        "withheld-model-42",
                        "withheld-model-43",
                        "withheld-model-44",
                    ]
                }
            },
        }
    }
    updated = update_arc_config_slots(cfg)
    assert [c["model"] for c in updated["arc"]["stage_one"]["candidates"]] == [
        "qwen3.8-flash-next",
        "qwen3.5-9b",
        "qwen3.8-27b",
    ]
    assert updated["arc"]["stage_two"]["arms"]["models"] == [
        "claude-opus-5.5",
        "qwen3.8-flash-next",
        "qwen3.5-9b",
        "qwen3.8-27b",
    ]

    old_aid = hashlib.sha256(
        json.dumps(
            {"control_id": CONTROL_ID_DEFAULT, "model": "withheld-model-44"},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    new_aid = hashlib.sha256(
        json.dumps(
            {"control_id": CONTROL_ID_DEFAULT, "model": "qwen3.8-27b"},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    sample = json.dumps(
        {
            "available_action_ids": [old_aid],
            "rates": {"withheld/withheld-model-44": {"input": 0.1}},
        }
    )
    translated = unmask_arc_goldens_text(sample)
    assert new_aid in translated
    assert old_aid not in translated
    assert "qwen/qwen3.8-27b" in translated
