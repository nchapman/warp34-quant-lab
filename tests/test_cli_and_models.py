"""Unit tests for warp_quant CLI, model registry, and release stamping."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from warp_quant.cli import build_parser, main
from warp_quant.models import MODELS, quant_recipe, resolve, serving_params
from warp_quant.stamp import RELEASE_TEMPLATE_SHA, stamp_release_template


def test_models_registry_defaults_to_w4a16_hgptq():
    for key in ("0.8b", "2b", "4b", "9b", "27b", "35b", "flash-next"):
        hub_id, info = resolve(key)
        assert hub_id == MODELS[key]["hub"]
        assert info["quant_repo"].endswith("-W4A16-HGPTQ")
        rec = quant_recipe(info)
        assert rec["algo"] == "hgptq"
        assert rec["calib"] == "data/calib-cascade-v1.jsonl"
        assert rec["num_samples"] == 999
        assert rec["max_seq_length"] == 4096
        assert rec["group_size"] == 64
        assert rec["hadamard_block_size"] == 64
        assert rec["actorder"] == "static"
        assert rec["quantize_mtp"] is True
        assert rec["quantize_vision"] is True

    _, mimo_info = resolve("9b-mimo")
    assert serving_params(mimo_info)["temperature"] == 0.6
    _, base_info = resolve("35b")
    assert serving_params(base_info) == {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def test_cli_parser_defaults(capsys):
    parser = build_parser()
    q_args = parser.parse_args(["quantize", "--model", "flash-next", "--output-dir", "/tmp/out"])
    assert q_args.model == "flash-next"
    assert q_args.group_size == 64
    assert q_args.hadamard_block_size == 64
    assert q_args.ple_group_size == 32
    assert q_args.ple_cd_passes == 1
    assert q_args.actorder == "static"
    assert q_args.quantize_mtp is True
    assert q_args.quantize_vision is True
    assert q_args.component_quant_mode == "gptq"

    p_args = parser.parse_args(["probe", "--model", "35b", "--quant", "/tmp/out", "--eval-mtp"])
    assert p_args.model == "35b"
    assert p_args.eval_mtp is True
    assert p_args.eval_sets == ["data/probe-v4.jsonl", "data/probe-guard.jsonl"]

    main(["list-models"])
    out = capsys.readouterr().out
    assert "Qwen/Qwen3.6-35B-A3B" in out
    assert "Qwen/Qwen3.8-Flash-Next" in out


def test_stamp_release_template_strips_embedded_chat_template():
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "tokenizer_config.json").write_text(
            json.dumps({"eos_token": "<|im_end|>", "chat_template": "OLD_EMBEDDED"}),
            encoding="utf-8",
        )
        sha = stamp_release_template(d)
        assert sha == RELEASE_TEMPLATE_SHA
        assert (d / "chat_template.jinja").is_file()
        tc = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
        assert "chat_template" not in tc


def test_frozen_data_sha256_and_disjointness():
    import hashlib

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    sums_lines = (data_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert len(sums_lines) >= 8
    for line in sums_lines:
        expected_sha, rel_name = line.split(maxsplit=1)
        rel_name = rel_name.strip()
        actual_sha = hashlib.sha256((data_dir / rel_name).read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"Checksum mismatch on {rel_name}"

    first_user_by_file: dict[str, set[str]] = {}
    for fname, expected_rows in (
        ("calib-cascade-v1.jsonl", 999),
        ("probe-v4.jsonl", 378),
        ("probe-guard.jsonl", 76),
        ("traj-v1.jsonl", 301),
    ):
        rows = [
            json.loads(line)
            for line in (data_dir / fname).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == expected_rows
        u_set = set()
        for r in rows:
            msgs = r.get("messages") or r.get("conversations")
            user_turns = [m["content"] for m in msgs if m["role"] == "user"]
            assert user_turns
            u_set.add(hashlib.sha256(user_turns[0].strip().encode()).hexdigest())
        first_user_by_file[fname] = u_set

    files = list(first_user_by_file)
    for i, a in enumerate(files):
        for b in files[i + 1 :]:
            assert not (first_user_by_file[a] & first_user_by_file[b]), f"Prompt overlap between {a} and {b}"


def test_normalize_messages_is_idempotent_with_tool_calls():
    from warp_quant.data import normalize_messages

    raw = [
        {"role": "user", "content": "Find the weather in Tokyo."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                }
            ],
        },
    ]
    once = normalize_messages(raw)
    twice = normalize_messages(once)
    assert once == twice
    assert twice[1]["tool_calls"] == [{"name": "get_weather", "arguments": {"city": "Tokyo"}}]


def test_aggregate_weighted_kl_includes_chat_long():
    from warp_quant.data import aggregate_weighted_kl

    without_chat_long = aggregate_weighted_kl(
        {"chat": {"kl": 1.0, "kl_n": 100}, "math": {"kl": 2.0, "kl_n": 100}}
    )
    with_chat_long = aggregate_weighted_kl(
        {
            "chat": {"kl": 1.0, "kl_n": 100},
            "chat_long": {"kl": 5.0, "kl_n": 100},
            "math": {"kl": 2.0, "kl_n": 100},
        }
    )
    assert with_chat_long > without_chat_long


def test_readme_summary_table_matches_published_receipts():
    root = Path(__file__).resolve().parents[1]
    readme_text = (root / "README.md").read_text(encoding="utf-8")

    expected = [
        ("Qwen/Qwen3.5-0.8B", root / "results/dense-w4a16-hgptq/0.8b-g64-h64/probe_results.json"),
        ("Qwen/Qwen3.5-2B", root / "results/dense-w4a16-hgptq/g64-h64/probe.json"),
        ("Qwen/Qwen3.5-4B", root / "results/dense-w4a16-hgptq/4b-g64-h64/probe_results.json"),
        ("Qwen/Qwen3.5-9B", root / "results/dense-w4a16-hgptq/9b-g64-h64/probe.json"),
        (
            "ornith-ai/Ornith-1.5-9B",
            root / "results/dense-w4a16-hgptq/9b-ornith-g64-h64/probe_results.json",
        ),
        ("Qwen/Qwen3.8-27B", root / "results/dense-w4a16-hgptq/27b-g64-h64/probe.json"),
        (
            "ukisai/Swift-1.5-Qwen3.8-27b",
            root / "results/dense-w4a16-hgptq/27b-swift-g64-h64/probe_results.json",
        ),
        ("Qwen/Qwen3.6-35B-A3B", root / "results/qwen36-35b-a3b-w4a16-hgptq/probe_results.json"),
        (
            "ornith-ai/Ornith-1.5-35B-A3B",
            root / "results/ornith-15-35b-a3b-w4a16-hgptq/probe_results.json",
        ),
        ("Qwen/Qwen3.8-Flash-Next", root / "results/qwen38-flash-next-w4a16-hgptq/probe_results.json"),
    ]
    for model_name, probe_path in expected:
        row_line = next(
            line for line in readme_text.splitlines() if f"`{model_name}`" in line and "|" in line
        )
        d = json.loads(probe_path.read_text(encoding="utf-8"))
        v4 = d["probe-v4"]
        gd = d["probe-guard"]
        mtp_iso = d["mtp_fidelity"]["isolated"]
        vis_iso = d["vision_fidelity"]["isolated"]
        assert f"{v4['ppl_base']:.4f} → {v4['ppl_quant']:.4f}" in row_line, (
            f"{model_name} PPL mismatch: {row_line}"
        )
        assert f"{v4['weighted_kl']:.6f}" in row_line, f"{model_name} v4 wkl mismatch: {row_line}"
        assert f"{gd['weighted_kl']:.6f}" in row_line, f"{model_name} guard wkl mismatch: {row_line}"
        assert f"{mtp_iso['mean_kl']:.6f}" in row_line, f"{model_name} mtp kl mismatch: {row_line}"
        assert f"{mtp_iso['top1_agreement'] * 100:.2f}%" in row_line, (
            f"{model_name} mtp top1 mismatch: {row_line}"
        )
        assert f"{vis_iso['mean_kl']:.6f}" in row_line, f"{model_name} vis kl mismatch: {row_line}"
        assert f"{vis_iso['top1_agreement'] * 100:.2f}%" in row_line, (
            f"{model_name} vis top1 mismatch: {row_line}"
        )


def test_model_cards_match_published_receipts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pairs = [
        (
            "results/dense-w4a16-hgptq/0.8b-g64-h64/TECHNICAL.md",
            "results/dense-w4a16-hgptq/0.8b-g64-h64/probe_results.json",
        ),
        (
            "results/dense-w4a16-hgptq/g64-h64/TECHNICAL.md",
            "results/dense-w4a16-hgptq/g64-h64/probe.json",
        ),
        (
            "results/dense-w4a16-hgptq/4b-g64-h64/TECHNICAL.md",
            "results/dense-w4a16-hgptq/4b-g64-h64/probe_results.json",
        ),
        (
            "results/dense-w4a16-hgptq/9b-g64-h64/TECHNICAL.md",
            "results/dense-w4a16-hgptq/9b-g64-h64/probe.json",
        ),
        (
            "results/dense-w4a16-hgptq/9b-ornith-g64-h64/TECHNICAL.md",
            "results/dense-w4a16-hgptq/9b-ornith-g64-h64/probe_results.json",
        ),
        (
            "results/dense-w4a16-hgptq/27b-g64-h64/TECHNICAL.md",
            "results/dense-w4a16-hgptq/27b-g64-h64/probe.json",
        ),
        (
            "results/dense-w4a16-hgptq/27b-swift-g64-h64/TECHNICAL.md",
            "results/dense-w4a16-hgptq/27b-swift-g64-h64/probe_results.json",
        ),
        (
            "results/qwen36-35b-a3b-w4a16-hgptq/TECHNICAL.md",
            "results/qwen36-35b-a3b-w4a16-hgptq/probe_results.json",
        ),
        (
            "results/ornith-15-35b-a3b-w4a16-hgptq/TECHNICAL.md",
            "results/ornith-15-35b-a3b-w4a16-hgptq/probe_results.json",
        ),
        (
            "results/rco-9b-w2w4/TECHNICAL-W2.6A16-HGPTQ-RCO.md",
            "results/rco-9b-w2w4/v2_receipts/probe_9b_w26_v2.json",
        ),
        (
            "results/ornith-15-9b-w26a16-hgptq-rco/TECHNICAL.md",
            "results/ornith-15-9b-w26a16-hgptq-rco/v2_receipts/probe_9b_ornith_w26_v2.json",
        ),
        (
            "results/rco-27b-w2w4/TECHNICAL-W2.6A16-HGPTQ-RCO.md",
            "results/rco-27b-w2w4/v2_receipts/probe_37pct_rco.json",
        ),
        (
            "results/rco-35b-w2w4/TECHNICAL-W2.6A16-HGPTQ-RCO.md",
            "results/rco-35b-w2w4/v2_receipts/probe_35b_w26_v2.json",
        ),
        (
            "results/ornith-15-35b-a3b-w26a16-hgptq-rco/TECHNICAL.md",
            "results/ornith-15-35b-a3b-w26a16-hgptq-rco/v2_receipts/probe_35b_ornith_w26_v2.json",
        ),
        (
            "results/qwen38-flash-next-w4a16-hgptq/TECHNICAL.md",
            "results/qwen38-flash-next-w4a16-hgptq/probe_results.json",
        ),
        (
            "results/qwen38-flash-next-w26a16-hgptq-rco/TECHNICAL.md",
            "results/qwen38-flash-next-w26a16-hgptq-rco/v2_receipts/probe_flash_next_w26_v2.json",
        ),
    ]
    for md_rel, json_rel in pairs:
        md = (repo_root / md_rel).read_text(encoding="utf-8")
        d = json.loads((repo_root / json_rel).read_text(encoding="utf-8"))
        t = d["probe-v4"]
        g = d["probe-guard"]
        m = d["mtp_fidelity"]
        v = d["vision_fidelity"]
        vg = d["vision_guard_fidelity"]
        fidelity_bullet = next(line for line in md.splitlines() if line.startswith("- **Fidelity"))
        for expected in (
            f"{t['weighted_kl']:.4f}",
            f"{t['mean_kl']:.4f}",
            f"{g['mean_kl']:.4f}",
            f"{m['isolated']['top1_agreement'] * 100:.2f}%",
            f"{m['e2e']['top1_agreement'] * 100:.2f}%",
            f"{v['isolated']['top1_agreement'] * 100:.2f}%",
            f"{v['e2e']['top1_agreement'] * 100:.2f}%",
        ):
            assert expected in fidelity_bullet, f"{md_rel} bullet missing {expected}: {fidelity_bullet}"
            assert expected in md, f"{md_rel} missing {expected}"
        for expected_table in (
            f"{t['ppl_quant']:.4f}",
            f"{g['ppl_quant']:.4f}",
            f"{vg['isolated']['top1_agreement'] * 100:.2f}%",
            f"{vg['e2e']['top1_agreement'] * 100:.2f}%",
        ):
            assert expected_table in md, f"{md_rel} table missing {expected_table}"


def test_published_user_model_cards_have_frontmatter_and_technical_reports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    card_pairs = [
        (
            "results/dense-w4a16-hgptq/0.8b-g64-h64/README.md",
            "results/dense-w4a16-hgptq/0.8b-g64-h64/TECHNICAL.md",
        ),
        ("results/dense-w4a16-hgptq/g64-h64/README.md", "results/dense-w4a16-hgptq/g64-h64/TECHNICAL.md"),
        (
            "results/dense-w4a16-hgptq/4b-g64-h64/README.md",
            "results/dense-w4a16-hgptq/4b-g64-h64/TECHNICAL.md",
        ),
        (
            "results/dense-w4a16-hgptq/9b-g64-h64/README.md",
            "results/dense-w4a16-hgptq/9b-g64-h64/TECHNICAL.md",
        ),
        (
            "results/dense-w4a16-hgptq/27b-g64-h64/README.md",
            "results/dense-w4a16-hgptq/27b-g64-h64/TECHNICAL.md",
        ),
        ("results/qwen36-35b-a3b-w4a16-hgptq/README.md", "results/qwen36-35b-a3b-w4a16-hgptq/TECHNICAL.md"),
        (
            "results/qwen38-flash-next-w4a16-hgptq/README.md",
            "results/qwen38-flash-next-w4a16-hgptq/TECHNICAL.md",
        ),
        (
            "results/rco-9b-w2w4/README-W2.6A16-HGPTQ-RCO.md",
            "results/rco-9b-w2w4/TECHNICAL-W2.6A16-HGPTQ-RCO.md",
        ),
        (
            "results/ornith-15-9b-w26a16-hgptq-rco/README.md",
            "results/ornith-15-9b-w26a16-hgptq-rco/TECHNICAL.md",
        ),
        (
            "results/rco-27b-w2w4/README-W2.6A16-HGPTQ-RCO.md",
            "results/rco-27b-w2w4/TECHNICAL-W2.6A16-HGPTQ-RCO.md",
        ),
        (
            "results/rco-35b-w2w4/README-W2.6A16-HGPTQ-RCO.md",
            "results/rco-35b-w2w4/TECHNICAL-W2.6A16-HGPTQ-RCO.md",
        ),
        (
            "results/ornith-15-35b-a3b-w26a16-hgptq-rco/README.md",
            "results/ornith-15-35b-a3b-w26a16-hgptq-rco/TECHNICAL.md",
        ),
        (
            "results/qwen38-flash-next-w26a16-hgptq-rco/README.md",
            "results/qwen38-flash-next-w26a16-hgptq-rco/TECHNICAL.md",
        ),
        (
            "results/dense-w4a16-hgptq/9b-ornith-g64-h64/README.md",
            "results/dense-w4a16-hgptq/9b-ornith-g64-h64/TECHNICAL.md",
        ),
        (
            "results/ornith-15-35b-a3b-w4a16-hgptq/README.md",
            "results/ornith-15-35b-a3b-w4a16-hgptq/TECHNICAL.md",
        ),
        ("results/apex-35b-v1-w4a16-hgptq/README.md", "results/apex-35b-v1-w4a16-hgptq/TECHNICAL.md"),
        (
            "results/dense-w4a16-hgptq/27b-swift-g64-h64/README.md",
            "results/dense-w4a16-hgptq/27b-swift-g64-h64/TECHNICAL.md",
        ),
        ("results/arete-27b-w4a16-hgptq/README.md", "results/arete-27b-w4a16-hgptq/TECHNICAL.md"),
        ("results/daisyrose-27b-w4a16-hgptq/README.md", "results/daisyrose-27b-w4a16-hgptq/TECHNICAL.md"),
        ("results/sabi-noema-w4a16-hgptq/README.md", "results/sabi-noema-w4a16-hgptq/TECHNICAL.md"),
        (
            "results/dense-w4a16-hgptq/dflash2-27b-README.md",
            "results/dense-w4a16-hgptq/dflash2-27b-TECHNICAL.md",
        ),
        (
            "results/dense-w4a16-hgptq/dflash2-flash-next-README.md",
            "results/dense-w4a16-hgptq/dflash2-flash-next-TECHNICAL.md",
        ),
        ("results/decision-w4a16-hgptq/d3/README.md", "results/decision-w4a16-hgptq/d3/TECHNICAL.md"),
        (
            "results/decision-w4a16-hgptq/d3-ultra/README.md",
            "results/decision-w4a16-hgptq/d3-ultra/TECHNICAL.md",
        ),
        (
            "results/decision-w4a16-hgptq/d3-turbo/README.md",
            "results/decision-w4a16-hgptq/d3-turbo/TECHNICAL.md",
        ),
        (
            "results/decision-w4a16-hgptq/d3-flash/README.md",
            "results/decision-w4a16-hgptq/d3-flash/TECHNICAL.md",
        ),
        (
            "results/decision-w4a16-hgptq/d3-edge/README.md",
            "results/decision-w4a16-hgptq/d3-edge/TECHNICAL.md",
        ),
        (
            "results/decision-w4a16-hgptq/d3-mini/README.md",
            "results/decision-w4a16-hgptq/d3-mini/TECHNICAL.md",
        ),
        (
            "results/decision-w4a16-hgptq/d3-nano/README.md",
            "results/decision-w4a16-hgptq/d3-nano/TECHNICAL.md",
        ),
        (
            "results/decision-w4a16-hgptq/d3-lite/README.md",
            "results/decision-w4a16-hgptq/d3-lite/TECHNICAL.md",
        ),
        (
            "results/decision-w4a16-hgptq/glide-27b/README.md",
            "results/decision-w4a16-hgptq/glide-27b/TECHNICAL.md",
        ),
        (
            "results/decision-w4a16-hgptq/glide-9b/README.md",
            "results/decision-w4a16-hgptq/glide-9b/TECHNICAL.md",
        ),
        (
            "results/decision-w4a16-hgptq/glide-2b/README.md",
            "results/decision-w4a16-hgptq/glide-2b/TECHNICAL.md",
        ),
        (
            "results/decision-w4a16-hgptq/glide-0.8b/README.md",
            "results/decision-w4a16-hgptq/glide-0.8b/TECHNICAL.md",
        ),
    ]
    for readme_rel, tech_rel in card_pairs:
        readme_text = (repo_root / readme_rel).read_text(encoding="utf-8")
        tech_path = repo_root / tech_rel
        assert tech_path.exists(), f"Missing technical report: {tech_rel}"
        assert readme_text.startswith("---\n"), f"{readme_rel} missing YAML frontmatter"
        assert "base_model_relation: quantized" in readme_text, f"{readme_rel} missing base_model_relation"
        assert "quantized_by: nchapman" in readme_text, f"{readme_rel} missing quantized_by"
        assert "## Quickstart" in readme_text, f"{readme_rel} missing Quickstart"
        assert "## About this quant" in readme_text, f"{readme_rel} missing About this quant"
        assert "[Technical Report](TECHNICAL.md)" in readme_text, (
            f"{readme_rel} missing Technical Report link"
        )
        tech_text = tech_path.read_text(encoding="utf-8")
        assert not tech_text.startswith("---\n"), f"{tech_rel} should not carry duplicate YAML frontmatter"
