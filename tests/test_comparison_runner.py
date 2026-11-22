"""Unit tests for multi-format comparison byte-budget accounting, config export, and paired bootstrap."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from warp_quant.comparison_runner import (
    SUITES,
    build_text_only_config,
    count_bf16_trunk_params,
    extract_evalscope_metrics_and_samples,
    inspect_gguf_byte_budget,
    inspect_safetensors_byte_budget,
    list_model_safetensors,
    normalize_text_weight_key,
    paired_evalscope_bootstrap_map,
    paired_probe_bootstrap,
)


def _write_synthetic_gguf(path: Path, tensors: list[tuple[str, bytes]]) -> None:
    """Write a minimal valid GGUF v3 file with given (name, raw_bytes) tensor payloads."""
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<IQQ", 3, len(tensors), 1)  # version=3, n_tensors, n_kv=1
    # KV: general.alignment = 32 (uint32 type=4)
    key = b"general.alignment"
    buf += struct.pack("<Q", len(key)) + key
    buf += struct.pack("<II", 4, 32)
    # Tensor table
    offset = 0
    aligned_payloads: list[bytes] = []
    for name, payload in tensors:
        nb = name.encode("utf-8")
        buf += struct.pack("<Q", len(nb)) + nb
        buf += struct.pack("<I", 1)  # ndim=1
        buf += struct.pack("<Q", len(payload))
        buf += struct.pack("<IQ", 0, offset)
        pad = (32 - (len(payload) % 32)) % 32
        aligned_payloads.append(payload + b"\x00" * pad)
        offset += len(payload) + pad
    # Align data section start to 32 bytes
    pad_hdr = (32 - (len(buf) % 32)) % 32
    buf += b"\x00" * pad_hdr
    for p in aligned_payloads:
        buf += p
    path.write_bytes(bytes(buf))


class ComparisonRunnerTests(unittest.TestCase):
    def test_suites_define_seven_distinct_arms_per_model(self):
        for key in ("27b", "9b"):
            arms = SUITES[key]
            self.assertEqual(len(arms), 7)
            ids = [a.arm_id for a in arms]
            self.assertEqual(len(set(ids)), 7)
            self.assertEqual(ids[0], "bf16")
            self.assertIn("w4_hgptq", ids)
            self.assertIn("exl3_4bpw", ids)
            self.assertIn("gguf_q4_k_xl", ids)
            self.assertIn("w26_rco", ids)
            self.assertIn("exl3_3bpw", ids)
            self.assertIn("gguf_q2_k_xl", ids)

    def test_list_model_safetensors_ignores_cal_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            save_file(
                {"a": torch.zeros(4, dtype=torch.bfloat16)}, str(d / "model-00001-of-00001.safetensors")
            )
            save_file({"input_ids": torch.zeros(4, dtype=torch.int64)}, str(d / "cal_trace.safetensors"))
            found = list_model_safetensors(d)
            self.assertEqual([p.name for p in found], ["model-00001-of-00001.safetensors"])

            # With index file, only indexed shards are returned
            (d / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}})
            )
            found_idx = list_model_safetensors(d)
            self.assertEqual([p.name for p in found_idx], ["model-00001-of-00001.safetensors"])

            # Path traversal and absolute paths in index files are rejected
            for bad in ("../escape.safetensors", "/etc/passwd"):
                (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": bad}}))
                with self.assertRaises(ValueError):
                    list_model_safetensors(d)

    def test_safetensors_byte_budget_separates_text_backbone_from_vision_and_mtp(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            tensors = {
                "model.language_model.layers.0.mlp.gate_proj.weight": torch.zeros(
                    (64, 64), dtype=torch.bfloat16
                ),
                "model.language_model.norm.weight": torch.zeros((64,), dtype=torch.bfloat16),
                "model.language_model.embed_tokens.weight": torch.zeros((128, 64), dtype=torch.bfloat16),
                "lm_head.weight": torch.zeros((128, 64), dtype=torch.bfloat16),
                "model.visual.blocks.0.attn.qkv.weight": torch.zeros((32, 32), dtype=torch.bfloat16),
                "mtp.layers.0.mlp.down_proj.weight": torch.zeros((32, 32), dtype=torch.bfloat16),
            }
            save_file(tensors, str(d / "model.safetensors"))
            trunk_params = count_bf16_trunk_params(d)
            self.assertEqual(trunk_params, 64 * 64 + 64)

            budget = inspect_safetensors_byte_budget(d, trunk_params=trunk_params)
            self.assertEqual(budget["trunk_bytes"], (64 * 64 + 64) * 2)
            self.assertAlmostEqual(budget["trunk_bpw"], 16.0, places=3)
            self.assertEqual(budget["embed_bytes"], 128 * 64 * 2)
            self.assertEqual(budget["head_bytes"], 128 * 64 * 2)
            self.assertEqual(budget["vision_bytes"], 32 * 32 * 2)
            self.assertEqual(budget["mtp_bytes"], 32 * 32 * 2)
            self.assertEqual(
                budget["text_total_bytes"],
                budget["trunk_bytes"] + budget["embed_bytes"] + budget["head_bytes"],
            )

    def test_gguf_byte_budget_separates_trunk_mtp_embed_and_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpath = Path(tmp) / "toy.gguf"
            _write_synthetic_gguf(
                gpath,
                [
                    ("token_embd.weight", b"\x01" * 128),
                    ("blk.0.ffn_gate.weight", b"\x02" * 256),
                    ("blk.1.ffn_gate.weight", b"\x03" * 256),
                    ("blk.2.ffn_gate.weight", b"\x04" * 96),  # layer >= n_layers=2 -> MTP
                    ("output_norm.weight", b"\x05" * 32),
                    ("output.weight", b"\x06" * 160),
                ],
            )
            budget = inspect_gguf_byte_budget(gpath, n_layers=2, trunk_params=1024)
            self.assertEqual(budget["embed_bytes"], 128)
            self.assertEqual(budget["trunk_bytes"], 256 + 256 + 32)
            self.assertEqual(budget["mtp_bytes"], 96)
            self.assertEqual(budget["head_bytes"], 160)
            self.assertEqual(budget["vision_bytes"], 0)
            self.assertAlmostEqual(budget["trunk_bpw"], (544 * 8) / 1024, places=3)

    def test_build_text_only_config_inherits_top_level_tie_word_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_p = Path(tmp) / "config.json"
            cfg_p.write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                        "tie_word_embeddings": False,
                        "text_config": {
                            "model_type": "qwen3_5_text",
                            "hidden_size": 4096,
                            "mtp_num_hidden_layers": 1,
                            "quantization_config": {"quant_method": "compressed-tensors"},
                        },
                    }
                )
            )
            out = build_text_only_config(cfg_p)
            self.assertEqual(out["model_type"], "qwen3_5_text")
            self.assertEqual(out["architectures"], ["Qwen3_5ForCausalLM"])
            self.assertFalse(out["tie_word_embeddings"])
            self.assertEqual(out["dtype"], "bfloat16")
            self.assertEqual(out["mtp_num_hidden_layers"], 0)
            self.assertNotIn("quantization_config", out)

    def test_normalize_text_weight_key(self):
        self.assertEqual(
            normalize_text_weight_key("model.language_model.layers.0.mlp.down_proj.weight"),
            "model.layers.0.mlp.down_proj.weight",
        )
        self.assertEqual(normalize_text_weight_key("lm_head.weight"), "lm_head.weight")
        self.assertIsNone(normalize_text_weight_key("model.visual.blocks.0.attn.qkv.weight"))
        self.assertIsNone(normalize_text_weight_key("mtp.layers.0.self_attn.q_proj.weight"))

    def test_paired_probe_bootstrap_mean_and_weighted_kl(self):
        rows_a = [{"row": i, "domain": "chat", "tokens": 100, "mean_kl": 0.10} for i in range(10)] + [
            {"row": 10 + i, "domain": "belebele_fra_Latn", "tokens": 50, "mean_kl": 0.20} for i in range(10)
        ]
        rows_b = [{"row": i, "domain": "chat", "tokens": 100, "mean_kl": 0.15} for i in range(10)] + [
            {"row": 10 + i, "domain": "belebele_fra_Latn", "tokens": 50, "mean_kl": 0.30} for i in range(10)
        ]
        res = paired_probe_bootstrap(rows_a, rows_b, n_boot=1000, seed=7)
        self.assertEqual(res["rows"], 20)
        self.assertEqual(res["tokens"], 1500)
        self.assertGreater(res["mean_kl_diff_b_minus_a"], 0.0)
        self.assertGreater(res["weighted_kl_diff_b_minus_a"], 0.0)
        self.assertEqual(res["row_wins_a"], 20)
        self.assertEqual(res["row_wins_b"], 0)
        self.assertEqual(res["p_a_beats_b_mean_kl"], 1.0)
        self.assertEqual(res["p_a_beats_b_weighted_kl"], 1.0)

    def test_extract_evalscope_metrics_and_paired_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_a = Path(tmp) / "arm_a" / "20261122_000001"
            run_b = Path(tmp) / "arm_b" / "20261122_000001"
            for rdir, he_pass, math_ok, if_ok in [
                (run_a, [True, True, True, False], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]),
                (run_b, [True, False, False, False], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]),
            ]:
                (rdir / "reports" / "m").mkdir(parents=True)
                (rdir / "reviews" / "m").mkdir(parents=True)
                (rdir / "reports" / "m" / "ifeval.json").write_text(
                    json.dumps(
                        {
                            "dataset_name": "ifeval",
                            "num": 4,
                            "metrics": [
                                {
                                    "identity": {"name": "prompt_level_strict"},
                                    "score": sum(if_ok) / 4,
                                    "num": 4,
                                },
                                {"identity": {"name": "inst_level_strict"}, "score": 0.8, "num": 6},
                                {"identity": {"name": "prompt_level_loose"}, "score": 0.75, "num": 4},
                                {"identity": {"name": "inst_level_loose"}, "score": 0.85, "num": 6},
                            ],
                        }
                    )
                )
                (rdir / "reports" / "m" / "math_500.json").write_text(
                    json.dumps(
                        {
                            "dataset_name": "math_500",
                            "num": 4,
                            "metrics": [
                                {"identity": {"name": "accuracy"}, "score": sum(math_ok) / 4, "num": 4}
                            ],
                        }
                    )
                )
                (rdir / "reports" / "m" / "humaneval_plus.json").write_text(
                    json.dumps(
                        {
                            "dataset_name": "humaneval_plus",
                            "num": 4,
                            "metrics": [
                                {
                                    "identity": {"name": "accuracy", "aggregation": "mean"},
                                    "score": sum(he_pass) / 4,
                                    "num": 4,
                                },
                                {
                                    "identity": {"name": "accuracy", "aggregation": "pass_at_k"},
                                    "score": sum(he_pass) / 4,
                                    "num": 4,
                                },
                            ],
                        }
                    )
                )
                with (rdir / "reviews" / "m" / "ifeval_default.jsonl").open("w") as f:
                    for i, v in enumerate(if_ok):
                        f.write(
                            json.dumps(
                                {
                                    "index": i,
                                    "sample_score": {
                                        "score": {"value": {"prompt_level_strict": v}},
                                        "sample_metadata": {"key": 1000 + i},
                                    },
                                }
                            )
                            + "\n"
                        )
                with (rdir / "reviews" / "m" / "math_500_Level 1.jsonl").open("w") as f:
                    for i, v in enumerate(math_ok):
                        f.write(
                            json.dumps(
                                {
                                    "index": i,
                                    "messages": [
                                        {
                                            "role": "assistant",
                                            "perf_metrics": {"output_tokens": 2048 if i == 2 else 500},
                                        }
                                    ],
                                    "sample_score": {
                                        "score": {"value": {"accuracy": v}},
                                        "sample_metadata": {"question_id": f"q/{i}.json"},
                                    },
                                }
                            )
                            + "\n"
                        )
                with (rdir / "reviews" / "m" / "humaneval_plus_default.jsonl").open("w") as f:
                    for i, v in enumerate(he_pass):
                        f.write(
                            json.dumps(
                                {
                                    "index": i,
                                    "sample_score": {
                                        "score": {"value": {"acc": v}},
                                        "sample_metadata": {"task_id": f"HumanEval/{i}"},
                                    },
                                }
                            )
                            + "\n"
                        )

            sum_a, samp_a = extract_evalscope_metrics_and_samples(Path(tmp) / "arm_a", max_tokens=2048)
            sum_b, samp_b = extract_evalscope_metrics_and_samples(Path(tmp) / "arm_b", max_tokens=2048)
            self.assertEqual(sum_a["math_500"]["truncated_at_max_tokens"], 1)
            self.assertEqual(sum_a["humaneval_plus"]["passed"], 3)
            self.assertEqual(sum_b["humaneval_plus"]["passed"], 1)

            paired = paired_evalscope_bootstrap_map(samp_a, samp_b, n_boot=1000, seed=1)
            self.assertEqual(paired["humaneval_plus"]["a_only_wins"], 2)
            self.assertEqual(paired["humaneval_plus"]["b_only_wins"], 0)
            self.assertAlmostEqual(paired["math_500"]["diff_b_minus_a"], -0.25)


if __name__ == "__main__":
    unittest.main()
