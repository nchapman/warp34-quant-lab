"""Unit tests for EvalScope downstream benchmark configuration and paired comparison helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from warp_quant.cli import build_parser
from warp_quant.evalscope_runner import (
    build_dataset_args,
    build_evalscope_task_dict,
    extract_evalscope_reports,
    paired_binary_bootstrap,
)
from warp_quant.hadamard_gptq import _hadamard_rotate_weight, fuse_quant_model_for_generation


class EvalScopeRunnerTests(unittest.TestCase):
    def test_build_dataset_args_attaches_remove_until_filter_and_zero_shot(self):
        ds_args = build_dataset_args(
            ["ifeval", "math_500", "gpqa_diamond"],
            remove_thinking_tag="</think>",
            math_levels=["Level 5"],
        )
        self.assertEqual(ds_args["ifeval"], {"filters": {"remove_until": "</think>"}})
        self.assertEqual(
            ds_args["math_500"],
            {
                "few_shot_num": 0,
                "subset_list": ["Level 5"],
                "filters": {"remove_until": "</think>"},
            },
        )
        self.assertEqual(
            ds_args["gpqa_diamond"],
            {"few_shot_num": 0, "filters": {"remove_until": "</think>"}},
        )

    def test_build_dataset_args_rejects_unknown_dataset(self):
        with self.assertRaises(ValueError):
            build_dataset_args(["unknown_bench"])

    def test_build_evalscope_task_dict_thinking_and_non_thinking(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_think = build_evalscope_task_dict(
                model="nchapman/Qwen3.8-27B-W4A16-HGPTQ",
                api_url="http://127.0.0.1:8000/v1",
                datasets=["ifeval", "humaneval_plus"],
                work_dir=tmp,
                thinking=True,
            )
            self.assertEqual(cfg_think["generation_config"]["max_tokens"], 16384)
            self.assertNotIn("extra_body", cfg_think["generation_config"])
            self.assertEqual(cfg_think["sandbox"], {"enabled": True})

            cfg_nothink = build_evalscope_task_dict(
                model="nchapman/Qwen3.8-27B-W4A16-HGPTQ",
                api_url="http://127.0.0.1:8000/v1",
                datasets=["ifeval"],
                work_dir=tmp,
                thinking=False,
                presence_penalty=1.5,
            )
            self.assertEqual(cfg_nothink["generation_config"]["max_tokens"], 4096)
            self.assertEqual(cfg_nothink["generation_config"]["presence_penalty"], 1.5)
            self.assertEqual(
                cfg_nothink["generation_config"]["extra_body"],
                {"chat_template_kwargs": {"enable_thinking": False}},
            )
            self.assertNotIn("sandbox", cfg_nothink)

    def test_extract_evalscope_reports_reads_nested_report_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep_dir = Path(tmp) / "20261114_120000" / "reports" / "model_a"
            rep_dir.mkdir(parents=True)
            (rep_dir / "ifeval.json").write_text(json.dumps({"dataset_name": "ifeval", "score": 0.875}))
            reports = extract_evalscope_reports(tmp)
            self.assertEqual(reports["ifeval"]["score"], 0.875)

    def test_paired_binary_bootstrap_computes_diff_and_discordant_wins(self):
        scores_a = [1.0, 1.0, 1.0, 0.0, 1.0] * 10
        scores_b = [1.0, 0.0, 1.0, 0.0, 0.0] * 10
        res = paired_binary_bootstrap(scores_a, scores_b, n_boot=2000, seed=1)
        self.assertEqual(res["n"], 50)
        self.assertAlmostEqual(res["mean_a"], 0.8)
        self.assertAlmostEqual(res["mean_b"], 0.4)
        self.assertAlmostEqual(res["diff_b_minus_a"], -0.4)
        self.assertEqual(res["a_only_wins"], 20)
        self.assertEqual(res["b_only_wins"], 0)
        self.assertLess(res["ci95_diff_b_minus_a"][1], 0.0)

    def test_cli_eval_parser_wiring(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "eval",
                "--model",
                "nchapman/Qwen3.8-27B-W4A16-HGPTQ",
                "--output-dir",
                "/tmp/out_eval",
                "--datasets",
                "ifeval",
                "math_500",
                "--no-thinking",
            ]
        )
        self.assertEqual(args.cmd, "eval")
        self.assertFalse(args.thinking)
        self.assertEqual(args.datasets, ["ifeval", "math_500"])

    def test_fuse_quant_model_for_generation_matches_online_hadamard(self):
        import torch
        import torch.nn as nn

        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 32, bias=False)

            def forward(self, x):
                return self.fc(x)

        m = ToyModel()
        x = torch.randn(3, 64)
        # Simulate online Hadamard input pre-hook: y = (x @ H) @ W_rot^T
        w_orig = m.fc.weight.data.clone()
        m.fc.weight.data = _hadamard_rotate_weight(w_orig, 64)
        m.fc.register_forward_pre_hook(lambda mod, inp: (_hadamard_rotate_weight(inp[0], 64),))
        y_hook = m(x).clone()

        fused = fuse_quant_model_for_generation(m)
        self.assertEqual(fused, 1)
        self.assertEqual(len(m.fc._forward_pre_hooks), 0)
        y_fused = m(x)
        self.assertTrue(torch.allclose(y_hook, y_fused, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
