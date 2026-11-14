"""EvalScope downstream benchmark runner and paired comparison analyzer for Qwen3.5 / 3.6 / 3.8."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from warp_quant.lablog import log_event, run_done

SUPPORTED_EVALSCOPE_DATASETS: tuple[str, ...] = (
    "ifeval",
    "math_500",
    "gpqa_diamond",
    "humaneval_plus",
    "live_code_bench",
    "mmlu_pro",
)

CODE_SANDBOX_DATASETS: frozenset[str] = frozenset({"humaneval_plus", "live_code_bench"})


def build_dataset_args(
    datasets: list[str],
    *,
    remove_thinking_tag: str | None = "</think>",
    math_levels: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build per-dataset EvalScope arguments with deterministic <think> stripping."""
    dataset_args: dict[str, dict[str, Any]] = {}
    for ds in datasets:
        if ds not in SUPPORTED_EVALSCOPE_DATASETS:
            raise ValueError(
                f"Unsupported EvalScope dataset {ds!r}; supported: {list(SUPPORTED_EVALSCOPE_DATASETS)}"
            )
        cfg: dict[str, Any] = {}
        if ds in {"math_500", "gpqa_diamond"}:
            cfg["few_shot_num"] = 0
        if ds == "math_500" and math_levels:
            cfg["subset_list"] = list(math_levels)
        if remove_thinking_tag:
            cfg["filters"] = {"remove_until": remove_thinking_tag}
        dataset_args[ds] = cfg
    return dataset_args


def build_evalscope_task_dict(
    *,
    model: str,
    api_url: str,
    datasets: list[str],
    work_dir: str | Path,
    api_key: str = "EMPTY",
    eval_batch_size: int = 64,
    max_tokens: int | None = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    presence_penalty: float | None = None,
    thinking: bool = True,
    remove_thinking_tag: str | None = "</think>",
    math_levels: list[str] | None = None,
    limit: int | None = None,
    seed: int = 42,
    timeout: int = 60000,
    stream: bool = True,
) -> dict[str, Any]:
    """Construct a serializable dictionary of kwargs for `evalscope.TaskConfig`."""
    if not datasets:
        raise ValueError("datasets must not be empty")
    if eval_batch_size < 1:
        raise ValueError(f"eval_batch_size must be >= 1, got {eval_batch_size}")
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive when set, got {limit}")

    eff_max_tokens = max_tokens if max_tokens is not None else (16384 if thinking else 4096)
    gen_cfg: dict[str, Any] = {
        "max_tokens": eff_max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "n": 1,
    }
    if presence_penalty is not None:
        gen_cfg["presence_penalty"] = presence_penalty
    if not thinking:
        gen_cfg["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    ds_args = build_dataset_args(
        datasets,
        remove_thinking_tag=remove_thinking_tag,
        math_levels=math_levels,
    )
    task_dict: dict[str, Any] = {
        "model": model,
        "api_url": api_url,
        "api_key": api_key,
        "eval_type": "openai_api",
        "datasets": list(datasets),
        "dataset_args": ds_args,
        "work_dir": str(Path(work_dir)),
        "eval_batch_size": eval_batch_size,
        "generation_config": gen_cfg,
        "timeout": timeout,
        "stream": stream,
        "seed": seed,
    }
    if limit is not None:
        task_dict["limit"] = limit
    if any(ds in CODE_SANDBOX_DATASETS for ds in datasets):
        task_dict["sandbox"] = {"enabled": True}
    return task_dict


def extract_evalscope_reports(work_dir: str | Path) -> dict[str, Any]:
    """Collect all EvalScope JSON report summaries written under `work_dir`."""
    root = Path(work_dir)
    reports_dir = root / "reports"
    if not reports_dir.exists():
        # Search nested timestamped run directories if EvalScope created one
        candidates = sorted(root.glob("*/reports"))
        if candidates:
            reports_dir = candidates[-1]
    summary: dict[str, Any] = {}
    if not reports_dir.exists():
        return summary
    for report_file in sorted(reports_dir.rglob("*.json")):
        try:
            data = json.loads(report_file.read_text())
        except json.JSONDecodeError:
            continue
        ds_name = data.get("dataset_name") or report_file.stem
        summary[ds_name] = data
    return summary


def paired_binary_bootstrap(
    scores_a: list[float],
    scores_b: list[float],
    *,
    n_boot: int = 20000,
    seed: int = 20261114,
) -> dict[str, Any]:
    """Compute paired mean difference, 95% bootstrap CI, and discordant counts for two paired score lists."""
    if len(scores_a) != len(scores_b) or not scores_a:
        raise ValueError("scores_a and scores_b must be non-empty lists of equal length")
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)
    n = len(arr_a)
    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    diff_b_minus_a = mean_b - mean_a

    rng = np.random.default_rng(seed)
    diffs = arr_b - arr_a
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_diffs = diffs[idx].mean(axis=1)

    a_wins = int(np.sum(arr_a > arr_b))
    b_wins = int(np.sum(arr_b > arr_a))
    ties = int(n - a_wins - b_wins)
    return {
        "n": n,
        "mean_a": round(mean_a, 6),
        "mean_b": round(mean_b, 6),
        "diff_b_minus_a": round(diff_b_minus_a, 6),
        "ci95_diff_b_minus_a": [
            round(float(np.percentile(boot_diffs, 2.5)), 6),
            round(float(np.percentile(boot_diffs, 97.5)), 6),
        ],
        "p_a_ge_b": round(float(np.mean(boot_diffs <= 0)), 5),
        "a_only_wins": a_wins,
        "b_only_wins": b_wins,
        "ties": ties,
    }


def run_evalscope_cli(args: argparse.Namespace) -> dict[str, Any]:
    """Execute an EvalScope evaluation task against an OpenAI-compatible server."""
    try:
        from evalscope import TaskConfig, run_task
    except ImportError as exc:
        raise RuntimeError(
            "EvalScope is not installed in this environment. Install it with: "
            "`uv pip install 'evalscope[app]' math_verify immutabledict langdetect nltk`"
        ) from exc

    work_dir = Path(args.output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    task_dict = build_evalscope_task_dict(
        model=args.model,
        api_url=args.api_url,
        api_key=args.api_key,
        datasets=args.datasets,
        work_dir=work_dir,
        eval_batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        thinking=args.thinking,
        remove_thinking_tag=args.remove_thinking_tag if args.remove_thinking_tag != "none" else None,
        math_levels=args.math_levels,
        limit=args.limit,
        seed=args.seed,
        timeout=args.timeout,
        stream=args.stream,
    )
    (work_dir / "task_config.json").write_text(json.dumps(task_dict, indent=2) + "\n")
    log_event(
        "EVALSCOPE_START",
        model=args.model,
        api_url=args.api_url,
        datasets=args.datasets,
        batch_size=args.batch_size,
        thinking=args.thinking,
        limit=args.limit,
    )
    task_cfg = TaskConfig(**task_dict)
    run_task(task_cfg=task_cfg)
    reports = extract_evalscope_reports(work_dir)
    summary_path = work_dir / "evalscope_summary.json"
    summary_payload = {
        "task_config": task_dict,
        "reports": reports,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")
    run_done(output_dir=str(work_dir), datasets=list(reports.keys()))
    return summary_payload
