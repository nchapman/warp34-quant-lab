#!/usr/bin/env python3
"""Multi-GPU reproducible comparison runner for BF16 / W4A16-HGPTQ / W2.6A16-HGPTQ-RCO / Unsloth GGUF / EXL3.

Usage:
    uv run --no-sync python scripts/run_comparison_suite.py \
        --models 27b 9b \
        --gpus 0 1 \
        --work-dir /workspace/comparison_run \
        --vllm-venv /tmp/.venv-vllm \
        --exl3-venv /tmp/.venv-exl3
"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from warp_quant.comparison_runner import (
    SUITES,
    ComparisonArm,
    count_bf16_trunk_params,
    dequantize_gguf_to_hf_dir,
    extract_evalscope_metrics_and_samples,
    inspect_gguf_byte_budget,
    inspect_safetensors_byte_budget,
    load_probe_rows_jsonl,
    paired_evalscope_bootstrap_map,
    paired_probe_bootstrap,
)
from warp_quant.models import MODELS

REPO_ROOT = Path(__file__).resolve().parents[1]


def _log(msg: str) -> None:
    ts = datetime.now(UTC).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _build_ld_library_path(venv_bin: Path) -> str:
    venv_root = venv_bin.parent
    lib_dirs = glob.glob(str(venv_root / "lib" / "python*" / "site-packages" / "nvidia" / "*" / "lib"))
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [p for p in lib_dirs if os.path.isdir(p)]
    if existing:
        parts.append(existing)
    return ":".join(parts)


def _wait_for_server(port: int, proc: subprocess.Popen, timeout_s: float = 420.0) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2.0)
    raise TimeoutError(f"vLLM server on port {port} did not become ready within {timeout_s}s")


def _download_arm(arm: ComparisonArm, cache_root: Path) -> dict[str, Any]:
    """Download checkpoint files from HF and return local path + exact commit SHA provenance."""
    api = HfApi()
    info = api.model_info(arm.repo, revision=arm.revision)
    sha = info.sha

    if arm.family == "unsloth_gguf":
        assert arm.gguf_filename is not None
        local_file = Path(
            hf_hub_download(
                repo_id=arm.repo,
                filename=arm.gguf_filename,
                revision=sha,
            )
        )
        return {
            "raw_path": str(local_file),
            "sha": sha,
            "gguf_filename": arm.gguf_filename,
        }

    ignore_patterns = ["*.png", "*.jpg", "cal_trace.*", "qbench_*", "*.md"]
    local_dir = Path(
        snapshot_download(
            repo_id=arm.repo,
            revision=sha,
            ignore_patterns=ignore_patterns,
        )
    )
    return {
        "raw_path": str(local_dir),
        "sha": sha,
    }


def _dequantize_exl3_subprocess(
    exl3_python: Path,
    exl3_dir: Path,
    base_dir: Path,
    out_dir: Path,
    gpu: str,
) -> dict[str, Any]:
    """Run EXL3 reconstruction in the dedicated exllamav3 venv so CUDA/torch ABI stays clean."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    code = (
        "import json; from pathlib import Path; "
        "from warp_quant.comparison_runner import dequantize_exl3_to_hf_dir; "
        f"res = dequantize_exl3_to_hf_dir(Path({str(exl3_dir)!r}), Path({str(base_dir)!r}), "
        f"Path({str(out_dir)!r}), device='cuda:0', verify_first_linear=True); "
        "print('EXL3_DEQUANT_JSON ' + json.dumps(res))"
    )
    proc = subprocess.run(
        [str(exl3_python), "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"EXL3 dequantization failed for {exl3_dir} (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout[-2000:]}\nSTDERR:\n{proc.stderr[-2000:]}"
        )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("EXL3_DEQUANT_JSON "):
            return json.loads(line[len("EXL3_DEQUANT_JSON ") :])
    raise RuntimeError(f"Missing EXL3_DEQUANT_JSON output for {exl3_dir}:\n{proc.stdout[-1000:]}")


def prepare_model_suite(
    model_key: str,
    work_dir: Path,
    exl3_python: Path,
    gpu: str,
) -> dict[str, Any]:
    """Download all 7 arms, measure exact text-backbone byte budgets, and dequantize GGUF/EXL3."""
    arms = SUITES[model_key]
    model_dir = work_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    prov_path = model_dir / "provenance.json"

    n_layers = int(MODELS[model_key]["num_hidden_layers"])
    _log(f"[{model_key}] Preparing {len(arms)} arms (GPU {gpu} for EXL3 dequant)")

    arm_records: dict[str, dict[str, Any]] = {}
    # Step 1: Download BF16 teacher first (needed for exact trunk param count & dequantization assets)
    bf16_arm = next(a for a in arms if a.arm_id == "bf16")
    bf16_dl = _download_arm(bf16_arm, model_dir)
    base_dir = Path(bf16_dl["raw_path"])
    trunk_params = count_bf16_trunk_params(base_dir)
    _log(f"[{model_key}] Base BF16 ready at {base_dir} (trunk_params={trunk_params:,})")

    for arm in arms:
        _log(f"[{model_key}] Downloading / verifying {arm.arm_id}: {arm.label}")
        dl = bf16_dl if arm.arm_id == "bf16" else _download_arm(arm, model_dir)
        raw_path = Path(dl["raw_path"])

        if arm.family == "unsloth_gguf":
            budget = inspect_gguf_byte_budget(raw_path, n_layers=n_layers, trunk_params=trunk_params)
        else:
            budget = inspect_safetensors_byte_budget(raw_path, trunk_params=trunk_params)

        eval_path = raw_path
        dequant_meta: dict[str, Any] | None = None

        if arm.family in ("unsloth_gguf", "exl3"):
            deq_dir = model_dir / "dequant" / arm.arm_id
            done_marker = deq_dir / ".dequant_done.json"
            if done_marker.is_file():
                dequant_meta = json.loads(done_marker.read_text())
                _log(f"[{model_key}] Reusing cached dequant for {arm.arm_id}: {deq_dir}")
            else:
                t0 = time.time()
                if arm.family == "unsloth_gguf":
                    _log(f"[{model_key}] Dequantizing GGUF {arm.arm_id} -> {deq_dir}")
                    dequant_meta = dequantize_gguf_to_hf_dir(raw_path, base_dir, deq_dir)
                else:
                    _log(f"[{model_key}] Dequantizing EXL3 {arm.arm_id} on GPU {gpu} -> {deq_dir}")
                    dequant_meta = _dequantize_exl3_subprocess(exl3_python, raw_path, base_dir, deq_dir, gpu)
                dequant_meta["elapsed_s"] = round(time.time() - t0, 2)
                done_marker.write_text(json.dumps(dequant_meta, indent=2) + "\n")
                _log(
                    f"[{model_key}] Dequantized {arm.arm_id} in {dequant_meta['elapsed_s']}s "
                    f"(verify={dequant_meta.get('verification')})"
                )
            eval_path = deq_dir

        arm_records[arm.arm_id] = {
            "spec": arm.to_dict(),
            "sha": dl["sha"],
            "raw_path": str(raw_path),
            "eval_path": str(eval_path),
            "byte_budget": budget,
            "dequant": dequant_meta,
        }

    payload = {
        "model_key": model_key,
        "base_repo": bf16_arm.repo,
        "base_sha": bf16_dl["sha"],
        "base_dir": str(base_dir),
        "n_layers": n_layers,
        "trunk_params": trunk_params,
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "arms": arm_records,
    }
    prov_path.write_text(json.dumps(payload, indent=2) + "\n")
    _log(f"[{model_key}] Prepare complete -> {prov_path}")
    return payload


def run_probe_suite(model_key: str, work_dir: Path, gpu: str) -> None:
    """Run single-pass cached-teacher warp-quant probe across all 6 quantized arms on one GPU."""
    model_dir = work_dir / model_key
    prov = json.loads((model_dir / "provenance.json").read_text())
    probe_dir = model_dir / "probe"
    rows_dir = model_dir / "probe_rows"
    done_marker = probe_dir / ".probe_suite_done.json"
    if done_marker.is_file():
        _log(f"[{model_key}] Reusing completed probe suite: {done_marker}")
        return

    probe_dir.mkdir(parents=True, exist_ok=True)
    rows_dir.mkdir(parents=True, exist_ok=True)

    arms = [a for a in SUITES[model_key] if a.arm_id != "bf16"]
    quant_paths: list[str] = []
    name_to_arm_id: dict[str, str] = {}
    for arm in arms:
        ep = Path(prov["arms"][arm.arm_id]["eval_path"]).resolve()
        quant_paths.append(str(ep))
        if ep.name in name_to_arm_id:
            raise RuntimeError(f"Duplicate resolved basename {ep.name!r} for {arm.arm_id}")
        name_to_arm_id[ep.name] = arm.arm_id

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"

    cmd = [
        sys.executable,
        "-m",
        "warp_quant.cli",
        "probe",
        "--model",
        prov["base_dir"],
        "--quant",
        *quant_paths,
        "--eval-sets",
        str(REPO_ROOT / "data" / "probe-v4.jsonl"),
        str(REPO_ROOT / "data" / "probe-guard.jsonl"),
        "--output-dir",
        str(probe_dir),
        "--dump-rows-dir",
        str(rows_dir),
    ]
    _log(f"[{model_key}] Starting cached-teacher probe on GPU {gpu} across {len(quant_paths)} arms")
    t0 = time.time()
    log_path = probe_dir / "probe_suite.log"
    with log_path.open("w") as log_f:
        proc = subprocess.run(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = log_path.read_text()[-3000:]
        raise RuntimeError(f"[{model_key}] Probe failed (rc={proc.returncode}):\n{tail}")

    # Normalize output files to canonical arm_id names
    probe_map: dict[str, str] = {}
    rows_map: dict[str, str] = {}
    for base_name, arm_id in name_to_arm_id.items():
        src_json = probe_dir / f"probe_{base_name}.json"
        if not src_json.is_file():
            raise FileNotFoundError(f"Expected probe output missing: {src_json}")
        dst_json = probe_dir / f"arm_{arm_id}.json"
        dst_json.write_text(src_json.read_text())
        probe_map[arm_id] = str(dst_json)

        src_rows = rows_dir / base_name
        if not src_rows.is_dir():
            raise FileNotFoundError(f"Expected probe rows dir missing: {src_rows}")
        rows_map[arm_id] = str(src_rows)

    meta = {
        "model_key": model_key,
        "gpu": gpu,
        "elapsed_s": round(time.time() - t0, 2),
        "probe_files": probe_map,
        "rows_dirs": rows_map,
    }
    done_marker.write_text(json.dumps(meta, indent=2) + "\n")
    _log(f"[{model_key}] Probe suite completed in {meta['elapsed_s']}s")


def run_evalscope_arm(
    model_key: str,
    arm_id: str,
    work_dir: Path,
    vllm_venv: Path,
    gpu: str,
    port: int,
    *,
    datasets: list[str],
    max_tokens: int,
    limit: int | None,
    batch_size: int,
    max_model_len: int,
    gpu_mem_util: float,
    trust_remote_code: bool = False,
    allow_local_code_exec: bool = False,
) -> dict[str, Any]:
    """Serve one arm with vLLM and evaluate full EvalScope datasets on the assigned GPU."""
    model_dir = work_dir / model_key
    prov = json.loads((model_dir / "provenance.json").read_text())
    arm_rec = prov["arms"][arm_id]
    eval_path = Path(arm_rec["eval_path"])
    arm_out_dir = model_dir / "evalscope" / arm_id
    done_path = arm_out_dir / "done.json"
    if done_path.is_file():
        _log(f"[{model_key}/{arm_id}] Reusing completed EvalScope run: {done_path}")
        return json.loads(done_path.read_text())

    arm_out_dir.mkdir(parents=True, exist_ok=True)
    venv_bin = vllm_venv / "bin"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["LD_LIBRARY_PATH"] = _build_ld_library_path(venv_bin)
    env["NLTK_DATA"] = env.get("NLTK_DATA", "/root/nltk_data")
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VLLM_WARP_QUANT_W2_MARLIN"] = "1"

    cfg_path = eval_path / "config.json"
    cfg_data = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    is_vlm = cfg_data.get("model_type") == "qwen3_5"

    served_name = f"{model_key}-{arm_id}"
    cmd = [
        str(venv_bin / "vllm"),
        "serve",
        str(eval_path),
        "--served-model-name",
        served_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--gdn-prefill-backend",
        "triton",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_mem_util),
        "--disable-log-requests",
    ]
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if is_vlm:
        cmd.append("--language-model-only")

    vllm_log = arm_out_dir / "vllm.log"
    t_start = time.time()
    _log(f"[{model_key}/{arm_id}] Launching vLLM on GPU {gpu} port {port} (vlm={is_vlm}): {eval_path}")
    log_f = vllm_log.open("w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        _wait_for_server(port, proc, timeout_s=420.0)
        t_ready = round(time.time() - t_start, 2)
        _log(
            f"[{model_key}/{arm_id}] vLLM ready in {t_ready}s; running EvalScope "
            f"(datasets={datasets}, limit={limit}, max_tokens={max_tokens}, bs={batch_size})"
        )

        es_work_dir = arm_out_dir / "es_runs"
        es_log = arm_out_dir / "evalscope.log"
        local_exec_patch = (
            """
import os, resource, tempfile
from evalscope.api.mixin.code_execution_sandbox_mixin import CodeExecutionSandboxMixin

def _apply_rlimits():
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

def _local_subprocess_exec(self, code, timeout=30, language="python"):
    try:
        eff_t = max(15, min(60, int(timeout or 30)))
        clean_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONNOUSERSITE": "1"}
        with tempfile.TemporaryDirectory(prefix="es_code_") as tmpdir:
            r = subprocess.run(
                [sys.executable, "-I", "-"],
                input=code,
                cwd=tmpdir,
                env=clean_env,
                preexec_fn=_apply_rlimits,
                capture_output=True,
                text=True,
                timeout=eff_t,
            )
        return {"status": "success" if r.returncode == 0 else "failed", "returncode": r.returncode, "stderr": r.stderr[-500:]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

CodeExecutionSandboxMixin.execute_code_in_sandbox = _local_subprocess_exec
"""
            if allow_local_code_exec
            else ""
        )
        eval_code = f"""
import json, subprocess, sys, time
from evalscope import TaskConfig, run_task
{local_exec_patch}
t0 = time.time()
cfg = TaskConfig(
    model={served_name!r},
    api_url="http://127.0.0.1:{port}/v1",
    api_key="EMPTY",
    eval_type="openai_api",
    datasets={datasets!r},
    dataset_args={{
        "ifeval": {{"filters": {{"remove_until": "</think>"}}}},
        "math_500": {{"few_shot_num": 0, "filters": {{"remove_until": "</think>"}}}},
        "humaneval_plus": {{"filters": {{"remove_until": "</think>"}}}},
    }},
    eval_batch_size={batch_size},
    limit={limit!r},
    generation_config={{
        "max_tokens": {max_tokens},
        "temperature": 0.0,
        "extra_body": {{"chat_template_kwargs": {{"enable_thinking": False}}}},
    }},
    seed=42,
    work_dir={str(es_work_dir)!r},
    use_cache=None,
)
run_task(cfg)
print("EVALSCOPE_ELAPSED_S", round(time.time() - t0, 2))
"""
        t_eval0 = time.time()
        with es_log.open("w") as ef:
            res = subprocess.run(
                [str(venv_bin / "python"), "-c", eval_code],
                env=env,
                stdout=ef,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if res.returncode != 0:
            tail = es_log.read_text()[-3000:]
            raise RuntimeError(f"[{model_key}/{arm_id}] EvalScope failed (rc={res.returncode}):\n{tail}")
        eval_elapsed_s = round(time.time() - t_eval0, 2)
        summary, samples = extract_evalscope_metrics_and_samples(es_work_dir, max_tokens=max_tokens)
        (arm_out_dir / "samples.json").write_text(json.dumps(samples))
        done_payload = {
            "model_key": model_key,
            "arm_id": arm_id,
            "gpu": gpu,
            "vllm_ready_s": t_ready,
            "eval_elapsed_s": eval_elapsed_s,
            "max_tokens": max_tokens,
            "limit": limit,
            "summary": summary,
        }
        done_path.write_text(json.dumps(done_payload, indent=2) + "\n")
        _log(f"[{model_key}/{arm_id}] EvalScope done in {eval_elapsed_s}s: {summary}")
        return done_payload
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=15)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        log_f.close()
        time.sleep(3.0)


def aggregate_model_receipts(
    model_key: str,
    work_dir: Path,
    *,
    datasets: list[str],
    max_tokens: int,
    limit: int | None,
) -> tuple[Path, Path]:
    """Aggregate provenance, byte budgets, probes, EvalScope metrics, and paired bootstrap CIs."""
    model_dir = work_dir / model_key
    prov = json.loads((model_dir / "provenance.json").read_text())
    probe_meta = json.loads((model_dir / "probe" / ".probe_suite_done.json").read_text())

    arms = SUITES[model_key]
    arm_ids = [a.arm_id for a in arms]

    # 1. Load probe results & rows
    probes_by_arm: dict[str, Any] = {}
    rows_by_set: dict[str, dict[str, list[dict[str, Any]]]] = {"probe-v4": {}, "probe-guard": {}}
    for arm in arms:
        if arm.arm_id == "bf16":
            continue
        p_file = Path(probe_meta["probe_files"][arm.arm_id])
        probes_by_arm[arm.arm_id] = json.loads(p_file.read_text())
        r_dir = Path(probe_meta["rows_dirs"][arm.arm_id])
        for eval_set in ("probe-v4", "probe-guard"):
            rf = r_dir / f"{eval_set}_rows.jsonl"
            if rf.is_file():
                rows_by_set[eval_set][arm.arm_id] = load_probe_rows_jsonl(rf)

    # Paired probe bootstrap comparisons (20,000 draws)
    probe_pairs = [
        ("w4_hgptq", "exl3_4bpw"),
        ("w4_hgptq", "gguf_q4_k_xl"),
        ("exl3_4bpw", "gguf_q4_k_xl"),
        ("w26_rco", "exl3_3bpw"),
        ("w26_rco", "gguf_q2_k_xl"),
        ("exl3_3bpw", "gguf_q2_k_xl"),
        ("w4_hgptq", "w26_rco"),
    ]
    paired_probe: dict[str, dict[str, Any]] = {}
    for eval_set in ("probe-v4", "probe-guard"):
        paired_probe[eval_set] = {}
        for a_id, b_id in probe_pairs:
            if a_id in rows_by_set[eval_set] and b_id in rows_by_set[eval_set]:
                paired_probe[eval_set][f"{a_id}_vs_{b_id}"] = paired_probe_bootstrap(
                    rows_by_set[eval_set][a_id],
                    rows_by_set[eval_set][b_id],
                )

    # 2. Load EvalScope results & per-sample scores
    es_by_arm: dict[str, Any] = {}
    es_samples: dict[str, dict[str, dict[str, float]]] = {}
    for arm_id in arm_ids:
        es_dir = model_dir / "evalscope" / arm_id
        done = json.loads((es_dir / "done.json").read_text())
        es_by_arm[arm_id] = {
            "vllm_ready_s": done["vllm_ready_s"],
            "eval_elapsed_s": done["eval_elapsed_s"],
            **done["summary"],
        }
        es_samples[arm_id] = json.loads((es_dir / "samples.json").read_text())

    paired_vs_bf16: dict[str, Any] = {}
    for arm_id in arm_ids:
        if arm_id == "bf16":
            continue
        paired_vs_bf16[arm_id] = paired_evalscope_bootstrap_map(es_samples["bf16"], es_samples[arm_id])

    paired_peers: dict[str, Any] = {}
    for a_id, b_id in probe_pairs:
        if a_id in es_samples and b_id in es_samples:
            paired_peers[f"{a_id}_vs_{b_id}"] = paired_evalscope_bootstrap_map(
                es_samples[a_id], es_samples[b_id]
            )

    arms_meta = {
        arm_id: {
            "spec": prov["arms"][arm_id]["spec"],
            "sha": prov["arms"][arm_id]["sha"],
            "byte_budget": prov["arms"][arm_id]["byte_budget"],
            "exl3_verification": (prov["arms"][arm_id].get("dequant") or {}).get("verification"),
        }
        for arm_id in arm_ids
    }

    receipts_dir = work_dir / "receipts" / model_key
    receipts_dir.mkdir(parents=True, exist_ok=True)
    probe_receipt = {
        "model_key": model_key,
        "base_model": {
            "repo": prov["base_repo"],
            "sha": prov["base_sha"],
            "n_layers": prov["n_layers"],
            "trunk_params": prov["trunk_params"],
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "probe_dtype": "bfloat16",
        "arms": arms_meta,
        "probes": probes_by_arm,
        "paired_bootstrap_20000_draws": paired_probe,
    }
    evalscope_receipt = {
        "model_key": model_key,
        "base_model": {
            "repo": prov["base_repo"],
            "sha": prov["base_sha"],
            "n_layers": prov["n_layers"],
            "trunk_params": prov["trunk_params"],
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "eval_config": {
            "engine": "vllm",
            "dtype": "bfloat16",
            "gdn_prefill_backend": "triton",
            "datasets": datasets,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "thinking": False,
            "limit": limit,
            "seed": 42,
        },
        "arms_meta": arms_meta,
        "arms": es_by_arm,
        "paired_vs_bf16_20000_draws": paired_vs_bf16,
        "paired_peers_20000_draws": paired_peers,
    }

    probe_out = receipts_dir / "probe_comparison.json"
    es_out = receipts_dir / "evalscope_comparison.json"
    probe_out.write_text(json.dumps(probe_receipt, indent=2) + "\n")
    es_out.write_text(json.dumps(evalscope_receipt, indent=2) + "\n")
    _log(f"[{model_key}] Wrote receipts: {probe_out} & {es_out}")
    return probe_out, es_out


def main() -> None:
    p = argparse.ArgumentParser(description="Run 7-arm multi-format comparison suite across GPUs")
    p.add_argument("--models", nargs="+", choices=["27b", "9b"], default=["27b", "9b"])
    p.add_argument("--gpus", nargs="+", default=["0", "1"])
    p.add_argument("--work-dir", default="/workspace/comparison_run")
    p.add_argument("--vllm-venv", default="/tmp/.venv-vllm")
    p.add_argument("--exl3-venv", default="/tmp/.venv-exl3")
    p.add_argument("--datasets", nargs="+", default=["ifeval", "math_500", "humaneval_plus"])
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--limit", type=int, default=0, help="0 means full dataset (limit=None)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-mem-util", type=float, default=0.85)
    p.add_argument("--base-port", type=int, default=18200)
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Opt-in to passing --trust-remote-code to vLLM (not needed for built-in Qwen3.5/3.8 architectures).",
    )
    p.add_argument(
        "--allow-local-code-exec",
        action="store_true",
        help="Opt-in to running HumanEval+ code execution in a resource-limited local subprocess instead of EvalScope's Docker sandbox (only use inside an ephemeral container).",
    )
    p.add_argument(
        "--phases",
        nargs="+",
        choices=["prepare", "probe", "evalscope", "aggregate"],
        default=["prepare", "probe", "evalscope", "aggregate"],
    )
    args = p.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    vllm_venv = Path(args.vllm_venv)
    exl3_python = Path(args.exl3_venv) / "bin" / "python"
    eff_limit = None if args.limit <= 0 else args.limit

    # Phase 1: Prepare (download + byte budgets + dequantize GGUF/EXL3) in parallel per model size
    if "prepare" in args.phases:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.models)) as ex:
            futs = {}
            for idx, mk in enumerate(args.models):
                gpu = args.gpus[idx % len(args.gpus)]
                futs[ex.submit(prepare_model_suite, mk, work_dir, exl3_python, gpu)] = mk
            for fut in concurrent.futures.as_completed(futs):
                fut.result()

    # Phase 2: Teacher-forced probes in parallel across GPUs (1 GPU per model size)
    if "probe" in args.phases:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(args.models), len(args.gpus))) as ex:
            futs = {}
            for idx, mk in enumerate(args.models):
                gpu = args.gpus[idx % len(args.gpus)]
                futs[ex.submit(run_probe_suite, mk, work_dir, gpu)] = mk
            for fut in concurrent.futures.as_completed(futs):
                fut.result()

    # Phase 3: vLLM + EvalScope across all arms using a shared multi-GPU queue (27b first for balance)
    if "evalscope" in args.phases:
        jobs: queue.Queue[tuple[str, str]] = queue.Queue()
        ordered_models = sorted(args.models, key=lambda m: 0 if m == "27b" else 1)
        for mk in ordered_models:
            for arm in SUITES[mk]:
                jobs.put((mk, arm.arm_id))

        errors: list[tuple[str, str, BaseException]] = []
        err_lock = threading.Lock()

        def worker(worker_idx: int, gpu: str) -> None:
            port = args.base_port + worker_idx
            while True:
                try:
                    mk, arm_id = jobs.get_nowait()
                except queue.Empty:
                    return
                try:
                    run_evalscope_arm(
                        mk,
                        arm_id,
                        work_dir,
                        vllm_venv,
                        gpu,
                        port,
                        datasets=args.datasets,
                        max_tokens=args.max_tokens,
                        limit=eff_limit,
                        batch_size=args.batch_size,
                        max_model_len=args.max_model_len,
                        gpu_mem_util=args.gpu_mem_util,
                        trust_remote_code=args.trust_remote_code,
                        allow_local_code_exec=args.allow_local_code_exec,
                    )
                except BaseException as exc:
                    _log(f"ERROR [{mk}/{arm_id}] on GPU {gpu}: {exc}")
                    with err_lock:
                        errors.append((mk, arm_id, exc))
                finally:
                    jobs.task_done()

        threads = [
            threading.Thread(target=worker, args=(i, gpu), daemon=False) for i, gpu in enumerate(args.gpus)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if errors:
            summary_err = "; ".join(f"{mk}/{aid}: {exc}" for mk, aid, exc in errors)
            raise RuntimeError(f"EvalScope phase had {len(errors)} failure(s): {summary_err}")

    # Phase 4: Aggregate receipts + paired bootstrap CIs
    if "aggregate" in args.phases:
        for mk in args.models:
            aggregate_model_receipts(
                mk,
                work_dir,
                datasets=args.datasets,
                max_tokens=args.max_tokens,
                limit=eff_limit,
            )

    _log("ALL PHASES COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
