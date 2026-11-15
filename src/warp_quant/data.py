"""Calibration datasets, text/MTP/Vision probe loaders, and v2 vision fixture materializer."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

from warp_quant.lablog import log_event

MAX_LEN = 4096
CHUNK = 512

CASCADE_WEIGHTS: dict[str, float] = {
    "chat": 0.586,
    "math": 0.185,
    "science": 0.114,
    "instruction_following": 0.052,
    "conversational_agent": 0.035,
    "terminal_agent": 0.020,
    "swe": 0.006,
}
AGG_WEIGHTS: dict[str, float] = {
    **{
        domain: 0.9 * weight / math.fsum(CASCADE_WEIGHTS.values())
        for domain, weight in CASCADE_WEIGHTS.items()
    },
    "ML": 0.10,
}
WEIGHTED_KL_PROTOCOL = "cascade90-ml10-language-macro"

SOURCES_V2_CORE = (
    (
        "chartqa",
        "HuggingFaceM4/ChartQA",
        "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5",
        "data/test-00000-of-00001-e2cd0b7a0f9eb20d.parquet",
    ),
    (
        "textvqa",
        "lmms-lab/TextVQA",
        "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "data/validation-00000-of-00003.parquet",
    ),
    (
        "mmstar",
        "Lin-Chen/MMStar",
        "bc98d668301da7b14f648724866e57302778ab27",
        "mmstar.parquet",
    ),
    (
        "ocrbench",
        "echo840/OCRBench",
        "92a54bd1384387c178d5a07140a2d85e0a3d12e1",
        "data/test-00000-of-00001.parquet",
    ),
)

SOURCES_V2_GUARD = (
    (
        "screenspot",
        "rootsautomation/ScreenSpot",
        "0be08781e2e188582f6131625ae1598d443b4d5d",
        "data/test-00000-of-00003.parquet",
    ),
)

OCR_V2_BUCKET_MAP = {
    "docVQA": "docvqa",
    "infographicVQA": "infographic_vqa",
    "FUNSD": "funsd_forms",
    "SROIE": "kie_receipts_labels",
    "POIE": "kie_receipts_labels",
    "ocrVQA": "ocrvqa_scene",
    "ESTVQA": "ocrvqa_scene",
    "STVQA": "ocrvqa_scene",
}

MMSTAR_V2_BUCKET_MAP = {
    "coarse perception": "mmstar_perception",
    "fine-grained perception": "mmstar_perception",
    "instance reasoning": "mmstar_reasoning",
    "logical reasoning": "mmstar_reasoning",
    "math": "mmstar_math",
    "science & technology": "mmstar_science",
}

V2_IN_DIST_DOMAINS = (
    "chartqa_human",
    "chartqa_machine",
    "docvqa",
    "funsd_forms",
    "infographic_vqa",
    "kie_receipts_labels",
    "mmstar_math",
    "mmstar_perception",
    "mmstar_reasoning",
    "mmstar_science",
    "ocrvqa_scene",
    "textvqa",
)

V2_GUARD_DOMAINS = (
    "ocrbench_artistic_math",
    "screenspot_ios",
    "screenspot_macos_icon",
    "screenspot_macos_text",
    "screenspot_windows_icon",
    "screenspot_windows_text",
)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_messages(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = []
    for message in conversation:
        msg: dict[str, Any] = {"role": message["role"], "content": message.get("content") or ""}
        if message.get("tool_calls"):
            tool_calls = []
            for call in message["tool_calls"]:
                fn = call.get("function") if isinstance(call.get("function"), dict) else call
                call = dict(fn or {})
                args = call.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                call["arguments"] = args
                tool_calls.append(call)
            msg["tool_calls"] = tool_calls
        messages.append(msg)
    return messages


def load_jsonl_calib(path: str | Path, num_samples: int) -> list[list[dict[str, Any]]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            msgs = r.get("messages") or r.get("conversations")
            rows.append(normalize_messages(msgs))
    return rows[:num_samples]


def get_calibration_dataset(
    processor: Any,
    calib: str | Path,
    num_samples: int,
    max_seq_length: int,
    window: str = "head",
    seed: int = 0,
) -> Dataset:
    tokenizer = processor.tokenizer
    path = Path(calib)
    conversations = load_jsonl_calib(path, num_samples)

    def tokenize(sample: dict[str, Any], index: int) -> dict[str, Any]:
        try:
            text = processor.apply_chat_template(normalize_messages(sample["messages"]), tokenize=False)
        except Exception:
            return {"input_ids": [], "attention_mask": []}
        if window == "random":
            rng = random.Random(hashlib.sha256(text.encode()).digest()[:8] + seed.to_bytes(4, "big"))
            enc = tokenizer(text, padding=False, truncation=False, add_special_tokens=False)
            ids = enc["input_ids"]
            if len(ids) > max_seq_length:
                off = rng.randrange(0, len(ids) - max_seq_length + 1)
                ids = ids[off : off + max_seq_length]
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}
            return {"input_ids": ids, "attention_mask": enc["attention_mask"]}
        tokenizer.truncation_side = "left" if window == "tail" else "right"
        return tokenizer(
            text,
            padding=False,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )

    ds = Dataset.from_list([{"messages": m} for m in conversations])
    ds = ds.map(tokenize, with_indices=True, remove_columns=ds.column_names)
    ds = ds.filter(lambda x: len(x["input_ids"]) > 0)
    return ds


def aggregate_weighted_kl(domains: dict[str, dict[str, Any]]) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    for domain, stats in domains.items():
        if domain.startswith("belebele"):
            group = "ML"
        elif domain == "chat_long":
            group = "chat"
        else:
            group = domain
        if group not in AGG_WEIGHTS:
            continue
        if stats["kl_n"] <= 0:
            raise ValueError(f"domain {domain!r} has no KL tokens")
        groups[group].append(stats["kl"] / stats["kl_n"])
    if not groups:
        raise ValueError("weighted KL requires at least one recognized domain")
    total_weight = math.fsum(AGG_WEIGHTS[group] for group in groups)
    return (
        math.fsum(AGG_WEIGHTS[group] * math.fsum(means) / len(means) for group, means in groups.items())
        / total_weight
    )


def load_batches(
    processor: Any,
    path: str | Path,
    limit: int | None = None,
) -> list[tuple[str, torch.Tensor]]:
    batches: list[tuple[str, torch.Tensor]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            msgs = row.get("messages") or row.get("conversations")
            domain = row.get("domain", "all")
            text = processor.apply_chat_template(msgs, tokenize=False)
            enc = processor.tokenizer(
                text,
                max_length=MAX_LEN,
                truncation=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
            batches.append((domain, enc["input_ids"]))
            if limit and len(batches) >= limit:
                break
    return batches


class FixedInputMTP(nn.Module):
    """Thin fusion wrapper around Qwen decoder/norm/RoPE modules for Native MTP evaluation."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        config = copy.deepcopy(config)
        config.num_hidden_layers = 1
        config.layer_types = ["full_attention"]
        config._attn_implementation = "eager"
        h, eps = config.hidden_size, config.rms_norm_eps
        self.fc = nn.Linear(2 * h, h, bias=False)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(h, eps=eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(h, eps=eps)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, layer_idx=0)])
        self.norm = Qwen3_5RMSNorm(h, eps=eps)
        self.rotary = Qwen3_5TextRotaryEmbedding(config)

    def forward(
        self,
        hidden: torch.Tensor,
        embedded: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if hidden.shape != embedded.shape or positions.shape != (3, hidden.shape[0], hidden.shape[1]):
            raise ValueError("MTP hidden/embedding/position alignment mismatch")
        x = self.fc(
            torch.cat([self.pre_fc_norm_embedding(embedded), self.pre_fc_norm_hidden(hidden)], dim=-1)
        )
        length = x.shape[1]
        mask = torch.zeros((1, 1, length, length), device=x.device, dtype=x.dtype)
        mask.masked_fill_(
            torch.ones((length, length), device=x.device, dtype=torch.bool).triu(1), float("-inf")
        )
        x = self.layers[0](
            x,
            position_embeddings=self.rotary(x, positions),
            attention_mask=mask,
            past_key_values=None,
            use_cache=False,
        )
        return self.norm(x)


def align_mtp(
    ids: torch.Tensor,
    postnorm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 3 or postnorm.shape[:2] != ids.shape:
        raise ValueError("requires one unpadded sequence >=3 tokens with matching postnorm states")
    positions = torch.arange(ids.shape[1] - 2, device=ids.device).view(1, 1, -1).expand(3, 1, -1)
    return postnorm[:, :-2].detach().clone(), ids[:, 1:-1].clone(), positions, ids[:, 2:].clone()


def compare_logits(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.shape[:-1] != labels.shape:
        raise ValueError("logits/label shape mismatch")
    p, q = F.log_softmax(reference.float(), -1), F.log_softmax(candidate.float(), -1)
    if not torch.isfinite(p).all() or not torch.isfinite(q).all():
        raise ValueError("nonfinite logits")
    kl = (p.exp() * (p - q)).sum(-1).flatten()
    tv = (0.5 * (p.exp() - q.exp()).abs().sum(-1)).flatten()
    return {
        "kl": kl.cpu().tolist(),
        "tv": tv.cpu().tolist(),
        "same_top1": int((reference.argmax(-1) == candidate.argmax(-1)).sum()),
        "reference_nll_sum": float(-p.gather(-1, labels.unsqueeze(-1)).sum()),
        "candidate_nll_sum": float(-q.gather(-1, labels.unsqueeze(-1)).sum()),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty probe")
    kl = torch.tensor([x for r in rows for x in r["kl"]], dtype=torch.float64)
    tv = torch.tensor([x for r in rows for x in r["tv"]], dtype=torch.float64)
    if not len(kl):
        raise ValueError("empty scored tokens")
    return {
        "rows": len(rows),
        "tokens": len(kl),
        "mean_kl": kl.mean().item(),
        "median_kl": kl.median().item(),
        "p99_kl": torch.quantile(kl, 0.99).item(),
        "max_kl": kl.max().item(),
        "mean_tv": tv.mean().item(),
        "top1_agreement": sum(r["same_top1"] for r in rows) / len(kl),
        "reference_nll": sum(r["reference_nll_sum"] for r in rows) / len(kl),
        "candidate_nll": sum(r["candidate_nll_sum"] for r in rows) / len(kl),
    }


def feature_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("invalid encoder outputs")
    a_f, b_f = a.float(), b.float()
    error = (a_f - b_f).square().sum().item()
    energy = a_f.square().sum().item()
    if energy == 0:
        raise ValueError("zero-energy encoder reference; relative error is undefined")
    return {
        "elements": a.numel(),
        "squared_error": error,
        "reference_energy": energy,
        "relative_rmse": (error / energy) ** 0.5,
    }


def encode_image_row(
    processor: Any,
    data_dir: str | Path,
    row: dict[str, Any],
    max_pixels: int,
    blank: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    image_path = Path(data_dir) / row["image"]
    if sha256(image_path) != row["image_sha256"]:
        raise ValueError("image checksum mismatch")
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    if blank:
        image = Image.new("RGB", image.size, (127, 127, 127))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": row["question"] + "\nAnswer with only the answer, without explanation.",
                },
            ],
        }
    ]
    prefix = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    kwargs = dict(
        images=[image],
        return_tensors="pt",
        add_special_tokens=False,
        min_pixels=128 * 128,
        max_pixels=max_pixels,
    )
    prompt = processor(text=[prefix], **kwargs)
    full = processor(text=[prefix + row["answer"]], **kwargs)
    count = prompt["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][:, :count], prompt["input_ids"]):
        raise ValueError("answer changed prefix tokenization")
    labels = full["input_ids"][:, count:].clone()
    if not labels.numel():
        raise ValueError("empty answer tokens")
    keep = torch.arange(count - 1, full["input_ids"].shape[1] - 1)
    return full, keep, labels


def device_inputs(inputs: dict[str, torch.Tensor], device: str | torch.device) -> dict[str, torch.Tensor]:
    return {
        k: v.to(device=device, dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
        for k, v in inputs.items()
    }


def select_text_rows(path: str | Path, per_domain: int) -> list[tuple[int, str, list[dict[str, Any]]]]:
    counts: Counter[str] = Counter()
    result: list[tuple[int, str, list[dict[str, Any]]]] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        domain = row.get("domain", "all")
        if counts[domain] < per_domain:
            result.append((index, domain, row.get("messages") or row["conversations"]))
            counts[domain] += 1
    if not result or min(counts.values()) != per_domain:
        raise ValueError("insufficient text fixtures per domain")
    return result


def _select_v2_split(
    candidates: list[tuple[str, str, int, bytes, str, list[str]]],
    seed_tag: str,
    budgets: dict[str, int],
    seen: set[str],
    min_dim: int,
) -> list[tuple[str, str, int, str, bytes, int, int, str, list[str]]]:
    ranked = []
    for dom, cat, idx, blob, q, ans in candidates:
        digest = hashlib.sha256(blob).hexdigest()
        rk = hashlib.sha256(f"{seed_tag}:{dom}:{digest}:{idx}".encode()).hexdigest()
        ranked.append((rk, dom, cat, idx, digest, blob, q, ans))
    counts: Counter[str] = Counter()
    picked = []
    for _, dom, cat, idx, digest, blob, q, ans in sorted(ranked):
        if digest in seen or counts[dom] >= budgets.get(dom, 0):
            continue
        with Image.open(io.BytesIO(blob)) as image:
            width, height = image.size
            image.verify()
        if min(width, height) < min_dim:
            continue
        counts[dom] += 1
        seen.add(digest)
        picked.append((dom, cat, idx, digest, blob, width, height, q, ans))
    if dict(counts) != budgets:
        raise ValueError(f"incomplete v2 {seed_tag} selection: {dict(counts)} != {budgets}")
    return picked


def prepare_v2_vision_split(output: str | Path, split: str = "probe") -> Path:
    if split not in ("probe", "calibration", "guard"):
        raise ValueError(f"unsupported v2 split: {split!r}")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()

    sources = SOURCES_V2_CORE + (SOURCES_V2_GUARD if split == "guard" else ())
    provenance = []
    source_paths: dict[str, str] = {}
    for name, repo, revision, file in sources:
        path = hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision, filename=file)
        provenance.append(
            {"domain": name, "repo": repo, "revision": revision, "file": file, "sha256": sha256(path)}
        )
        source_paths[name] = path

    in_dist: list[tuple[str, str, int, bytes, str, list[str]]] = []
    guard_cands: list[tuple[str, str, int, bytes, str, list[str]]] = []

    for idx, row in enumerate(pq.ParquetFile(source_paths["chartqa"]).read().to_pylist()):
        blob = row["image"]["bytes"]
        ans = [str(a).strip() for a in (row["label"] or []) if isinstance(a, str) and str(a).strip()]
        q = (row["query"] or "").strip()
        if ans and q:
            dom = "chartqa_human" if str(row["human_or_machine"]) == "0" else "chartqa_machine"
            in_dist.append((dom, f"chartqa:{row['human_or_machine']}", idx, blob, q, ans))

    for idx, row in enumerate(pq.ParquetFile(source_paths["textvqa"]).read().to_pylist()):
        blob = row["image"]["bytes"]
        ans = [str(a).strip() for a in (row["answers"] or []) if isinstance(a, str) and str(a).strip()]
        q = (row["question"] or "").strip()
        if ans and q:
            in_dist.append(("textvqa", "textvqa:all", idx, blob, q, ans))

    for idx, row in enumerate(pq.ParquetFile(source_paths["ocrbench"]).read().to_pylist()):
        ds = row["dataset"]
        blob = row["image"]["bytes"]
        ans = [str(a).strip() for a in (row["answer"] or []) if isinstance(a, str) and str(a).strip()]
        q = (row["question"] or "").strip()
        if not (ans and q):
            continue
        if ds in OCR_V2_BUCKET_MAP:
            in_dist.append((OCR_V2_BUCKET_MAP[ds], f"ocrbench:{ds}", idx, blob, q, ans))
        elif ds in ("WordArt", "HME100k"):
            guard_cands.append(("ocrbench_artistic_math", f"ocrbench:{ds}", idx, blob, q, ans))

    for idx, row in enumerate(pq.ParquetFile(source_paths["mmstar"]).read().to_pylist()):
        cat = row["category"]
        if cat not in MMSTAR_V2_BUCKET_MAP:
            continue
        img = row["image"]
        blob = img if isinstance(img, bytes) else img["bytes"]
        ans = [str(row["answer"]).strip()] if row.get("answer") and str(row["answer"]).strip() else []
        q = (row["question"] or "").strip()
        if ans and q:
            in_dist.append((MMSTAR_V2_BUCKET_MAP[cat], f"mmstar:{cat}", idx, blob, q, ans))

    seen: set[str] = set()
    probe_budgets = {d: 8 for d in V2_IN_DIST_DOMAINS}
    calib_budgets = {d: 10 for d in V2_IN_DIST_DOMAINS}
    probe_picked = _select_v2_split(in_dist, "vision-probe-v2-0", probe_budgets, seen, min_dim=128)

    if split == "probe":
        selected = probe_picked
        selection_desc = (
            "suite v2 probe: hash seed vision-probe-v2-0; 12 domains x 8 images (96 total, min_dim>=128)"
        )
    else:
        calib_picked = _select_v2_split(in_dist, "vision-calib-v2-1", calib_budgets, seen, min_dim=128)
        if split == "calibration":
            selected = calib_picked
            selection_desc = (
                "suite v2 calibration: hash seed vision-calib-v2-1 (disjoint from vision-probe-v2-0); "
                "12 domains x 10 images (120 total, min_dim>=128)"
            )
        else:
            for idx, row in enumerate(pq.ParquetFile(source_paths["screenspot"]).read().to_pylist()):
                img = row["image"]
                blob = img if isinstance(img, bytes) else img["bytes"]
                src = str(row["data_source"] or "").strip()
                dtype = str(row["data_type"] or "").strip()
                inst = str(row["instruction"] or "").strip()
                if not (src and dtype and inst):
                    continue
                dom = f"screenspot_{src}_{dtype}" if src in ("macos", "windows") else "screenspot_ios"
                q = (
                    f"In this {src} interface screenshot, is the UI element for '{inst}' "
                    f"a text element or an icon?\nOptions: text, icon"
                )
                guard_cands.append((dom, f"screenspot:{src}:{dtype}", idx, blob, q, [dtype]))
            guard_budgets = {d: 8 for d in V2_GUARD_DOMAINS}
            selected = _select_v2_split(guard_cands, "vision-guard-v2-0", guard_budgets, seen, min_dim=96)
            selection_desc = (
                "suite v2 guard: hash seed vision-guard-v2-0 (disjoint from probe and calibration); "
                "6 OOD domains x 8 images (48 total, min_dim>=96)"
            )

    records = []
    for dom, cat, idx, digest, blob, width, height, q, ans in selected:
        filename = f"images/{digest}.image"
        (out / filename).write_bytes(blob)
        records.append(
            {
                "domain": dom,
                "category": cat,
                "source_row": idx,
                "image": filename,
                "image_sha256": digest,
                "width": width,
                "height": height,
                "question": q,
                "answers": ans,
                "answer": Counter(ans).most_common(1)[0][0],
            }
        )
    log_event("SOURCE_SELECTED", suite="v2", split=split, rows=len(records))
    rows_path = out / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    manifest = {
        "suite": "v2",
        "split": split,
        "sources": provenance,
        "rows": len(records),
        "rows_sha256": sha256(rows_path),
        "selection": selection_desc,
        "calibration": split == "calibration",
        "guard": split == "guard",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


SOURCES_IMAGE_GEN_V1_CORE = (
    (
        "oneig_en",
        "OneIG-Bench/OneIG-Bench",
        "f80e6317806bf0b1d92da3113633a62451b0fd4f",
        "OneIG-Bench.json",
    ),
    (
        "oneig_zh",
        "OneIG-Bench/OneIG-Bench",
        "f80e6317806bf0b1d92da3113633a62451b0fd4f",
        "OneIG-Bench-ZH.json",
    ),
    (
        "dpg_bench",
        "Jialuo21/DPG-Bench",
        "f7f1626d7a5f0a4dc062c3d9715959325d5039bd",
        "data/test-00000-of-00001.parquet",
    ),
    (
        "parti_prompts",
        "nateraw/parti-prompts",
        "944b156abfdad7627c3221b5ec4f6a6fb060a197",
        "PartiPrompts.tsv",
    ),
    (
        "gedit_shard0",
        "stepfun-ai/GEdit-Bench",
        "50766778e2a737474c7e9bdf84cdce82c3ea3f4f",
        "data-00000-of-00009.arrow",
    ),
    (
        "gedit_shard2",
        "stepfun-ai/GEdit-Bench",
        "50766778e2a737474c7e9bdf84cdce82c3ea3f4f",
        "data-00002-of-00009.arrow",
    ),
    (
        "gedit_shard4",
        "stepfun-ai/GEdit-Bench",
        "50766778e2a737474c7e9bdf84cdce82c3ea3f4f",
        "data-00004-of-00009.arrow",
    ),
    (
        "gedit_shard7",
        "stepfun-ai/GEdit-Bench",
        "50766778e2a737474c7e9bdf84cdce82c3ea3f4f",
        "data-00007-of-00009.arrow",
    ),
)

SOURCES_IMAGE_GEN_V1_GUARD = (
    (
        "magicbrush_dev0",
        "osunlp/MagicBrush",
        "6c254560879df7851233ea0fd2ae2a62764d0ef2",
        "data/dev-00000-of-00004-f147d414270a90e1.parquet",
    ),
    (
        "gedit_shard8",
        "stepfun-ai/GEdit-Bench",
        "50766778e2a737474c7e9bdf84cdce82c3ea3f4f",
        "data-00008-of-00009.arrow",
    ),
)

IMAGE_GEN_V1_IN_DIST_DOMAINS = (
    "oneig_en_text",
    "oneig_en_knowledge",
    "oneig_en_portrait",
    "oneig_en_general",
    "oneig_en_anime",
    "dpg_dense_en",
    "parti_writing_complex",
    "oneig_zh_text",
    "oneig_zh_multilingual",
    "oneig_zh_knowledge",
    "oneig_zh_portrait",
    "oneig_zh_general_anime",
    "gedit_text_change",
    "gedit_subject_style",
    "gedit_human_motion",
    "gedit_scene_color",
)

IMAGE_GEN_V1_GUARD_DOMAINS = (
    "parti_world_knowledge",
    "parti_arts_illustrations",
    "parti_vehicles_artifacts",
    "parti_animals_food",
    "magicbrush_edit",
    "gedit_tone_transfer",
)

IMAGE_GEN_V1_ASPECT_RATIOS: tuple[tuple[int, int], ...] = (
    (1024, 1024),
    (1344, 768),
    (768, 1344),
    (1152, 896),
    (896, 1152),
)

_ONEIG_EN_MAP = {
    "Text_Rendering": "oneig_en_text",
    "Knowledge_Reasoning": "oneig_en_knowledge",
    "Portrait": "oneig_en_portrait",
    "General_Object": "oneig_en_general",
    "Anime_Stylization": "oneig_en_anime",
}

_ONEIG_ZH_MAP = {
    "Text_Rendering": "oneig_zh_text",
    "Multilingualism": "oneig_zh_multilingual",
    "Knowledge_Reasoning": "oneig_zh_knowledge",
    "Portrait": "oneig_zh_portrait",
    "General_Object": "oneig_zh_general_anime",
    "Anime_Stylization": "oneig_zh_general_anime",
}

_PARTI_IN_DIST_CHALLENGES = frozenset(
    {
        "Writing & Symbols",
        "Complex",
        "Linguistic Structures",
        "Properties & Positioning",
    }
)


def _calculate_32_aligned_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    width = math.sqrt(float(target_area) * float(ratio))
    height = width / float(ratio)
    return max(32, int(round(width / 32.0) * 32)), max(32, int(round(height / 32.0) * 32))


def _read_arrow_stream(path: str | Path) -> list[dict[str, Any]]:
    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_stream(source).read_all().to_pylist()


def _select_image_gen_split(
    candidates: list[tuple[str, str, int, bytes | None, str, str]],
    seed_tag: str,
    budgets: dict[str, int],
    domain_order: tuple[str, ...],
    seen: set[str],
    min_dim: int = 128,
) -> list[dict[str, Any]]:
    by_dom: dict[str, list[tuple[str, str, int, str, bytes | None, int | None, int | None, str, str]]] = (
        defaultdict(list)
    )
    for dom, cat, idx, blob, prompt, lang in candidates:
        if dom not in budgets:
            continue
        p_clean = prompt.strip()
        if len(p_clean) < 8:
            continue
        if blob is not None:
            img_digest = hashlib.sha256(blob).hexdigest()
            if img_digest in seen:
                continue
            try:
                with Image.open(io.BytesIO(blob)) as image:
                    src_w, src_h = image.size
                    image.verify()
            except (UnidentifiedImageError, OSError, SyntaxError):
                continue
            if min(src_w, src_h) < min_dim:
                continue
            digest = img_digest
        else:
            src_w, src_h = None, None
            digest = hashlib.sha256(p_clean.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
        rk = hashlib.sha256(f"{seed_tag}:{dom}:{lang}:{digest}:{idx}".encode()).hexdigest()
        by_dom[dom].append((rk, cat, idx, digest, blob, src_w, src_h, p_clean, lang))

    picked_records: list[dict[str, Any]] = []
    for d_idx, dom in enumerate(domain_order):
        budget = budgets.get(dom, 0)
        pool = sorted(by_dom.get(dom, []), key=lambda x: x[0])
        langs_present = {item[8] for item in pool}
        is_bilingual = "en" in langs_present and "zh" in langs_present

        dom_picked: list[tuple[str, str, int, str, bytes | None, int | None, int | None, str, str]] = []
        if is_bilingual:
            en_pool = [it for it in pool if it[8] == "en"]
            zh_pool = [it for it in pool if it[8] == "zh"]
            first_pool, second_pool = (en_pool, zh_pool) if d_idx % 2 == 0 else (zh_pool, en_pool)
            i1, i2 = 0, 0
            while len(dom_picked) < budget and (i1 < len(first_pool) or i2 < len(second_pool)):
                take_first = (len(dom_picked) % 2 == 0 and i1 < len(first_pool)) or i2 >= len(second_pool)
                cand = first_pool[i1] if take_first else second_pool[i2]
                if take_first:
                    i1 += 1
                else:
                    i2 += 1
                if cand[3] in seen:
                    continue
                seen.add(cand[3])
                dom_picked.append(cand)
        else:
            for cand in pool:
                if len(dom_picked) >= budget:
                    break
                if cand[3] in seen:
                    continue
                seen.add(cand[3])
                dom_picked.append(cand)

        if len(dom_picked) != budget:
            raise ValueError(
                f"incomplete image-gen-v1 {seed_tag} selection for {dom}: {len(dom_picked)} != {budget}"
            )

        for item_idx, (_, cat, idx, digest, blob, src_w, src_h, p_clean, lang) in enumerate(dom_picked):
            if blob is not None and src_w is not None and src_h is not None:
                width, height = _calculate_32_aligned_dimensions(1024 * 1024, src_w / src_h)
                mode = "ti2i"
            else:
                width, height = IMAGE_GEN_V1_ASPECT_RATIOS[
                    (d_idx + item_idx) % len(IMAGE_GEN_V1_ASPECT_RATIOS)
                ]
                mode = "t2i"
            picked_records.append(
                {
                    "domain": dom,
                    "category": cat,
                    "mode": mode,
                    "lang": lang,
                    "source_row": idx,
                    "prompt": p_clean,
                    "prompt_sha256": hashlib.sha256(p_clean.encode("utf-8")).hexdigest(),
                    "image": f"images/{digest}.image" if blob is not None else None,
                    "image_sha256": digest if blob is not None else None,
                    "source_width": src_w,
                    "source_height": src_h,
                    "width": width,
                    "height": height,
                    "_blob": blob,
                }
            )
    return picked_records


def prepare_image_gen_split(output: str | Path, split: str = "probe") -> Path:
    """Materialize disjoint `image-gen-{calib,probe,guard}-v1` fixtures for Qwen-Image-2.1."""
    if split not in ("probe", "calibration", "guard"):
        raise ValueError(f"unsupported image-gen-v1 split: {split!r}")
    out = Path(output)
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    try:
        return _populate_image_gen_split(out, split=split)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise


def _populate_image_gen_split(out: Path, split: str) -> Path:
    (out / "images").mkdir()

    sources = SOURCES_IMAGE_GEN_V1_CORE + (SOURCES_IMAGE_GEN_V1_GUARD if split == "guard" else ())
    provenance = []
    source_paths: dict[str, str] = {}
    for name, repo, revision, file in sources:
        path = hf_hub_download(repo_id=repo, repo_type="dataset", revision=revision, filename=file)
        provenance.append(
            {"domain": name, "repo": repo, "revision": revision, "file": file, "sha256": sha256(path)}
        )
        source_paths[name] = path

    in_dist: list[tuple[str, str, int, bytes | None, str, str]] = []
    guard_cands: list[tuple[str, str, int, bytes | None, str, str]] = []

    for idx, row in enumerate(json.loads(Path(source_paths["oneig_en"]).read_text(encoding="utf-8"))):
        cat = str(row.get("category") or "").strip()
        prompt = str(row.get("prompt_en") or "").strip()
        if cat in _ONEIG_EN_MAP and prompt:
            in_dist.append((_ONEIG_EN_MAP[cat], f"oneig_en:{cat}", idx, None, prompt, "en"))

    for idx, row in enumerate(json.loads(Path(source_paths["oneig_zh"]).read_text(encoding="utf-8"))):
        cat = str(row.get("category") or "").strip()
        prompt = str(row.get("prompt_cn") or "").strip()
        if cat in _ONEIG_ZH_MAP and prompt:
            in_dist.append((_ONEIG_ZH_MAP[cat], f"oneig_zh:{cat}", idx, None, prompt, "zh"))

    for idx, row in enumerate(pq.ParquetFile(source_paths["dpg_bench"]).read().to_pylist()):
        prompt = str(row.get("prompt") or "").strip()
        item_id = str(row.get("item_id") or idx)
        if prompt:
            in_dist.append(("dpg_dense_en", f"dpg_bench:{item_id}", idx, None, prompt, "en"))

    with Path(source_paths["parti_prompts"]).open(encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f, delimiter="\t")):
            prompt = str(row.get("Prompt") or "").strip()
            cat = str(row.get("Category") or "").strip()
            chal = str(row.get("Challenge") or "").strip()
            if not prompt:
                continue
            if chal in _PARTI_IN_DIST_CHALLENGES:
                in_dist.append(("parti_writing_complex", f"parti:{cat}:{chal}", idx, None, prompt, "en"))
            else:
                if cat in ("World Knowledge", "Outdoor Scenes"):
                    guard_cands.append(
                        ("parti_world_knowledge", f"parti:{cat}:{chal}", idx, None, prompt, "en")
                    )
                elif cat in ("Arts", "Illustrations"):
                    guard_cands.append(
                        ("parti_arts_illustrations", f"parti:{cat}:{chal}", idx, None, prompt, "en")
                    )
                elif cat in ("Vehicles", "Artifacts"):
                    guard_cands.append(
                        ("parti_vehicles_artifacts", f"parti:{cat}:{chal}", idx, None, prompt, "en")
                    )
                elif cat in ("Animals", "Food & Beverage"):
                    guard_cands.append(("parti_animals_food", f"parti:{cat}:{chal}", idx, None, prompt, "en"))

    gedit_shards = (
        ("gedit_shard0", {"background_change": "gedit_scene_color", "color_alter": "gedit_scene_color"}),
        ("gedit_shard2", {"ps_human": "gedit_human_motion", "motion_change": "gedit_human_motion"}),
        ("gedit_shard4", {"subject-add": "gedit_subject_style", "style_change": "gedit_subject_style"}),
        ("gedit_shard7", {"text_change": "gedit_text_change"}),
    )
    for shard_key, task_map in gedit_shards:
        for idx, row in enumerate(_read_arrow_stream(source_paths[shard_key])):
            tt = str(row.get("task_type") or "").strip()
            if tt not in task_map:
                continue
            inst = str(row.get("instruction") or "").strip()
            raw_lang = str(row.get("instruction_language") or "en").strip().lower()
            lang = "zh" if raw_lang in ("cn", "zh") else "en"
            img_obj = row.get("input_image_raw") or row.get("input_image")
            blob = img_obj["bytes"] if isinstance(img_obj, dict) else img_obj
            if inst and isinstance(blob, bytes):
                in_dist.append((task_map[tt], f"gedit:{tt}:{lang}", idx, blob, inst, lang))

    seen: set[str] = set()
    probe_budgets = {d: 4 for d in IMAGE_GEN_V1_IN_DIST_DOMAINS}
    calib_budgets = {d: 6 for d in IMAGE_GEN_V1_IN_DIST_DOMAINS}
    probe_picked = _select_image_gen_split(
        in_dist,
        "image-gen-probe-v1-0",
        probe_budgets,
        IMAGE_GEN_V1_IN_DIST_DOMAINS,
        seen,
        min_dim=128,
    )

    if split == "probe":
        selected = probe_picked
        selection_desc = (
            "suite image-gen-v1 probe: hash seed image-gen-probe-v1-0; "
            "16 domains (12 T2I + 4 TI2I, EN+ZH, 5 aspect ratios) x 4 rows (64 total)"
        )
    else:
        calib_picked = _select_image_gen_split(
            in_dist,
            "image-gen-calib-v1-1",
            calib_budgets,
            IMAGE_GEN_V1_IN_DIST_DOMAINS,
            seen,
            min_dim=128,
        )
        if split == "calibration":
            selected = calib_picked
            selection_desc = (
                "suite image-gen-v1 calibration: hash seed image-gen-calib-v1-1 (disjoint from probe); "
                "16 domains (12 T2I + 4 TI2I, EN+ZH, 5 aspect ratios) x 6 rows (96 total)"
            )
        else:
            for idx, row in enumerate(pq.ParquetFile(source_paths["magicbrush_dev0"]).read().to_pylist()):
                inst = str(row.get("instruction") or "").strip()
                img_obj = row.get("source_img")
                blob = img_obj["bytes"] if isinstance(img_obj, dict) else img_obj
                if inst and isinstance(blob, bytes):
                    guard_cands.append(("magicbrush_edit", "magicbrush:dev", idx, blob, inst, "en"))
            for idx, row in enumerate(_read_arrow_stream(source_paths["gedit_shard8"])):
                tt = str(row.get("task_type") or "").strip()
                if tt != "tone_transfer":
                    continue
                inst = str(row.get("instruction") or "").strip()
                raw_lang = str(row.get("instruction_language") or "en").strip().lower()
                lang = "zh" if raw_lang in ("cn", "zh") else "en"
                img_obj = row.get("input_image_raw") or row.get("input_image")
                blob = img_obj["bytes"] if isinstance(img_obj, dict) else img_obj
                if inst and isinstance(blob, bytes):
                    guard_cands.append(("gedit_tone_transfer", f"gedit:{tt}:{lang}", idx, blob, inst, lang))
            guard_budgets = {d: 4 for d in IMAGE_GEN_V1_GUARD_DOMAINS}
            selected = _select_image_gen_split(
                guard_cands,
                "image-gen-guard-v1-0",
                guard_budgets,
                IMAGE_GEN_V1_GUARD_DOMAINS,
                seen,
                min_dim=128,
            )
            selection_desc = (
                "suite image-gen-v1 guard: hash seed image-gen-guard-v1-0 (disjoint from probe and calibration); "
                "6 OOD domains (4 T2I + 2 TI2I) x 4 rows (24 total)"
            )

    records = []
    for item in selected:
        rec = dict(item)
        blob = rec.pop("_blob")
        if blob is not None and rec["image"] is not None:
            (out / rec["image"]).write_bytes(blob)
        records.append(rec)

    log_event("SOURCE_SELECTED", suite="image-gen-v1", split=split, rows=len(records))
    rows_path = out / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    manifest = {
        "suite": "image-gen-v1",
        "split": split,
        "sources": provenance,
        "rows": len(records),
        "t2i_rows": sum(1 for r in records if r["mode"] == "t2i"),
        "ti2i_rows": sum(1 for r in records if r["mode"] == "ti2i"),
        "en_rows": sum(1 for r in records if r["lang"] == "en"),
        "zh_rows": sum(1 for r in records if r["lang"] == "zh"),
        "rows_sha256": sha256(rows_path),
        "selection": selection_desc,
        "calibration": split == "calibration",
        "guard": split == "guard",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out
