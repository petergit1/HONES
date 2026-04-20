from __future__ import annotations

"""Dataset helpers for the unified multi-task benchmark.

The code assumes the compact JSON structure described in README.md and supports
optional image-disjoint split files for discovery/test evaluation.
"""

import json
import os
import random
import re
from pathlib import Path
from typing import List, Optional, Set

from PIL import Image


def _image_key(item: dict) -> str:
    """Return a stable image identifier used for split filtering."""
    info = item.get("image_info", {})
    return str(info.get("id") or info.get("image_id") or info.get("file_name") or "")


def load_split_ids(path: str | None) -> Optional[Set[str]]:
    """Load optional split IDs.

    Accepted formats are either a plain list or a dict containing one of:
    ``ids``, ``image_ids``, ``file_names``, or ``images``.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        for key in ["ids", "image_ids", "file_names", "images"]:
            if key in obj:
                obj = obj[key]
                break
    return {str(x) for x in obj}


def load_items(data_file: str, task: str, limit: int | None = None, split_ids: Set[str] | None = None) -> List[dict]:
    """Load valid examples for a task from the unified JSON file."""
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = []
    for item in data.get("data", []):
        file_name = item.get("image_info", {}).get("file_name")
        if split_ids is not None and _image_key(item) not in split_ids and file_name not in split_ids:
            continue
        if _valid_item(item, task):
            out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def _valid_item(item: dict, task: str) -> bool:
    """Check whether an item has the annotation fields required by the task."""
    task = task.lower()
    if task == "vqa":
        return bool(item.get("vqa", {}).get("question", "").strip()) and bool(item.get("vqa", {}).get("answer", "").strip())
    if task == "ocr":
        return bool(item.get("ocr", {}).get("text", "").strip())
    if task in {"caption", "retrieval"}:
        caps = item.get("captions", [])
        return isinstance(caps, list) and any(c.get("caption", "").strip() for c in caps)
    raise ValueError(f"Unknown task: {task}")


def image_path(image_dir: str, item: dict) -> str:
    """Return the absolute image path for an item."""
    fn = item.get("image_info", {}).get("file_name")
    if not fn:
        raise KeyError("item['image_info']['file_name'] is missing")
    return os.path.join(image_dir, fn)


def open_image(image_dir: str, item: dict) -> Image.Image:
    """Open an image as RGB for both LLaVA and Qwen processors."""
    return Image.open(image_path(image_dir, item)).convert("RGB")


def references_for_caption(item: dict, max_refs: int = 5) -> list[str]:
    """Return up to five reference captions."""
    refs = [c.get("caption", "").strip() for c in item.get("captions", []) if c.get("caption", "").strip()]
    return refs[:max_refs]


def vqa_gold_answers(item: dict) -> list[str]:
    """Return all VQA references when available; otherwise fallback to one answer."""
    vqa = item.get("vqa", {})
    for key in ["answers", "answer_list", "gt_answers"]:
        vals = vqa.get(key)
        if isinstance(vals, list) and vals:
            out = []
            for a in vals:
                if isinstance(a, dict):
                    a = a.get("answer", "")
                if str(a).strip():
                    out.append(str(a).strip())
            if out:
                return out
    ans = str(vqa.get("answer", "")).strip()
    return [ans] if ans else []


def build_retrieval_items(items: list[dict], n_neg: int = 49, seed: int = 3407, limit: int | None = None) -> list[dict]:
    """Build 1:49 image-to-text retrieval items from caption annotations.

    This follows the pairwise re-ranking protocol: one positive caption
    plus 49 hard negatives from other images. The model later scores each pair
    with the next-token Yes-vs-No logit margin and NDCG@5 is reported.
    """
    rng = random.Random(seed)
    all_caps = []
    for it in items:
        all_caps.extend(references_for_caption(it, max_refs=5))
    all_caps = list(dict.fromkeys(all_caps))

    out = []
    for it in items:
        refs = references_for_caption(it, max_refs=5)
        if not refs:
            continue
        pos = rng.choice(refs)
        neg_pool = [c for c in all_caps if c not in set(refs)]
        if len(neg_pool) < n_neg:
            continue
        negs = _hard_negative_sample(pos, neg_pool, n_neg, rng)
        candidates = [pos] + negs
        rng.shuffle(candidates)
        out.append({
            "image_info": it["image_info"],
            "candidates": candidates,
            "positive_index": candidates.index(pos),
        })
        if limit is not None and len(out) >= limit:
            break
    return out


def _hard_negative_sample(pos: str, neg_pool: list[str], n_neg: int, rng: random.Random) -> list[str]:
    """hard-negative mining based on word overlap.
    """
    def words(s: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", s.lower()))

    pos_w = words(pos)
    sample_pool = rng.sample(neg_pool, min(len(neg_pool), max(n_neg * 50, n_neg)))
    scored = sorted(sample_pool, key=lambda c: len(pos_w & words(c)), reverse=True)
    chosen = scored[:n_neg]
    if len(chosen) < n_neg:
        rest = [c for c in neg_pool if c not in set(chosen)]
        chosen += rng.sample(rest, min(len(rest), n_neg - len(chosen)))
    return chosen[:n_neg]
