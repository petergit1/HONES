from __future__ import annotations

"""Dataset adapters for HONES steering.
"""

import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

from ..data import load_items, load_split_ids, build_retrieval_items


def load_steering_items(
    dataset_cfg: dict,
    task: str,
    split: str,
    limit: int | None = None,
    retrieval_negatives: int = 49,
    seed: int = 3407,
) -> list[dict]:
    """Load task items for steering.

    Parameters
    ----------
    dataset_cfg:
        Dataset entry from the YAML config.  It contains ``name``,
        ``data_file`` and ``image_dir``.  Optional ``dev_split``/``test_split``
        fields are used only by COCO-style unified files.
    task:
        One of ``vqa``, ``ocr``, ``caption`` or ``retrieval``.
    split:
        ``dev`` for learning scaling factors or ``test`` for final evaluation.
    limit:
        Optional sample cap for quick debugging.
    retrieval_negatives:
        Number of negative captions used for pairwise retrieval.
    """
    name = str(dataset_cfg.get("name", "coco")).lower()
    task = task.lower()
    data_file = dataset_cfg["data_file"]

    if name in {"coco", "unified", "hones"}:
        split_path = dataset_cfg.get(f"{split}_split")
        split_ids = load_split_ids(split_path) if split_path else None
        base = load_items(data_file, "caption" if task == "retrieval" else task, limit=limit, split_ids=split_ids)
        if task == "retrieval":
            return build_retrieval_items(base, n_neg=retrieval_negatives, seed=seed, limit=limit)
        return base

    if name == "gqa":
        return _load_gqa(data_file, limit=limit)
    if name == "textvqa":
        return _load_textvqa(data_file, limit=limit)
    if name in {"flickr30k", "flickr"}:
        items = _load_flickr30k(data_file, limit=None)
        if task == "retrieval":
            return build_retrieval_items(items, n_neg=retrieval_negatives, seed=seed, limit=limit)
        return items[:limit] if limit else items

    raise ValueError(f"Unsupported dataset adapter: {name}")


def image_dir_from_dataset(dataset_cfg: dict) -> str:
    """Return the image directory from a dataset config entry."""
    return str(dataset_cfg["image_dir"])


def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_records(obj: Any) -> list[dict]:
    """Normalize common JSON containers to a list of dict records."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ["data", "annotations", "questions", "images"]:
            if isinstance(obj.get(key), list):
                return obj[key]
    raise ValueError("Cannot find a list of records in the dataset JSON")


def _load_gqa(data_file: str, limit: int | None = None) -> list[dict]:
    """Load GQA-like records and convert them to the unified VQA fields."""
    obj = _load_json(data_file)
    records = _as_records(obj)
    out = []
    for r in records:
        q = r.get("question") or r.get("sent") or r.get("prompt")
        a = r.get("answer") or r.get("label")
        image_id = r.get("imageId") or r.get("image_id") or r.get("img_id")
        file_name = r.get("file_name") or r.get("image") or (f"{image_id}.jpg" if image_id is not None else None)
        if q and a and file_name:
            out.append({"image_info": {"file_name": str(file_name), "id": str(image_id or file_name)}, "vqa": {"question": str(q), "answer": str(a)}})
        if limit is not None and len(out) >= limit:
            break
    return out


def _load_textvqa(data_file: str, limit: int | None = None) -> list[dict]:
    """Load TextVQA-like records.
    """
    obj = _load_json(data_file)
    records = _as_records(obj)
    out = []
    for r in records:
        answers = r.get("answers") or r.get("answer") or r.get("gt_answers")
        if isinstance(answers, list):
            ans = str(answers[0]) if answers else ""
        else:
            ans = str(answers or "")
        image_id = r.get("image_id") or r.get("imageId") or r.get("img_id")
        file_name = r.get("file_name") or r.get("image") or (f"{image_id}.jpg" if image_id is not None else None)
        q = r.get("question", "Transcribe the text in the image.")
        if ans.strip() and file_name:
            out.append({
                "image_info": {"file_name": str(file_name), "id": str(image_id or file_name)},
                "ocr": {"text": ans.strip(), "question": str(q)},
                "vqa": {"question": str(q), "answer": ans.strip()},
            })
        if limit is not None and len(out) >= limit:
            break
    return out


def _load_flickr30k(data_file: str, limit: int | None = None) -> list[dict]:
    """Load Flickr30k-style caption records into unified caption fields."""
    obj = _load_json(data_file)
    records = _as_records(obj)
    out = []
    for r in records:
        file_name = r.get("file_name") or r.get("filename") or r.get("image") or r.get("image_path")
        caps = r.get("captions") or r.get("sentences") or r.get("caption")
        if isinstance(caps, str):
            caps = [caps]
        norm_caps = []
        if isinstance(caps, list):
            for c in caps:
                if isinstance(c, dict):
                    c = c.get("caption") or c.get("raw") or c.get("sentence")
                if str(c or "").strip():
                    norm_caps.append({"caption": str(c).strip()})
        if file_name and norm_caps:
            out.append({"image_info": {"file_name": str(file_name), "id": str(file_name)}, "captions": norm_caps})
        if limit is not None and len(out) >= limit:
            break
    return out
