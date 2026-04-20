from __future__ import annotations

"""Task evaluation functions used for head localization and masking evaluation."""

import numpy as np
import torch

from .data import open_image, references_for_caption, vqa_gold_answers
from .metrics import anls, bleu4, ndcg_at_k_from_scores, vqa_accuracy
from .prompts import build_prompt, target_text


def clean_generation(text: str) -> str:
    """Remove common assistant prefixes and keep the first generated line."""
    text = (text or "").strip()
    for prefix in ["Answer:", "A:", "assistant:", "Assistant:", "Caption:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    return text.split("\n")[0].strip().strip('"').strip("'")


@torch.no_grad()
def evaluate_task(vlm, task: str, items: list[dict], image_dir: str, max_new_tokens: int = 24, ndcg_k: int = 5) -> dict:
    """Evaluate a VLM on one task using the paper-aligned metric.

    This scalar utility is the implementation counterpart of P_t(x, y; theta).
    Higher values always mean better task performance.
    """
    task = task.lower()
    if task == "retrieval":
        return evaluate_retrieval(vlm, items, image_dir, ndcg_k=ndcg_k)

    preds, refs_for_bleu = [], []
    scores = []
    for item in items:
        try:
            image = open_image(image_dir, item)
            pred = clean_generation(vlm.generate_text(image, build_prompt(task, item), max_new_tokens=max_new_tokens))
            if task == "vqa":
                scores.append(vqa_accuracy(pred, vqa_gold_answers(item)))
            elif task == "ocr":
                scores.append(anls(pred, target_text(task, item)))
            elif task == "caption":
                preds.append(pred)
                refs_for_bleu.append(references_for_caption(item))
            else:
                raise ValueError(task)
        except Exception:
            continue

    if task == "caption":
        metric = bleu4(preds, refs_for_bleu) if preds else 0.0
        return {"metric": metric, "metric_name": "BLEU-4", "n": len(preds)}
    return {"metric": float(np.mean(scores)) if scores else 0.0, "metric_name": "Accuracy" if task == "vqa" else "ANLS", "n": len(scores)}


@torch.no_grad()
def evaluate_retrieval(vlm, items: list[dict], image_dir: str, ndcg_k: int = 5) -> dict:
    """Evaluate pairwise image-to-text retrieval with NDCG@k.

    Each candidate caption is converted into a Yes/No verification prompt. The
    score is logit(Yes) - logit(No), and candidates are ranked by this score.
    """
    vals = []
    yes_id = _first_token_id(vlm.tokenizer, " Yes")
    no_id = _first_token_id(vlm.tokenizer, " No")
    for item in items:
        try:
            image = open_image(image_dir, item)
            scores = []
            for cand in item["candidates"]:
                logits = vlm.next_token_logits(image, build_prompt("retrieval", item, cand))[0]
                scores.append(float(logits[yes_id] - logits[no_id]))
            vals.append(ndcg_at_k_from_scores(scores, item["positive_index"], k=ndcg_k))
        except Exception:
            continue
    return {"metric": float(np.mean(vals)) if vals else 0.0, "metric_name": f"NDCG@{ndcg_k}", "n": len(vals)}


def _first_token_id(tokenizer, text: str) -> int:
    """Get the first token ID for a decision string such as ' Yes'."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(text.strip(), add_special_tokens=False)
    if not ids:
        raise RuntimeError(f"Cannot tokenize decision token: {text}")
    return int(ids[0])
