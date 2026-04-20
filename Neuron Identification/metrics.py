from __future__ import annotations

"""Task metrics used in evaluation protocol."""

import math
import re
from typing import List

import numpy as np


def normalize_answer(s: str) -> str:
    """Normalize generated answers for exact/soft VQA matching."""
    s = str(s).lower().strip()
    s = re.sub(r"^(answer\s*:|assistant\s*:|a\s*:)", "", s).strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return " ".join(s.split())


def vqa_accuracy(pred: str, gold: str | list[str]) -> float:
    """VQA-style accuracy.

    If multiple human references are available, use the standard soft score
    min(#matching/3, 1). For compact JSON files with one answer, use normalized
    exact/containment matching.
    """
    p = normalize_answer(pred)
    if not p:
        return 0.0
    if isinstance(gold, list):
        refs = [normalize_answer(g) for g in gold if normalize_answer(g)]
        if len(refs) > 1:
            return float(min(sum(p == g for g in refs) / 3.0, 1.0))
        gold = refs[0] if refs else ""
    g = normalize_answer(str(gold))
    return float(p == g or p in g or g in p) if g else 0.0


def _levenshtein(a: str, b: str) -> int:
    """Compute edit distance without requiring an external package."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, gold: str, threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity for OCR."""
    p = re.sub(r"[^\w]", "", str(pred).lower())
    g = re.sub(r"[^\w]", "", str(gold).lower())
    if not p or not g:
        return 0.0
    dist = _levenshtein(p, g)
    score = 1.0 - dist / max(len(p), len(g))
    return float(score if score >= threshold else 0.0)


def bleu4(predictions: List[str], references: List[List[str]]) -> float:
    """BLEU-4 for captioning.
    """
    if not predictions:
        return 0.0
    try:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        from pycocoevalcap.bleu.bleu import Bleu
        gts, res = {}, {}
        for i, (p, refs) in enumerate(zip(predictions, references)):
            gts[i] = [{"caption": r} for r in refs if r.strip()]
            res[i] = [{"caption": p}]
        tokenizer = PTBTokenizer()
        gts_t = tokenizer.tokenize(gts)
        res_t = tokenizer.tokenize(res)
        score, _ = Bleu(4).compute_score(gts_t, res_t)
        return float(score[3])
    except Exception:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        smoothie = SmoothingFunction().method1
        refs_tok = [[r.lower().split() for r in refs if r.strip()] for refs in references]
        preds_tok = [p.lower().split() for p in predictions]
        return float(corpus_bleu(refs_tok, preds_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))


def ndcg_at_k_from_scores(scores: List[float], positive_index: int, k: int = 5) -> float:
    """NDCG@k with one relevant positive caption."""
    order = list(np.argsort(np.asarray(scores))[::-1])
    k = min(k, len(order))
    dcg = 0.0
    for rank in range(k):
        idx = int(order[rank])
        rel = 1.0 if idx == positive_index else 0.0
        dcg += (2.0**rel - 1.0) / math.log2(rank + 2)
    idcg = 1.0  # one relevant item at rank 1
    return float(dcg / idcg)
