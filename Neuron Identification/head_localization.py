from __future__ import annotations

"""Critical attention-head localization.

For each head, we apply the mean-replacement causal intervention and measure the
expected degradation in the task utility. The top-K heads form H_t^*.
"""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import save_json
from .eval import evaluate_task
from .hooks import register_head_mean_replacement, remove_hooks
from .models import get_attention, get_lm_layers, get_num_heads


def localize_heads(vlm, task: str, items: list[dict], image_dir: str, output_dir: str, top_k: int = 30,
                   max_new_tokens: int = 24, ndcg_k: int = 5) -> list[dict]:
    """Run head localization and save ``heads.json`` plus ``head_scores.npy``."""
    layers = get_lm_layers(vlm.model)
    n_layers = len(layers)
    n_heads = get_num_heads(get_attention(layers[0]), getattr(vlm.model, "config", None))

    # Baseline utility P_t(x, y; theta), averaged over the discovery split.
    base = evaluate_task(vlm, task, items, image_dir, max_new_tokens=max_new_tokens, ndcg_k=ndcg_k)
    matrix = np.zeros((n_layers, n_heads), dtype=np.float32)
    rows = []

    for li in tqdm(range(n_layers), desc="head layers"):
        for hi in range(n_heads):
            # theta_{I_h}: model under mean-replacement intervention on one head.
            h = register_head_mean_replacement(vlm.model, layers, li, hi)
            try:
                intervened = evaluate_task(vlm, task, items, image_dir, max_new_tokens=max_new_tokens, ndcg_k=ndcg_k)
                drop = float(base["metric"] - intervened["metric"])
                matrix[li, hi] = drop
                rows.append({
                    "layer": li,
                    "head": hi,
                    "score": drop,
                    "baseline": base["metric"],
                    "intervened": intervened["metric"],
                    "metric_name": base["metric_name"],
                })
            finally:
                remove_hooks([h])

    rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)
    top = [dict(r, rank=i + 1) for i, r in enumerate(rows_sorted[:top_k])]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "head_scores.npy", matrix)
    save_json({"baseline": base, "all_heads": rows_sorted, "top_heads": top}, out / "heads.json")
    return top
