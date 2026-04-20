from __future__ import annotations

"""Held-out causal evaluation by masking selected neurons."""

from pathlib import Path

from .config import load_json, save_json
from .eval import evaluate_task
from .hooks import register_neuron_zero_mask, remove_hooks
from .models import get_lm_layers


def evaluate_neuron_masking(vlm, task: str, items: list[dict], image_dir: str, neurons_path: str,
                            output_dir: str, max_new_tokens: int = 24, ndcg_k: int = 5) -> dict:
    """Measure the task-metric drop after masking selected neurons.

    This is the held-out verification step: discovery happens on D_disc, while
    this function should be run on D_test.
    """
    selected = load_json(neurons_path)
    layers = get_lm_layers(vlm.model)

    base = evaluate_task(vlm, task, items, image_dir, max_new_tokens=max_new_tokens, ndcg_k=ndcg_k)
    hooks = register_neuron_zero_mask(layers, selected)
    try:
        masked = evaluate_task(vlm, task, items, image_dir, max_new_tokens=max_new_tokens, ndcg_k=ndcg_k)
    finally:
        remove_hooks(hooks)

    drop = base["metric"] - masked["metric"]
    rel_drop = drop / max(abs(base["metric"]), 1e-12) * 100.0
    result = {"baseline": base, "masked": masked, "absolute_drop": float(drop), "relative_drop_percent": float(rel_drop)}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(result, out / "masking_eval.json")
    return result
