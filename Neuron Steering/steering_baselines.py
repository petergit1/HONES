from __future__ import annotations

"""Matched-budget baselines for HONES neuron steering.
"""

from pathlib import Path
from typing import Iterable

from ..config import save_json
from .lightweight_neuron_steering import (
    SteeringResult,
    evaluate_with_current_scales,
    export_neuron_scales,
    reset_ns_scale,
    set_uniform_amplification,
)


BASELINE_DESCRIPTIONS = {
    "fixed_amp": "Uniform train-free amplification on HONES-identified neurons.",
    "grid_search": "Uniform train-free amplification selected on the development split.",
    "rand_neuron": "Learnable scaling on random neurons with a matched per-layer budget.",
    "ours_no_kl": "HONES learnable scaling with the KL regularizer removed.",
}


def normalize_method_name(name: str) -> str:
    """Map user-facing aliases to compact internal method names."""
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fixed": "fixed_amp",
        "fixedamp": "fixed_amp",
        "fixed_amp": "fixed_amp",
        "grid": "grid_search",
        "gridsearch": "grid_search",
        "grid_search": "grid_search",
        "random": "rand_neuron",
        "rand": "rand_neuron",
        "randneuron": "rand_neuron",
        "rand_neuron": "rand_neuron",
        "ours_w_o_kl": "ours_no_kl",
        "ours_wo_kl": "ours_no_kl",
        "ours_no_kl": "ours_no_kl",
        "ours": "ours",
        "base": "base",
    }
    return aliases.get(key, key)


def display_method_name(name: str) -> str:
    """Return the display name used in result tables."""
    mapping = {
        "base": "Base",
        "fixed_amp": "Fixed-Amp",
        "grid_search": "Grid-Search",
        "rand_neuron": "RandNeuron",
        "ours_no_kl": "Ours w/o KL",
        "ours": "Ours",
    }
    return mapping.get(normalize_method_name(name), name)


def parse_methods(methods: str | Iterable[str] | None, default: list[str]) -> list[str]:
    """Parse a comma-separated method list and normalize aliases."""
    if methods is None:
        raw = list(default)
    elif isinstance(methods, str):
        raw = [m for m in methods.split(",") if m.strip()]
    else:
        raw = list(methods)
    return [normalize_method_name(m) for m in raw]


def method_summary_rows() -> list[dict]:
    """Small helper used by documentation/tests to list implemented baselines."""
    return [{"method": display_method_name(k), "key": k, "description": v} for k, v in BASELINE_DESCRIPTIONS.items()]


def run_fixed_amp(vlm, task: str, test_items: list[dict], image_dir: str, out_dir: str | Path, amp: float, max_new_tokens: int, ndcg_k: int) -> SteeringResult:
    """Fixed-Amp baseline: uniformly amplify every selected neuron.

    This is a train-free control. It tests whether a single global scalar is
    enough to exploit the identified neuron bank, or whether neuron-wise learned
    modulation is needed.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reset_ns_scale(vlm, 0.0)
    set_uniform_amplification(vlm, amp)
    result = evaluate_with_current_scales(vlm, task, test_items, image_dir, max_new_tokens, ndcg_k)
    export_neuron_scales(vlm, out_dir / "fixed_amp_scales.json")
    save_json({"amp": amp, "description": BASELINE_DESCRIPTIONS["fixed_amp"]}, out_dir / "baseline_info.json")
    return SteeringResult("Fixed-Amp", float(result["metric"]), result["metric_name"], int(result["n"]), {"amp": amp})


def run_grid_search(vlm, task: str, dev_items: list[dict], test_items: list[dict], image_dir: str, out_dir: str | Path, amps: list[float], max_new_tokens: int, ndcg_k: int) -> SteeringResult:
    """Grid-Search baseline: choose one uniform amplifier on the dev split."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_scores = []
    best_amp, best_metric = None, -1e9
    for amp in amps:
        reset_ns_scale(vlm, 0.0)
        set_uniform_amplification(vlm, amp)
        res = evaluate_with_current_scales(vlm, task, dev_items, image_dir, max_new_tokens, ndcg_k)
        row = {"amp": float(amp), "dev_metric": float(res["metric"]), "metric_name": res["metric_name"]}
        dev_scores.append(row)
        if row["dev_metric"] > best_metric:
            best_metric = row["dev_metric"]
            best_amp = float(amp)

    reset_ns_scale(vlm, 0.0)
    set_uniform_amplification(vlm, float(best_amp))
    test = evaluate_with_current_scales(vlm, task, test_items, image_dir, max_new_tokens, ndcg_k)
    save_json(dev_scores, out_dir / "grid_search_dev_scores.json")
    export_neuron_scales(vlm, out_dir / "grid_search_scales.json")
    return SteeringResult("Grid-Search", float(test["metric"]), test["metric_name"], int(test["n"]), {"best_amp": best_amp, "dev_scores": dev_scores})


def save_baseline_catalog(out_dir: str | Path) -> None:
    """Write a simple catalog of implemented baselines to the run directory."""
    save_json(method_summary_rows(), Path(out_dir) / "baseline_catalog.json")
