#!/usr/bin/env python3
from __future__ import annotations

"""Command-line entry point for held-out neuron masking evaluation."""

import argparse
from pathlib import Path

from hones.config import load_config, resolve_path, set_seed
from hones.data import build_retrieval_items, load_items, load_split_ids
from hones.masking import evaluate_neuron_masking
from hones.models import load_vlm


def _default_config(model: str) -> str:
    """Resolve configs/qwen_hones.yaml or configs/llava_hones.yaml."""
    root = Path(__file__).resolve().parents[1]
    return str(root / "configs" / f"{model}_hones.yaml")


def main():
    parser = argparse.ArgumentParser(description="Evaluate causal importance by masking HONES-selected neurons.")
    parser.add_argument("--model", choices=["qwen", "llava"], default=None, help="Shortcut for configs/{model}_hones.yaml")
    parser.add_argument("--config", default=None, help="Path to yaml config. Overrides --model if both are given.")
    parser.add_argument("--task", default=None, choices=["vqa", "ocr", "caption", "retrieval"], help="Task to evaluate. Overrides config run.task.")
    parser.add_argument("--neurons", default=None, help="Path to neurons_top1pct.json. Defaults to output_dir/neurons_top1pct.json")
    args = parser.parse_args()

    config_path = args.config or _default_config(args.model or "qwen")
    cfg = load_config(config_path)
    task = args.task or cfg["run"]["task"]
    set_seed(int(cfg["data"].get("seed", 3407)))

    data_file = resolve_path(cfg, cfg["data"]["data_file"])
    image_dir = resolve_path(cfg, cfg["data"]["image_dir"])
    split_path = resolve_path(cfg, cfg["data"].get("test_split"))
    split_ids = load_split_ids(split_path)
    out_dir = Path(resolve_path(cfg, cfg["run"]["output_dir"]))
    neurons = args.neurons or str(out_dir / "neurons_top1pct.json")

    print(f"[HONES] config: {config_path}")
    print(f"[HONES] model: {cfg['model']['name']} | {cfg['model']['model_id']}")
    print(f"[HONES] task: {task}")

    vlm = load_vlm(cfg["model"])

    # Use the held-out test split for causal verification.
    limit = int(cfg["data"].get("eval_limit", 100))
    items = load_items(data_file, task if task != "retrieval" else "caption", limit=limit, split_ids=split_ids)
    if task == "retrieval":
        items = build_retrieval_items(
            items,
            n_neg=int(cfg["run"].get("retrieval_negatives", 49)),
            seed=int(cfg["data"].get("seed", 3407)),
            limit=limit,
        )

    result = evaluate_neuron_masking(
        vlm, task, items, image_dir, neurons, str(out_dir),
        max_new_tokens=int(cfg["run"].get("max_new_tokens", 24)),
        ndcg_k=int(cfg["run"].get("ndcg_k", 5)),
    )
    print(result)


if __name__ == "__main__":
    main()
