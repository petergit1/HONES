#!/usr/bin/env python3
from __future__ import annotations

"""Command-line entry point for HONES localization.

This script runs the HONES pipeline:
1. critical attention-head localization;
2. head-guided FFN neuron scoring.
"""

import argparse
import json
from pathlib import Path

from hones.config import load_config, resolve_path, set_seed
from hones.data import build_retrieval_items, load_items, load_split_ids
from hones.head_localization import localize_heads
from hones.models import load_vlm
from hones.neuron_scoring import score_neurons


def _default_config(model: str) -> str:
    """Resolve configs/qwen_hones.yaml or configs/llava_hones.yaml."""
    root = Path(__file__).resolve().parents[1]
    return str(root / "configs" / f"{model}_hones.yaml")


def main():
    parser = argparse.ArgumentParser(description="Run HONES head localization and head-guided neuron scoring.")
    parser.add_argument("--model", choices=["qwen", "llava"], default=None, help="Shortcut for configs/{model}_hones.yaml")
    parser.add_argument("--config", default=None, help="Path to yaml config. Overrides --model if both are given.")
    parser.add_argument("--task", default=None, choices=["vqa", "ocr", "caption", "retrieval"], help="Task to run. Overrides config run.task.")
    parser.add_argument("--stage", default="all", choices=["heads", "neurons", "all"], help="Run only head localization, only neuron scoring, or both.")
    args = parser.parse_args()

    config_path = args.config or _default_config(args.model or "qwen")
    cfg = load_config(config_path)
    task = args.task or cfg["run"]["task"]
    set_seed(int(cfg["data"].get("seed", 3407)))

    data_file = resolve_path(cfg, cfg["data"]["data_file"])
    image_dir = resolve_path(cfg, cfg["data"]["image_dir"])
    split_path = resolve_path(cfg, cfg["data"].get("discovery_split"))
    split_ids = load_split_ids(split_path)
    out_dir = Path(resolve_path(cfg, cfg["run"]["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[HONES] config: {config_path}")
    print(f"[HONES] model: {cfg['model']['name']} | {cfg['model']['model_id']}")
    print(f"[HONES] task: {task}")

    # Load full-parameter model from Hugging Face.
    vlm = load_vlm(cfg["model"])

    # Retrieval reuses caption annotations and constructs pairwise candidates.
    limit = int(cfg["data"].get("discovery_limit", 100))
    items = load_items(data_file, task if task != "retrieval" else "caption", limit=limit, split_ids=split_ids)
    if task == "retrieval":
        items = build_retrieval_items(
            items,
            n_neg=int(cfg["run"].get("retrieval_negatives", 49)),
            seed=int(cfg["data"].get("seed", 3407)),
            limit=limit,
        )
    print(f"[HONES] discovery items: {len(items)}")

    heads_path = out_dir / "heads.json"
    if args.stage in {"heads", "all"}:
        top_heads = localize_heads(
            vlm, task, items, image_dir, str(out_dir),
            top_k=int(cfg["run"].get("top_k_heads", 30)),
            max_new_tokens=int(cfg["run"].get("max_new_tokens", 24)),
            ndcg_k=int(cfg["run"].get("ndcg_k", 5)),
        )
    else:
        # Reuse a previous head-localization result for neuron scoring only.
        with open(heads_path, "r", encoding="utf-8") as f:
            top_heads = json.load(f)["top_heads"]

    if args.stage in {"neurons", "all"}:
        selected = score_neurons(
            vlm, task, items, image_dir, top_heads, str(out_dir),
            top_ratio=float(cfg["run"].get("top_neuron_ratio", 0.01)),
        )
        print(f"[HONES] saved {len(selected)} selected neurons to {out_dir / 'neurons_top1pct.json'}")


if __name__ == "__main__":
    main()
