#!/usr/bin/env python3
from __future__ import annotations

"""Run HONES neuron steering and matched-budget baselines.

Examples
--------
python scripts/run_steering.py --model llava --task vqa --dataset coco
python scripts/run_steering.py --model qwen --task retrieval --dataset coco
python scripts/run_steering.py --config configs/llava_hones.yaml --task vqa --dataset gqa
"""

import argparse
import copy
from pathlib import Path

from hones.config import load_config, resolve_path, save_json, set_seed
from hones.eval import evaluate_task
from hones.models import load_vlm
from hones.neuron_steering.lightweight_neuron_steering import (
    SteeringResult,
    inject_neuron_scaling,
    load_neuron_bank,
    make_random_bank_like,
    train_scaling,
    evaluate_with_current_scales,
)
from hones.neuron_steering.steering_baselines import (
    parse_methods,
    run_fixed_amp,
    run_grid_search,
    save_baseline_catalog,
)
from hones.neuron_steering.steering_data import image_dir_from_dataset, load_steering_items


MODEL_TO_CONFIG = {
    "llava": "configs/llava_hones.yaml",
    "qwen": "configs/qwen_hones.yaml",
}


def parse_args():
    p = argparse.ArgumentParser(description="HONES neuron steering")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")
    p.add_argument("--model", type=str, default="llava", choices=["llava", "qwen"], help="Model shortcut when --config is not provided")
    p.add_argument("--task", type=str, default=None, choices=["vqa", "ocr", "caption", "retrieval"])
    p.add_argument("--dataset", type=str, default=None, help="Dataset key from config: coco, gqa, textvqa, flickr30k")
    p.add_argument("--methods", type=str, default=None, help="Comma-separated methods: fixed_amp,grid_search,ours_no_kl,ours,rand_neuron")
    p.add_argument("--bank", type=str, default=None, help="Override task-specific neuron bank path")
    p.add_argument("--out", type=str, default=None, help="Override output directory")
    return p.parse_args()


def resolve_config_path(args) -> str:
    if args.config:
        return args.config
    return MODEL_TO_CONFIG[args.model]


def dataset_entry(cfg: dict, dataset_key: str) -> dict:
    """Resolve dataset config and repository-relative paths."""
    data_cfg = cfg.get("data", {})
    datasets = data_cfg.get("datasets", {})
    if dataset_key in datasets:
        ds = copy.deepcopy(datasets[dataset_key])
    else:

        ds = {
            "name": dataset_key,
            "data_file": data_cfg.get("data_file"),
            "image_dir": data_cfg.get("image_dir"),
            "dev_split": data_cfg.get("dev_split") or data_cfg.get("development_split"),
            "test_split": data_cfg.get("test_split"),
            "dev_limit": data_cfg.get("dev_limit") or data_cfg.get("eval_limit"),
            "test_limit": data_cfg.get("test_limit") or data_cfg.get("eval_limit"),
        }
    for key in ["data_file", "image_dir", "dev_split", "test_split"]:
        if ds.get(key):
            ds[key] = resolve_path(cfg, ds[key])
    return ds


def task_bank_path(cfg: dict, task: str, args) -> str:
    if args.bank:
        return resolve_path(cfg, args.bank)
    steering = cfg.get("steering", {})
    banks = steering.get("neuron_banks", {})
    if task in banks:
        return resolve_path(cfg, banks[task])
    run_out = cfg.get("run", {}).get("output_dir", f"artifacts/{cfg['model']['name']}/{task}")
    return resolve_path(cfg, f"{run_out}/neurons_top1pct.json")


def load_eval_data(cfg: dict, task: str, dataset_key: str):
    data_cfg = cfg.get("data", {})
    steer_cfg = cfg.get("steering", {})
    ds = dataset_entry(cfg, dataset_key)
    seed = int(data_cfg.get("seed", 3407))
    n_neg = int(cfg.get("run", {}).get("retrieval_negatives", 49))
    dev_limit = steer_cfg.get("dev_limit", ds.get("dev_limit", data_cfg.get("eval_limit")))
    test_limit = steer_cfg.get("test_limit", ds.get("test_limit", data_cfg.get("eval_limit")))
    dev_items = load_steering_items(ds, task, split="dev", limit=dev_limit, retrieval_negatives=n_neg, seed=seed)
    test_items = load_steering_items(ds, task, split="test", limit=test_limit, retrieval_negatives=n_neg, seed=seed + 1)
    return ds, dev_items, test_items


def run_learned_method(method_name: str, cfg: dict, task: str, dataset_key: str, bank: list[dict], dev_items, test_items, image_dir: str, out_dir: Path, beta_kl: float) -> SteeringResult:
    """Load a fresh model, inject bank neurons, train scales, then evaluate."""
    vlm = load_vlm(cfg["model"])
    stats = inject_neuron_scaling(vlm, bank)
    method_dir = out_dir / method_name.replace(" ", "_").lower()
    method_dir.mkdir(parents=True, exist_ok=True)

    st_cfg = cfg.get("steering", {})
    train_info = train_scaling(
        vlm=vlm,
        task=task,
        train_items=dev_items,
        image_dir=image_dir,
        output_dir=method_dir,
        epochs=int(st_cfg.get("epochs", 1)),
        lr=float(st_cfg.get("lr", 5e-4)),
        grad_accum_steps=int(st_cfg.get("grad_accum_steps", 4)),
        beta_kl=float(beta_kl),
        l2_weight=float(st_cfg.get("l2_weight", 1e-5)),
        clamp=float(st_cfg.get("clamp", 1.5)),
        max_new_tokens=int(cfg.get("run", {}).get("max_new_tokens", 24)),
        retrieval_train_negatives=int(st_cfg.get("retrieval_train_negatives", 4)),
        seed=int(cfg.get("data", {}).get("seed", 3407)),
    )
    result = evaluate_with_current_scales(
        vlm,
        task,
        test_items,
        image_dir,
        max_new_tokens=int(cfg.get("run", {}).get("max_new_tokens", 24)),
        ndcg_k=int(cfg.get("run", {}).get("ndcg_k", 5)),
    )
    extra = {"beta_kl": beta_kl, "inject_stats": stats, "train_info": train_info}
    save_json(extra, method_dir / "extra.json")
    return SteeringResult(method_name, float(result["metric"]), result["metric_name"], int(result["n"]), extra)


def main():
    args = parse_args()
    cfg = load_config(resolve_config_path(args))
    if args.task:
        cfg.setdefault("run", {})["task"] = args.task
    task = cfg.get("run", {}).get("task", "vqa").lower()
    dataset_key = (args.dataset or cfg.get("steering", {}).get("dataset") or "coco").lower()
    set_seed(int(cfg.get("data", {}).get("seed", 3407)))

    ds, dev_items, test_items = load_eval_data(cfg, task, dataset_key)
    image_dir = image_dir_from_dataset(ds)
    out_dir = Path(resolve_path(cfg, args.out or cfg.get("steering", {}).get("output_dir", f"artifacts/{cfg['model']['name']}/{task}/steering/{dataset_key}")))
    out_dir.mkdir(parents=True, exist_ok=True)

    bank_path = task_bank_path(cfg, task, args)
    bank = load_neuron_bank(bank_path, topk=cfg.get("steering", {}).get("bank_topk"))
    save_json({"task": task, "dataset": dataset_key, "bank_path": bank_path, "bank_size": len(bank), "dev_n": len(dev_items), "test_n": len(test_items)}, out_dir / "run_info.json")

    st_cfg = cfg.get("steering", {})
    methods = parse_methods(args.methods, st_cfg.get("methods", ["fixed_amp", "grid_search", "ours_no_kl", "ours", "rand_neuron"]))
    save_baseline_catalog(out_dir)

    results = []

 
    if "base" in methods or st_cfg.get("always_eval_base", True):
        base_vlm = load_vlm(cfg["model"])
        base = evaluate_task(base_vlm, task, test_items, image_dir, max_new_tokens=int(cfg.get("run", {}).get("max_new_tokens", 24)), ndcg_k=int(cfg.get("run", {}).get("ndcg_k", 5)))
        results.append(SteeringResult("Base", float(base["metric"]), base["metric_name"], int(base["n"]), {}). __dict__)
        del base_vlm

   
    if "fixed_amp" in methods or "grid_search" in methods:
        vlm = load_vlm(cfg["model"])
        inject_neuron_scaling(vlm, bank)
        if "fixed_amp" in methods:
            r = run_fixed_amp(vlm, task, test_items, image_dir, out_dir / "fixed_amp", float(st_cfg.get("fixed_amp", 2.0)), int(cfg.get("run", {}).get("max_new_tokens", 24)), int(cfg.get("run", {}).get("ndcg_k", 5)))
            results.append(r.__dict__)
        if "grid_search" in methods:
            amps = [float(x) for x in st_cfg.get("grid_amps", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])]
            r = run_grid_search(vlm, task, dev_items, test_items, image_dir, out_dir / "grid_search", amps, int(cfg.get("run", {}).get("max_new_tokens", 24)), int(cfg.get("run", {}).get("ndcg_k", 5)))
            results.append(r.__dict__)
        del vlm

    if "ours_no_kl" in methods:
        r = run_learned_method("Ours w/o KL", cfg, task, dataset_key, bank, dev_items, test_items, image_dir, out_dir, beta_kl=0.0)
        results.append(r.__dict__)

    if "ours" in methods:
        r = run_learned_method("Ours", cfg, task, dataset_key, bank, dev_items, test_items, image_dir, out_dir, beta_kl=float(st_cfg.get("beta_kl", 0.05)))
        results.append(r.__dict__)

    if "rand_neuron" in methods:
        vlm = load_vlm(cfg["model"])
        random_bank = make_random_bank_like(vlm, bank, seed=int(cfg.get("data", {}).get("seed", 3407)) + 13)
        del vlm
        r = run_learned_method("RandNeuron", cfg, task, dataset_key, random_bank, dev_items, test_items, image_dir, out_dir, beta_kl=float(st_cfg.get("beta_kl", 0.05)))
        results.append(r.__dict__)

    save_json(results, out_dir / "steering_summary.json")
    print("\n========= Steering Summary =========")
    for r in results:
        print(f"{r['method']}: {r['metric_name']}={r['metric']:.4f}, n={r['n']}")
    print(f"[SAVE] {out_dir / 'steering_summary.json'}")


if __name__ == "__main__":
    main()
