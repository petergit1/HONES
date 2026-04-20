from __future__ import annotations

"""Lightweight neuron steering for HONES.

This module implements the steering stage:

    freeze the VLM backbone
    learn only sparse neuron-wise scaling factors on discovered task neurons
    optimize task loss + KL(original || scaled)

It also implements the: Fixed-Amp, Grid-Search, RandNeuron and HONESs w/o KL.
"""

import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from ..config import save_json
from ..data import open_image, references_for_caption, vqa_gold_answers
from ..eval import evaluate_task, clean_generation
from ..metrics import anls, bleu4, ndcg_at_k_from_scores, vqa_accuracy
from ..models import VLM, get_lm_layers
from ..prompts import build_prompt, target_text

IGNORE_INDEX = -100


@dataclass
class SteeringResult:
    """Compact result object saved by ``run_steering.py``."""

    method: str
    metric: float
    metric_name: str
    n: int
    extra: dict


class NeuronScaleMLP(nn.Module):
    """FFN wrapper that scales selected intermediate neurons.
    """

    def __init__(self, base_mlp: nn.Module, mask_indices: torch.Tensor):
        super().__init__()
        self.base_mlp_type = type(base_mlp).__name__

        required = ["gate_proj", "up_proj", "down_proj", "act_fn"]
        missing = [name for name in required if not hasattr(base_mlp, name)]
        if missing:
            raise AttributeError(f"Unsupported MLP type {type(base_mlp)}; missing {missing}")

        self.gate_proj = base_mlp.gate_proj
        self.up_proj = base_mlp.up_proj
        self.down_proj = base_mlp.down_proj
        self.act_fn = base_mlp.act_fn

        for p in self.gate_proj.parameters():
            p.requires_grad = False
        for p in self.up_proj.parameters():
            p.requires_grad = False
        for p in self.down_proj.parameters():
            p.requires_grad = False

        try:
            base_dev = next(base_mlp.parameters()).device
        except StopIteration:
            base_dev = torch.device("cpu")
        inter_dim = int(self.gate_proj.out_features)

        mask_indices = mask_indices.to(device=base_dev, dtype=torch.long)
        mask_indices = mask_indices[(mask_indices >= 0) & (mask_indices < inter_dim)]
        mask_indices = torch.unique(mask_indices)

        mask = torch.zeros(inter_dim, dtype=torch.float32, device=base_dev)
        if mask_indices.numel() > 0:
            mask[mask_indices] = 1.0

        self.register_buffer("ns_mask", mask.view(1, 1, -1))
        self.register_buffer("bank_indices", mask_indices.clone())
        self.ns_scale = nn.Parameter(torch.zeros(inter_dim, dtype=torch.float32, device=base_dev))
        self.scaling_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        if self.scaling_enabled:
            scale = 1.0 + self.ns_mask * self.ns_scale.view(1, 1, -1)
            h = h * scale.to(dtype=h.dtype)
        out = self.down_proj(h)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def load_neuron_bank(path: str | Path, topk: int | None = None) -> list[dict]:
    """Load a neuron bank.

    Accepted formats:
    1. ``[{"layer": 16, "neuron": 123, "score": ...}, ...]``
    2. ``{"selected_neurons": [...]}``
    3. ``{"neurons": [...]}``
    4. ``{"groups": {"VQA+Retrieval": [...]}}`` with optional group selection
       already handled by pointing to the desired group file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Neuron bank does not exist: {p}")
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        raw = obj
    elif isinstance(obj, dict):
        if isinstance(obj.get("selected_neurons"), list):
            raw = obj["selected_neurons"]
        elif isinstance(obj.get("neurons"), list):
            raw = obj["neurons"]
        elif isinstance(obj.get("groups"), dict):
            groups = obj["groups"]
            if len(groups) != 1:
                raise ValueError("Bank JSON contains multiple groups. Please pass a task-specific dominant group file.")
            raw = next(iter(groups.values()))
        else:
            raise ValueError(f"Cannot find neurons in bank file: {p}")
    else:
        raise ValueError(f"Invalid bank JSON: {p}")

    parsed = []
    for it in raw:
        try:
            d = {"layer": int(it["layer"]), "neuron": int(it["neuron"])}
            if "score" in it:
                d["score"] = float(it["score"])
            elif "importance" in it:
                d["score"] = float(it["importance"])
            parsed.append(d)
        except Exception:
            continue
    parsed.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return parsed[:topk] if topk else parsed


def inject_neuron_scaling(vlm: VLM, neurons: list[dict]) -> dict:
    """Inject ``NeuronScaleMLP`` into all layers touched by the neuron bank."""
    layers = get_lm_layers(vlm.model)
    by_layer: dict[int, list[int]] = {}
    for n in neurons:
        by_layer.setdefault(int(n["layer"]), []).append(int(n["neuron"]))

    # Freeze everything first.  The wrapper will re-enable only ns_scale.
    for p in vlm.model.parameters():
        p.requires_grad = False

    stats = {"layers": {}, "total_neurons": 0}
    for layer_idx, idxs in sorted(by_layer.items()):
        if layer_idx < 0 or layer_idx >= len(layers):
            continue
        unique = sorted(set(idxs))
        mlp = layers[layer_idx].mlp
        device = next(mlp.parameters()).device
        mask = torch.tensor(unique, dtype=torch.long, device=device)
        layers[layer_idx].mlp = NeuronScaleMLP(mlp, mask)
        hit = int(layers[layer_idx].mlp.ns_mask.sum().item())
        stats["layers"][str(layer_idx)] = hit
        stats["total_neurons"] += hit

    if stats["total_neurons"] == 0:
        raise RuntimeError("No neurons were injected. Check bank layer indices and model architecture.")

    # Make only ns_scale trainable.
    for name, p in vlm.model.named_parameters():
        p.requires_grad = name.endswith("ns_scale")
    return stats


def scaling_modules(vlm: VLM) -> list[NeuronScaleMLP]:
    """Return all injected scaling modules."""
    mods = []
    for layer in get_lm_layers(vlm.model):
        if isinstance(getattr(layer, "mlp", None), NeuronScaleMLP):
            mods.append(layer.mlp)
    return mods


@contextmanager
def scaling_enabled(vlm: VLM, enabled: bool):
    """Temporarily enable/disable scaling modules.

    This lets one full-parameter model act as both the frozen teacher
    (scaling disabled) and the student (scaling enabled), avoiding the memory
    cost of loading two 7B backbones.
    """
    mods = scaling_modules(vlm)
    old = [m.scaling_enabled for m in mods]
    try:
        for m in mods:
            m.scaling_enabled = enabled
        yield
    finally:
        for m, v in zip(mods, old):
            m.scaling_enabled = v


@torch.no_grad()
def reset_ns_scale(vlm: VLM, value: float = 0.0) -> None:
    """Reset all learnable scaling factors to a scalar value."""
    for m in scaling_modules(vlm):
        m.ns_scale.data.fill_(float(value))


@torch.no_grad()
def set_uniform_amplification(vlm: VLM, amp: float) -> None:
    """Set every selected neuron to the same multiplicative factor ``amp``."""
    for m in scaling_modules(vlm):
        # Only selected entries matter because ns_mask zeros out the rest.
        m.ns_scale.data.fill_(float(amp) - 1.0)


@torch.no_grad()
def clamp_ns_scale(vlm: VLM, clamp_val: float) -> None:
    """Clamp scaling offsets to avoid overly aggressive edits."""
    for m in scaling_modules(vlm):
        m.ns_scale.data.clamp_(-float(clamp_val), float(clamp_val))


def ns_l2_regularization(vlm: VLM) -> torch.Tensor:
    """L2 regularization over selected scaling factors."""
    regs = []
    for m in scaling_modules(vlm):
        mask = m.ns_mask.view(-1).to(dtype=m.ns_scale.dtype)
        regs.append(((m.ns_scale * mask) ** 2).mean())
    if not regs:
        return torch.tensor(0.0, device=vlm.device)
    return torch.stack(regs).mean()


@torch.no_grad()
def export_neuron_scales(vlm: VLM, out_path: str | Path) -> list[dict]:
    """Export learned scale factors as ``layer/neuron/scale`` JSON."""
    rows = []
    layers = get_lm_layers(vlm.model)
    for layer_idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, NeuronScaleMLP):
            continue
        ns = mlp.ns_scale.detach().float().cpu()
        mask = mlp.ns_mask.detach().cpu().view(-1).bool()
        for neuron_idx in torch.nonzero(mask, as_tuple=False).view(-1).tolist():
            rows.append({"layer": int(layer_idx), "neuron": int(neuron_idx), "scale": float(1.0 + ns[neuron_idx].item())})
    save_json(rows, out_path)
    return rows


def make_random_bank_like(vlm: VLM, source_neurons: list[dict], seed: int = 3407) -> list[dict]:
    """Create a random neuron bank with the same per-layer budget as source."""
    rng = random.Random(seed)
    layers = get_lm_layers(vlm.model)
    counts: dict[int, int] = {}
    for n in source_neurons:
        counts[int(n["layer"])] = counts.get(int(n["layer"]), 0) + 1
    out = []
    for layer_idx, count in counts.items():
        if layer_idx < 0 or layer_idx >= len(layers):
            continue
        mlp = layers[layer_idx].mlp
        inter_dim = int(mlp.gate_proj.out_features)
        chosen = rng.sample(range(inter_dim), min(count, inter_dim))
        out.extend({"layer": layer_idx, "neuron": i} for i in chosen)
    return out



@torch.no_grad()
def collect_scale_statistics(vlm: VLM) -> dict:
    """Return a small diagnostic summary for injected steering factors.
    """
    rows = []
    for layer_idx, layer in enumerate(get_lm_layers(vlm.model)):
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, NeuronScaleMLP):
            continue
        mask = mlp.ns_mask.detach().cpu().view(-1).bool()
        if not mask.any():
            continue
        ns = mlp.ns_scale.detach().float().cpu()[mask]
        rows.append({
            "layer": int(layer_idx),
            "n": int(mask.sum().item()),
            "mean_abs_offset": float(ns.abs().mean().item()),
            "max_abs_offset": float(ns.abs().max().item()),
            "mean_scale": float((1.0 + ns).mean().item()),
        })
    total = sum(r["n"] for r in rows)
    return {"total_selected_neurons": int(total), "layers": rows}


def describe_training_objective(task: str, beta_kl: float, l2_weight: float) -> str:
    return (
        f"Task={task}: optimize answer-token CE + {beta_kl:g} * KL(original||scaled) "
        f"+ {l2_weight:g} * L2(ns_scale), with the backbone frozen."
    )

def train_scaling(
    vlm: VLM,
    task: str,
    train_items: list[dict],
    image_dir: str,
    output_dir: str | Path,
    *,
    epochs: int = 1,
    lr: float = 5e-4,
    batch_size: int = 1,
    grad_accum_steps: int = 4,
    beta_kl: float = 0.05,
    l2_weight: float = 1e-5,
    clamp: float = 1.5,
    max_new_tokens: int = 24,
    retrieval_train_negatives: int = 4,
    seed: int = 3407,
    log_every: int = 25,
) -> dict:
    """Train only neuron scaling factors on the development split.
    """
    task = task.lower()
    rng = random.Random(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = [p for p in vlm.model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable ns_scale parameters found.")

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    total_steps = max(1, math.ceil(len(train_items) * epochs / max(1, grad_accum_steps)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.1)

    logs = []
    global_update = 0
    vlm.model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(epochs):
        items = list(train_items)
        rng.shuffle(items)
        pbar = tqdm(items, desc=f"train-{task}-epoch{epoch+1}")
        accum = 0
        running = {"ce": 0.0, "kl": 0.0, "l2": 0.0}

        for item in pbar:
            examples = build_training_examples(task, item, retrieval_train_negatives=retrieval_train_negatives, rng=rng)
            if not examples:
                continue


            loss_total = torch.tensor(0.0, device=vlm.device)
            ce_total = 0.0
            kl_total = 0.0
            used = 0
            for ex in examples:
                try:
                    image = open_image(image_dir, ex["item"])
                    loss_ce, loss_kl = compute_sequence_ce_kl(vlm, image, ex["prompt"], ex["target"])
                    loss_total = loss_total + loss_ce + float(beta_kl) * loss_kl
                    ce_total += float(loss_ce.detach().item())
                    kl_total += float(loss_kl.detach().item())
                    used += 1
                except Exception:
                    continue
            if used == 0:
                continue

            loss_total = loss_total / used
            loss_l2 = ns_l2_regularization(vlm)
            loss = (loss_total + float(l2_weight) * loss_l2) / max(1, grad_accum_steps)
            loss.backward()
            accum += 1
            running["ce"] += ce_total / used
            running["kl"] += kl_total / used
            running["l2"] += float(loss_l2.detach().item())

            if accum % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                clamp_ns_scale(vlm, clamp)
                scheduler.step()
                global_update += 1

                if global_update % log_every == 0:
                    row = {
                        "update": global_update,
                        "epoch": epoch + 1,
                        "ce": running["ce"] / max(1, log_every * grad_accum_steps),
                        "kl": running["kl"] / max(1, log_every * grad_accum_steps),
                        "l2": running["l2"] / max(1, log_every * grad_accum_steps),
                        "lr": optimizer.param_groups[0]["lr"],
                        "ns_abs_mean": ns_abs_mean(vlm),
                    }
                    logs.append(row)
                    pbar.set_postfix({"ce": f"{row['ce']:.3f}", "kl": f"{row['kl']:.3f}", "ns": f"{row['ns_abs_mean']:.2e}"})
                    running = {"ce": 0.0, "kl": 0.0, "l2": 0.0}

    vlm.model.eval()
    save_json(logs, out_dir / "train_logs.json")
    export_neuron_scales(vlm, out_dir / "neuron_scales.json")
    return {
        "updates": global_update,
        "logs": logs,
        "objective": describe_training_objective(task, beta_kl, l2_weight),
        "ns_abs_mean": ns_abs_mean(vlm),
        "ns_abs_max": ns_abs_max(vlm),
        "scale_statistics": collect_scale_statistics(vlm),
    }


def build_training_examples(task: str, item: dict, retrieval_train_negatives: int, rng: random.Random) -> list[dict]:
    """Build prompt/target examples for task-specific CE training."""
    task = task.lower()
    if task == "vqa":
        return [{"item": item, "prompt": build_prompt("vqa", item), "target": target_text("vqa", item)}]
    if task == "ocr":
        return [{"item": item, "prompt": build_prompt("ocr", item), "target": target_text("ocr", item)}]
    if task == "caption":
        refs = references_for_caption(item)
        if not refs:
            return []
        return [{"item": item, "prompt": build_prompt("caption", item), "target": rng.choice(refs)}]
    if task == "retrieval":
        candidates = list(item.get("candidates", []))
        pos_idx = int(item.get("positive_index", -1))
        if pos_idx < 0 or pos_idx >= len(candidates):
            return []
        negs = [i for i in range(len(candidates)) if i != pos_idx]
        rng.shuffle(negs)
        keep = [pos_idx] + negs[:retrieval_train_negatives]
        examples = []
        for idx in keep:
            target = "Yes" if idx == pos_idx else "No"
            examples.append({"item": item, "prompt": build_prompt("retrieval", item, candidates[idx]), "target": target})
        return examples
    raise ValueError(task)


def compute_sequence_ce_kl(vlm: VLM, image: Image.Image, prompt: str, target: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute answer-token CE and KL(original||scaled) for one example."""
    target = str(target).strip()
    if not target:
        z = torch.tensor(0.0, device=vlm.device)
        return z, z

  
    inputs = vlm.build_response_inputs(image, prompt, target)
    prompt_inputs = vlm.build_inputs(image, prompt)
    input_ids = inputs["input_ids"]
    prompt_len = int(prompt_inputs["input_ids"].shape[1])

    labels = input_ids.clone()
    labels[:, :min(prompt_len, labels.shape[1])] = IGNORE_INDEX
    if getattr(vlm.tokenizer, "pad_token_id", None) is not None:
        labels[labels == vlm.tokenizer.pad_token_id] = IGNORE_INDEX

    # Teacher distribution: same model with scaling disabled, no gradient.
    with scaling_enabled(vlm, False):
        with torch.no_grad():
            teacher_logits = vlm.model(**inputs, use_cache=False, return_dict=True).logits.detach()

    # Student distribution: scaling enabled, gradients flow only to ns_scale.
    with scaling_enabled(vlm, True):
        student_logits = vlm.model(**inputs, use_cache=False, return_dict=True).logits

    labels_mm = align_labels_to_logits(labels, input_ids, student_logits.shape[1], vlm)
    ce = shifted_token_ce(student_logits, labels_mm)
    kl = shifted_token_kl(student_logits, teacher_logits, labels_mm)
    return ce, kl


def align_labels_to_logits(labels: torch.Tensor, input_ids: torch.Tensor, target_seq_len: int, vlm: VLM) -> torch.Tensor:
    """Align text-token labels to multimodal logits length.
    """
    if labels.shape[1] == target_seq_len:
        return labels
    if labels.shape[1] > target_seq_len:
        return labels[:, -target_seq_len:]

    extra = target_seq_len - labels.shape[1]
    image_token_id = find_image_token_id(vlm)
    rows = []
    for b in range(labels.shape[0]):
        ids = input_ids[b].tolist()
        labs = labels[b].tolist()
        insert_pos = None
        if image_token_id is not None:
            for i, tok in enumerate(ids):
                if int(tok) == int(image_token_id):
                    insert_pos = i
                    break
        if insert_pos is None:
            insert_pos = 1 if len(labs) > 1 else 0
        new = labs[:insert_pos] + [IGNORE_INDEX] * extra + labs[insert_pos:]
        if len(new) < target_seq_len:
            new = [IGNORE_INDEX] * (target_seq_len - len(new)) + new
        rows.append(new[-target_seq_len:])
    return torch.tensor(rows, dtype=labels.dtype, device=labels.device)


def find_image_token_id(vlm: VLM) -> Optional[int]:
    """Best-effort lookup of a model's image placeholder token id."""
    cfg = getattr(vlm.model, "config", None)
    for attr in ["image_token_index", "image_token_id"]:
        if cfg is not None and hasattr(cfg, attr):
            try:
                return int(getattr(cfg, attr))
            except Exception:
                pass
    for tok in ["<image>", "<|image_pad|>", "<|vision_start|>"]:
        try:
            tid = vlm.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and int(tid) >= 0:
                return int(tid)
        except Exception:
            pass
    return None


def shifted_token_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on answer tokens after autoregressive shifting."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels != IGNORE_INDEX
    if valid.sum().item() == 0:
        return logits.new_tensor(0.0)
    return F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum") / valid.sum().clamp(min=1)


def shifted_token_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """KL(original||scaled) on answer-token positions only."""
    s = student_logits[:, :-1, :].float()
    t = teacher_logits[:, :-1, :].float()
    lab = labels[:, 1:]
    valid = lab != IGNORE_INDEX
    if valid.sum().item() == 0:
        return student_logits.new_tensor(0.0)
    s_sel = s[valid]
    t_sel = t[valid]
    log_s = F.log_softmax(s_sel, dim=-1)
    p_t = F.softmax(t_sel, dim=-1)
    return F.kl_div(log_s, p_t, reduction="batchmean")


@torch.no_grad()
def ns_abs_mean(vlm: VLM) -> float:
    vals = []
    for m in scaling_modules(vlm):
        mask = m.ns_mask.detach().cpu().view(-1).bool()
        if mask.any():
            vals.append(m.ns_scale.detach().float().cpu()[mask].abs().mean().item())
    return float(np.mean(vals)) if vals else 0.0


@torch.no_grad()
def ns_abs_max(vlm: VLM) -> float:
    mx = 0.0
    for m in scaling_modules(vlm):
        mask = m.ns_mask.detach().cpu().view(-1).bool()
        if mask.any():
            mx = max(mx, float(m.ns_scale.detach().float().cpu()[mask].abs().max().item()))
    return mx


def evaluate_with_current_scales(vlm: VLM, task: str, items: list[dict], image_dir: str, max_new_tokens: int, ndcg_k: int) -> dict:
    """Evaluate the currently scaled model with the standard task metric."""
    with scaling_enabled(vlm, True):
        return evaluate_task(vlm, task, items, image_dir, max_new_tokens=max_new_tokens, ndcg_k=ndcg_k)

