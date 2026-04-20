from __future__ import annotations

"""Head-guided FFN neuron attribution with direct vocabulary projection (DVP)."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import save_json
from .data import open_image, references_for_caption
from .hooks import register_head_mean_replacement, remove_hooks
from .models import get_lm_layers, get_mlp, get_unembedding
from .prompts import build_prompt, target_text


def compute_idf(items: list[dict], tokenizer) -> dict[int, float]:
    """Compute smoothed IDF weights for caption target tokens on D_disc."""
    docs = []
    for it in items:
        refs = references_for_caption(it)
        toks = set()
        for r in refs:
            toks.update(tokenizer.encode(r, add_special_tokens=False))
        if toks:
            docs.append(toks)

    df = Counter()
    for d in docs:
        for t in d:
            df[int(t)] += 1
    n = max(1, len(docs))
    return {tid: float(np.log((n + 1) / (c + 1)) + 1.0) for tid, c in df.items()}


class DVPCollector:
    """Collect sample-level FFN write-in contributions c_{l,i}(x,y; theta).

    For each MLP neuron, the hook computes:
        Delta r_i = z_i * W_down[i, :]
        c_i = <Delta r_i, u_y>
    where u_y is the normalized target unembedding direction.
    """

    def __init__(self, vlm, task: str, idf: dict[int, float] | None = None):
        self.vlm = vlm
        self.task = task
        self.layers = get_lm_layers(vlm.model)
        self.unembed = get_unembedding(vlm.model)
        self.idf = idf or {}
        self.target_ids: list[int] = []
        self.target_weights: list[float] = []
        self.effects: dict[int, list[np.ndarray]] = defaultdict(list)
        self.handles = []

    def _target_direction(self, device, dtype) -> torch.Tensor | None:
        """Construct u_y from fixed-set or IDF-weighted caption targets."""
        if not self.target_ids:
            return None
        ids = torch.tensor(self.target_ids, device=self.unembed.device, dtype=torch.long)
        u = self.unembed.index_select(0, ids).to(device=device, dtype=dtype)
        w = torch.tensor(self.target_weights, device=device, dtype=dtype).view(-1, 1)
        vec = (u * w).sum(dim=0)
        return vec / (vec.norm(p=2) + 1e-12)

    def _hook(self, layer_idx: int):
        """MLP hook that records neuron-wise DVP contributions at the last token."""
        def hook_fn(module, inputs, output):
            x = inputs[0]
            if not isinstance(x, torch.Tensor) or x.dim() != 3:
                return
            act_fn = getattr(module, "act_fn", None) or getattr(module, "activation_fn", None)
            if act_fn is None or not all(hasattr(module, a) for a in ["gate_proj", "up_proj", "down_proj"]):
                return

            # z is the FFN intermediate activation. Use the last token as in autoregressive next-token prediction.
            z = act_fn(module.gate_proj(x)) * module.up_proj(x)  # [B, S, I]
            z_last = z[:, -1, :]
            u_y = self._target_direction(z_last.device, z_last.dtype)
            if u_y is None:
                return

            # align_i = <W_down[i, :], u_y>, then c_i = z_i * align_i.
            w_down = module.down_proj.weight.to(device=z_last.device, dtype=z_last.dtype)  # [H, I]
            align = torch.matmul(w_down.transpose(0, 1), u_y)  # [I]
            contrib = z_last * align.unsqueeze(0)
            self.effects[layer_idx].append(contrib.detach().float().cpu().numpy())
        return hook_fn

    def register(self):
        """Register MLP hooks on all decoder layers."""
        self.handles = []
        for li, layer in enumerate(self.layers):
            self.handles.append(get_mlp(layer).register_forward_hook(self._hook(li)))

    def clear_effects(self):
        """Clear cached contribution arrays before a new forward pass batch."""
        self.effects = defaultdict(list)

    def remove(self):
        """Remove all MLP hooks."""
        remove_hooks(self.handles)
        self.handles = []

    def set_target(self, text: str):
        """Tokenize y and assign weights for the target direction u_y."""
        ids = self.vlm.tokenizer.encode(text, add_special_tokens=False)
        special = set(getattr(self.vlm.tokenizer, "all_special_ids", []) or [])
        ids = [int(i) for i in ids if i not in special]
        if self.task == "caption":
            weights = [self.idf.get(i, 1.0) for i in ids]
        else:
            weights = [1.0 for _ in ids]
        self.target_ids = ids[:128]
        self.target_weights = weights[:128]

    @torch.no_grad()
    def run_items(self, items: list[dict], image_dir: str):
        """Run all examples once and return mean contribution per layer."""
        self.clear_effects()
        for item in tqdm(items, desc="collect DVP", leave=False):
            try:
                image = open_image(image_dir, item)
                self.set_target(target_text(self.task, item))
                if not self.target_ids:
                    continue
                if self.task == "retrieval":
                    cand = item["candidates"][item["positive_index"]]
                    prompt = build_prompt(self.task, item, cand)
                else:
                    prompt = build_prompt(self.task, item)
                inputs = self.vlm.build_inputs(image, prompt)
                _ = self.vlm.model(**inputs, use_cache=False, return_dict=True)
            except Exception:
                continue
        return self.mean_effects()

    def mean_effects(self) -> dict[int, np.ndarray]:
        """Aggregate sample-level contributions into one vector per layer."""
        out = {}
        for li, arrs in self.effects.items():
            if not arrs:
                continue
            arr = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrs], axis=0)
            out[li] = np.nan_to_num(arr.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        return out


def score_neurons(vlm, task: str, items: list[dict], image_dir: str, top_heads: list[dict], output_dir: str,
                  top_ratio: float = 0.01) -> list[dict]:
    """Compute head-guided neuron importance I_{l,i} and select Top-K neurons."""
    layers = get_lm_layers(vlm.model)
    idf = compute_idf(items, vlm.tokenizer) if task == "caption" else None
    collector = DVPCollector(vlm, task, idf=idf)
    collector.register()

    try:
        # Baseline DVP contribution c_{l,i}(x,y; theta).
        base = collector.run_items(items, image_dir)
        importance = None
        weight_sum = 0.0

        for h in tqdm(top_heads, desc="head-guided neuron scoring"):
            # Head scores are used as weights w_h; negative heads are ignored.
            score = max(float(h.get("score", 0.0)), 0.0)
            if score <= 0:
                continue

            handle = register_head_mean_replacement(vlm.model, layers, int(h["layer"]), int(h["head"]))
            try:
                # DVP contributions under theta_{I_h}.
                intervened = collector.run_items(items, image_dir)
            finally:
                remove_hooks([handle])

            mat = _positive_drop_matrix(base, intervened)
            if importance is None:
                importance = np.zeros_like(mat, dtype=np.float32)
            importance += score * mat
            weight_sum += score

        if importance is None:
            # Fallback for pathological cases where no positive head is found.
            importance = _base_matrix(base)
        else:
            importance /= max(weight_sum, 1e-12)
    finally:
        collector.remove()

    selected = select_top_neurons(importance, top_ratio)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "neuron_importance.npy", importance)
    save_json(selected, out / "neurons_top1pct.json")
    return selected


def _base_matrix(effects: dict[int, np.ndarray]) -> np.ndarray:
    """Pack a layer->vector dict into a dense [layers, neurons] matrix."""
    max_layer = max(effects.keys())
    width = max(v.shape[-1] for v in effects.values())
    out = np.zeros((max_layer + 1, width), dtype=np.float32)
    for li, v in effects.items():
        out[li, :v.shape[-1]] = v
    return out


def _positive_drop_matrix(base: dict[int, np.ndarray], intervened: dict[int, np.ndarray]) -> np.ndarray:
    """Compute [c(theta) - c(theta_Ih)]_+ for all available layers."""
    max_layer = max(set(base.keys()) | set(intervened.keys()))
    width = max([v.shape[-1] for v in list(base.values()) + list(intervened.values())])
    out = np.zeros((max_layer + 1, width), dtype=np.float32)
    for li in set(base.keys()) & set(intervened.keys()):
        b, m = base[li], intervened[li]
        n = min(b.shape[-1], m.shape[-1])
        out[li, :n] = np.maximum(b[:n] - m[:n], 0.0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def select_top_neurons(importance: np.ndarray, top_ratio: float = 0.01) -> list[dict]:
    """Select the globally highest-scoring neuron indices."""
    flat = importance.reshape(-1)
    k = max(1, int(round(flat.size * float(top_ratio))))
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    width = importance.shape[1]
    return [
        {"rank": r + 1, "layer": int(i // width), "neuron": int(i % width), "score": float(flat[i]), "global_index": int(i)}
        for r, i in enumerate(idx)
    ]
