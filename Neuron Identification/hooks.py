from __future__ import annotations

"""Forward hooks for causal interventions.

1. mean-replacement attention-head intervention for routing localization;
2. FFN neuron zero-masking for held-out causal evaluation.
"""

from collections import defaultdict
from typing import Iterable

import torch

from .models import get_attention, get_mlp, get_num_heads, get_o_proj


def make_mean_replacement_slice_hook(head_idx: int, num_heads: int):
    """Create hooks for the mean-replacement head intervention.
    """
    def replace_slice(x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            return x
        bsz, seq_len, hidden = x.shape
        if hidden % num_heads != 0 or not (0 <= head_idx < num_heads):
            return x

        head_dim = hidden // num_heads
        v = x.view(bsz, seq_len, num_heads, head_dim).clone()
        if num_heads > 1:
            mean_other = (v.sum(dim=2, keepdim=True) - v[:, :, head_idx:head_idx + 1, :]) / float(num_heads - 1)
            v[:, :, head_idx:head_idx + 1, :] = mean_other
        else:
            v[:, :, head_idx, :] = 0
        return v.view(bsz, seq_len, hidden)

    def forward_hook(module, inputs, output):
        if isinstance(output, tuple):
            attn_out, others = output[0], output[1:]
        else:
            attn_out, others = output, None
        new_out = replace_slice(attn_out)
        return (new_out, *others) if others is not None else new_out

    def pre_hook(module, args):
        if not args:
            return args
        x = replace_slice(args[0])
        return (x,) + tuple(args[1:])

    return forward_hook, pre_hook


def register_head_mean_replacement(model, layers: list, layer_idx: int, head_idx: int):
    """Register a mean-replacement hook for one attention head."""
    layer = layers[layer_idx]
    attn = get_attention(layer)
    n_heads = get_num_heads(attn, getattr(model, "config", None))
    forward_hook, pre_hook = make_mean_replacement_slice_hook(head_idx, n_heads)

    o_proj = get_o_proj(attn)
    if o_proj is not None:
        return o_proj.register_forward_pre_hook(pre_hook)
    return attn.register_forward_hook(forward_hook)


def register_neuron_zero_mask(layers: list, selected_neurons: list[dict]):
    """Mask selected FFN neurons by setting their intermediate activations to 0.

    This is used only for causal evaluation on held-out data, not for discovery.
    """
    by_layer = defaultdict(list)
    for n in selected_neurons:
        by_layer[int(n["layer"])].append(int(n.get("neuron", n.get("neuron_idx"))))

    handles = []
    for layer_idx, neuron_ids in by_layer.items():
        if layer_idx < 0 or layer_idx >= len(layers):
            continue
        mlp = get_mlp(layers[layer_idx])
        ids = torch.tensor(sorted(set(neuron_ids)), dtype=torch.long)

        def hook_fn(module, inputs, output, ids=ids):
            # Recompute the MLP output with selected z_i suppressed.
            x = inputs[0]
            if not isinstance(x, torch.Tensor) or x.dim() != 3:
                return output
            if not all(hasattr(module, a) for a in ["gate_proj", "up_proj", "down_proj"]):
                return output
            act_fn = getattr(module, "act_fn", None) or getattr(module, "activation_fn", None)
            if act_fn is None:
                return output
            z = act_fn(module.gate_proj(x)) * module.up_proj(x)
            ids_dev = ids.to(z.device)
            z[..., ids_dev] = 0
            return module.down_proj(z)

        handles.append(mlp.register_forward_hook(hook_fn))
    return handles


def remove_hooks(handles: Iterable) -> None:
    """Safely remove a collection of PyTorch hooks."""
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass
