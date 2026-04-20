from __future__ import annotations

"""Configuration and utilities for the HONES."""

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config and attach repository-root metadata.

    All relative paths in the YAML file are resolved against the repository root.
    """
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["_config_path"] = str(path)
    cfg["_repo_root"] = str(path.parent.parent if path.parent.name == "configs" else path.parent)
    return cfg


def resolve_path(cfg: Dict[str, Any], value: str | Path | None) -> str | None:
    """Resolve a path from config.
    """
    if value is None:
        return None
    p = Path(value).expanduser()
    if p.is_absolute():
        return str(p)
    return str(Path(cfg.get("_repo_root", ".")) / p)


def save_json(obj: Any, path: str | Path) -> None:
    """Write an object as UTF-8 JSON with pretty indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int = 3407) -> None:
    """Set all random seeds used by this compact release."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # TF32 can speed up Ampere+ GPUs without changing the algorithmic flow.
        torch.backends.cuda.matmul.allow_tf32 = True


def pick_dtype(name: str = "auto"):
    """Choose the tensor dtype used for full-parameter model loading."""
    name = (name or "auto").lower()
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16

    # Auto mode: CPU uses fp32; CUDA uses bf16 on Ampere+ and fp16 otherwise.
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16
