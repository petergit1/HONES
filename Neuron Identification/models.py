from __future__ import annotations

"""Model loading and architecture accessors.
"""

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .config import pick_dtype

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:  # pragma: no cover - depends on transformers version
    Qwen2_5_VLForConditionalGeneration = None

try:
    from qwen_vl_utils import process_vision_info
except Exception:  # pragma: no cover - Qwen users should install qwen-vl-utils
    process_vision_info = None


@dataclass
class VLM:
    """Small wrapper that standardizes LLaVA and Qwen inference calls."""

    name: str
    model: Any
    processor: Any
    device: torch.device

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    def build_inputs(self, image: Image.Image, prompt: str) -> dict:
        """Build model inputs using each backbone's official chat format."""
        if self.name == "llava":
            # LLaVA-HF expects the image placeholder in a text-style chat prompt.
            text = f"USER: <image>\n{prompt}\nASSISTANT:"
            inputs = self.processor(images=image, text=text, return_tensors="pt", padding=True)
        elif self.name == "qwen":
            # Qwen2.5-VL uses structured multimodal messages serialized by the processor.
            if process_vision_info is None:
                raise ImportError("qwen-vl-utils is required for Qwen2.5-VL")
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        else:
            raise ValueError(f"Unsupported model name: {self.name}")

        return move_to_device(inputs, self.device, getattr(self.model, "dtype", None))

    @torch.no_grad()
    def generate_text(self, image: Image.Image, prompt: str, max_new_tokens: int = 24) -> str:
        """Greedy decoding used by VQA/OCR/Caption evaluation."""
        inputs = self.build_inputs(image, prompt)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_id,
            return_dict_in_generate=True,
        )
        in_len = inputs["input_ids"].shape[1]
        gen_ids = out.sequences[0, in_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    @torch.no_grad()
    def next_token_logits(self, image: Image.Image, prompt: str) -> torch.Tensor:
        """Return next-token logits at the prompt end.

        Retrieval uses these logits to score Yes/No decisions for each candidate.
        """
        inputs = self.build_inputs(image, prompt)
        out = self.model(**inputs, use_cache=False, return_dict=True)
        return out.logits[:, -1, :].float()


def load_vlm(model_cfg: dict) -> VLM:
    """Load a full-parameter VLM from an online Hugging Face model ID."""
    name = model_cfg["name"].lower()
    model_id = model_cfg["model_id"]
    dtype = pick_dtype(model_cfg.get("dtype", "auto"))
    device_map = model_cfg.get("device_map", "auto")
    device_map_arg = "auto" if device_map == "auto" and torch.cuda.is_available() else None

    if name == "llava":
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map_arg,
        )
    elif name == "qwen":
        if Qwen2_5_VLForConditionalGeneration is None:
            raise ImportError("Your transformers version does not expose Qwen2_5_VLForConditionalGeneration")
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=model_cfg.get("min_pixels"),
            max_pixels=model_cfg.get("max_pixels"),
            use_fast=False,
        )
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "left"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map_arg,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

    if device_map_arg is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    else:
        device = next(model.parameters()).device
    model.eval()
    return VLM(name=name, model=model, processor=processor, device=device)


def move_to_device(inputs: dict, device: torch.device, dtype=None) -> dict:
    """Move processor outputs to the model device and cast float tensors."""
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if v.is_floating_point() and dtype is not None:
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def get_lm_layers(model) -> list:
    """Locate decoder layers across LLaVA/Qwen wrappers."""
    candidates = []
    if hasattr(model, "language_model"):
        candidates.append(model.language_model)
    if hasattr(model, "model"):
        candidates.append(model.model)
        if hasattr(model.model, "language_model"):
            candidates.append(model.model.language_model)
    candidates.append(model)
    for m in candidates:
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return list(m.model.layers)
        if hasattr(m, "layers"):
            return list(m.layers)
    raise AttributeError("Cannot locate decoder layers")


def get_attention(layer):
    """Return the self-attention module from a decoder layer."""
    for name in ["self_attn", "attn", "attention"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise AttributeError("Cannot locate attention module")


def get_o_proj(attn):
    """Return attention output projection if the module exposes it."""
    for name in ["o_proj", "out_proj", "proj"]:
        if hasattr(attn, name):
            return getattr(attn, name)
    return None


def get_mlp(layer):
    """Return the FFN/MLP block from a decoder layer."""
    if hasattr(layer, "mlp"):
        return layer.mlp
    raise AttributeError("Cannot locate MLP module")


def get_num_heads(attn, config=None) -> int:
    """Infer the number of attention heads from module attributes or config."""
    for name in ["num_heads", "num_attention_heads", "n_heads", "n_head"]:
        if hasattr(attn, name):
            v = getattr(attn, name)
            if isinstance(v, int) and v > 0:
                return v
    if config is not None and hasattr(config, "num_attention_heads"):
        return int(config.num_attention_heads)
    raise AttributeError("Cannot infer number of heads")


def get_unembedding(model) -> torch.Tensor:
    """Return the language-model unembedding matrix used by DVP."""
    emb = model.get_output_embeddings()
    if emb is not None and hasattr(emb, "weight"):
        return emb.weight
    if hasattr(model, "lm_head"):
        return model.lm_head.weight
    if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head.weight
    raise AttributeError("Cannot locate output embedding / lm_head")
