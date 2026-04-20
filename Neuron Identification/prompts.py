from __future__ import annotations

"""Prompt templates used by both localization and evaluation."""


def build_prompt(task: str, item: dict, candidate: str | None = None) -> str:
    """Build task prompts aligned with the paper appendix."""
    task = task.lower()
    if task == "vqa":
        q = item["vqa"]["question"].strip()
        return f"Question: {q}\nAnswer the question with a short phrase."
    if task == "ocr":
        return "Transcribe the text inside the blue box.\nOutput text only."
    if task == "caption":
        return "Describe the image in one sentence."
    if task == "retrieval":
        if candidate is None:
            raise ValueError("retrieval prompt requires candidate caption")
        return f"Candidate: {candidate}\nDoes the candidate correctly describe the image?\nAnswer exactly Yes or No."
    raise ValueError(f"Unknown task: {task}")


def target_text(task: str, item: dict) -> str:
    """Return the text target used to build the DVP target direction u_y."""
    task = task.lower()
    if task == "vqa":
        return str(item["vqa"]["answer"]).strip()
    if task == "ocr":
        return str(item["ocr"]["text"]).strip()
    if task == "caption":
        # Use all available references to form an IDF-weighted semantic center.
        caps = [c.get("caption", "").strip() for c in item.get("captions", []) if c.get("caption", "").strip()]
        return " ".join(caps[:5])
    if task == "retrieval":
        # Retrieval is scored as pairwise Yes/No verification.
        return "Yes"
    raise ValueError(f"Unknown task: {task}")
