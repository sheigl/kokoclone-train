"""Shared model downloads and device utilities for RVC v2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import hf_hub_download

_MODEL_CACHE = Path.home() / ".cache" / "kokoclone"


def detect_device() -> torch.device:
    """Detect the best available device (MPS → CUDA → XPU → CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def empty_cache(device: torch.device) -> None:
    """Clear framework-specific cache for the given device."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()


def _ensure_cache_dir() -> Path:
    _MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    return _MODEL_CACHE


def get_hubert_path() -> str:
    """Return the repo ID for the HuBERT encoder.

    Returns the HuggingFace repo ID string, which can be used directly with
    transformers.AutoModel.from_pretrained(). The model will be cached automatically
    by the transformers library.
    """
    return "facebook/hubert-base-ls960"


def get_pretrained_generator(name: Literal["f0G48k.pth"]) -> Path | None:
    """Download a pretrained RVC v2 generator if not cached.

    Args:
        name: One of ``"f0G48k.pth"``.

    Returns:
        Path to the downloaded checkpoint file, or None if unavailable.
    """
    cache = _ensure_cache_dir()
    local_path = cache / name

    if local_path.exists():
        return local_path

    repo_map = {
        "f0G48k.pth": ("liujing04", "Retrieval-based-Voice-Conversion-WebUI", "f0G48k.pth"),
    }
    repo_id, repo_name, filename = repo_map[name]

    try:
        path = hf_hub_download(
            repo_id=f"{repo_id}/{repo_name}",
            filename=filename,
            local_dir=str(cache),
        )
        downloaded = cache / name
        if not downloaded.exists() and Path(path).exists():
            import shutil
            shutil.copy2(Path(path), downloaded)

        return downloaded
    except Exception:
        # If download fails, training will use random initialization
        print(f"[WARN] Could not download pretrained generator {name} — using random init.")
        return None