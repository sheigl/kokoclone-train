"""Shared model downloads and device utilities for RVC v2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import hf_hub_download

_MODEL_CACHE = Path.home() / ".cache" / "kokoclone"


_VALID_DEVICES = ("auto", "cuda", "xpu", "vulkan", "mps", "cpu")  # "cuda:N"/"xpu:N" parsed separately


def detect_device(requested: str | None = None) -> torch.device:
    """Detect or validate the compute device.

    Args:
        requested: ``"auto"`` (default) | ``"cuda"`` | ``"cuda:N"`` |
            ``"xpu"`` | ``"xpu:N"`` | ``"mps"`` | ``"cpu"``
            (case/whitespace normalized).

    ``None``/``"auto"`` behaves exactly like the legacy auto-detect chain
    (MPS → CUDA → XPU → CPU).  Explicit requests that name an unavailable,
    out-of-range, or unknown device raise ``ValueError`` with an actionable
    message (mentioning what was requested, what is available, and how to
    fix it).
    """
    req = (requested or "auto").strip().lower()

    if req in ("auto", None, ""):
        return _auto_detect()

    # Parse optional ":N" index suffix.  The format check runs before the
    # availability checks so garbage like "cuda:abc" produces a clear error.
    base, _, idx_str = req.partition(":")
    idx = None
    if idx_str:
        if not idx_str.isdigit():
            raise ValueError(
                f"Invalid device '{requested}': index '{idx_str}' is not a "
                f"non-negative integer. Valid values: auto, cuda[:N], xpu[:N], "
                f"mps, cpu."
            )
        idx = int(idx_str)

    if base == "cpu":
        return torch.device("cpu")

    if base == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(
                f"Requested device '{requested}' but this torch build has no MPS "
                f"backend (MPS is Apple-Silicon only). Available: "
                f"{_available_summary()}. Use --device auto or --device cpu."
            )
        return torch.device("mps")

    if base == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                f"Requested device '{requested}' but no CUDA device is available "
                f"in this torch build. Available: {_available_summary()}. If you "
                f"intend to use an AMD GPU (e.g. RX 470), install the ROCm build "
                f"of torch — see docs/ROCM_RX470_SETUP.md — or use --device auto."
            )
        n = torch.cuda.device_count()
        if idx is not None and idx >= n:
            raise ValueError(
                f"Requested 'cuda:{idx}' but this machine has {n} CUDA "
                f"device(s) (indices 0..{n - 1}). Use --device auto or a "
                f"valid index."
            )
        return torch.device(f"cuda:{idx}" if idx is not None else "cuda")

    if base == "xpu":
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise ValueError(
                f"Requested device '{requested}' but no Intel XPU device is "
                f"available in this torch build. Available: "
                f"{_available_summary()}. Use the XPU build of torch "
                f"(pytorch-xpu index) or use --device auto."
            )
        n = torch.xpu.device_count() if hasattr(torch.xpu, "device_count") else 1
        if idx is not None and idx >= n:
            raise ValueError(
                f"Requested 'xpu:{idx}' but this machine has {n} XPU "
                f"device(s) (indices 0..{n - 1}). Use --device auto or a "
                f"valid index."
            )
        return torch.device(f"xpu:{idx}" if idx is not None else "xpu")

    if base == "vulkan":
        if not _vulkan_available():
            raise ValueError(
                f"Requested device '{requested}' but the pytorch-vulkan backend "
                f"is not available. Install it (pip install pytorch-vulkan or "
                f"add ~/code/pytorch-vulkan to PYTHONPATH) and ensure the GPU "
                f"exposes a Vulkan compute device. Available: "
                f"{_available_summary()}. Use --device auto or --device cpu."
            )
        if idx is not None:
            raise ValueError(
                f"Requested 'vulkan:{idx}' but the vulkan backend exposes a "
                f"single device (index 0 only). Use --device vulkan."
            )
        return torch.device("vulkan")

    raise ValueError(
        f"Unknown device '{requested}'. Valid values: {', '.join(_VALID_DEVICES)} "
        f"(with optional :N index for cuda/xpu). Available now: "
        f"{_available_summary()}."
    )


def _auto_detect() -> torch.device:
    """Current MPS → CUDA → XPU → CPU chain (extracted, unchanged semantics)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _vulkan_available() -> bool:
    """Best-effort probe for the pytorch-vulkan backend.

    Imports lazily (so kokoclone-train still works without it) and calls
    ``pytorch_vulkan.init()`` to register the device + Vulkan context. Returns
    False if the package is missing or no Vulkan compute device is usable.
    """
    try:
        import pytorch_vulkan
    except ImportError:
        return False
    try:
        pytorch_vulkan.init()
        return pytorch_vulkan.is_available()
    except Exception:
        return False


def _available_summary() -> str:
    """Human-readable list of currently available devices for error messages."""
    parts = []
    if torch.backends.mps.is_available():
        parts.append("mps")
    if torch.cuda.is_available():
        parts.append(f"cuda:{torch.cuda.device_count()} device(s)")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        parts.append("xpu")
    if _vulkan_available():
        parts.append("vulkan")
    parts.append("cpu")
    return ", ".join(parts) if parts else "none"


def empty_cache(device: torch.device) -> None:
    """Clear framework-specific cache for the given device.

    ``cpu`` is a no-op; unknown device types are ignored defensively.
    The MPS branch is guarded with ``hasattr`` so non-Apple builds never crash.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "vulkan":
        try:
            import pytorch_vulkan
            pytorch_vulkan.empty_cache()
        except Exception:
            pass
    elif device.type == "mps" and hasattr(torch.backends, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    # cpu: no-op


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
        "f0G48k.pth": ("lj1995", "VoiceConversionWebUI", "pretrained_v2/f0G48k.pth"),
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