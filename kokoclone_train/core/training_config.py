"""Per-voice training configuration.

Load order (later overrides earlier):
    1. Default config: ``training_config.json`` in the kokoclone-train root
    2. Voice-specific: ``voices/{name}/training_config.json``
    3. CLI arguments (highest priority, applied by caller)

Example voice-specific override::

    voices/am_jefferson/training_config.json::

        {
            "epochs": 1000,
            "batch_size": 2,
            "loss_weights": {"lambda_audio_l1": 300.0},
            "xtts_augment": true,
            "aug_ratio": 0.4
        }

Only the keys you want to change need to be present; missing keys fall
through to defaults.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default path (next to pyproject.toml)
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "training_config.json"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns new dict)."""
    merged = base.copy()
    for k, v in override.items():
        if k.startswith("_"):
            continue  # skip comments
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_default_config(path: Optional[Path] = None) -> dict[str, Any]:
    """Load the project-level default training config."""
    cfg_path = path or _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        logger.warning("No default training config at %s — using hardcoded defaults", cfg_path)
        return {}
    with open(cfg_path) as f:
        return json.load(f)


def load_voice_config(voice_name: str, voices_dir: Path = Path("voices")) -> dict[str, Any]:
    """Load a voice-specific training config (if it exists).

    Looks for ``{voices_dir}/{voice_name}/training_config.json``.
    Returns empty dict if not found (caller merges with defaults).
    """
    cfg_path = voices_dir / voice_name / "training_config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        cfg = json.load(f)
    logger.info("Loaded voice-specific training config from %s", cfg_path)
    return cfg


def resolve_training_config(
    voice_name: str,
    voices_dir: Path = Path("voices"),
    default_path: Optional[Path] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Resolve the final training config by merging defaults + voice-specific + CLI.

    Args:
        voice_name: Voice name (used to find per-voice override).
        voices_dir: Parent directory containing voice subdirectories.
        default_path: Override path for project-level defaults.
        cli_overrides: Dict of values from CLI args (highest priority).

    Returns:
        Merged config dict with all keys resolved.
    """
    cfg = load_default_config(default_path)
    cfg = _deep_merge(cfg, load_voice_config(voice_name, voices_dir))

    if cli_overrides:
        cfg = _deep_merge(cfg, {k: v for k, v in cli_overrides.items() if v is not None})

    return cfg


def get_epochs(config: dict) -> int:
    return int(config.get("epochs", 100))


def get_batch_size(config: dict) -> Optional[int]:
    bs = config.get("batch_size")
    return int(bs) if bs is not None else None


def get_checkpoint_interval(config: dict) -> int:
    return int(config.get("checkpoint_interval", 10))


def get_use_decoder(config: dict) -> bool:
    return bool(config.get("use_decoder", False))


def get_device(config: dict) -> str:
    """Resolve the requested device string from config.

    Falls back to ``"auto"`` when unset or empty (the effective CLI default).
    """
    device = config.get("device")
    return str(device) if device else "auto"


def get_lr_g(config: dict) -> float:
    opt = config.get("optimizer", {})
    if isinstance(opt, dict):
        return float(opt.get("lr_g", 1e-4))
    return float(config.get("lr_g", 1e-4))


def get_lr_d(config: dict) -> float:
    opt = config.get("optimizer", {})
    if isinstance(opt, dict):
        return float(opt.get("lr_d", 2e-4))
    return float(config.get("lr_d", 2e-4))


def get_betas(config: dict) -> tuple[float, float]:
    opt = config.get("optimizer", {})
    if isinstance(opt, dict):
        b = opt.get("betas", [0.8, 0.99])
    else:
        b = config.get("betas", [0.8, 0.99])
    return (float(b[0]), float(b[1]))


def get_weight_decay(config: dict) -> float:
    opt = config.get("optimizer", {})
    if isinstance(opt, dict):
        return float(opt.get("weight_decay", 1e-5))
    return float(config.get("weight_decay", 1e-5))


def get_scheduler_gamma(config: dict) -> float:
    opt = config.get("optimizer", {})
    if isinstance(opt, dict):
        return float(opt.get("scheduler_gamma", 0.998))
    return float(config.get("scheduler_gamma", 0.998))


def get_lambda_mel(config: dict) -> float:
    lw = config.get("loss_weights", {})
    if isinstance(lw, dict):
        return float(lw.get("lambda_mel", 100.0))
    return float(config.get("lambda_mel", 100.0))


def get_lambda_audio_l1(config: dict) -> float:
    lw = config.get("loss_weights", {})
    if isinstance(lw, dict):
        return float(lw.get("lambda_audio_l1", 200.0))
    return float(config.get("lambda_audio_l1", 200.0))


def get_lambda_fm(config: dict) -> float:
    lw = config.get("loss_weights", {})
    if isinstance(lw, dict):
        return float(lw.get("lambda_fm", 2.0))
    return float(config.get("lambda_fm", 2.0))


def get_xtts_augment(config: dict) -> bool:
    return bool(config.get("xtts_augment", False))


def get_aug_ratio(config: dict) -> float:
    return float(config.get("aug_ratio", 0.3))


def get_xtts_pool_size(config: dict) -> int:
    return int(config.get("xtts_pool_size", 200))


def get_pitch_adjustment(config: dict) -> float:
    return float(config.get("pitch_adjustment", 0.0))
