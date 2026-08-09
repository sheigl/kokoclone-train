"""RVC training pipeline.

Full PyTorch training loop for RVC v2 voice conversion models.
Pipeline stages:
    1. Preprocess audio → split into chunks, normalise
    2. Extract HuBERT content features (fairseq)
    3. Extract F0 pitch data (pyworld / harvest)
    4. Train SynthesizerTrn generator with Adam + gradient clipping
    5. Build FAISS index for feature retrieval during inference

Supports:
    - Audio trimming (--start/--end) to extract segments from long files
    - Resume training from checkpoints (--resume flag)
    - Checkpoint saving every N epochs (--checkpoint-interval)
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Time parsing helper ────────────────────────────────────────────────────

def parse_time(value: str) -> float:
    """Parse time string to seconds. Accepts '120', '2:30', '01:05:30' formats."""
    if re.match(r"^\d+(\.\d+)?$", value):
        return float(value)
    parts = value.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Invalid time format: {value} (use seconds, MM:SS, or HH:MM:SS)")


# ── State tracking helpers ─────────────────────────────────────────────────

def _write_state(temp_dir: Path, state: dict[str, Any]) -> None:
    """Write training state to temp dir for resume/status tracking."""
    state_path = temp_dir / "state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def _read_state(temp_dir: Path) -> Optional[dict[str, Any]]:
    """Read training state from temp dir. Returns None if no state file."""
    state_path = temp_dir / "state.json"
    if not state_path.exists():
        return None
    try:
        with open(state_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def _check_completed_stages(temp_dir: Path) -> dict[str, bool]:
    """Check which pipeline stages have already been completed."""
    gt_wavs = temp_dir / "0_gt_wavs"
    feature = temp_dir / "2_feature"
    f0_dir = temp_dir / "3_f0"
    mel_dir = temp_dir / "4_mel"

    return {
        "preprocess": gt_wavs.exists() and len(list(gt_wavs.glob("*.wav"))) > 0,
        "features":   feature.exists() and len(list(feature.glob("*.npy"))) > 0,
        "f0":         f0_dir.exists() and len(list(f0_dir.glob("*.npy"))) > 0,
        "mel":        mel_dir.exists() and len(list(mel_dir.glob("*.npy"))) > 0,
    }


def _sort_checkpoints(checkpoints: list[Path]) -> list[Path]:
    """Sort checkpoints by epoch number (natural sort, not lexicographic)."""
    def _epoch_key(p: Path) -> int:
        parts = p.stem.rsplit("_epoch_", 1)
        return int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0
    return sorted(checkpoints, key=_epoch_key)


def _find_latest_checkpoint(temp_dir: Path, voice_name: str) -> Optional[dict]:
    """Find the latest training checkpoint in temp dir."""
    checkpoints = _sort_checkpoints(list(temp_dir.glob(f"{voice_name}_epoch_*.pth")))
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    try:
        ckpt = torch.load(str(latest), map_location="cpu", weights_only=False)
        ckpt["_checkpoint_path"] = str(latest)
        return ckpt
    except Exception as e:
        logger.warning("Failed to load checkpoint %s: %s", latest, e)
        return None


def _cleanup_old_checkpoints(temp_dir: Path, voice_name: str, keep: int = 3) -> None:
    """Remove old checkpoints, keeping only the last `keep`."""
    checkpoints = _sort_checkpoints(list(temp_dir.glob(f"{voice_name}_epoch_*.pth")))
    while len(checkpoints) > keep:
        oldest = checkpoints.pop(0)
        oldest.unlink(missing_ok=True)


# ── Training dataset ───────────────────────────────────────────────────────

class ContentFeatureDataset(Dataset):
    """Per-chunk dataset for RVC training.

    Each sample is one audio chunk with content features, F0 pitch, and
    (optional) mel targets and (optional) raw audio.

    Content features are resampled on-the-fly to match F0/mel frame rate
    via linear interpolation (standard practice in speech processing).

    Two return shapes are supported, depending on which optional fields
    are present in the chunk dict (which the caller constructs based on
    the ``use_decoder`` flag in ``_train_model``):

    * Mel-only chunks (``{"content", "f0", "mel"}``):
        Returns ``(content, f0, mel)`` tensors.

    * End-to-end chunks (``{"content", "f0", "mel", "audio"}``):
        Returns ``(content, f0, mel, audio)`` tensors where ``audio``
        is the raw waveform ``[T_audio]`` and is length-aligned to the
        decoder's output expectation: ``T_audio = T_mel * 160``,
        where ``160 = product(decoder.upsample_rates)``.

    Audio is loaded via ``soundfile`` (sidesteps the torchaudio
    torchcodec dependency issue documented in CHANGELOG.md) and
    truncated/padded to exactly ``T_mel * 160`` samples so the L1 +
    adversarial losses can compute elementwise differences.
    """

    def __init__(
        self,
        chunks: list[dict],
        seq_len: int = 512,
        audio_upsample: int = 160,
        xtts_chunks: list[dict] | None = None,
        aug_ratio: float = 0.0,
    ):
        super().__init__()
        self.chunks = chunks
        self.seq_len = seq_len
        self.audio_upsample = audio_upsample
        self.xtts_chunks = xtts_chunks or []
        self.aug_ratio = aug_ratio
        self.aug_count = 0

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        from torch.nn import functional as F
        import soundfile as sf

        chunk = self.chunks[idx]

        # Load features
        content = np.load(chunk["content"])   # [T_c, 768] at ~50fps (e.g. 184 frames)
        f0 = np.load(chunk["f0"])             # [T_m] at mel rate (e.g. 493 or 925 frames)

        # Load mel target if available
        mel = None
        if "mel" in chunk and Path(chunk["mel"]).exists():
            mel = np.load(chunk["mel"])       # [T_m, 192]

        # Resample content to match F0 frame rate
        t_content = len(content)
        t_target = len(f0)

        if t_content != t_target:
            c_torch = torch.FloatTensor(content).T.unsqueeze(0)  # [1, 768, T_c]
            content_resampled = F.interpolate(
                c_torch, size=t_target, mode='linear', align_corners=False
            ).squeeze(0).transpose(0, 1)  # [T_target, 768]
            content = content_resampled.numpy()

        # Optionally replace source features with XTTS-generated features,
        # keeping the real mel/audio as targets.  This teaches the model
        # to map TTS-style inputs to the target speaker's voice.
        if self.xtts_chunks and self.aug_ratio > 0 and random.random() < self.aug_ratio:
            self.aug_count += 1
            xtts_idx = random.randint(0, len(self.xtts_chunks) - 1)
            xtts_content = np.load(self.xtts_chunks[xtts_idx]["content"])
            xtts_f0 = np.load(self.xtts_chunks[xtts_idx]["f0"])

            # Resample XTTS content to match target frames
            if len(xtts_content) != len(xtts_f0):
                xc = torch.FloatTensor(xtts_content).T.unsqueeze(0)
                xtts_content = F.interpolate(
                    xc, size=len(xtts_f0), mode='linear', align_corners=False
                ).squeeze(0).transpose(0, 1).numpy()

            # Use XTTS features as source, truncating to seq_len
            actual_len = self.seq_len
            c = torch.zeros(actual_len, xtts_content.shape[1])
            n = min(len(xtts_content), actual_len)
            c[:n] = torch.FloatTensor(xtts_content[:n])
            f = torch.zeros(actual_len, dtype=torch.long)
            f[:n] = torch.LongTensor(xtts_f0[:n])
        else:
            # Truncate or pad to seq_len so every sample has a uniform shape.
            # Variable-length chunks produce variable-length discriminator inputs,
            # which can trigger oneDNN JIT failures on Intel Arc GPUs.
            actual_len = self.seq_len
            c = torch.zeros(actual_len, content.shape[1])         # [seq_len, 768]
            n = min(len(content), actual_len)
            c[:n] = torch.FloatTensor(content[:n])
            f = torch.zeros(actual_len, dtype=torch.long)         # [seq_len]
            f[:n] = torch.LongTensor(f0[:n])

        result = [c, f]

        if mel is not None:
            # Pad mel with silence floor (-100 dB), NOT zeros.
            # Our mel is in dB scale where 0 dB = power 1.0 (loud),
            # so zero-padding would teach the model to produce loud
            # output for silent regions.
            m = torch.full((actual_len, mel.shape[1]), -100.0)  # [seq_len, 192]
            m[:n] = torch.FloatTensor(mel[:n])
            result.append(m)

        # Load raw audio if the chunk provides it (end-to-end mode).
        # Audio is padded/truncated to seq_len * audio_upsample samples
        # so all batches have identical shapes — required for stable
        # oneDNN GPU convolution in the discriminators.
        if "audio" in chunk and Path(chunk["audio"]).exists():
            audio_np, _sr = sf.read(chunk["audio"], dtype='float32')
            t_target_audio = actual_len * self.audio_upsample
            if len(audio_np) >= t_target_audio:
                audio_np = audio_np[:t_target_audio]
            else:
                audio_np = np.pad(audio_np, (0, t_target_audio - len(audio_np)))
            result.append(torch.FloatTensor(audio_np))         # [T_audio]

        return tuple(result)


# ── Public API ─────────────────────────────────────────────────────────────

def train_voice(
    voice_name: str,
    audio_dir: Path,
    output_dir: Path,
    epochs: int | None = None,
    pitch_adjustment: float | None = None,
    callback: Optional[Callable[[str, float], None]] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    resume: bool = False,
    checkpoint_interval: int | None = None,
    use_decoder: bool | None = None,
    batch_size: int | None = None,
    xtts_augment: bool | None = None,
    aug_ratio: float | None = None,
    training_config: Optional[dict] = None,
    device: str | None = None,
) -> dict:
    """Train a new RVC voice model from audio files.

    Args:
        voice_name: Name for the trained voice (used as directory name).
        audio_dir: Directory containing training audio (.wav/.mp3/.flac).
        output_dir: Parent directory where ``voices/{voice_name}/`` is created.
        epochs: Number of training epochs.  None = from config.
        pitch_adjustment: Pitch shift in semitones applied during training
            (-24 to +24).  None = from config.
        callback: Optional progress hook ``(message: str, percent: float)``.
        start_time: Start time in seconds (or MM:SS / HH:MM:SS string) for trimming audio.
        end_time: End time in seconds for trimming audio. None means use full file after start.
        resume: If True, resume from existing temp dir and checkpoints.
        checkpoint_interval: Save training checkpoint every N epochs.  None = from config.
        use_decoder: If True, train end-to-end with an embedded HiFi-GAN
            decoder (see ``docs/END_TO_END_DECODER_PLAN.md``).  This sets
            ``hop_length=160`` (so mel frame rate = 250 fps, matching the
            decoder's ``product(upsample_rates)=160``), requires audio in
            the dataset (alongside content/F0/mel), and uses the
            adversarial training loop.  If False (legacy mel-only mode),
            uses ``hop_length=300``, dataset returns ``(content, f0, mel)``
            only, and training uses plain MSE on mel prediction.
            None = from config.
        batch_size: Override auto-detected batch size.  None = auto or from config.
        xtts_augment: Enable XTTS data augmentation.  None = from config.
        aug_ratio: Fraction of batches using XTTS source features.  None = from config.
        training_config: Merged training config dict (defaults + voice-specific + CLI).
            If None, uses hardcoded defaults only.
        device: Device override string ("auto", "cuda[:N]", "xpu[:N]", "mps",
            "cpu").  None = from ``training_config`` ("device" key), falling back
            to "auto".

    Returns:
        Dict with keys ``name``, ``model_path``, ``index_path``, ``config_path``.

    Raises:
        ValueError: If no audio files are found or training data is empty.
    """
    from kokoclone_train.core.training_config import (
        get_device, get_epochs, get_use_decoder,
    )

    # Resolve use_decoder before we know hop_length
    if use_decoder is None:
        use_decoder = get_use_decoder(training_config or {})

    # Resolve device once: explicit arg > training_config["device"] > "auto".
    # Stays a string here; detect_device() is called inside the stages that use it.
    resolved_device = device if device is not None else get_device(training_config or {})

    # hop_length depends on use_decoder — see docs/END_TO_END_DECODER_PLAN.md §3.2
    # and AGENTS.md Architecture Invariant §3.
    hop_length = 160 if use_decoder else 300
    voice_output = output_dir / voice_name
    voice_output.mkdir(parents=True, exist_ok=True)

    temp_dir = output_dir.parent / "training_temp" / voice_name

    # Handle resume — check for existing state
    if resume:
        existing_state = _read_state(temp_dir)
        completed = _check_completed_stages(temp_dir)
        if not temp_dir.exists() or (not any(completed.values()) and existing_state is None):
            raise ValueError(
                f"No previous training state found for '{voice_name}'. "
                "Run without --resume to start fresh."
            )
        _log(callback, f"Resuming training for '{voice_name}' from saved state", 0.0)
        if existing_state:
            _log(callback, f"Previous state: stage={existing_state.get('stage')}, epoch={existing_state.get('current_epoch', 'N/A')}", 0.0)

    temp_dir.mkdir(parents=True, exist_ok=True)

    # Parse time values (accept string or float)
    start_sec = parse_time(start_time) if start_time is not None else None
    end_sec = parse_time(end_time) if end_time is not None else None
    duration_sec = (end_sec - start_sec) if (start_sec is not None and end_sec is not None) else None

    # Warn about long audio
    if duration_sec and duration_sec > 3600:
        logger.warning(
            "Audio duration %.1f minutes exceeds recommended 60 min for RVC training. "
            "Quality may suffer with very long inputs.",
            duration_sec / 60,
        )

    try:
        # ── Step 1: Preprocess audio ────────────────────────────────────
        gt_wavs_dir = temp_dir / "0_gt_wavs"
        wavs16k_dir = temp_dir / "1_16k_wavs"
        gt_wavs_dir.mkdir(exist_ok=True)
        wavs16k_dir.mkdir(exist_ok=True)

        completed = _check_completed_stages(temp_dir)

        if not (resume and completed["preprocess"]):
            _log(callback, "Preprocessing audio...", 0.05)
            audio_files = (
                list(audio_dir.glob("*.wav"))
                + list(audio_dir.glob("*.mp3"))
                + list(audio_dir.glob("*.flac"))
            )
            if not audio_files:
                raise ValueError(f"No audio files found in {audio_dir}")

            _preprocess_audio(
                audio_files, gt_wavs_dir, wavs16k_dir, callback,
                start_time=start_sec, end_time=end_sec, duration=duration_sec,
            )
            _write_state(temp_dir, {"stage": "preprocess", "current_epoch": 0})
        else:
            _log(callback, "Skipping preprocessing (already completed)", 0.05)

        # ── Step 2: Extract HuBERT features ─────────────────────────────
        from kokoclone_train.core.rvc_assets import get_hubert_path

        hubert_path = str(get_hubert_path())
        feature_dir = temp_dir / "2_feature"
        feature_dir.mkdir(exist_ok=True)

        if not (resume and completed["features"]):
            _log(callback, "Extracting HuBERT content features...", 0.2)
            _extract_features(wavs16k_dir, feature_dir, hubert_path, callback, device=resolved_device)
            _write_state(temp_dir, {"stage": "features", "current_epoch": 0})
        else:
            _log(callback, "Skipping feature extraction (already completed)", 0.2)

        # ── Step 3: Extract F0 pitch ────────────────────────────────────
        f0_dir = temp_dir / "3_f0"
        f0_dir.mkdir(exist_ok=True)

        if not (resume and completed["f0"]):
            _log(callback, "Extracting F0 pitch data...", 0.4)
            _extract_f0_data(gt_wavs_dir, f0_dir, callback, hop_length=hop_length)
            _write_state(temp_dir, {"stage": "f0", "current_epoch": 0})
        else:
            _log(callback, "Skipping F0 extraction (already completed)", 0.4)

        # ── Step 3b: Extract mel spectrogram targets ────────────────────
        mel_dir = temp_dir / "4_mel"
        mel_dir.mkdir(exist_ok=True)

        if not (resume and completed["mel"]):
            _log(callback, "Extracting mel spectrogram targets...", 0.45)
            _extract_mel_targets(gt_wavs_dir, mel_dir, callback, hop_length=hop_length)
            _write_state(temp_dir, {"stage": "mel", "current_epoch": 0})
        else:
            _log(callback, "Skipping mel extraction (already completed)", 0.45)

        # ── Step 4: Train model ─────────────────────────────────────────
        _log(callback, "Training RVC model...", 0.5)

        # Resolve config values (CLI > training_config > hardcoded defaults)
        from kokoclone_train.core.training_config import (
            get_aug_ratio,
            get_batch_size,
            get_checkpoint_interval,
            get_epochs,
            get_pitch_adjustment,
            get_xtts_augment,
        )

        cfg = training_config or {}
        resolved_epochs = epochs if epochs is not None else get_epochs(cfg)
        resolved_pitch = pitch_adjustment if pitch_adjustment is not None else get_pitch_adjustment(cfg)
        resolved_ci = checkpoint_interval if checkpoint_interval is not None else get_checkpoint_interval(cfg)
        resolved_bs = batch_size if batch_size is not None else get_batch_size(cfg)
        resolved_xtts = xtts_augment if xtts_augment is not None else get_xtts_augment(cfg)
        resolved_ar = aug_ratio if aug_ratio is not None else get_aug_ratio(cfg)

        _train_model(
            voice_name=voice_name,
            temp_dir=temp_dir,
            output_dir=voice_output,
            epochs=resolved_epochs,
            pitch_adjustment=resolved_pitch,
            callback=callback,
            resume=resume,
            checkpoint_interval=resolved_ci,
            use_decoder=use_decoder,
            hop_length=hop_length,
            batch_size=resolved_bs,
            xtts_augment=resolved_xtts,
            aug_ratio=resolved_ar,
            training_config=training_config,
            device=resolved_device,
        )

        # ── Step 5: Build FAISS index ───────────────────────────────────
        _log(callback, "Building FAISS feature index...", 0.9)
        _build_faiss_index(feature_dir, voice_output / "v1.index")

        # ── Save config ─────────────────────────────────────────────────
        config = {
            "name": voice_name,
            "version": 2,
            "input_size": 768,  # HuBERT output dim (fixed for facebook/hubert-base-ls960)
            "output_size": 192,
            "sample_rate": 40000,
            "hop_length": hop_length,
            "f0": True,
            "upsample_rates": [1],  # identity — alignment done in dataset
            "use_decoder": use_decoder,  # True = end-to-end (Hopfield-GAN dec inside net_g)
            "epoch": epochs,
            "pitch_adjustment": pitch_adjustment,
        }
        with open(voice_output / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save training hyperparameters used (for reproducibility)
        if training_config:
            from kokoclone_train.core.training_config import (
                get_betas, get_lambda_audio_l1, get_lambda_fm, get_lambda_mel,
                get_lr_d, get_lr_g, get_scheduler_gamma, get_weight_decay,
                get_xtts_augment, get_aug_ratio,
            )
            hp = {
                "device": str(resolved_device),
                "optimizer": {
                    "lr_g": get_lr_g(training_config),
                    "lr_d": get_lr_d(training_config) if use_decoder else None,
                    "betas": list(get_betas(training_config)),
                    "weight_decay": get_weight_decay(training_config),
                    "scheduler_gamma": get_scheduler_gamma(training_config),
                },
                "loss_weights": {
                    "lambda_mel": get_lambda_mel(training_config),
                    "lambda_audio_l1": get_lambda_audio_l1(training_config),
                    "lambda_fm": get_lambda_fm(training_config),
                },
                "xtts_augment": get_xtts_augment(training_config),
                "aug_ratio": get_aug_ratio(training_config),
            }
            with open(voice_output / "training_hp.json", "w") as f:
                json.dump(hp, f, indent=2)

        _log(callback, "Training complete!", 1.0)
        _write_state(temp_dir, {"stage": "complete", "current_epoch": epochs})

        return {
            "name": voice_name,
            "model_path": str(voice_output / "v1.pth"),
            "index_path": str(voice_output / "v1.index"),
            "config_path": str(voice_output / "config.json"),
        }

    except Exception as e:
        # On failure, save partial state for resume capability
        _write_state(temp_dir, {"stage": "failed", "error": str(e)})
        raise


# ── Preprocessing ──────────────────────────────────────────────────────────

def _preprocess_audio(
    audio_files, gt_wavs_dir, wavs16k_dir, callback,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    duration: Optional[float] = None,
):
    """Split and normalise training audio into chunks.

    Args:
        audio_files: List of audio file paths to process.
        gt_wavs_dir: Output directory for 40kHz wav chunks.
        wavs16k_dir: Output directory for 16kHz wav chunks.
        callback: Progress callback function.
        start_time: Start offset in seconds (for trimming).
        end_time: End time in seconds (unused directly; duration is derived).
        duration: Duration to load from start_time (end_time - start_time).
    """
    from scipy import signal
    import torchaudio

    sr = 40000
    bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=sr)
    per = 3.7
    overlap = 0.3
    tail = per + overlap
    max_val = 0.9
    alpha = 0.75

    trim_info = ""
    if start_time is not None and duration is not None:
        trim_info = f" (trimming {start_time:.0f}s–{start_time + duration:.0f}s)"
    elif start_time is not None:
        trim_info = f" (from {start_time:.0f}s onward)"

    for i, audio_file in enumerate(audio_files):
        try:
            _log(callback, f"Loading file {i + 1}/{len(audio_files)}: {audio_file.name}{trim_info}", 0.05)

            # Load with torchaudio (replaces librosa to avoid numba/llvmlite dep)
            waveform, sr_loaded = torchaudio.load(str(audio_file))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # stereo → mono
            # Trim with offset/duration (in samples at original sr)
            if start_time is not None:
                start_sample = int(start_time * sr_loaded)
                waveform = waveform[:, start_sample:]
            if duration is not None:
                end_sample = int(duration * sr_loaded)
                waveform = waveform[:, :end_sample]
            # Resample to target sr
            if sr_loaded != sr:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr_loaded, new_freq=sr,
                )
            audio = waveform.squeeze().numpy()
            _log(callback, f"Loaded {len(audio)/sr:.1f}s of audio from {audio_file.name}", 0.05 + (i * 0.02))

            if len(audio) < sr:
                logger.warning("Audio too short after trimming (%s), skipping", audio_file.name)
                continue

            audio = signal.lfilter(bh, ah, audio)

            idx1 = 0
            chunk_size = int(sr * per)
            hop_size = int(sr * (per - overlap))

            for start in range(0, len(audio) - int(sr * tail), hop_size):
                end = start + chunk_size
                if end > len(audio):
                    break

                chunk = audio[start:end]
                tmp_max = np.abs(chunk).max()
                if tmp_max > 2.5:
                    continue

                normalized = (chunk / tmp_max * (max_val * alpha)) + (1 - alpha) * chunk

                wavfile.write(
                    str(gt_wavs_dir / f"{i}_{idx1}.wav"),
                    sr,
                    normalized.astype(np.float32),
                )

                # Resample to 16kHz for HuBERT (torchaudio replaces librosa)
                norm_tensor = torch.from_numpy(normalized).unsqueeze(0)  # [1, T]
                audio_16k = torchaudio.functional.resample(
                    norm_tensor, orig_freq=sr, new_freq=16000,
                ).squeeze().numpy()
                wavfile.write(
                    str(wavs16k_dir / f"{i}_{idx1}.wav"),
                    16000,
                    audio_16k.astype(np.float32),
                )
                idx1 += 1

            progress = 0.05 + (i / len(audio_files)) * 0.15
            _log(callback, f"Preprocessed {i + 1}/{len(audio_files)} files ({idx1} chunks from last)", progress)

        except Exception as e:
            logger.error("Error processing %s: %s", audio_file, e)


# ── Feature extraction ─────────────────────────────────────────────────────

def _extract_features(wavs16k_dir, feature_dir, hubert_path, callback, device: str | None = None):
    """Extract HuBERT content features from 16 kHz audio."""
    from kokoclone_train.core.rvc_assets import detect_device

    device = detect_device(device)
    import torchaudio

    # Use local HuBERTManager (already in core/)
    from kokoclone_train.core.hubert_manager import HuBERTManager

    hubert = HuBERTManager.get(hubert_path, device=device)

    wav_files = sorted(wavs16k_dir.glob("*.wav"))
    for i, wav_file in enumerate(tqdm(wav_files, desc="Extracting features")):
        try:
            # Load with torchaudio (replaces librosa to avoid numba/llvmlite dep)
            waveform, sr_loaded = torchaudio.load(str(wav_file))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # stereo → mono
            if sr_loaded != 16000:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr_loaded, new_freq=16000,
                )
            audio = waveform.squeeze().numpy()
            content = hubert.content(audio, 16000)

            np.save(
                str(feature_dir / f"{wav_file.stem}.npy"),
                content.cpu().numpy(),
            )
        except Exception as e:
            logger.error("Error extracting features for %s: %s", wav_file, e)

    _log(callback, "Feature extraction complete", 0.35)


def _extract_f0_single(
    wav_file_str: str, f0_dir_str: str, hop_length: int = 300,
) -> tuple[str, bool, str | None]:
    """Worker: extract F0 for a single WAV file.

    Must be module-level so multiprocessing can pickle it.

    Returns:
        (stem, success, error_msg) — error_msg is None on success.
    """
    import pyworld as pw
    import torchaudio

    wav_file = Path(wav_file_str)
    f0_dir = Path(f0_dir_str)
    stem = wav_file.stem

    try:
        sr = 40000
        frame_period = hop_length / sr * 1000

        waveform, sr_loaded = torchaudio.load(str(wav_file))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # stereo → mono
        if sr_loaded != sr:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sr_loaded, new_freq=sr,
            )
        audio = waveform.squeeze().numpy()
        f0, t = pw.harvest(
            audio.astype(np.float64),
            sr,
            f0_floor=50,
            f0_ceil=1100,
            frame_period=frame_period,
        )

        # Convert to coarse F0 (log2 scale for embedding lookup)
        f0_coarse = np.zeros(len(f0), dtype=np.int32)
        for j in range(len(f0)):
            if f0[j] != 0:
                f0_coarse[j] = int(np.round(12 * np.log2(f0[j] / 75))) + 1

        np.save(str(f0_dir / f"{stem}.npy"), f0_coarse)
        return (stem, True, None)
    except Exception as e:
        logger.error("Error extracting F0 for %s: %s", wav_file, e)
        return (stem, False, str(e))


def _extract_f0_data(gt_wavs_dir, f0_dir, callback, hop_length: int = 300):
    """Extract F0 (fundamental frequency) from audio using Harvest.

    Runs each file in a separate process via ProcessPoolExecutor for near-
    linear speedup on multi-core machines.

    Args:
        hop_length: Frame step in samples. ``300`` (legacy) gives 133.3fps
            at 40kHz; ``160`` (end-to-end mode) gives 250fps.
    """
    import multiprocessing

    mp_context = multiprocessing.get_context("spawn")
    wav_files = sorted(gt_wavs_dir.glob("*.wav"))
    max_workers = os.cpu_count() or 1

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, mp_context=mp_context
    ) as executor:
        futures = [
            executor.submit(_extract_f0_single, str(wf), str(f0_dir), hop_length)
            for wf in wav_files
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Extracting F0",
        ):
            stem, success, err = future.result()
            if not success:
                logger.error("F0 extraction failed for %s: %s", stem, err)

    _log(callback, "F0 extraction complete", 0.45)


def _extract_mel_single(
    wav_file_str: str, mel_dir_str: str, hop_length: int = 300,
) -> tuple[str, bool, str | None]:
    """Worker: extract mel spectrogram for a single WAV file.

    Must be module-level so multiprocessing can pickle it.

    Returns:
        (stem, success, error_msg) — error_msg is None on success.
    """
    import torch
    import torchaudio
    import soundfile as sf

    wav_file = Path(wav_file_str)
    mel_dir = Path(mel_dir_str)
    stem = wav_file.stem

    try:
        sr = 40000
        n_mels = 192
        n_fft = hop_length * 4  # 4× hop is standard for STFT window length

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            n_mels=n_mels,
            normalized=False,
        )

        audio_np, sr_loaded = sf.read(str(wav_file), dtype='float32')
        audio = torch.from_numpy(audio_np).unsqueeze(0)  # [1, T]
        if sr_loaded != sr:
            audio = torchaudio.functional.resample(audio, sr_loaded, sr)

        # Ensure [1, T] shape for MelSpectrogram
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        mel = mel_transform(audio)  # [1, n_mels, T_frames], raw power spectrogram
        mel_db = torch.clamp(mel, min=1e-10).log10() * 10  # Convert to dB scale for vocoder compatibility
        mel_np = mel_db.squeeze(0).transpose(0, 1).cpu().numpy()  # [T_frames, n_mels]

        np.save(str(mel_dir / f"{stem}.npy"), mel_np)
        return (stem, True, None)
    except Exception as e:
        logger.error("Error extracting mel for %s: %s", wav_file, e)
        return (stem, False, str(e))


def _extract_mel_targets(gt_wavs_dir: Path, mel_dir: Path, callback, hop_length: int = 300):
    """Extract mel spectrogram targets from ground truth wav files.

    Runs each file in a separate process via ProcessPoolExecutor for near-
    linear speedup on multi-core machines.

    Uses torchaudio or librosa to compute mel spectrograms at the model's
    target sample rate (40kHz) with matching hop_length and n_mels.

    Args:
        hop_length: Frame step in samples. ``300`` (legacy) gives 133.3fps
            at 40kHz; ``160`` (end-to-end mode) gives 250fps and matches
            the HiFi-GAN decoder's ``product(upsample_rates)=160``.
    """
    import multiprocessing

    mp_context = multiprocessing.get_context("spawn")
    wav_files = sorted(gt_wavs_dir.glob("*.wav"))
    max_workers = os.cpu_count() or 1

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, mp_context=mp_context
    ) as executor:
        futures = [
            executor.submit(_extract_mel_single, str(wf), str(mel_dir), hop_length)
            for wf in wav_files
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Extracting mel targets",
        ):
            stem, success, err = future.result()
            if not success:
                logger.error("Mel extraction failed for %s: %s", stem, err)

    _log(callback, "Mel target extraction complete", 0.5)


# ── Real training loop ─────────────────────────────────────────────────────

def _train_model(
    voice_name: str,
    temp_dir: Path,
    output_dir: Path,
    epochs: int = 100,
    pitch_adjustment: float = 0.0,
    callback: Optional[Callable[[str, float], None]] = None,
    resume: bool = False,
    checkpoint_interval: int = 10,
    use_decoder: bool = False,
    hop_length: int = 300,
    batch_size: int | None = None,
    xtts_augment: bool = False,
    aug_ratio: float = 0.3,
    training_config: Optional[dict] = None,
    device: str | None = None,
) -> None:
    """Train the RVC SynthesizerTrn generator.

    Two modes (must not be mixed — see ``docs/END_TO_END_DECODER_PLAN.md``):

    * Mel-only (``use_decoder=False``, default):
        - ``hop_length=300`` (133.3 fps)
        - Loss: MSE(pred_mel, gt_mel)
        - Optimizer: AdamW lr=1e-4

    * End-to-end (``use_decoder=True``):
        - ``hop_length=160`` (250 fps, matches decoder's upsample product=160)
        - Loss: mel MSE ×45 + audio L1 + LS-GAN adversarial + feature matching ×2
        - Optimizer: AdamW lr=2e-4 + discriminators
        - Requires audio in dataset (alongside content/F0/mel)

    Steps:
        1. Load per-chunk features (content, F0, mel, audio) into a Dataset/DataLoader.
        2. Build SynthesizerTrn from config and initialise with pretrained weights.
        3. Run the appropriate training loop (mel-only or end-to-end adversarial).
        4. Save checkpoint as ``{voice_name}.pth``.

    Args:
        voice_name: Name for the output model file.
        temp_dir: Temporary directory containing extracted features and F0.
        output_dir: Where to save the trained .pth file.
        epochs: Number of training epochs.
        pitch_adjustment: Semitone shift applied to F0 during training.
        callback: Progress hook ``(message, percent)``.
        resume: If True, attempt to load latest checkpoint and continue training.
        checkpoint_interval: Save checkpoint every N epochs (default 10).
        use_decoder: If True, run the end-to-end adversarial training loop
            with the HiFi-GAN decoder; requires ``hop_length=160``.
        hop_length: Frame step used during F0/mel extraction (must match
            the data on disk; the training loop checks aligned shapes).
        device: Device override string ("auto", "cuda[:N]", "xpu[:N]", "mps",
            "cpu").  None/"auto" = auto-detect (MPS → CUDA → XPU → CPU).
    """
    from kokoclone_train.core.rvc_assets import detect_device

    device = detect_device(device)

    def _unwrap_model(m):
        return m.module if isinstance(m, torch.nn.DataParallel) else m

def _state_dict_to_cpu(sd):
    """Move every tensor in a state dict to CPU for serialization.

    torch.save() calls .cpu() on each tensor internally, which our vulkan
    backend cannot do via storage.set_. Moving here avoids that path.
    """
    return {k: v.detach().to("cpu") for k, v in sd.items()}


def _vulkan_adam_step(optimizer, device_type: str) -> bool:
    """Perform an Adam optimizer step via the native vulkan fused kernel.

    torch.optim.Adam on vulkan tensors dispatches ~8 elementwise ops per
    param (each with a GPU submit+wait), which dominates step time for
    ~45M params. When `device_type == "vulkan"` and pytorch_vulkan is
    available, apply the fused ``adam_update`` kernel instead.

    Returns True if the native path ran (and the caller should skip the
    standard ``optimizer.step()``).
    """
    if device_type != "vulkan":
        return False
    try:
        import pytorch_vulkan
        from pytorch_vulkan._C import adam_update
    except Exception:
        return False

    state = optimizer.state
    for group in optimizer.param_groups:
        lr = group.get("lr", 1e-3)
        betas = group.get("betas", (0.9, 0.999))
        eps = group.get("eps", 1e-8)
        for p in group["params"]:
            if p.grad is None or p.device.type != "vulkan":
                continue
            if p not in state:
                state[p] = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                }
            st = state[p]
            st["step"] += 1
            adam_update(
                p, p.grad, st["exp_avg"], st["exp_avg_sq"],
                st["step"], lr, betas[0], betas[1], eps,
            )
    return True

    # ── Load per-chunk metadata ───────────────────────────────────────────────
    feature_dir = temp_dir / "2_feature"
    f0_dir = temp_dir / "3_f0"
    mel_dir = temp_dir / "4_mel"
    wav_dir = temp_dir / "0_gt_wavs"  # end-to-end mode only

    chunks: list[dict] = []
    for feat_file in sorted(feature_dir.glob("*.npy")):
        stem = feat_file.stem
        f0_path = f0_dir / f"{stem}.npy"
        mel_path = mel_dir / f"{stem}.npy"
        wav_path = wav_dir / f"{stem}.wav"

        if not f0_path.exists():
            continue

        chunk = {
            "content": str(feat_file),
            "f0": str(f0_path),
        }
        if mel_path.exists():
            chunk["mel"] = str(mel_path)

        # End-to-end mode also requires the raw audio chunk for L1 + adversarial loss.
        if use_decoder and wav_path.exists():
            chunk["audio"] = str(wav_path)

        chunks.append(chunk)

    if not chunks:
        raise ValueError("No valid training data found after feature extraction")

    if use_decoder:
        n_with_audio = sum(1 for c in chunks if "audio" in c)
        if n_with_audio == 0:
            raise ValueError(
                "use_decoder=True requires audio chunks in 0_gt_wavs/ but none "
                "were found. Run 'kokoclone-train preprocess' first."
            )

    # Count total frames for logging
    total_frames = 0
    n_with_mel = 0
    for chunk in chunks:
        feat = np.load(chunk["content"])
        f0_data_chunk = np.load(chunk["f0"])
        # After resampling, content matches F0 length
        total_frames += len(f0_data_chunk)
        if "mel" in chunk and Path(chunk["mel"]).exists():
            n_with_mel += 1

    _log(callback, f"Loaded {len(chunks)} chunks ({total_frames} total frames, {n_with_mel} with mel targets, {sum(1 for c in chunks if 'audio' in c)} with audio)", 0.55)

    # Apply pitch adjustment to F0 files (semitone shift → frequency ratio)
    if pitch_adjustment != 0.0:
        freq_ratio = 2 ** (pitch_adjustment / 12)
        for chunk in chunks:
            f0_path = Path(chunk["f0"])
            f0_data_chunk = np.load(str(f0_path))
            f0_data_chunk[f0_data_chunk > 0] = np.clip(
                (f0_data_chunk[f0_data_chunk > 0].astype(np.float32)) * freq_ratio,
                a_min=1,
                a_max=255,
            ).astype(np.int64)
            # Save back in-place for this training session
            np.save(str(f0_path), f0_data_chunk)

    # ── Build dataset and dataloader ──────────────────────────────────────────
    # Mel-only mode: chunks are ~493 frames (hop=300 @ 3.7s).
    # End-to-end mode: chunks are ~925-926 frames (hop=160 @ 3.7s).
    # Use 925 (the F0 frame count) so mel loses only 1 frame (926→925)
    # and audio is exactly 148000 samples (3.7s @ 40kHz).  No padding
    # for normal chunks; only edge-case short chunks get padded.
    seq_len = 925 if use_decoder else 512
    batch_size = min(batch_size or 8, max(2, len(chunks) // 32))

    # Load XTTS augmentation data if enabled — hard fail if missing
    xtts_chunks: list[dict] | None = None
    if xtts_augment:
        xtts_content_dir = temp_dir / "xtts_aug" / "content"
        xtts_f0_dir = temp_dir / "xtts_aug" / "f0"
        if not xtts_content_dir.exists() or not xtts_f0_dir.exists():
            raise ValueError(
                f"xtts_augment=True but XTTS feature dirs not found: "
                f"{xtts_content_dir}, {xtts_f0_dir}. "
                f"Run build_xtts_aug.py first to generate them."
            )
        xtts_chunks = []
        for feat_file in sorted(xtts_content_dir.glob("*.npy")):
            stem = feat_file.stem
            f0_path = xtts_f0_dir / f"{stem}.npy"
            if f0_path.exists():
                xtts_chunks.append({
                    "content": str(feat_file),
                    "f0": str(f0_path),
                })
        if not xtts_chunks:
            raise ValueError(
                f"xtts_augment=True but no XTTS feature files found in "
                f"{xtts_content_dir}. Run build_xtts_aug.py first."
            )
        _log(callback, f"XTTS augmentation: {len(xtts_chunks)} samples loaded (ratio={aug_ratio})", 0.55)

    dataset = ContentFeatureDataset(
        chunks=chunks, seq_len=seq_len, audio_upsample=160 if use_decoder else 1,
        xtts_chunks=xtts_chunks, aug_ratio=aug_ratio,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type in ("cuda", "xpu"),
        # drop_last=True avoids partial last batches that produce
        # variable-length discriminator inputs.  The Intel Arc GPU's
        # oneDNN JIT compiler can't handle certain 2D conv backward
        # shapes that arise from edge-case batch sizes.
        drop_last=use_decoder,
    )

    _log(callback, f"Dataset: {len(dataset)} samples, batch_size={batch_size}, seq_len={seq_len}", 0.56)

    # ── Build model from pretrained weights ─────────────────────────────
    from kokoclone_train.core.rvc_models import build_rvc_model

    config = {
        "sample_rate": 40000,
        "hop_length": hop_length,
        "input_size": 768,
        "output_size": 192,
        "f0": True,
        "use_decoder": use_decoder,
    }

    net_g = build_rvc_model(config).to(device)

    # Load pretrained generator weights as initialization (optional).
    # use_decoder=True checkpoints have a ``dec.*`` sub-dict that is rare in
    # the public f0G48k.pth init — strict=False lets pretrained encoder
    # weights transfer while the HiFi-GAN decoder starts from random init.
    from kokoclone_train.core.rvc_assets import get_pretrained_generator

    pretrained_path = get_pretrained_generator("f0G48k.pth")
    if pretrained_path is not None:
        ckpt = torch.load(str(pretrained_path), map_location="cpu", weights_only=False)
        sd = ckpt.get("weight", ckpt.get("model", ckpt))
        if sd is not None:
            # Filter to only keys with matching shapes (skip shape mismatches that
            # raise RuntimeError even with strict=False, e.g. kernel size diffs in
            # dec.ups.0/.1 between pretrained RVC v2 and our config)
            model_sd = _unwrap_model(net_g).state_dict()
            filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
            n_loaded = len(filtered)
            n_skipped = len(sd) - n_loaded
            _unwrap_model(net_g).load_state_dict(filtered, strict=False)
            if n_skipped:
                n_dec_skipped = sum(1 for k in sd if k not in filtered and k.startswith("dec."))
                logger.warning(
                    "Pretrained init: loaded %d / %d keys (%d dec keys skipped due to shape mismatch)",
                    n_loaded, len(sd), n_dec_skipped,
                )
    else:
        _log(callback, "No pretrained generator found — using random initialization", 0.58)

    # Disable tanh in decoder during training so gradients can flow freely
    # through the conv_post layer (avoids saturation from pretrained VITS
    # decoder weights tuned for latent input, not mel spectrograms).
    # Tanh is re-enabled at inference (see vocoder.py forward()).
    m = _unwrap_model(net_g)
    if use_decoder and hasattr(m, "dec") and hasattr(m.dec, "use_tanh"):
        m.dec.use_tanh = False
        _log(callback, "Decoder tanh disabled during training (re-enabled at inference)", 0.58)

    _log(callback, f"Generator: {sum(p.numel() for p in m.parameters()):,} params (use_decoder={use_decoder})", 0.58)

    # ── Discriminators (end-to-end mode only) ───────────────────────────
    if use_decoder:
        from kokoclone_train.core.rvc_models import (
            MultiPeriodDiscriminator, MultiScaleDiscriminator,
        )
        net_d_mpd = MultiPeriodDiscriminator().to(device)
        net_d_msd = MultiScaleDiscriminator().to(device)
        _log(
            callback,
            f"Discriminators: MPD={sum(p.numel() for p in net_d_mpd.parameters()):,}, "
            f"MSD={sum(p.numel() for p in net_d_msd.parameters()):,}",
            0.58,
        )
    else:
        net_d_mpd = None
        net_d_msd = None

    # ── Optimisers and schedulers ──────────────────────────────────────
    # HiFi-GAN recommended lr=2e-4 for both G and D (literature confirmed,
    # see CHANGELOG July 6). Earlier D LR reduction to 5e-5 compensated for
    # LAMBDA_FM=20 being 10× the standard — fixed together.
    # Values can be overridden per-voice via training_config.json.
    cfg = training_config or {}
    from kokoclone_train.core.training_config import (
        get_betas, get_lambda_audio_l1, get_lambda_fm, get_lambda_mel,
        get_lr_d, get_lr_g, get_scheduler_gamma, get_weight_decay,
    )

    lr_g = get_lr_g(cfg)
    lr_d = get_lr_d(cfg) if use_decoder else None
    gamma = get_scheduler_gamma(cfg)
    betas_cfg = get_betas(cfg)
    wd = get_weight_decay(cfg)

    optimizer_g = torch.optim.AdamW(
        net_g.parameters(),
        lr=lr_g,
        weight_decay=wd,
        betas=betas_cfg,
    )
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=gamma)

    if use_decoder:
        disc_params = list(net_d_mpd.parameters()) + list(net_d_msd.parameters())
        optimizer_d = torch.optim.AdamW(
            disc_params,
            lr=lr_d,
            weight_decay=wd,
            betas=betas_cfg,
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=gamma)
    else:
        optimizer_d = None
        scheduler_d = None

    # ── Resume from checkpoint if available ─────────────────────────────
    start_epoch = 1
    best_loss = float("inf")

    if resume:
        latest_ckpt = _find_latest_checkpoint(temp_dir, voice_name)
        if latest_ckpt and "epoch" in latest_ckpt:
            # New-format checkpoints have weight_g/weight_d; old ones just have weight.
            if "weight_g" in latest_ckpt:
                _unwrap_model(net_g).load_state_dict(latest_ckpt["weight_g"])
            elif "weight" in latest_ckpt:
                # Legacy checkpoint from mel-only mode.  Strict=False because
                # the new end-to-end model has dec.* keys that the legacy ckpt lacks.
                _unwrap_model(net_g).load_state_dict(latest_ckpt["weight"], strict=False)
            optimizer_g.load_state_dict(latest_ckpt["optimizer_g"])
            scheduler_g.load_state_dict(latest_ckpt["scheduler_g"])
            if use_decoder and "weight_d" in latest_ckpt and optimizer_d is not None:
                net_d_mpd.load_state_dict(latest_ckpt["weight_d_mpd"])
                net_d_msd.load_state_dict(latest_ckpt["weight_d_msd"])
                optimizer_d.load_state_dict(latest_ckpt["optimizer_d"])
                scheduler_d.load_state_dict(latest_ckpt["scheduler_d"])
            start_epoch = latest_ckpt["epoch"] + 1

            # Re-apply current LR config so HP changes (e.g. D LR 5e-5→2e-4)
            # take effect on resume instead of being overwritten by checkpoint.
            for pg in optimizer_g.param_groups:
                pg["lr"] = lr_g
            if use_decoder and optimizer_d is not None:
                for pg in optimizer_d.param_groups:
                    pg["lr"] = lr_d
            best_loss = latest_ckpt.get("best_loss", float("inf"))
            _log(callback, f"Resumed from epoch {start_epoch - 1} (best_loss={best_loss:.4f})", 0.58)

    # ── Training loop ───────────────────────────────────────────────────
    _unwrap_model(net_g).train()
    if use_decoder:
        net_d_mpd.train()
        net_d_msd.train()
    has_mel_targets = n_with_mel > 0

    # Loss weights (adjusted for end-to-end training):
    # - LAMBDA_MEL=100: pushes spectral accuracy (increased from HiFi-GAN V1's 45
    #   because our mel is internal prediction, not mel-of-audio).
    # - LAMBDA_AUDIO_L1=200: doubled from 100 to push decoder harder (see
    #   AGENTS.md § Epoch 300 test — audio std only 55% of GT).
    # - LAMBDA_FM=2: matches HiFi-GAN paper (was 20, which drowned out adversarial
    #   signal and required D LR reduction as compensation — see July 6 changelog).
    # - LAMBDA_AUDIO_L1=200: our primary waveform supervision (no standard
    #   counterpart in vanilla HiFi-GAN which uses mel-of-audio).
    # - LAMBDA_MEL=100: internal mel prediction loss (not mel-of-audio).
    # All three can be overridden per-voice via training_config.json.
    LAMBDA_MEL = get_lambda_mel(cfg)
    LAMBDA_AUDIO_L1 = get_lambda_audio_l1(cfg)
    LAMBDA_FM = get_lambda_fm(cfg)

    for epoch in range(start_epoch, epochs + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        n_batches = 0

        for batch_data in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            if use_decoder:
                batch_c, batch_f0, batch_mel, batch_audio = batch_data
                batch_audio = batch_audio.to(device)      # [B, T_audio]
            else:
                if has_mel_targets:
                    batch_c, batch_f0, batch_mel = batch_data
                else:
                    batch_c, batch_f0 = batch_data

            batch_c = batch_c.to(device)       # [B, T, 768] — already resampled to match F0/mel frames
            batch_f0 = batch_f0.to(device)     # [B, T]

            optimizer_g.zero_grad(set_to_none=True)

            # Forward pass: content + F0 → (mel, audio or None)
            pred_mel, pred_audio = net_g(batch_c, f0=batch_f0)   # [B, 192, T], [B, 1, T_a] or None

            if not has_mel_targets:
                raise RuntimeError(
                    "No mel targets found — cannot train. Run 'kokoclone-train extract --step mel' first."
                )

            target_mel = batch_mel.transpose(1, 2).to(device)    # [B, T, 192] → [B, 192, T]
            loss_mel = torch.nn.functional.mse_loss(pred_mel, target_mel)

            if not use_decoder or pred_audio is None:
                # ─── Mel-only backward path (legacy) ───────────────────────
                loss = loss_mel
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=1.0)
                if not _vulkan_adam_step(optimizer_g, device.type):
                    optimizer_g.step()
                epoch_g_loss += float(loss.item())
            else:
                # ─── End-to-end adversarial path ───────────────────────────
                # 1) Discriminator step first (see plan §5) — D learns on real vs fake.detach()
                audio_pred_squeezed = pred_audio.squeeze(1)   # [B, T_audio]
                real_audio = batch_audio.unsqueeze(1)         # [B, 1, T_audio]
                # The decoder output has a constant +4 sample offset from conv
                # padding (see docs/END_TO_END_DECODER_PLAN.md §6.2).  Align
                # by truncating both to the shorter length.
                T_pred = pred_audio.shape[2]
                T_gt = real_audio.shape[2]
                T_min = min(T_pred, T_gt)
                audio_pred_aligned = pred_audio[:, :, :T_min]
                real_audio_aligned = real_audio[:, :, :T_min]

                # Discriminator loss: LS-GAN form (MSE, not Sigmoid+BCE)
                optimizer_d.zero_grad(set_to_none=True)
                mpd_real_s, mpd_fake_s, mpd_real_f, mpd_fake_f = net_d_mpd(real_audio_aligned, audio_pred_aligned.detach())
                msd_real_s, msd_fake_s, msd_real_f, msd_fake_f = net_d_msd(real_audio_aligned, audio_pred_aligned.detach())
                loss_d_mpd = sum(
                    torch.nn.functional.mse_loss(r, torch.ones_like(r))
                    + torch.nn.functional.mse_loss(f, torch.zeros_like(f))
                    for r, f in zip(mpd_real_s, mpd_fake_s)
                )
                loss_d_msd = sum(
                    torch.nn.functional.mse_loss(r, torch.ones_like(r))
                    + torch.nn.functional.mse_loss(f, torch.zeros_like(f))
                    for r, f in zip(msd_real_s, msd_fake_s)
                )
                loss_d = loss_d_mpd + loss_d_msd
                loss_d.backward()
                grad_norm_d = torch.nn.utils.clip_grad_norm_(
                    list(net_d_mpd.parameters()) + list(net_d_msd.parameters()),
                    max_norm=1.0,
                )
                if not _vulkan_adam_step(optimizer_d, device.type):
                    optimizer_d.step()
                epoch_d_loss += float(loss_d.item())

                # 2) Generator step — adversarial + feature-matching + L1 audio + mel MSE
                mpd_real_s, mpd_fake_s, mpd_real_f, mpd_fake_f = net_d_mpd(real_audio_aligned, audio_pred_aligned)
                msd_real_s, msd_fake_s, msd_real_f, msd_fake_f = net_d_msd(real_audio_aligned, audio_pred_aligned)

                # Adversarial loss: push discriminator to classify fake as real (1.0)
                loss_adv_mpd = sum(
                    torch.nn.functional.mse_loss(f, torch.ones_like(f))
                    for f in mpd_fake_s
                )
                loss_adv_msd = sum(
                    torch.nn.functional.mse_loss(f, torch.ones_like(f))
                    for f in msd_fake_s
                )
                loss_adv = loss_adv_mpd + loss_adv_msd

                # Feature matching: L1 between intermediate features of real vs fake
                loss_fm = 0.0
                for rf, ff in zip(mpd_real_f, mpd_fake_f):
                    for r, f in zip(rf, ff):
                        loss_fm = loss_fm + torch.nn.functional.l1_loss(r, f)
                for rf, ff in zip(msd_real_f, msd_fake_f):
                    for r, f in zip(rf, ff):
                        loss_fm = loss_fm + torch.nn.functional.l1_loss(r, f)

                # Audio L1 (length-aligned)
                loss_audio_l1 = torch.nn.functional.l1_loss(audio_pred_aligned, real_audio_aligned)

                loss_g = (
                    LAMBDA_MEL * loss_mel
                    + LAMBDA_AUDIO_L1 * loss_audio_l1
                    + LAMBDA_FM * loss_fm
                    + loss_adv
                )
                loss_g.backward()
                grad_norm_g = torch.nn.utils.clip_grad_norm_(net_g.parameters(), max_norm=1.0)
                if not _vulkan_adam_step(optimizer_g, device.type):
                    optimizer_g.step()
                epoch_g_loss += float(loss_g.item())

            n_batches += 1

        scheduler_g.step()
        if use_decoder and scheduler_d is not None:
            scheduler_d.step()

        avg_g_loss = epoch_g_loss / max(n_batches, 1)
        aug_count = getattr(dataset, 'aug_count', 0)
        if use_decoder:
            avg_d_loss = epoch_d_loss / max(n_batches, 1)
            log_msg = (
                f"Epoch {epoch}/{epochs} — "
                f"G={avg_g_loss:.4f} (mel={float(loss_mel.item()):.3f}), D={avg_d_loss:.4f}"
            )
        else:
            log_msg = f"Epoch {epoch}/{epochs} — loss={avg_g_loss:.4f}"

        if aug_count > 0:
            log_msg += f"  [XTTS aug: {aug_count}]"

        if avg_g_loss < best_loss:
            best_loss = avg_g_loss

        progress = 0.58 + (epoch / epochs) * 0.32
        _log(callback, log_msg, progress)
        _write_state(temp_dir, {
            "stage": "training",
            "current_epoch": epoch,
            "best_loss": best_loss,
            "use_decoder": use_decoder,
        })

        # Save intermediate checkpoint every N epochs
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
            ckpt_path = temp_dir / f"{voice_name}_epoch_{epoch}.pth"
            ckpt_data = {
                "weight_g": _state_dict_to_cpu(_unwrap_model(net_g).state_dict()),
                "optimizer_g": _state_dict_to_cpu(optimizer_g.state_dict()),
                "scheduler_g": _state_dict_to_cpu(scheduler_g.state_dict()),
                "epoch": epoch,
                "best_loss": best_loss,
                "use_decoder": use_decoder,
                "hop_length": hop_length,
            }
            if use_decoder:
                ckpt_data["weight_d_mpd"] = _state_dict_to_cpu(net_d_mpd.state_dict())
                ckpt_data["weight_d_msd"] = _state_dict_to_cpu(net_d_msd.state_dict())
                ckpt_data["optimizer_d"] = _state_dict_to_cpu(optimizer_d.state_dict())
                ckpt_data["scheduler_d"] = _state_dict_to_cpu(scheduler_d.state_dict())
            torch.save(ckpt_data, str(ckpt_path))
            _cleanup_old_checkpoints(temp_dir, voice_name)
            _log(callback, f"Checkpoint saved: {ckpt_path.name}", progress)

    # ── Save final model ────────────────────────────────────────────────
    _unwrap_model(net_g).eval()
    state_dict = _state_dict_to_cpu(_unwrap_model(net_g).state_dict())

    model_path = output_dir / "v1.pth"
    # Keep the public ``{"weight": ...}`` envelope so legacy inference paths
    # (kokoclone/core/rvc_engine.load_voice) keep working for mel-only models.
    # End-to-end checkpoints expose the dec sub-dict via this same "weight" key
    # so the inference engine can extract it as ``net_g.dec``.
    torch.save({"weight": state_dict}, str(model_path))

    _log(callback, f"Model saved to {model_path}", 0.92)


# ── FAISS index ────────────────────────────────────────────────────────────

def _build_faiss_index(feature_dir: Path, index_path: Path) -> None:
    """Build a flat inner-product FAISS index from content features."""
    import faiss

    all_features: list[np.ndarray] = []
    for feat_file in sorted(feature_dir.glob("*.npy")):
        feat = np.load(feat_file)
        if len(feat) > 0:
            all_features.append(feat)

    if not all_features:
        logger.warning("No features to index — skipping FAISS build")
        return

    features = np.concatenate(all_features, axis=0).astype(np.float32)
    d = features.shape[1]

    # L2-normalise for inner-product similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    features_normalized = features / norms

    index = faiss.IndexFlatIP(d)
    index.add(features_normalized)
    faiss.write_index(index, str(index_path))


# ── Helpers ────────────────────────────────────────────────────────────────

def _log(callback: Optional[Callable[[str, float], None]], msg: str, progress: float) -> None:
    """Log a message and invoke the progress callback if provided."""
    logger.info("[RVC Train] %s", msg)
    print(f"[RVC Train] {msg}")
    if callback is not None:
        try:
            callback(msg, progress)
        except Exception as e:
            logger.warning("Progress callback failed: %s", e)


# ── Public step entry points (for individual CLI commands) ─────────────────

def run_preprocess(
    voice_name: str,
    audio_dir: Path,
    output_dir: Path = Path("voices"),
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    callback: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """Split and normalize training audio into chunks.

    Creates ``training_temp/{voice_name}/0_gt_wavs/`` (40kHz) and
    ``1_16k_wavs/`` (16kHz for HuBERT).

    Returns a dict with ``chunks_created``, ``gt_wavs_dir``, ``wavs16k_dir``.
    """
    temp_dir = output_dir.parent / "training_temp" / voice_name
    gt_wavs_dir = temp_dir / "0_gt_wavs"
    wavs16k_dir = temp_dir / "1_16k_wavs"

    # Check if already done
    completed = _check_completed_stages(temp_dir)
    if completed.get("preprocess"):
        existing = len(list(gt_wavs_dir.glob("*.wav")))
        _log(callback, f"Preprocessing already done ({existing} chunks).", 0.0)
        return {
            "chunks_created": 0,
            "gt_wavs_dir": str(gt_wavs_dir),
            "wavs16k_dir": str(wavs16k_dir),
        }

    gt_wavs_dir.mkdir(parents=True, exist_ok=True)
    wavs16k_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        list(audio_dir.glob("*.wav"))
        + list(audio_dir.glob("*.mp3"))
        + list(audio_dir.glob("*.flac"))
    )
    if not audio_files:
        raise ValueError(f"No audio files found in {audio_dir}")

    start_sec = parse_time(start_time) if start_time is not None else None
    end_sec = parse_time(end_time) if end_time is not None else None
    duration_sec = (end_sec - start_sec) if (start_sec is not None and end_sec is not None) else None

    _preprocess_audio(
        audio_files, gt_wavs_dir, wavs16k_dir, callback,
        start_time=start_sec, end_time=end_sec, duration=duration_sec,
    )
    _write_state(temp_dir, {"stage": "preprocess", "current_epoch": 0})

    chunks = len(list(gt_wavs_dir.glob("*.wav")))
    _log(callback, f"Preprocessing complete: {chunks} chunks created", 1.0)

    return {
        "chunks_created": chunks,
        "gt_wavs_dir": str(gt_wavs_dir),
        "wavs16k_dir": str(wavs16k_dir),
    }


def run_extract(
    voice_name: str,
    output_dir: Path = Path("voices"),
    step: Optional[str] = None,
    use_decoder: bool | None = None,
    callback: Optional[Callable[[str, float], None]] = None,
    device: str | None = None,
) -> dict:
    """Extract HuBERT features, F0 pitch, and/or mel targets.

    Args:
        voice_name: Voice name (used to locate temp dir).
        output_dir: Parent directory for voices/ and training_temp/.
        step: Run a single step: "hubert", "f0", or "mel". None = all.
        use_decoder: If True, uses hop_length=160; else 300.
            None = read from voice config.
        callback: Progress hook.
        device: Device override string (CLI-only passthrough; no config merge
            for the extract path).  None/"auto" = auto-detect.

    Returns a dict with ``steps_run`` (list of step names that were executed).
    """
    temp_dir = output_dir.parent / "training_temp" / voice_name
    gt_wavs_dir = temp_dir / "0_gt_wavs"
    wavs16k_dir = temp_dir / "1_16k_wavs"

    # Check WAVs exist (preprocess must have run)
    if not gt_wavs_dir.exists() or len(list(gt_wavs_dir.glob("*.wav"))) == 0:
        raise ValueError(
            f"No WAV chunks found in {gt_wavs_dir}. "
            "Run 'kokoclone-train preprocess' first."
        )

    # Resolve hop_length from voice config if not provided
    cfg_path = output_dir / voice_name / "training_config.json"
    if use_decoder is None and cfg_path.exists():
        import json as _json
        with open(cfg_path) as f:
            cfg = _json.load(f)
        use_decoder = cfg.get("use_decoder", False)
    elif use_decoder is None:
        use_decoder = False

    hop_length = 160 if use_decoder else 300

    completed = _check_completed_stages(temp_dir)
    all_steps = ["hubert", "f0", "mel"]
    steps_to_run = [step] if step else all_steps

    for s in steps_to_run:
        if s not in all_steps:
            raise ValueError(f"Invalid step '{s}'. Must be one of: {', '.join(all_steps)}")

    from kokoclone_train.core.rvc_assets import get_hubert_path
    hubert_path = str(get_hubert_path())

    steps_run = []
    progress_base = 0.0

    # HuBERT features
    if "hubert" in steps_to_run:
        feature_dir = temp_dir / "2_feature"
        feature_dir.mkdir(exist_ok=True)
        if completed.get("features"):
            _log(callback, "Skipping HuBERT extraction (already done)", progress_base)
        else:
            _log(callback, "Extracting HuBERT content features...", progress_base)
            _extract_features(wavs16k_dir, feature_dir, hubert_path, callback, device=device)
            _write_state(temp_dir, {"stage": "features", "current_epoch": 0})
            steps_run.append("hubert")
        progress_base = 0.33

    # F0 pitch
    if "f0" in steps_to_run:
        f0_dir = temp_dir / "3_f0"
        f0_dir.mkdir(exist_ok=True)
        if completed.get("f0"):
            _log(callback, "Skipping F0 extraction (already done)", progress_base)
        else:
            _log(callback, "Extracting F0 pitch data...", progress_base)
            _extract_f0_data(gt_wavs_dir, f0_dir, callback, hop_length=hop_length)
            _write_state(temp_dir, {"stage": "f0", "current_epoch": 0})
            steps_run.append("f0")
        progress_base = 0.66

    # Mel targets
    if "mel" in steps_to_run:
        mel_dir = temp_dir / "4_mel"
        mel_dir.mkdir(exist_ok=True)
        if completed.get("mel"):
            _log(callback, "Skipping mel extraction (already done)", progress_base)
        else:
            _log(callback, "Extracting mel spectrogram targets...", progress_base)
            _extract_mel_targets(gt_wavs_dir, mel_dir, callback, hop_length=hop_length)
            _write_state(temp_dir, {"stage": "mel", "current_epoch": 0})
            steps_run.append("mel")

    _log(callback, f"Extraction complete: ran {', '.join(steps_run) if steps_run else 'none (all done)'}", 1.0)
    return {"steps_run": steps_run}


def run_train_only(
    voice_name: str,
    output_dir: Path = Path("voices"),
    epochs: int | None = None,
    pitch_adjustment: float | None = None,
    checkpoint_interval: int | None = None,
    use_decoder: bool | None = None,
    batch_size: int | None = None,
    xtts_augment: bool | None = None,
    aug_ratio: float | None = None,
    resume: bool = False,
    training_config: Optional[dict] = None,
    callback: Optional[Callable[[str, float], None]] = None,
    device: str | None = None,
) -> dict:
    """Run the training loop and build FAISS index.

    Requires preprocessing + feature extraction to have completed first.

    Args:
        device: Device override string ("auto", "cuda[:N]", "xpu[:N]", "mps",
            "cpu").  None = from ``training_config`` ("device" key), falling back
            to "auto".

    Returns a dict with ``model_path``, ``index_path``.
    """
    temp_dir = output_dir.parent / "training_temp" / voice_name
    voice_output = output_dir / voice_name
    voice_output.mkdir(parents=True, exist_ok=True)

    # Check features exist (extract must have run)
    completed = _check_completed_stages(temp_dir)
    if not completed.get("features") or not completed.get("f0") or not completed.get("mel"):
        missing = []
        if not completed.get("features"):
            missing.append("HuBERT features")
        if not completed.get("f0"):
            missing.append("F0 data")
        if not completed.get("mel"):
            missing.append("mel targets")
        raise ValueError(
            f"Missing extracted data: {', '.join(missing)}. "
            "Run 'kokoclone-train extract' first."
        )

    # Resolve use_decoder and hop_length
    from kokoclone_train.core.training_config import (
        get_device, get_use_decoder, get_epochs, get_pitch_adjustment,
        get_checkpoint_interval, get_batch_size,
        get_xtts_augment, get_aug_ratio,
    )

    if use_decoder is None:
        use_decoder = get_use_decoder(training_config or {})
    hop_length = 160 if use_decoder else 300

    # Resolve device once: explicit arg > training_config["device"] > "auto".
    resolved_device = device if device is not None else get_device(training_config or {})

    resolved_epochs = epochs if epochs is not None else get_epochs(training_config or {})
    resolved_pitch = pitch_adjustment if pitch_adjustment is not None else get_pitch_adjustment(training_config or {})
    resolved_ci = checkpoint_interval if checkpoint_interval is not None else get_checkpoint_interval(training_config or {})
    resolved_bs = batch_size if batch_size is not None else get_batch_size(training_config or {})
    resolved_xtts = xtts_augment if xtts_augment is not None else get_xtts_augment(training_config or {})
    resolved_ar = aug_ratio if aug_ratio is not None else get_aug_ratio(training_config or {})

    _log(callback, "Training RVC model...", 0.0)
    _train_model(
        voice_name=voice_name,
        temp_dir=temp_dir,
        output_dir=voice_output,
        epochs=resolved_epochs,
        pitch_adjustment=resolved_pitch,
        callback=callback,
        resume=resume,
        checkpoint_interval=resolved_ci,
        use_decoder=use_decoder,
        hop_length=hop_length,
        batch_size=resolved_bs,
        xtts_augment=resolved_xtts,
        aug_ratio=resolved_ar,
        training_config=training_config,
        device=resolved_device,
    )

    # Build FAISS index
    _log(callback, "Building FAISS feature index...", 0.95)
    index_path = voice_output / "v1.index"
    feature_dir = temp_dir / "2_feature"
    _build_faiss_index(feature_dir, index_path)

    model_path = voice_output / f"{voice_name}.pth"
    _log(callback, f"Training complete. Model: {model_path}", 1.0)

    return {
        "model_path": str(model_path),
        "index_path": str(index_path),
    }
