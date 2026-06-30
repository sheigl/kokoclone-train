"""HuBERT content feature extraction using transformers library."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


_HUBERT_CACHE: dict[str, "HuBERTWrapper"] = {}


class HuBERTWrapper:
    """Lightweight wrapper around a HuBERT encoder loaded from a checkpoint.

    Uses transformers.AutoModel for loading — works with both fairseq checkpoints
    and HuggingFace model format (facebook/hubert-base-ls960).
    """

    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        from transformers import AutoModel

        # Try loading as a HF model first (works for facebook/hubert-base-ls960)
        try:
            model = AutoModel.from_pretrained(checkpoint_path).to(self.device).eval()
            return model
        except Exception:
            pass

        # Fallback: load state dict directly from .pt file
        import os
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            # Try to reconstruct a HuBERT model from the state dict
            try:
                from transformers import HubertModel
                model = HubertModel.from_pretrained(
                    "facebook/hubert-base-ls960",
                    state_dict=state_dict if isinstance(state_dict, dict) else None,
                ).to(self.device).eval()
                return model
            except Exception:
                pass

        raise RuntimeError(f"Failed to load HuBERT model from {checkpoint_path}")

    def content(self, audio: torch.Tensor | list[float], sr: int = 16000) -> torch.Tensor:
        """Extract HuBERT content features from audio.

        Args:
            audio: Audio samples (float32) at 16 kHz, or a pre-loaded tensor.
            sr: Sample rate (must be 16 kHz for HuBERT).

        Returns:
            Content feature tensor of shape [T, C].
        """
        if not isinstance(audio, torch.Tensor):
            audio = torch.FloatTensor(audio)

        audio = audio.to(self.device)

        # Ensure 2D tensor [batch=1, sequence_length] for transformers HuBERT
        while audio.dim() < 2:
            audio = audio.unsqueeze(0)
        if audio.dim() > 2:
            audio = audio.squeeze()

        if sr != 16000:
            import librosa

            audio_np = audio.cpu().numpy()
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)
            audio = torch.from_numpy(audio_np).to(self.device)

        with torch.no_grad():
            # transformers HuBERT expects 2D input [batch, sequence_length]
            features = self.model(audio).last_hidden_state

        return features.squeeze(0).cpu()


class HuBERTManager:
    """Singleton manager for HuBERT encoder instances."""

    _cache: dict[str, HuBERTWrapper] = {}

    @classmethod
    def get(cls, checkpoint_path: str, device: torch.device | None = None) -> HuBERTWrapper:
        """Return a cached HuBERTWrapper, creating one if needed."""
        if device is None:
            from kokoclone_train.core.rvc_assets import detect_device

            device = detect_device()

        key = f"{checkpoint_path}:{device}"
        if key not in cls._cache:
            cls._cache[key] = HuBERTWrapper(checkpoint_path, device)
        return cls._cache[key]
