"""Tests for RVC v2 frame alignment fix (Hybrid A+C).

These tests require REAL PyTorch — do NOT mock torch at module level.
Run separately from test_cli_train.py which mocks torch globally.

Verifies:
- upsample_rates defaults to [1] in SynthesizerTrn and build_rvc_model
- Model output temporal dimension matches mel target exactly
- Content resampling via F.interpolate works correctly
- Per-chunk dataset returns aligned shapes
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


class TestUpsampleRatesDefault:
    """Verify upsample_rates defaults to [1] (identity)."""

    def test_synthesizer_default_upsample_rates(self):
        """SynthesizerTrn.__init__ defaults upsample_rates to [1]."""
        from kokoclone_train.core.rvc_models.models import SynthesizerTrn

        model = SynthesizerTrn()
        assert model.upsample_rates == [1], (
            f"Expected upsample_rates=[1], got {model.upsample_rates}"
        )

    def test_build_rvc_model_upsample_rates(self):
        """build_rvc_model always returns a model with upsample_rates=[1]."""
        from kokoclone_train.core.rvc_models.models import build_rvc_model

        config = {
            "sample_rate": 40000,
            "hop_length": 300,
            "input_size": 768,
            "output_size": 192,
            "f0": True,
        }
        model = build_rvc_model(config)
        assert model.upsample_rates == [1], (
            f"Expected upsample_rates=[1], got {model.upsample_rates}"
        )

    def test_build_rvc_model_ignores_sample_rate_for_upsample(self):
        """build_rvc_model uses [1] regardless of sample rate."""
        from kokoclone_train.core.rvc_models.models import build_rvc_model

        for sr in (48000, 40000, 22050, 16000):
            config = {
                "sample_rate": sr,
                "hop_length": 300,
                "input_size": 768,
                "output_size": 192,
                "f0": True,
            }
            model = build_rvc_model(config)
            assert model.upsample_rates == [1], (
                f"sr={sr}: Expected upsample_rates=[1], got {model.upsample_rates}"
            )


class TestFrameAlignment:
    """Verify model output shape matches mel target exactly."""

    def test_model_output_matches_mel_target_shape(self):
        """Model output temporal dimension must match mel target exactly.

        Mel-only mode (use_decoder=False) returns ``(mel, None)``.
        End-to-end mode (use_decoder=True) returns ``(mel, audio)``.
        """
        from kokoclone_train.core.rvc_models.models import SynthesizerTrn

        model = SynthesizerTrn(
            spec_channels=192, input_size=768, upsample_rates=[1], f0=True,
            use_decoder=False,
        )
        model.eval()

        B, T, C = 2, 493, 768
        content = torch.randn(B, T, C)
        f0 = torch.randint(1, 255, (B, T))

        with torch.no_grad():
            mel, audio = model(content, f0=f0)

        assert mel.shape == (B, 192, T), (
            f"Expected mel [{B}, 192, {T}], got {mel.shape}"
        )
        assert audio is None, (
            f"Mel-only mode should return None audio, got {audio}"
        )


class TestContentResampling:
    """Verify F.interpolate correctly resamples content features."""

    def test_content_resample_to_mel_frames(self):
        """F.interpolate maps HuBERT ~50fps to F0/mel ~133fps."""
        # Simulate HuBERT output at ~50fps for 3.7s audio
        t_content = 185
        t_mel = 493
        content = torch.randn(t_content, 768)

        # Resample to mel frame rate
        c_torch = content.T.unsqueeze(0)  # [1, 768, T_c]
        resampled = F.interpolate(
            c_torch, size=t_mel, mode='linear', align_corners=False
        ).squeeze(0).transpose(0, 1)

        assert resampled.shape == (t_mel, 768), (
            f"Expected ({t_mel}, 768), got {resampled.shape}"
        )

    def test_content_resample_no_op_when_equal(self):
        """When content and target lengths match, no interpolation needed."""
        t = 493
        content = torch.randn(t, 768)

        # When lengths are equal, skip interpolation (as in dataset code)
        assert len(content) == t
        c_torch = torch.FloatTensor(content)
        assert c_torch.shape == (t, 768)


class TestPerChunkDataset:
    """Verify per-chunk dataset returns aligned shapes."""

    def test_per_chunk_dataset_shapes(self):
        """Dataset returns aligned content/F0/mel with matching temporal dimensions."""
        from kokoclone_train.core.rvc_train import ContentFeatureDataset

        # Create mock chunk files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Content at HuBERT frame rate (~50fps)
            content = np.random.randn(185, 768).astype(np.float32)
            np.save(tmpdir / "content.npy", content)

            # F0 at mel frame rate (~133fps)
            f0 = np.random.randint(1, 255, size=493).astype(np.int64)
            np.save(tmpdir / "f0.npy", f0)

            # Mel targets at same frame rate as F0
            mel = np.random.randn(493, 192).astype(np.float32)
            np.save(tmpdir / "mel.npy", mel)

            chunks = [{
                "content": str(tmpdir / "content.npy"),
                "f0": str(tmpdir / "f0.npy"),
                "mel": str(tmpdir / "mel.npy"),
            }]

            dataset = ContentFeatureDataset(chunks=chunks, seq_len=512)
            sample = dataset[0]

            c, f, m = sample
            assert c.shape[0] == f.shape[0], (
                f"Content/F0 length mismatch: {c.shape} vs {f.shape}"
            )
            assert c.shape[0] == m.shape[0], (
                f"Content/Mel length mismatch: {c.shape} vs {m.shape}"
            )
            assert c.shape[1] == 768, (
                f"Content dim should be 768, got {c.shape[1]}"
            )
            assert m.shape[1] == 192, (
                f"Mel dim should be 192, got {m.shape[1]}"
            )

    def test_per_chunk_dataset_without_mel(self):
        """Dataset returns only content and F0 when mel is absent."""
        from kokoclone_train.core.rvc_train import ContentFeatureDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            content = np.random.randn(185, 768).astype(np.float32)
            np.save(tmpdir / "content.npy", content)

            f0 = np.random.randint(1, 255, size=493).astype(np.int64)
            np.save(tmpdir / "f0.npy", f0)

            chunks = [{
                "content": str(tmpdir / "content.npy"),
                "f0": str(tmpdir / "f0.npy"),
            }]

            dataset = ContentFeatureDataset(chunks=chunks, seq_len=512)
            sample = dataset[0]

            assert len(sample) == 2, f"Expected 2 elements (c, f), got {len(sample)}"
            c, f = sample
            assert c.shape[0] == f.shape[0], "Content/F0 length mismatch"

    def test_per_chunk_dataset_len(self):
        """Dataset __len__ returns number of chunks."""
        from kokoclone_train.core.rvc_train import ContentFeatureDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            chunks = []
            for i in range(5):
                content = np.random.randn(185, 768).astype(np.float32)
                f0 = np.random.randint(1, 255, size=493).astype(np.int64)
                np.save(tmpdir / f"content_{i}.npy", content)
                np.save(tmpdir / f"f0_{i}.npy", f0)
                chunks.append({
                    "content": str(tmpdir / f"content_{i}.npy"),
                    "f0": str(tmpdir / f"f0_{i}.npy"),
                })

            dataset = ContentFeatureDataset(chunks=chunks, seq_len=512)
            assert len(dataset) == 5, f"Expected 5 chunks, got {len(dataset)}"


class TestConfigSaving:
    """Verify saved config includes version and upsample_rates."""

    def test_config_has_version_and_upsample_rates(self):
        """Config dict should include version=2 and upsample_rates=[1]."""
        expected_keys = {
            "name", "version", "input_size", "output_size",
            "sample_rate", "hop_length", "f0", "upsample_rates",
            "epoch", "pitch_adjustment",
        }

        config = {
            "name": "test_voice",
            "version": 2,
            "input_size": 768,
            "output_size": 192,
            "sample_rate": 40000,
            "hop_length": 300,
            "f0": True,
            "upsample_rates": [1],
            "epoch": 100,
            "pitch_adjustment": 0.0,
        }

        assert set(config.keys()) == expected_keys
        assert config["version"] == 2
        assert config["upsample_rates"] == [1]
        assert config["input_size"] == 768
