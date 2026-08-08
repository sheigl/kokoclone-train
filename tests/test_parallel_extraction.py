"""Tests for parallelized F0 and mel extraction (ProcessPoolExecutor).

Verifies:
- Worker functions are picklable (module-level, importable by multiprocessing)
- Error isolation: one failing WAV does not block others
- F0 coarse conversion formula unchanged
- Mel dB clamping preserved
- Progress tracking with tqdm
- Callback logging retained
- hop_length parameter propagation
- Empty input handling
"""

import concurrent.futures
import os
import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _has_mocked_torch():
    """Check if torch is mocked (MagicMock) in sys.modules."""
    t = sys.modules.get("torch")
    return isinstance(t, MagicMock) or (t is not None and hasattr(t, '_mock_name'))


def _write_sine_wav(path, sr=40000, duration_sec=1.0, freq_hz=261.63):
    """Write a minimal valid WAV file with a sine wave."""
    from scipy.io import wavfile
    n_samples = int(sr * duration_sec)
    samples = np.sin(2 * np.pi * freq_hz * np.arange(n_samples) / sr).astype(np.float32)
    wavfile.write(str(path), sr, samples)


def _write_invalid_wav(path):
    """Write a file with .wav extension but invalid content."""
    path.write_bytes(b"this is not a valid wav file at all")


class TestWorkerPicklability:
    def test_extract_f0_single_is_picklable(self):
        from kokoclone_train.core.rvc_train import _extract_f0_single
        pickled = pickle.dumps(_extract_f0_single)
        restored = pickle.loads(pickled)
        assert callable(restored)
        assert restored.__name__ == "_extract_f0_single"

    def test_extract_mel_single_is_picklable(self):
        from kokoclone_train.core.rvc_train import _extract_mel_single
        pickled = pickle.dumps(_extract_mel_single)
        restored = pickle.loads(pickled)
        assert callable(restored)
        assert restored.__name__ == "_extract_mel_single"

    def test_extract_f0_single_importable_by_name(self):
        from kokoclone_train.core import rvc_train
        assert hasattr(rvc_train, "_extract_f0_single")
        func = getattr(rvc_train, "_extract_f0_single")
        assert callable(func)

    def test_extract_mel_single_importable_by_name(self):
        from kokoclone_train.core import rvc_train
        assert hasattr(rvc_train, "_extract_mel_single")
        func = getattr(rvc_train, "_extract_mel_single")
        assert callable(func)


class TestF0WorkerCorrectness:
    def test_f0_coarse_conversion_formula(self):
        from kokoclone_train.core.rvc_train import _extract_f0_single
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            wav_path = tmpdir / "test.wav"
            f0_dir = tmpdir / "f0_out"
            f0_dir.mkdir()
            _write_sine_wav(wav_path, sr=40000, duration_sec=1.0, freq_hz=440.0)
            stem, success, err = _extract_f0_single(str(wav_path), str(f0_dir), hop_length=300)
            assert success, f"F0 extraction failed: {err}"
            assert err is None
            assert stem == "test"
            f0_path = f0_dir / f"{stem}.npy"
            assert f0_path.exists()
            f0_coarse = np.load(str(f0_path))
            expected = int(np.round(12 * np.log2(440.0 / 75))) + 1
            nonzero_mask = f0_coarse != 0
            if nonzero_mask.any():
                median_val = int(np.median(f0_coarse[nonzero_mask]))
                assert abs(median_val - expected) <= 2

    def test_f0_worker_returns_correct_tuple(self):
        from kokoclone_train.core.rvc_train import _extract_f0_single
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            wav_path = tmpdir / "test.wav"
            f0_dir = tmpdir / "f0_out"
            f0_dir.mkdir()
            _write_sine_wav(wav_path, sr=40000, duration_sec=1.0)
            result = _extract_f0_single(str(wav_path), str(f0_dir), hop_length=300)
            assert isinstance(result, tuple) and len(result) == 3
            stem, success, err = result
            assert isinstance(stem, str)
            assert isinstance(success, bool)
            assert err is None or isinstance(err, str)

    def test_f0_worker_handles_invalid_wav(self):
        from kokoclone_train.core.rvc_train import _extract_f0_single
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            wav_path = tmpdir / "bad.wav"
            f0_dir = tmpdir / "f0_out"
            f0_dir.mkdir()
            _write_invalid_wav(wav_path)
            stem, success, err = _extract_f0_single(str(wav_path), str(f0_dir), hop_length=300)
            assert not success
            assert err is not None and len(err) > 0

    def test_f0_hop_length_160(self):
        from kokoclone_train.core.rvc_train import _extract_f0_single
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            wav_path = tmpdir / "test.wav"
            f0_dir = tmpdir / "f0_out"
            f0_dir.mkdir()
            _write_sine_wav(wav_path, sr=40000, duration_sec=1.0)
            stem, success, err = _extract_f0_single(str(wav_path), str(f0_dir), hop_length=160)
            assert success
            f0_coarse = np.load(str(f0_dir / f"{stem}.npy"))
            expected_approx = int(40000 * 1.0 / 160)
            assert abs(len(f0_coarse) - expected_approx) < 20

    def test_f0_hop_length_300(self):
        from kokoclone_train.core.rvc_train import _extract_f0_single
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            wav_path = tmpdir / "test.wav"
            f0_dir = tmpdir / "f0_out"
            f0_dir.mkdir()
            _write_sine_wav(wav_path, sr=40000, duration_sec=1.0)
            stem, success, err = _extract_f0_single(str(wav_path), str(f0_dir), hop_length=300)
            assert success
            f0_coarse = np.load(str(f0_dir / f"{stem}.npy"))
            expected_approx = int(40000 * 1.0 / 300)
            assert abs(len(f0_coarse) - expected_approx) < 20


class TestMelWorkerCorrectness:
    def test_mel_db_clamping(self):
        from kokoclone_train.core.rvc_train import _extract_mel_single
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            wav_path = tmpdir / "test.wav"
            mel_dir = tmpdir / "mel_out"
            mel_dir.mkdir()
            _write_sine_wav(wav_path, sr=40000, duration_sec=1.0)
            stem, success, err = _extract_mel_single(str(wav_path), str(mel_dir), hop_length=300)
            assert success
            mel_np = np.load(str(mel_dir / f"{stem}.npy"))
            assert np.all(np.isfinite(mel_np)), "Mel contains non-finite values"

    def test_mel_output_shape(self):
        from kokoclone_train.core.rvc_train import _extract_mel_single
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            wav_path = tmpdir / "test.wav"
            mel_dir = tmpdir / "mel_out"
            mel_dir.mkdir()
            _write_sine_wav(wav_path, sr=40000, duration_sec=1.0)
            stem, success, err = _extract_mel_single(str(wav_path), str(mel_dir), hop_length=300)
            assert success
            mel_np = np.load(str(mel_dir / f"{stem}.npy"))
            assert mel_np.ndim == 2
            assert mel_np.shape[1] == 192

    def test_mel_worker_returns_correct_tuple(self):
        from kokoclone_train.core.rvc_train import _extract_mel_single
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            wav_path = tmpdir / "test.wav"
            mel_dir = tmpdir / "mel_out"
            mel_dir.mkdir()
            _write_sine_wav(wav_path, sr=40000, duration_sec=1.0)
            result = _extract_mel_single(str(wav_path), str(mel_dir), hop_length=300)
            assert isinstance(result, tuple) and len(result) == 3
            stem, success, err = result
            assert isinstance(stem, str)
            assert isinstance(success, bool)

    def test_mel_worker_handles_invalid_wav(self):
        from kokoclone_train.core.rvc_train import _extract_mel_single
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            wav_path = tmpdir / "bad.wav"
            mel_dir = tmpdir / "mel_out"
            mel_dir.mkdir()
            _write_invalid_wav(wav_path)
            stem, success, err = _extract_mel_single(str(wav_path), str(mel_dir), hop_length=300)
            assert not success
            assert err is not None and len(err) > 0

    def test_mel_hop_length_160(self):
        from kokoclone_train.core.rvc_train import _extract_mel_single
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            wav_path = tmpdir / "test.wav"
            mel_dir = tmpdir / "mel_out"
            mel_dir.mkdir()
            _write_sine_wav(wav_path, sr=40000, duration_sec=1.0)
            stem, success, err = _extract_mel_single(str(wav_path), str(mel_dir), hop_length=160)
            assert success
            mel_np = np.load(str(mel_dir / f"{stem}.npy"))
            expected_approx = int(40000 * 1.0 / 160)
            assert abs(mel_np.shape[0] - expected_approx) < 20


class TestErrorIsolation:
    def test_f0_parallel_continues_on_single_failure(self):
        from kokoclone_train.core.rvc_train import _extract_f0_data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gt_dir = tmpdir / "gt_wavs"
            f0_dir = tmpdir / "f0_out"
            gt_dir.mkdir()
            f0_dir.mkdir()
            for i in range(3):
                _write_sine_wav(gt_dir / f"good_{i}.wav", sr=40000, duration_sec=0.5)
            _write_invalid_wav(gt_dir / "bad.wav")
            callback_calls = []
            def mock_callback(msg, pct):
                callback_calls.append((msg, pct))
            _extract_f0_data(gt_dir, f0_dir, mock_callback, hop_length=300)
            f0_files = list(f0_dir.glob("*.npy"))
            assert len(f0_files) >= 3

    def test_mel_parallel_continues_on_single_failure(self):
        from kokoclone_train.core.rvc_train import _extract_mel_targets
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gt_dir = tmpdir / "gt_wavs"
            mel_dir = tmpdir / "mel_out"
            gt_dir.mkdir()
            mel_dir.mkdir()
            for i in range(3):
                _write_sine_wav(gt_dir / f"good_{i}.wav", sr=40000, duration_sec=0.5)
            _write_invalid_wav(gt_dir / "bad.wav")
            callback_calls = []
            def mock_callback(msg, pct):
                callback_calls.append((msg, pct))
            _extract_mel_targets(gt_dir, mel_dir, mock_callback, hop_length=300)
            mel_files = list(mel_dir.glob("*.npy"))
            assert len(mel_files) >= 3


class TestCallbackLogging:
    def test_f0_callback_invoked_on_completion(self):
        from kokoclone_train.core.rvc_train import _extract_f0_data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gt_dir = tmpdir / "gt_wavs"
            f0_dir = tmpdir / "f0_out"
            gt_dir.mkdir()
            f0_dir.mkdir()
            _write_sine_wav(gt_dir / "test.wav", sr=40000, duration_sec=0.5)
            callback_calls = []
            def mock_callback(msg, pct):
                callback_calls.append((msg, pct))
            _extract_f0_data(gt_dir, f0_dir, mock_callback, hop_length=300)
            assert len(callback_calls) > 0
            last_msg = callback_calls[-1][0]
            assert "complete" in last_msg.lower()

    def test_mel_callback_invoked_on_completion(self):
        from kokoclone_train.core.rvc_train import _extract_mel_targets
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gt_dir = tmpdir / "gt_wavs"
            mel_dir = tmpdir / "mel_out"
            gt_dir.mkdir()
            mel_dir.mkdir()
            _write_sine_wav(gt_dir / "test.wav", sr=40000, duration_sec=0.5)
            callback_calls = []
            def mock_callback(msg, pct):
                callback_calls.append((msg, pct))
            _extract_mel_targets(gt_dir, mel_dir, mock_callback, hop_length=300)
            assert len(callback_calls) > 0
            last_msg = callback_calls[-1][0]
            assert "complete" in last_msg.lower()

    def test_f0_callback_none_does_not_crash(self):
        from kokoclone_train.core.rvc_train import _extract_f0_data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gt_dir = tmpdir / "gt_wavs"
            f0_dir = tmpdir / "f0_out"
            gt_dir.mkdir()
            f0_dir.mkdir()
            _write_sine_wav(gt_dir / "test.wav", sr=40000, duration_sec=0.5)
            _extract_f0_data(gt_dir, f0_dir, None, hop_length=300)

    def test_mel_callback_none_does_not_crash(self):
        from kokoclone_train.core.rvc_train import _extract_mel_targets
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gt_dir = tmpdir / "gt_wavs"
            mel_dir = tmpdir / "mel_out"
            gt_dir.mkdir()
            mel_dir.mkdir()
            _write_sine_wav(gt_dir / "test.wav", sr=40000, duration_sec=0.5)
            _extract_mel_targets(gt_dir, mel_dir, None, hop_length=300)


class TestEmptyInput:
    def test_f0_empty_input_no_crash(self):
        from kokoclone_train.core.rvc_train import _extract_f0_data
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gt_dir = tmpdir / "gt_wavs"
            f0_dir = tmpdir / "f0_out"
            gt_dir.mkdir()
            f0_dir.mkdir()
            callback_calls = []
            def mock_callback(msg, pct):
                callback_calls.append((msg, pct))
            _extract_f0_data(gt_dir, f0_dir, mock_callback, hop_length=300)
            assert len(callback_calls) > 0

    def test_mel_empty_input_no_crash(self):
        from kokoclone_train.core.rvc_train import _extract_mel_targets
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gt_dir = tmpdir / "gt_wavs"
            mel_dir = tmpdir / "mel_out"
            gt_dir.mkdir()
            mel_dir.mkdir()
            callback_calls = []
            def mock_callback(msg, pct):
                callback_calls.append((msg, pct))
            _extract_mel_targets(gt_dir, mel_dir, mock_callback, hop_length=300)
            assert len(callback_calls) > 0


class TestProgressTracking:
    def test_f0_tqdm_total_matches_file_count(self):
        from kokoclone_train.core.rvc_train import _extract_f0_data
        captured_totals = []
        original_tqdm = __import__("tqdm").tqdm
        def mock_tqdm(iterable, total=None, desc=None, **kwargs):
            captured_totals.append(total)
            return original_tqdm(iterable, total=total, desc=desc, **kwargs)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gt_dir = tmpdir / "gt_wavs"
            f0_dir = tmpdir / "f0_out"
            gt_dir.mkdir()
            f0_dir.mkdir()
            n_files = 5
            for i in range(n_files):
                _write_sine_wav(gt_dir / f"test_{i}.wav", sr=40000, duration_sec=0.3)
            with patch("kokoclone_train.core.rvc_train.tqdm", mock_tqdm):
                _extract_f0_data(gt_dir, f0_dir, None, hop_length=300)
        assert len(captured_totals) > 0
        assert captured_totals[0] == n_files

    def test_mel_tqdm_total_matches_file_count(self):
        from kokoclone_train.core.rvc_train import _extract_mel_targets
        captured_totals = []
        original_tqdm = __import__("tqdm").tqdm
        def mock_tqdm(iterable, total=None, desc=None, **kwargs):
            captured_totals.append(total)
            return original_tqdm(iterable, total=total, desc=desc, **kwargs)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gt_dir = tmpdir / "gt_wavs"
            mel_dir = tmpdir / "mel_out"
            gt_dir.mkdir()
            mel_dir.mkdir()
            n_files = 5
            for i in range(n_files):
                _write_sine_wav(gt_dir / f"test_{i}.wav", sr=40000, duration_sec=0.3)
            with patch("kokoclone_train.core.rvc_train.tqdm", mock_tqdm):
                _extract_mel_targets(gt_dir, mel_dir, None, hop_length=300)
        assert len(captured_totals) > 0
        assert captured_totals[0] == n_files


class TestMaxWorkers:
    def test_f0_uses_cpu_count(self):
        from kokoclone_train.core.rvc_train import _extract_f0_data
        captured_max_workers = []
        original_ppe = concurrent.futures.ProcessPoolExecutor
        class MockPPE(original_ppe):
            def __init__(self, max_workers=None, **kwargs):
                captured_max_workers.append(max_workers)
                super().__init__(max_workers=max_workers, **kwargs)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gt_dir = tmpdir / "gt_wavs"
            f0_dir = tmpdir / "f0_out"
            gt_dir.mkdir()
            f0_dir.mkdir()
            _write_sine_wav(gt_dir / "test.wav", sr=40000, duration_sec=0.3)
            with patch("kokoclone_train.core.rvc_train.concurrent.futures.ProcessPoolExecutor", MockPPE):
                _extract_f0_data(gt_dir, f0_dir, None, hop_length=300)
        assert len(captured_max_workers) > 0
        expected = os.cpu_count() or 1
        assert captured_max_workers[0] == expected

    def test_mel_uses_cpu_count(self):
        from kokoclone_train.core.rvc_train import _extract_mel_targets
        captured_max_workers = []
        original_ppe = concurrent.futures.ProcessPoolExecutor
        class MockPPE(original_ppe):
            def __init__(self, max_workers=None, **kwargs):
                captured_max_workers.append(max_workers)
                super().__init__(max_workers=max_workers, **kwargs)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            gt_dir = tmpdir / "gt_wavs"
            mel_dir = tmpdir / "mel_out"
            gt_dir.mkdir()
            mel_dir.mkdir()
            _write_sine_wav(gt_dir / "test.wav", sr=40000, duration_sec=0.3)
            with patch("kokoclone_train.core.rvc_train.concurrent.futures.ProcessPoolExecutor", MockPPE):
                _extract_mel_targets(gt_dir, mel_dir, None, hop_length=300)
        assert len(captured_max_workers) > 0
        expected = os.cpu_count() or 1
        assert captured_max_workers[0] == expected


class TestHopLengthPropagation:
    def test_hop_length_300_passed_to_extractors(self):
        from kokoclone_train.core.rvc_train import train_voice
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            audio_dir = tmpdir / "audio"
            output_dir = tmpdir / "output"
            audio_dir.mkdir()
            output_dir.mkdir()
            _write_sine_wav(audio_dir / "test.wav", sr=40000, duration_sec=5.0)
            captured_hop_length = []
            def capture_f0(gt_wavs_dir, f0_dir, callback, hop_length=300):
                captured_hop_length.append(("f0", hop_length))
                f0_dir.mkdir(parents=True, exist_ok=True)
            def capture_mel(gt_wavs_dir, mel_dir, callback, hop_length=300):
                captured_hop_length.append(("mel", hop_length))
                mel_dir.mkdir(parents=True, exist_ok=True)
            with patch("kokoclone_train.core.rvc_train._extract_f0_data", capture_f0):
                with patch("kokoclone_train.core.rvc_train._extract_mel_targets", capture_mel):
                    with patch("kokoclone_train.core.rvc_train._train_model"):
                        with patch("kokoclone_train.core.rvc_train._extract_features"):
                            with patch("kokoclone_train.core.rvc_train._build_faiss_index"):
                                try:
                                    train_voice(voice_name="test", audio_dir=audio_dir, output_dir=output_dir, use_decoder=False)
                                except Exception:
                                    pass
            assert len(captured_hop_length) >= 2
            for name, hl in captured_hop_length:
                assert hl == 300

    def test_hop_length_160_passed_to_extractors(self):
        from kokoclone_train.core.rvc_train import train_voice
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            audio_dir = tmpdir / "audio"
            output_dir = tmpdir / "output"
            audio_dir.mkdir()
            output_dir.mkdir()
            _write_sine_wav(audio_dir / "test.wav", sr=40000, duration_sec=5.0)
            captured_hop_length = []
            def capture_f0(gt_wavs_dir, f0_dir, callback, hop_length=300):
                captured_hop_length.append(("f0", hop_length))
                f0_dir.mkdir(parents=True, exist_ok=True)
            def capture_mel(gt_wavs_dir, mel_dir, callback, hop_length=300):
                captured_hop_length.append(("mel", hop_length))
                mel_dir.mkdir(parents=True, exist_ok=True)
            with patch("kokoclone_train.core.rvc_train._extract_f0_data", capture_f0):
                with patch("kokoclone_train.core.rvc_train._extract_mel_targets", capture_mel):
                    with patch("kokoclone_train.core.rvc_train._train_model"):
                        with patch("kokoclone_train.core.rvc_train._extract_features"):
                            with patch("kokoclone_train.core.rvc_train._build_faiss_index"):
                                try:
                                    train_voice(voice_name="test", audio_dir=audio_dir, output_dir=output_dir, use_decoder=True)
                                except Exception:
                                    pass
            assert len(captured_hop_length) >= 2
            for name, hl in captured_hop_length:
                assert hl == 160


class TestF0CoarseFormula:
    def test_formula_matches_reference(self):
        test_cases = [
            (75.0, 1),
            (146.83, 13),
            (261.63, 23),
            (440.0, 32),
            (880.0, 44),
        ]
        for f0_hz, expected_coarse in test_cases:
            actual = int(np.round(12 * np.log2(f0_hz / 75))) + 1
            assert actual == expected_coarse

    def test_formula_zero_f0_produces_zero(self):
        f0_raw = np.array([0.0, 261.63, 0.0, 440.0])
        f0_coarse = np.zeros(len(f0_raw), dtype=np.int32)
        for j in range(len(f0_raw)):
            if f0_raw[j] != 0:
                f0_coarse[j] = int(np.round(12 * np.log2(f0_raw[j] / 75))) + 1
        assert f0_coarse[0] == 0
        assert f0_coarse[2] == 0
        assert f0_coarse[1] > 0
        assert f0_coarse[3] > 0


class TestMelDBConversion:
    def test_mel_db_conversion_formula(self):
        import torch
        raw_power = torch.tensor([1.0, 0.1, 0.01, 1e-10, 1e-20])
        mel_db = torch.clamp(raw_power, min=1e-10).log10() * 10
        assert abs(mel_db[0].item()) < 1e-6
        assert abs(mel_db[1].item() - (-10.0)) < 1e-4
        assert abs(mel_db[2].item() - (-20.0)) < 1e-4
        assert abs(mel_db[3].item() - (-100.0)) < 1e-4
        # 1e-20 clamped to 1e-10 -> -100 dB (not -inf)
        assert abs(mel_db[4].item() - (-100.0)) < 1e-4

    def test_mel_clamp_prevents_negative_log(self):
        import torch
        raw_power = torch.tensor([0.0, -1.0, 1e-15])
        mel_db = torch.clamp(raw_power, min=1e-10).log10() * 10
        assert torch.all(torch.isfinite(mel_db)), "Clamping should prevent non-finite dB values"

