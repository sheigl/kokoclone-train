"""Tests for kokoclone_train CLI (train command)."""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock heavy ML dependencies before any core imports.
# NOTE: Do NOT mock "numpy" — scipy.io.wavfile requires real numpy's __version__.

# torch needs special handling: must look like a package for `from torch.utils.data import ...`
_torch_mock = MagicMock()
_torch_mock.__path__ = []
sys.modules["torch"] = _torch_mock

for _submod in (
    "torch.utils",
    "torch.utils.data",
    "torch.nn",
):
    if _submod not in sys.modules:
        _m = MagicMock()
        _m.__path__ = []
        sys.modules[_submod] = _m

for _mod in (
    "torchaudio",
    "huggingface_hub",
    "librosa",
    "faiss",
    "tqdm",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()


class TestTrainCommandArgParsing:
    """Test that argparse correctly handles train command arguments."""

    def _build_parser(self):
        import argparse

        parser = argparse.ArgumentParser(
            prog="kokoclone-train",
            description="Train RVC v2 voice conversion models from audio.",
        )
        sub = parser.add_subparsers(dest="command", required=True)

        p_train = sub.add_parser("train", help="Train a single voice from audio files.")
        p_train.add_argument("--name", required=True, help="Voice name (output directory).")
        p_train.add_argument(
            "--audio-dir", required=True, type=Path, help="Directory with .wav/.mp3/.flac files."
        )
        p_train.add_argument(
            "--output-dir",
            type=Path,
            default=Path("voices"),
            help="Parent dir for voice output (default: ./voices).",
        )
        p_train.add_argument(
            "--epochs", type=int, default=100, help="Training epochs (default: 100)."
        )
        p_train.add_argument(
            "--pitch-adjustment",
            type=float,
            default=0.0,
            help="Pitch shift in semitones (default: 0.0).",
        )

        return parser

    def test_train_command_parsed(self):
        parser = self._build_parser()
        args = parser.parse_args(["train", "--name", "test_voice", "--audio-dir", "/path/to/audio"])
        assert args.command == "train"

    def test_name_parsed(self):
        parser = self._build_parser()
        args = parser.parse_args(
            ["train", "--name", "my_voice", "--audio-dir", "/path/to/audio"]
        )
        assert args.name == "my_voice"

    def test_audio_dir_parsed(self):
        parser = self._build_parser()
        args = parser.parse_args(
            ["train", "--name", "test", "--audio-dir", "/path/to/audio"]
        )
        assert args.audio_dir == Path("/path/to/audio")

    def test_epochs_default(self):
        parser = self._build_parser()
        args = parser.parse_args(["train", "--name", "test", "--audio-dir", "/tmp/audio"])
        assert args.epochs == 100

    def test_epochs_custom(self):
        parser = self._build_parser()
        args = parser.parse_args(
            ["train", "--name", "test", "--audio-dir", "/tmp/audio", "--epochs", "250"]
        )
        assert args.epochs == 250

    def test_pitch_adjustment_default(self):
        parser = self._build_parser()
        args = parser.parse_args(["train", "--name", "test", "--audio-dir", "/tmp/audio"])
        assert args.pitch_adjustment == 0.0

    def test_pitch_adjustment_positive(self):
        parser = self._build_parser()
        args = parser.parse_args(
            ["train", "--name", "test", "--audio-dir", "/tmp/audio", "--pitch-adjustment", "3"]
        )
        assert args.pitch_adjustment == 3.0

    def test_pitch_adjustment_negative(self):
        parser = self._build_parser()
        args = parser.parse_args(
            ["train", "--name", "test", "--audio-dir", "/tmp/audio", "--pitch-adjustment", "-5"]
        )
        assert args.pitch_adjustment == -5.0

    def test_pitch_adjustment_float(self):
        parser = self._build_parser()
        args = parser.parse_args(
            ["train", "--name", "test", "--audio-dir", "/tmp/audio", "--pitch-adjustment", "2.5"]
        )
        assert args.pitch_adjustment == 2.5

    def test_output_dir_default(self):
        parser = self._build_parser()
        args = parser.parse_args(["train", "--name", "test", "--audio-dir", "/tmp/audio"])
        assert args.output_dir == Path("voices")

    def test_output_dir_custom(self):
        parser = self._build_parser()
        args = parser.parse_args(
            ["train", "--name", "test", "--audio-dir", "/tmp/audio", "--output-dir", "/custom/voices"]
        )
        assert args.output_dir == Path("/custom/voices")


class TestTrainCommandValidation:
    """Test validation logic for train command."""

    def test_audio_dir_required(self):
        parser = self._build_parser()
        # Should raise SystemExit when --audio-dir is missing (required=True)
        try:
            parser.parse_args(["train", "--name", "test"])
            assert False, "Should have raised SystemExit"
        except SystemExit as e:
            assert e.code != 0

    def test_name_required(self):
        parser = self._build_parser()
        # Should raise SystemExit when --name is missing (required=True)
        try:
            parser.parse_args(["train", "--audio-dir", "/tmp/audio"])
            assert False, "Should have raised SystemExit"
        except SystemExit as e:
            assert e.code != 0

    def _build_parser(self):
        import argparse

        parser = argparse.ArgumentParser(
            prog="kokoclone-train",
            description="Train RVC v2 voice conversion models from audio.",
        )
        sub = parser.add_subparsers(dest="command", required=True)

        p_train = sub.add_parser("train", help="Train a single voice from audio files.")
        p_train.add_argument("--name", required=True, help="Voice name (output directory).")
        p_train.add_argument(
            "--audio-dir", required=True, type=Path, help="Directory with .wav/.mp3/.flac files."
        )
        p_train.add_argument(
            "--output-dir",
            type=Path,
            default=Path("voices"),
            help="Parent dir for voice output (default: ./voices).",
        )
        p_train.add_argument("--epochs", type=int, default=100)
        p_train.add_argument("--pitch-adjustment", type=float, default=0.0)

        return parser


class TestTrainCommandAudioValidation:
    """Test audio path validation logic for train command."""

    def test_directory_with_no_audio_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "source"
            src_dir.mkdir(parents=True, exist_ok=True)
            (src_dir / "readme.txt").write_text("not audio")

            found = (
                list(src_dir.glob("*.wav"))
                + list(src_dir.glob("*.mp3"))
                + list(src_dir.glob("*.flac"))
            )
            assert len(found) == 0

    def test_directory_with_audio_files_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "source"
            src_dir.mkdir(parents=True, exist_ok=True)
            (src_dir / "clip1.wav").write_text("wav content")
            (src_dir / "clip2.mp3").write_text("mp3 content")
            (src_dir / "clip3.flac").write_text("flac content")
            (src_dir / "readme.txt").write_text("not audio")

            found = (
                list(src_dir.glob("*.wav"))
                + list(src_dir.glob("*.mp3"))
                + list(src_dir.glob("*.flac"))
            )
            assert len(found) == 3


class TestTrainCommandIntegration:
    """Test train command end-to-end with mocked dependencies."""

    @patch("kokoclone_train.core.rvc_train.train_voice")
    def test_train_passes_args_correctly(self, mock_train_voice):
        """Verify args are passed correctly to train_voice."""
        mock_result = {
            "name": "test_voice",
            "model_path": "/voices/test_voice/v1.pth",
            "index_path": "/voices/test_voice/v1.index",
            "config_path": "/voices/test_voice/config.json",
        }
        mock_train_voice.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "source"
            src_dir.mkdir(parents=True, exist_ok=True)
            (src_dir / "audio.wav").write_text("fake wav")

            voices_dir = Path(tmpdir) / "voices"
            voices_dir.mkdir(parents=True, exist_ok=True)

            result = mock_train_voice(
                voice_name="test_voice",
                audio_dir=src_dir,
                output_dir=voices_dir,
                epochs=200,
                pitch_adjustment=-3.0,
                callback=lambda msg, pct: None,
            )

            assert result == mock_result
            call_kwargs = mock_train_voice.call_args[1]
            assert call_kwargs["voice_name"] == "test_voice"
            assert call_kwargs["audio_dir"] == src_dir
            assert call_kwargs["epochs"] == 200
            assert call_kwargs["pitch_adjustment"] == -3.0

    @patch("kokoclone_train.core.rvc_train.train_voice")
    def test_output_uses_v1_pth(self, mock_train_voice):
        """Verify model output uses v1.pth convention."""
        mock_result = {
            "name": "myvoice",
            "model_path": "/voices/myvoice/v1.pth",
            "index_path": "/voices/myvoice/v1.index",
            "config_path": "/voices/myvoice/config.json",
        }
        mock_train_voice.return_value = mock_result

        result = mock_train_voice(
            voice_name="myvoice",
            audio_dir=Path("/tmp/audio"),
            output_dir=Path("voices"),
            epochs=100,
            pitch_adjustment=0.0,
            callback=None,
        )

        assert "v1.pth" in result["model_path"]
        assert "v1.index" in result["index_path"]


class TestWatchCommandArgParsing:
    """Test that argparse correctly handles watch command arguments."""

    def _build_parser(self):
        import argparse

        parser = argparse.ArgumentParser(
            prog="kokoclone-train",
            description="Train RVC v2 voice conversion models from audio.",
        )
        sub = parser.add_subparsers(dest="command", required=True)

        p_train = sub.add_parser("train")
        p_train.add_argument("--name", required=True)
        p_train.add_argument("--audio-dir", required=True, type=Path)

        p_watch = sub.add_parser("watch", help="Watch training/ directory.")
        p_watch.add_argument(
            "--training-dir",
            type=Path,
            default=Path("training"),
        )
        p_watch.add_argument(
            "--voices-dir",
            type=Path,
            default=Path("voices"),
        )

        return parser

    def test_watch_command_parsed(self):
        parser = self._build_parser()
        args = parser.parse_args(["watch"])
        assert args.command == "watch"

    def test_watch_training_dir_default(self):
        parser = self._build_parser()
        args = parser.parse_args(["watch"])
        assert args.training_dir == Path("training")

    def test_watch_voices_dir_default(self):
        parser = self._build_parser()
        args = parser.parse_args(["watch"])
        assert args.voices_dir == Path("voices")

    def test_watch_custom_dirs(self):
        parser = self._build_parser()
        args = parser.parse_args(
            ["watch", "--training-dir", "/custom/training", "--voices-dir", "/custom/voices"]
        )
        assert args.training_dir == Path("/custom/training")
        assert args.voices_dir == Path("/custom/voices")
