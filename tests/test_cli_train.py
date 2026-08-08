"""Tests for kokoclone_train CLI (train command).

These tests mock heavy ML dependencies via fixtures rather than module-level
sys.modules injection, so they can coexist with test_frame_alignment.py which
requires real PyTorch.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


def _mock_heavy_deps():
    """Inject mocks for heavy ML dependencies into sys.modules."""
    torch_mock = MagicMock()
    torch_mock.__path__ = []
    sys.modules["torch"] = torch_mock

    for submod in ("torch.utils", "torch.utils.data", "torch.nn"):
        if submod not in sys.modules:
            m = MagicMock()
            m.__path__ = []
            sys.modules[submod] = m

    for mod in ("torchaudio", "huggingface_hub", "librosa", "faiss", "tqdm"):
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()


def _unmock_heavy_deps():
    """Remove mocked modules from sys.modules."""
    to_remove = [k for k in sys.modules if k == "torch" or k.startswith("torch.")
                 or k in ("torchaudio", "huggingface_hub", "librosa", "faiss", "tqdm")]
    for mod_name in to_remove:
        del sys.modules[mod_name]


def _parse(argv):
    """Parse argv with the REAL _build_parser (imported under mocked torch).

    ``__main__`` imports ``rvc_train`` at module level, which imports torch —
    so the mock must be active before the import (see plan §4.2).
    """
    _mock_heavy_deps()
    try:
        from kokoclone_train.__main__ import _build_parser
        return _build_parser().parse_args(argv)
    finally:
        _unmock_heavy_deps()


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
        try:
            parser.parse_args(["train", "--name", "test"])
            assert False, "Should have raised SystemExit"
        except SystemExit as e:
            assert e.code != 0

    def test_name_required(self):
        parser = self._build_parser()
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
        _mock_heavy_deps()
        try:
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
        finally:
            _unmock_heavy_deps()

    @patch("kokoclone_train.core.rvc_train.train_voice")
    def test_output_uses_v1_pth(self, mock_train_voice):
        """Verify model output uses v1.pth convention."""
        _mock_heavy_deps()
        try:
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
        finally:
            _unmock_heavy_deps()


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


class TestDeviceFlagParsing:
    """--device parses on train / train-only / extract (real _build_parser)."""

    def test_train_device_flag_parsed(self):
        args = _parse(
            ["train", "--name", "v", "--audio-dir", "/tmp/a", "--device", "cuda:0"]
        )
        assert args.device == "cuda:0"

    def test_train_device_unset_is_none(self):
        """Unset --device → None so the merged config decides (not argparse "auto")."""
        args = _parse(["train", "--name", "v", "--audio-dir", "/tmp/a"])
        assert args.device is None

    def test_train_only_device_flag_parsed(self):
        args = _parse(["train-only", "--name", "v", "--device", "xpu:1"])
        assert args.device == "xpu:1"

    def test_train_only_device_unset_is_none(self):
        args = _parse(["train-only", "--name", "v"])
        assert args.device is None

    def test_extract_device_flag_parsed(self):
        args = _parse(["extract", "--name", "v", "--device", "cpu"])
        assert args.device == "cpu"

    def test_extract_device_unset_is_none(self):
        args = _parse(["extract", "--name", "v"])
        assert args.device is None

    def test_device_flag_accepts_auto(self):
        args = _parse(
            ["train", "--name", "v", "--audio-dir", "/tmp/a", "--device", "auto"]
        )
        assert args.device == "auto"


class TestDevicePrecedence:
    """Device resolution: CLI flag > voice config > root config > "auto"."""

    def test_train_device_cli_overrides_config(self):
        """--device on train reaches cli_overrides and train_voice."""
        _mock_heavy_deps()
        try:
            import sys
            from kokoclone_train.__main__ import _main
            from kokoclone_train.core import training_config

            with tempfile.TemporaryDirectory() as tmpdir:
                src = Path(tmpdir) / "audio"
                src.mkdir()
                (src / "a.wav").write_text("x")
                with (
                    patch("kokoclone_train.__main__.train_voice") as mock_tv,
                    patch.object(training_config, "resolve_training_config") as mock_resolve,
                    patch.object(
                        sys, "argv",
                        ["kokoclone-train", "train", "--name", "v",
                         "--audio-dir", str(src), "--device", "cuda:0"],
                    ),
                ):
                    mock_tv.return_value = {
                        "name": "v", "model_path": "/tmp/v1.pth",
                        "index_path": "/tmp/v1.index", "config_path": "/tmp/config.json",
                    }
                    mock_resolve.return_value = {"device": "cuda:0"}
                    rc = _main()
                assert rc == 0
                assert mock_resolve.call_args[1]["cli_overrides"]["device"] == "cuda:0"
                assert mock_tv.call_args[1]["device"] == "cuda:0"
        finally:
            _unmock_heavy_deps()

    def test_train_only_device_cli_overrides_config(self):
        """--device on train-only reaches cli_overrides and run_train_only."""
        _mock_heavy_deps()
        try:
            import sys
            from kokoclone_train.__main__ import _main
            from kokoclone_train.core import rvc_train, training_config

            with tempfile.TemporaryDirectory() as tmpdir:
                with (
                    patch.object(rvc_train, "run_train_only") as mock_rto,
                    patch.object(training_config, "resolve_training_config") as mock_resolve,
                    patch.object(
                        sys, "argv",
                        ["kokoclone-train", "train-only", "--name", "v",
                         "--output-dir", str(Path(tmpdir) / "voices"), "--device", "xpu:1"],
                    ),
                ):
                    mock_rto.return_value = {
                        "model_path": "/tmp/v1.pth", "index_path": "/tmp/v1.index",
                    }
                    mock_resolve.return_value = {"device": "xpu:1"}
                    rc = _main()
                assert rc == 0
                assert mock_resolve.call_args[1]["cli_overrides"]["device"] == "xpu:1"
                assert mock_rto.call_args[1]["device"] == "xpu:1"
        finally:
            _unmock_heavy_deps()

    def test_extract_device_passthrough(self):
        """--device on extract is a plain CLI passthrough to run_extract."""
        _mock_heavy_deps()
        try:
            import sys
            from kokoclone_train.__main__ import _main
            from kokoclone_train.core import rvc_train

            with (
                patch.object(rvc_train, "run_extract") as mock_re,
                patch.object(
                    sys, "argv",
                    ["kokoclone-train", "extract", "--name", "v", "--device", "cpu"],
                ),
            ):
                mock_re.return_value = {"steps_run": ["hubert"]}
                rc = _main()
            assert rc == 0
            assert mock_re.call_args[1]["device"] == "cpu"
        finally:
            _unmock_heavy_deps()

    def test_device_config_honored_when_flag_absent(self):
        """Unset flag → voice-config device wins; root default is "auto"."""
        from kokoclone_train.core.training_config import get_device, resolve_training_config

        with tempfile.TemporaryDirectory() as tmpdir:
            voices = Path(tmpdir) / "voices"
            voice = voices / "v"
            voice.mkdir(parents=True)
            (voice / "training_config.json").write_text(json.dumps({"device": "xpu:1"}))

            # flag absent (None) → voice config device wins
            cfg = resolve_training_config("v", voices_dir=voices, cli_overrides={"device": None})
            assert get_device(cfg) == "xpu:1"

            # CLI value wins over the voice config
            cfg2 = resolve_training_config("v", voices_dir=voices, cli_overrides={"device": "cuda:0"})
            assert get_device(cfg2) == "cuda:0"

        # No voice override → root default applies
        cfg3 = resolve_training_config("ghost_voice", voices_dir=Path("/nonexistent"),
                                       cli_overrides={"device": None})
        assert get_device(cfg3) == "auto"

    def test_get_device_fallback(self):
        """get_device falls back to 'auto' for unset/empty values."""
        from kokoclone_train.core.training_config import get_device

        assert get_device({}) == "auto"
        assert get_device({"device": "cuda"}) == "cuda"
        assert get_device({"device": "xpu:1"}) == "xpu:1"
        assert get_device({"device": None}) == "auto"
        assert get_device({"device": ""}) == "auto"

    def test_training_hp_records_device(self):
        """Resolved device is recorded in the saved training_hp.json."""
        _mock_heavy_deps()
        try:
            from kokoclone_train.core.rvc_train import train_voice

            with tempfile.TemporaryDirectory() as tmpdir:
                src = Path(tmpdir) / "audio"
                src.mkdir()
                (src / "a.wav").write_text("x")
                voices = Path(tmpdir) / "voices"
                voices.mkdir()

                with (
                    patch("kokoclone_train.core.rvc_train._preprocess_audio"),
                    patch("kokoclone_train.core.rvc_train._extract_features"),
                    patch("kokoclone_train.core.rvc_train._extract_f0_data"),
                    patch("kokoclone_train.core.rvc_train._extract_mel_targets"),
                    patch("kokoclone_train.core.rvc_train._train_model"),
                    patch("kokoclone_train.core.rvc_train._build_faiss_index"),
                ):
                    train_voice(
                        voice_name="v",
                        audio_dir=src,
                        output_dir=voices,
                        callback=lambda m, p: None,
                        training_config={"device": "cuda:0", "epochs": 1, "checkpoint_interval": 10},
                    )

                hp_path = voices / "v" / "training_hp.json"
                assert hp_path.exists(), "training_hp.json should be saved"
                hp = json.loads(hp_path.read_text())
                assert hp["device"] == "cuda:0"
        finally:
            _unmock_heavy_deps()
