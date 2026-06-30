"""Command-line interface for kokoclone-train."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from kokoclone_train.core.rvc_train import train_voice
from kokoclone_train.core.training_watcher import TrainingWatcher


def _build_parser() -> argparse.ArgumentParser:
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
    p_train.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start time for audio trimming (seconds or MM:SS/HH:MM:SS).",
    )
    p_train.add_argument(
        "--end",
        type=str,
        default=None,
        help="End time for audio trimming (seconds or MM:SS/HH:MM:SS).",
    )
    p_train.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint or skip completed stages.",
    )
    p_train.add_argument(
        "--force",
        action="store_true",
        help="Force retraining: clear all checkpoints and state.",
    )
    p_train.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (default: 10). Set to 0 to disable.",
    )

    p_watch = sub.add_parser(
        "watch",
        help="Watch training/ directory and auto-train new voices.",
    )
    p_watch.add_argument(
        "--training-dir",
        type=Path,
        default=Path("training"),
        help="Directory to watch for new voices (default: ./training).",
    )
    p_watch.add_argument(
        "--voices-dir",
        type=Path,
        default=Path("voices"),
        help="Where trained voices are saved (default: ./voices).",
    )

    return parser


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        if not args.audio_dir.is_dir():
            print(f"Error: audio-dir not found: {args.audio_dir}")
            return 1

        def progress(msg: str, pct: float) -> None:
            bar = "#" * int(pct * 40)
            print(f"\r[{bar:<40}] {pct*100:5.1f}%  {msg}", end="", flush=True)

        try:
            result = train_voice(
                voice_name=args.name,
                audio_dir=args.audio_dir,
                output_dir=args.output_dir,
                epochs=args.epochs,
                pitch_adjustment=args.pitch_adjustment,
                callback=progress,
                start_time=args.start,
                end_time=args.end,
                resume=args.resume,
                force=args.force,
                checkpoint_interval=args.checkpoint_interval,
            )
            print()
            print(f"Done! Model: {result['model_path']}")
            return 0
        except Exception as e:
            print(f"\nError: {e}")
            return 1

    if args.command == "watch":
        print(f"Watching {args.training_dir} for new voices...")
        watcher = TrainingWatcher(
            training_dir=str(args.training_dir),
            voices_dir=str(args.voices_dir),
        )
        watcher.start()
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            watcher.stop()
            print("Stopped.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(_main())