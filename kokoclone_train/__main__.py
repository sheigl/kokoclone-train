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
        "--epochs", type=int, default=None,
        help="Training epochs (default: from training_config.json, typically 100).",
    )
    p_train.add_argument(
        "--pitch-adjustment",
        type=float,
        default=None,
        help="Pitch shift in semitones (default: from training_config.json, typically 0.0).",
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
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Save checkpoint every N epochs (default: from training_config.json, typically 10). Set to 0 to disable.",
    )
    p_train.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: auto — min(8, len(dataset)//32)).",
    )
    p_train.add_argument(
        "--use-decoder",
        action="store_true",
        help=(
            "Train end-to-end with an embedded HiFi-GAN decoder "
            "(see docs/END_TO_END_DECODER_PLAN.md).  Sets hop_length=160 "
            "and enables adversarial + feature-matching + L1 losses."
        ),
    )
    p_train.add_argument(
        "--xtts-augment",
        action="store_true",
        help="Enable XTTS data augmentation (requires XTTS features in temp_dir/xtts_aug/).",
    )
    p_train.add_argument(
        "--aug-ratio",
        type=float,
        default=None,
        help="Fraction of batches using XTTS source features (default: from training_config.json).",
    )
    p_train.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to a training config JSON file.  Merged with defaults and"
            " voice-specific config (voices/{name}/training_config.json)."
            " CLI flags override all configs."
        ),
    )
    p_train.add_argument(
        "--device",
        metavar="{auto,cuda,cpu,xpu,vulkan,mps}",
        default=None,
        help=(
            "Device to run on: auto (default, MPS→CUDA→XPU→CPU), "
            "cuda[:N] (NVIDIA/AMD via ROCm), xpu[:N] (Intel Arc), vulkan (AMD/Intel via Vulkan), mps, cpu."
        ),
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

    p_transcribe = sub.add_parser(
        "transcribe",
        help="Transcribe audio chunks via a remote Whisper server.",
    )
    p_transcribe.add_argument("--name", required=True, help="Voice name.")
    p_transcribe.add_argument(
        "--server-url",
        default="http://localhost:8100/",
        help="Whisper server URL (default: http://localhost:8100/).",
    )
    p_transcribe.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("training_temp"),
        help="Root of training_temp directory.",
    )
    p_transcribe.add_argument(
        "--language",
        default=None,
        help="Language code for whisper (e.g. en). Auto-detected if not set.",
    )
    p_transcribe.add_argument(
        "--output-dir",
        type=Path,
        default=Path("voices"),
        help="Voices directory for permanent transcript storage (default: ./voices).",
    )
    p_transcribe.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="Number of parallel requests to the Whisper server (default: 0 = auto/cpu_count).",
    )

    # ── preprocess subcommand ────────────────────────────────────────
    p_preprocess = sub.add_parser(
        "preprocess",
        help="Split and normalize audio into training chunks.",
    )
    p_preprocess.add_argument("--name", required=True, help="Voice name.")
    p_preprocess.add_argument(
        "--audio-dir", required=True, type=Path,
        help="Directory with .wav/.mp3/.flac files.",
    )
    p_preprocess.add_argument(
        "--output-dir", type=Path, default=Path("voices"),
        help="Parent dir for voice output (default: ./voices).",
    )
    p_preprocess.add_argument(
        "--start", type=str, default=None,
        help="Start time for audio trimming (seconds or MM:SS/HH:MM:SS).",
    )
    p_preprocess.add_argument(
        "--end", type=str, default=None,
        help="End time for audio trimming.",
    )
    # (--force removed — use --resume to continue; clear training_temp/ manually to re-run)

    # ── extract subcommand ───────────────────────────────────────────
    p_extract = sub.add_parser(
        "extract",
        help="Extract HuBERT features, F0 pitch, and/or mel targets.",
    )
    p_extract.add_argument("--name", required=True, help="Voice name.")
    p_extract.add_argument(
        "--output-dir", type=Path, default=Path("voices"),
        help="Parent dir for voice output (default: ./voices).",
    )
    p_extract.add_argument(
        "--step", choices=["hubert", "f0", "mel"], default=None,
        help="Run a single step only. Omit to run all.",
    )
    p_extract.add_argument(
        "--use-decoder", action="store_true", default=None,
        help="Use hop_length=160 (for end-to-end decoder training).",
    )
    p_extract.add_argument(
        "--device",
        metavar="{auto,cuda,cpu,xpu,vulkan,mps}",
        default=None,
        help=(
            "Device to run on: auto (default, MPS→CUDA→XPU→CPU), "
            "cuda[:N] (NVIDIA/AMD via ROCm), xpu[:N] (Intel Arc), vulkan (AMD/Intel via Vulkan), mps, cpu."
        ),
    )
    # (--force removed — delete training_temp/{name}/stage files to re-run)

    # ── train-only subcommand ────────────────────────────────────────
    p_trainonly = sub.add_parser(
        "train-only",
        help="Run the training loop and build FAISS index (requires preprocess + extract first).",
    )
    p_trainonly.add_argument("--name", required=True, help="Voice name.")
    p_trainonly.add_argument(
        "--output-dir", type=Path, default=Path("voices"),
        help="Parent dir for voice output (default: ./voices).",
    )
    p_trainonly.add_argument(
        "--epochs", type=int, default=None,
        help="Training epochs (default: from training_config.json).",
    )
    p_trainonly.add_argument(
        "--pitch-adjustment", type=float, default=None,
        help="Pitch shift in semitones.",
    )
    p_trainonly.add_argument(
        "--checkpoint-interval", type=int, default=None,
        help="Save checkpoint every N epochs. Set to 0 to disable.",
    )
    p_trainonly.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (default: auto).",
    )
    p_trainonly.add_argument(
        "--use-decoder", action="store_true", default=None,
        help="Train end-to-end with HiFi-GAN decoder.",
    )
    p_trainonly.add_argument(
        "--xtts-augment", action="store_true", default=None,
        help="Enable XTTS data augmentation.",
    )
    p_trainonly.add_argument(
        "--aug-ratio", type=float, default=None,
        help="Fraction of batches using XTTS source features.",
    )
    p_trainonly.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint.",
    )
    p_trainonly.add_argument(
        "--config", type=Path, default=None,
        help="Path to a training config JSON file.",
    )
    p_trainonly.add_argument(
        "--device",
        metavar="{auto,cuda,cpu,xpu,vulkan,mps}",
        default=None,
        help=(
            "Device to run on: auto (default, MPS→CUDA→XPU→CPU), "
            "cuda[:N] (NVIDIA/AMD via ROCm), xpu[:N] (Intel Arc), vulkan (AMD/Intel via Vulkan), mps, cpu."
        ),
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

        # Load merged training config (defaults + voice-specific + CLI overrides)
        from kokoclone_train.core.training_config import resolve_training_config

        # Only include boolean flags when explicitly set (True).
        # When False (argparse default), let config decide.
        cli_overrides = {
            "epochs": args.epochs,
            "pitch_adjustment": args.pitch_adjustment,
            "checkpoint_interval": args.checkpoint_interval,
            "batch_size": args.batch_size,
            "aug_ratio": args.aug_ratio,
            "device": args.device,
        }
        if args.use_decoder:
            cli_overrides["use_decoder"] = True
        if args.xtts_augment:
            cli_overrides["xtts_augment"] = True

        training_cfg = resolve_training_config(
            voice_name=args.name,
            voices_dir=args.output_dir,
            default_path=args.config if args.config else None,
            cli_overrides=cli_overrides,
        )

        def progress(msg: str, pct: float) -> None:
            bar = "#" * int(pct * 40)
            print(f"\r[{bar:<40}] {pct*100:5.1f}%  {msg}", end="", flush=True)

        try:
            # For boolean flags: pass None when not explicitly set (False = argparse default),
            # so the resolved config takes effect.  True means user explicitly requested it.
            result = train_voice(
                voice_name=args.name,
                audio_dir=args.audio_dir,
                output_dir=args.output_dir,
                epochs=args.epochs,                          # None if not passed
                pitch_adjustment=args.pitch_adjustment,      # None if not passed
                callback=progress,
                start_time=args.start,
                end_time=args.end,
                resume=args.resume,
                checkpoint_interval=args.checkpoint_interval,  # None if not passed
                use_decoder=True if args.use_decoder else None,  # True or None (not False)
                batch_size=args.batch_size,                  # None = auto/config
                xtts_augment=True if args.xtts_augment else None,  # True or None
                aug_ratio=args.aug_ratio,                    # None if not passed
                training_config=training_cfg,
                device=args.device,                          # None = config decides ("auto")
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

    if args.command == "transcribe":
        from kokoclone_train.core.transcribe_client import transcribe_voice

        try:
            result = transcribe_voice(
                voice_name=args.name,
                server_url=args.server_url,
                temp_dir=args.temp_dir,
                voices_dir=args.output_dir,
                language=args.language,
                concurrency=args.concurrency,
                callback=lambda chunk, text, done, total: print(
                    f"\r[{done}/{total}] {chunk}", end="", flush=True
                ),
            )
            print()
            print(
                f"Done! Transcribed: {result['transcribed']}, "
                f"Skipped: {result['skipped']}, Failed: {result['failed']}"
            )
            print(f"JSONL output: {result['jsonl_path']}")
            return 0 if result["failed"] == 0 else 1
        except Exception as e:
            print(f"\nError: {e}")
            return 1

    if args.command == "preprocess":
        from kokoclone_train.core.rvc_train import run_preprocess

        if not args.audio_dir.is_dir():
            print(f"Error: audio-dir not found: {args.audio_dir}")
            return 1

        def progress(msg, pct):
            bar = "#" * int(pct * 40)
            print(f"\r[{bar:<40}] {pct*100:5.1f}%  {msg}", end="", flush=True)

        try:
            result = run_preprocess(
                voice_name=args.name,
                audio_dir=args.audio_dir,
                output_dir=args.output_dir,
                start_time=args.start,
                end_time=args.end,
                callback=progress,
            )
            print()
            print(f"Done! Chunks created: {result['chunks_created']}")
            return 0
        except Exception as e:
            print(f"\nError: {e}")
            return 1

    if args.command == "extract":
        from kokoclone_train.core.rvc_train import run_extract

        def progress(msg, pct):
            bar = "#" * int(pct * 40)
            print(f"\r[{bar:<40}] {pct*100:5.1f}%  {msg}", end="", flush=True)

        try:
            result = run_extract(
                voice_name=args.name,
                output_dir=args.output_dir,
                step=args.step,
                use_decoder=args.use_decoder,
                callback=progress,
                device=args.device,  # CLI-only passthrough (no config merge for extract)
            )
            print()
            steps = ", ".join(result["steps_run"]) if result["steps_run"] else "none (all done)"
            print(f"Done! Steps run: {steps}")
            return 0
        except Exception as e:
            print(f"\nError: {e}")
            return 1

    if args.command == "train-only":
        from kokoclone_train.core.training_config import resolve_training_config

        cli_overrides = {
            "epochs": args.epochs,
            "pitch_adjustment": args.pitch_adjustment,
            "checkpoint_interval": args.checkpoint_interval,
            "batch_size": args.batch_size,
            "aug_ratio": args.aug_ratio,
            "device": args.device,
        }
        if args.use_decoder:
            cli_overrides["use_decoder"] = True
        if args.xtts_augment:
            cli_overrides["xtts_augment"] = True

        training_cfg = resolve_training_config(
            voice_name=args.name,
            voices_dir=args.output_dir,
            default_path=args.config if args.config else None,
            cli_overrides=cli_overrides,
        )

        def progress(msg, pct):
            bar = "#" * int(pct * 40)
            print(f"\r[{bar:<40}] {pct*100:5.1f}%  {msg}", end="", flush=True)

        from kokoclone_train.core.rvc_train import run_train_only

        try:
            result = run_train_only(
                voice_name=args.name,
                output_dir=args.output_dir,
                epochs=args.epochs,
                pitch_adjustment=args.pitch_adjustment,
                checkpoint_interval=args.checkpoint_interval,
                use_decoder=True if args.use_decoder else None,
                batch_size=args.batch_size,
                xtts_augment=True if args.xtts_augment else None,
                aug_ratio=args.aug_ratio,
                resume=args.resume,
                training_config=training_cfg,
                callback=progress,
                device=args.device,  # None = config decides ("auto")
            )
            print()
            print(f"Done! Model: {result['model_path']}")
            return 0
        except Exception as e:
            print(f"\nError: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(_main())