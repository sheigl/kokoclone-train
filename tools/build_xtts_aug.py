"""Build XTTS augmentation pools for all voices.

Generates XTTS audio from real chunk transcripts, extracts HuBERT + F0 features,
and saves them to training_temp/{name}/xtts_aug/{content,f0}/ so the training
loop can use them for data augmentation.

Usage:
    cd /home/sheigl/code/auralis
    VIRTUAL_ENV=.venv ONEAPI_DEVICE_SELECTOR="level_zero:1" nohup uv run \
      /home/sheigl/code/kokoclone-train/tools/build_xtts_aug.py \
      > /home/sheigl/code/kokoclone-train/build_xtts_aug.log 2>&1 &
"""

import logging
import sys
from pathlib import Path

# Monkey-patch transformers BEFORE any TTS import
import torch
import transformers.pytorch_utils
if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
    transformers.pytorch_utils.isin_mps_friendly = torch.isin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path("/home/sheigl/code/kokoclone-train")


def build_for_voice(voice_name: str, pool_size: int = 250):
    from auralis.training.augment import build_xtts_augmentation
    from auralis.tts_engine import XTTSEngine
    from kokoclone_train.core.rvc_assets import get_hubert_path

    temp_dir = ROOT / "training_temp" / voice_name

    hubert_path = str(get_hubert_path())
    logger.info("HuBERT path: %s", hubert_path)

    transcripts_dir = ROOT / "voices" / voice_name / "transcripts"
    if not transcripts_dir.is_dir() or not any(transcripts_dir.glob("*.txt")):
        logger.warning("No transcripts found at %s", transcripts_dir)
        return

    tts = XTTSEngine.get_instance()
    builtin_speakers = tts.get_builtin_speakers()
    if not builtin_speakers:
        logger.warning("No built-in XTTS speakers available")
        return

    logger.info("Building XTTS augmentation for %s", voice_name)
    logger.info("  transcripts:  %s", transcripts_dir)
    logger.info("  built-in speakers: %d available (using random per sample)", len(builtin_speakers))
    logger.info("  pool_size:    %d", pool_size)

    aug_dir = build_xtts_augmentation(
        tts_engine=tts,
        voice_temp_dir=temp_dir,
        pool_size=pool_size,
        hop_length=160,
        hubert_path=hubert_path,
        transcripts_dir=transcripts_dir,
        builtin_speakers=builtin_speakers,
    )
    logger.info("Done: %s", aug_dir)


def main():
    build_for_voice("am_jefferson", pool_size=250)
    build_for_voice("am_welch", pool_size=250)


if __name__ == "__main__":
    main()
