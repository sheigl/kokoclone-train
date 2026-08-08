#!/usr/bin/env python3
"""Test different inference parameters for RVC voice conversion.

Runs on CPU, uses the full rvc_engine pipeline (with FAISS index),
tries f0_method and transpose combinations.

Usage:
    PYTHONPATH="/home/sheigl/code/kokoclone:$PYTHONPATH" \
      python tools/test_inference_tweaks.py
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
import time

sys.path.insert(0, str(Path.home() / "code/kokoclone"))
from core.rvc_engine import load_voice, convert_audio

voice_dir = str(Path.home() / "code/kokoclone-train/voices/am_jefferson")
audio_path = Path.home() / "code/kokoclone-train/training_temp/am_jefferson/0_gt_wavs/0_0.wav"
out_dir = Path.home() / "code/kokoclone-train/tools"

audio_40k, sr = sf.read(str(audio_path), dtype="float32")
print(f"Source: {audio_path.name} ({len(audio_40k)/sr:.1f}s @ {sr}Hz)")
print(f"  range=[{audio_40k.min():.4f}, {audio_40k.max():.4f}], std={audio_40k.std():.4f}")

voice = load_voice(voice_dir, device_select="cpu")

tests = [
    ("harvest", 0),
    ("harvest", -2),
    ("harvest", -4),
    ("pm", 0),
    ("pm", -2),
    ("pm", -4),
]

for f0_method, transpose in tests:
    label = f"{f0_method}_t{transpose:+d}"
    print(f"\n[{label}] Running inference...")
    t0 = time.time()
    wav, _ = convert_audio(
        audio_40k.copy(), sr, voice,
        transpose=transpose, f0_method=f0_method,
    )
    elapsed = time.time() - t0
    out_path = out_dir / f"test_tweak_{label}.wav"
    sf.write(str(out_path), wav, sr)
    print(f"  {out_path.name}: {len(wav)/sr:.1f}s, "
          f"mean={wav.mean():+.6f}, std={wav.std():.4f}, "
          f"sat={(np.abs(wav) > 0.99).mean()*100:.1f}%, "
          f"in {elapsed:.0f}s")
