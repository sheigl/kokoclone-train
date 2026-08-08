#!/usr/bin/env python3
"""Test a training checkpoint by running end-to-end inference on CPU.

Loads a checkpoint from training_temp/, extracts the generator weights,
builds the end-to-end SynthesizerTrn (with HiFi-GAN decoder), runs inference
on a ground-truth audio chunk, and saves the output as a WAV file.

Usage:
    PYTHONPATH="/path/to/kokoclone-train:$PYTHONPATH" \
      python tools/test_checkpoint.py \
        --ckpt training_temp/am_jefferson/am_jefferson_epoch_20.pth \
        --audio training_temp/am_jefferson/0_gt_wavs/0_0.wav

Runs entirely on CPU — does not interfere with GPU training.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from scipy.signal import resample_poly


def main():
    parser = argparse.ArgumentParser(description="Test a training checkpoint.")
    parser.add_argument(
        "--ckpt", required=True, type=Path,
        help="Path to checkpoint .pth file (e.g., am_jefferson_epoch_20.pth)",
    )
    parser.add_argument(
        "--audio", required=True, type=Path,
        help="Path to input audio .wav file (40kHz ground-truth chunk)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent,
        help="Where to save output WAV files (default: tools/)",
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    # ── 1. Load checkpoint ──────────────────────────────────────────────
    print(f"[1/7] Loading checkpoint: {args.ckpt}")
    t0 = time.time()
    ckpt = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)

    # New format: weight_g (generator only).  Legacy: weight.
    if "weight_g" in ckpt:
        sd = ckpt["weight_g"]
        epoch = ckpt.get("epoch", "?")
        print(f"      Epoch {epoch}, keys in state_dict: {len(sd)}")
    elif "weight" in ckpt:
        sd = ckpt["weight"]
        epoch = "?"
        print(f"      Legacy checkpoint, keys: {len(sd)}")
    else:
        print(f"ERROR: checkpoint has no 'weight_g' or 'weight' key")
        sys.exit(1)

    # ── 2. Build model ──────────────────────────────────────────────────
    print(f"[2/7] Building SynthesizerTrn (use_decoder=True) on CPU...")
    from kokoclone_train.core.rvc_models import build_rvc_model

    config = {
        "sample_rate": 40000,
        "hop_length": 160,
        "input_size": 768,
        "output_size": 192,
        "f0": True,
        "use_decoder": True,
    }
    net_g = build_rvc_model(config).to(device)

    missing, unexpected = net_g.load_state_dict(sd, strict=False)
    if missing:
        print(f"      WARNING: {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"      WARNING: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
    if not missing and not unexpected:
        print(f"      All keys loaded ✓")

    # Fuse weight_norm in the decoder (weight_v + weight_g → .weight)
    if hasattr(net_g, "dec"):
        net_g.dec.remove_weight_norm()
        print(f"      Decoder weight_norm removed ✓")

    net_g.eval()
    n_params = sum(p.numel() for p in net_g.parameters())
    print(f"      Generator: {n_params:,} params")
    print(f"      Checkpoint load + model build: {time.time()-t0:.1f}s")

    # ── 3. Load input audio ─────────────────────────────────────────────
    print(f"[3/7] Loading audio: {args.audio}")
    audio_40k, sr = sf.read(str(args.audio), dtype="float32")
    if audio_40k.ndim > 1:
        audio_40k = audio_40k.mean(axis=1)
    print(f"      {len(audio_40k)} samples @ {sr} Hz = {len(audio_40k)/sr:.2f}s")

    # Copy GT to output for comparison
    gt_out = args.output_dir / "test_gt.wav"
    sf.write(str(gt_out), audio_40k, sr)
    print(f"      GT saved: {gt_out}")

    # ── 4. Extract HuBERT content features ──────────────────────────────
    print(f"[4/7] Extracting HuBERT content features...")
    t1 = time.time()
    audio_16k = resample_poly(audio_40k, 16000, sr).astype(np.float32)

    from kokoclone_train.core.hubert_manager import HuBERTManager
    from kokoclone_train.core.rvc_assets import get_hubert_path

    hubert_path = str(get_hubert_path())
    hubert = HuBERTManager.get(hubert_path, device=device)
    content = hubert.content(audio_16k, sr=16000)
    print(f"      Content: {content.shape} @ ~50fps ({time.time()-t1:.1f}s)")

    # ── 5. Extract F0 ───────────────────────────────────────────────────
    print(f"[5/7] Extracting F0 (harvest, hop=160)...")
    t2 = time.time()
    import pyworld as pw

    hop_length = 160
    frame_period = hop_length / sr * 1000  # 4.0 ms
    f0_raw, _ = pw.harvest(
        audio_40k.astype(np.float64), sr,
        f0_floor=50, f0_ceil=1100,
        frame_period=frame_period,
    )

    # Convert to coarse F0 bins
    f0_coarse = np.zeros(len(f0_raw), dtype=np.int32)
    for i in range(len(f0_raw)):
        if f0_raw[i] != 0:
            f0_coarse[i] = int(np.round(12 * np.log2(max(f0_raw[i], 50) / 75))) + 1
    print(f"      F0: {len(f0_coarse)} frames, nonzero={int((f0_coarse != 0).sum())} ({time.time()-t2:.1f}s)")

    # ── 6. Align + run inference ────────────────────────────────────────
    print(f"[6/7] Running end-to-end inference...")
    t3 = time.time()

    # Resample content to match F0 frame rate (185 → 925)
    from torch.nn import functional as F
    c_torch = torch.FloatTensor(content).T.unsqueeze(0)  # [1, 768, T_c]
    content_resampled = F.interpolate(
        c_torch, size=len(f0_coarse), mode="linear", align_corners=False
    ).squeeze(0).transpose(0, 1)  # [T_m, 768]

    c_in = content_resampled.unsqueeze(0).to(device)  # [1, T, 768]
    f0_in = torch.LongTensor(f0_coarse).unsqueeze(0).to(device)  # [1, T]

    with torch.no_grad():
        mel, audio = net_g.infer(c_in, f0=f0_in)

    print(f"      mel:   {mel.shape}, mean={float(mel.mean()):.4f}, std={float(mel.std()):.4f}")
    if audio is not None:
        sat_pct = float((audio.abs() > 0.99).float().mean()) * 100
        print(f"      audio: {audio.shape}, mean={float(audio.mean()):+.6f}, std={float(audio.std()):.6f}, sat={sat_pct:.1f}%")
    else:
        print(f"      audio: None (use_decoder was False?)")
        sys.exit(1)
    print(f"      Inference: {time.time()-t3:.1f}s")

    # ── 7. Save output ──────────────────────────────────────────────────
    print(f"[7/7] Saving output...")
    wav_np = audio[0, 0].cpu().numpy().astype(np.float32)

    # Truncate to same length as GT for easy comparison
    min_len = min(len(wav_np), len(audio_40k))
    wav_np = wav_np[:min_len]

    epoch_str = f"epoch{epoch}" if epoch != "?" else "unknown"
    out_path = args.output_dir / f"test_{epoch_str}.wav"
    sf.write(str(out_path), wav_np, sr)
    print(f"      Output: {out_path}")
    print(f"      GT:     {gt_out}")
    print()

    # ── Summary ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Checkpoint:  {args.ckpt.name} (epoch {epoch})")
    print(f"  Input:       {args.audio.name} ({len(audio_40k)/sr:.1f}s)")
    print(f"  Output:      {out_path.name}")
    print(f"  Mel range:   [{float(mel.min()):.1f}, {float(mel.max()):.1f}] dB")
    print(f"  Audio mean:  {float(audio.mean()):+.6f}  (target: |mean| < 0.3)")
    print(f"  Audio std:   {float(audio.std()):.6f}  (target: > 0.01)")
    print(f"  Saturation:  {sat_pct:.1f}%           (target: < 5%)")
    print()

    if sat_pct > 50:
        print("  ⚠️  HIGH SATURATION — decoder is producing tanh-clipped output.")
        print("     This is expected at early epochs and should improve with training.")
    elif sat_pct < 5 and abs(float(audio.mean())) < 0.3:
        print("  ✓  Audio looks healthy — non-saturated with reasonable amplitude.")
    else:
        print("  ~  Audio has some saturation — keep training and check again.")

    total_time = time.time() - t0
    print(f"\n  Total time: {total_time:.1f}s (CPU)")


if __name__ == "__main__":
    main()
