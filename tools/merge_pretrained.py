"""Merge pretrained f0G48k.pth decoder resblock weights into training checkpoint.

Transfers all matching decoder weights (resblocks, conv_pre, conv_post, ups.2, ups.3)
from the pretrained RVC checkpoint into the current training checkpoint.
Skips mismatched layers (ups.0, ups.1 — different kernel sizes) and encoder/flow.
"""

import torch
import sys
from pathlib import Path

PRETRAINED = Path("/home/sheigl/.cache/kokoclone/f0G48k.pth")
CKPT = Path("/home/sheigl/code/kokoclone-train/training_temp/am_jefferson/am_jefferson_epoch_200.pth")
OUT = Path("/home/sheigl/code/kokoclone-train/training_temp/am_jefferson/am_jefferson_epoch_200_merged.pth")

print("Loading pretrained...")
p = torch.load(str(PRETRAINED), map_location="cpu", weights_only=False)
sd_p = p["model"]

print("Loading training checkpoint...")
ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
sd_t = ckpt["weight_g"]

transferred = 0
skipped_shape = 0
skipped_missing = 0

for k in sd_t:
    if k in sd_p:
        if sd_t[k].shape == sd_p[k].shape:
            sd_t[k] = sd_p[k]
            transferred += 1
        else:
            skipped_shape += 1
    else:
        skipped_missing += 1

# Save with same structure
ckpt["weight_g"] = sd_t
torch.save(ckpt, str(OUT))

print(f"Transferred: {transferred} keys")
print(f"Skipped (shape mismatch): {skipped_shape}")
print(f"Skipped (not in pretrained): {skipped_missing}")
print(f"Saved: {OUT}")
print(f"  Size: {OUT.stat().st_size / 1024 / 1024:.1f} MB")
