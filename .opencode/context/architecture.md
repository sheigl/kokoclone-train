# Architecture Context

> **Populated by the Document agent.** Source of truth: `docs/ARCHITECTURE.md`.

## Data flow (end-to-end mode, hop_length=160)

```
Audio (40kHz, ~3.7s chunks)
    ├──→ 16kHz resample → HuBERT → content [T_c≈185, 768] @ ~50fps
    ├──→ pyworld.harvest   → F0     [T_m≈925]           @ ~250fps
    └──→ torchaudio.MelSpec → mel    [T_m≈925, 192]      @ ~250fps
                              ↓ audio [T_audio = T_m × 160]

SynthesizerTrn (use_decoder=True):
    content+f0 → Linear(768→256) → Transformer ×6 → flows ×2 → mel [B,192,T]
    → HiFiGANDecoder (upsample_rates=[8,5,4,2]) → audio [B,1,T×160]
```

Frame alignment happens in the dataset via `F.interpolate` (content ~50fps →
~250fps); fixed-length padding to `seq_len=925` + `drop_last=True` works
around an Intel Arc oneDNN JIT limitation.

## Device handling

`kokoclone_train.core.rvc_assets.detect_device(requested=None)`:

- `None`/`"auto"` → auto chain: **MPS → CUDA → XPU → CPU** (unchanged).
- Explicit requests: `auto | cuda[:N] | xpu[:N] | mps | cpu`, case/whitespace
  normalized; unavailable / out-of-range / unknown values raise an actionable
  `ValueError`.
- `--device` CLI flag on `train`, `train-only`, `extract`. Precedence:
  CLI > voice config > root config > `"auto"`. `extract` is CLI-only passthrough.
- Resolved device recorded in per-voice `training_hp.json`.
- `empty_cache(device)`: cuda/xpu dispatch; mps guarded via `hasattr`; cpu no-op.

## ROCm 5.7 environment (AMD RX 470, `tools/setup_rocm_env.sh`)

- Second venv `.venv-rocm57` (sibling of `.venv`): torch 2.3.1 + torchaudio
  2.3.1 from `https://download.pytorch.org/whl/rocm5.7`; deps pinned for torch
  2.3 (`numpy<2`, `transformers<4.48`, `huggingface-hub<1.0`, …); package
  installed `-e . --no-deps`. Never `uv sync` here (`[tool.uv.sources]` maps
  torch → pytorch-xpu). `uv pip install` DOES read the project's
  `[[tool.uv.index]]` (where the pytorch-xpu index is defined), so every
  install passes `--no-project` to ignore the project config entirely and use
  only the rocm5.7 index / PyPI.
- In this env the RX 470 (Polaris/gfx803, PCI 0a:00.0) appears via the CUDA
  API as `cuda:0`; the auto chain reaches it (MPS off on Linux → CUDA). Story
  001's `--device cuda:0` pins runs to it. `tools/check_rocm.py` is the
  verification gate; system ROCm 5.7 userspace install is best-effort on
  Ubuntu 24.04 (guide: `docs/ROCM_RX470_SETUP.md`).
- Polaris is unsupported in ROCm 6.x/7.x — 5.7 is the last release with
  first-class gfx803 support; no amdgpu-dkms rebuild needed (kernel driver
  already supports Polaris and is loaded).

## Key invariants (do not regress)

1. **No G40k.pth as standalone vocoder** (see `docs/VOCODER_INVESTIGATION.md`).
2. **Inference owns its decoder** — `SynthesizerTrn.dec`; no `_load_vocoder` step.
3. **hop_length=160 with the HiFi-GAN decoder**; mel-only (hop=300) is superseded.
