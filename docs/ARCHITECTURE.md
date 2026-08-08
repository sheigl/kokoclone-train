# Architecture

## Overview

KokoClone Train implements an RVC v2 (Retrieval-based Voice Conversion) training pipeline with an **end-to-end HiFi-GAN decoder**. The core model is a `SynthesizerTrn` generator that converts HuBERT content embeddings and F0 pitch contours into **both mel spectrograms and raw audio**, enabling voice cloning without any external vocoder dependency.

## Data Flow

```
Audio (40kHz, ~3.7s chunks)
    │
    ├──→ 16kHz resample → HuBERT → content [T_c≈185, 768] @ ~50fps
    ├──→ pyworld.harvest   → F0     [T_m≈925]           @ ~250fps
    └──→ torchaudio.MelSpec → mel    [T_m≈925, 192]      @ ~250fps
                              ↓ audio [T_audio = T_m × 160]

Dataset __getitem__ (frame alignment, hop=160):
    content resampled: [185, 768] → F.interpolate → [925, 768]
    audio cropped to match: [148000]
    mel padded to seq_len=925

Model forward (use_decoder=True):
    input:   [B, T=925, C=768] + f0 [B, T=925]
    encoder: Linear(768→256) → Transformer × 6 layers → [B, 925, 256]
    flows:   ResidualCouplingBlock × 2 on [B, 256, 925]
    upsample: Conv1d(256→192, kernel=3, pad=1) → [B, 192, 925]
    mel_out: [B, 192, T=925]
    decoder: HiFiGANDecoder (product of upsample_rates=160) → audio [B, 1, 148000]

Training losses:
    G: L1(mel_pred, mel_gt) + L1(audio_pred, audio_gt) + LS-GAN(G) + FM(feats)
    D: LS-GAN(D)  (D-before-G per HiFi-GAN convention)

Inference:
    single forward pass through SynthesizerTrn → (mel, audio)
    weight_norm removed on decoder convs
```

## Key Frame Rates

| Source | Rate | Frames (3.7s audio) |
|--------|------|---------------------|
| HuBERT (`facebook/hubert-base-ls960`) | ~50 fps (stride=320 at 16kHz) | ~185 |
| F0 / Mel (pyworld/torchaudio, hop_length=160 at 40kHz) | ~250 fps | ~925 |
| Audio | 40kHz | 148000 samples |
| **Ratio** (content→mel) | **~5** | — |
| **Ratio** (mel→audio) | **160** (= product of upsample_rates) | — |

hop_length=160 matches the decoder's `product(upsample_rates)=160`, ensuring audio sample count aligns exactly: `148000 = 925 × 160`.

## Model Architecture

### SynthesizerTrn Generator

```
SynthesizerTrn(
    spec_channels=192,    # mel spectrogram bins
    input_size=768,       # HuBERT embedding dimension
    upsample_rates=[8,5,4,2],  # product=160, drives decoder stride
    f0=True,              # use pitch conditioning
    use_decoder=True,     # embedded HiFi-GAN decoder
)
```

#### Components

1. **Encoder**: `Linear(768→256)` → Transformer Encoder × 6 layers with MultiHeadAttention
   - Input: `[B, T, 768]`
   - Output: `[B, T, 256]`

2. **Flows**: ResidualCouplingBlock × 2 (WaveNet-based normalizing flows)
   - Operate on transposed tensor: `[B, 256, T]`
   - Learn invertible mapping for better latent representation

3. **Upsample**: `upsample_rates=[8,5,4,2]` with ConvTranspose1d layers
   - `[B, 256, T]` → interleave → ConvTranspose1d → `[B, spec_channels, T×stride]`
   - Final layer is Conv1d(256→192, kernel=3, pad=1) to set output channels
   - Output: `[B, 192, T]` ← mel prediction

4. **Decoder** (when `use_decoder=True`): `HiFiGANDecoder`
   - Conv1d(192→512, kernel=7) → LeakyReLU
   - 4 upsample blocks (stride: 8, 5, 4, 2), each followed by 3 ResBlocks
   - ResBlock stacks: kernel_sizes=[11,7,5] × dilations=[1,3,5]
   - Conv1d(512→1, kernel=7) → tanh activation
   - Output: `[B, 1, T × 160]`

### Discriminators

Two discriminator networks for GAN training (used when `use_decoder=True`):

1. **MultiPeriodDiscriminator (MPD)**: 5 sub-discriminators operating on audio at different periodicities (periods: 2, 3, 5, 7, 11). Each sub-D is a 4-layer 1D convnet with LeakyReLU. Checks audio waveform periodicity — catches periodic artifacts like buzzing.

2. **MultiScaleDiscriminator (MSD)**: 3 sub-discriminators operating on audio at different scales (original, ×2 avg-pooled, ×4 avg-pooled). Each sub-D is a 4-layer 1D convnet with LeakyReLU and group normalization. Checks spectral envelope realism at multiple temporal resolutions.

Training alternates: **D step first** (per HiFi-GAN convention), then G step. D is optimized on LS-GAN loss; G on LS-GAN + L1 mel + L1 audio + feature matching.

## Dataset Design

### Per-Chunk ContentFeatureDataset

Each audio chunk (~3.7s) is stored as separate `.npy` files plus a raw audio `.wav`:
```python
chunks = [
    {"content": "path/to/content.npy",  # [T_c, 768]  at ~50fps
     "f0":      "path/to/f0.npy",       # [T_m]       at ~250fps
     "mel":     "path/to/mel.npy",      # [T_m, 192]  at ~250fps
     "audio":   "path/to/audio.wav"},   # [T_audio]   at 40kHz
    ...
]
```

**On-the-fly resampling + fixed-length padding in `__getitem__`**:
```python
# Content features are interpolated to match F0/mel frame count
if len(content) != len(f0):
    c_torch = torch.FloatTensor(content).T.unsqueeze(0)  # [1, 768, T_c]
    content = F.interpolate(c_torch, size=len(f0), mode='linear',
                            align_corners=False).squeeze(0).transpose(0, 1)

# Fixed-length padding to seq_len
content = pad_to(content, seq_len, dim=0)
f0 = pad_to(f0, seq_len, dim=0, value=0)
mel = pad_to(mel, seq_len, dim=0, value=mel_pad)  # -100 dB for silence
audio = pad_to(audio, audio_len, dim=0)

# Matching lengths: content [seq_len, 768], f0 [seq_len], mel [seq_len, 192], audio [seq_len × 160]
```

### Why Linear Interpolation?

Linear interpolation (`F.interpolate` with `mode='linear'`) is standard practice in speech processing for resampling feature sequences. It preserves the smooth temporal structure of HuBERT embeddings while mapping them to the higher frame rate used by F0 and mel features.

### Fixed-Length Padding

When `use_decoder=True`, all chunks are padded to a fixed `seq_len=925` to avoid variable-length batches. This works around a oneDNN GPU limitation: Intel Arc B70 cannot JIT-compile certain 2D conv backward shapes from MPD. Combined with `drop_last=True` in the DataLoader, batches are always uniform size and the JIT compiler sees consistent shapes.

## Training Loop

### Mel-Only Mode (use_decoder=False, legacy)
1. Load per-chunk dataset (content, f0, mel paths)
2. For each batch:
   - Dataset returns aligned `[content, f0, mel]` with matching temporal dimensions
   - Model forward pass produces mel prediction
   - MSE loss on mel spectrograms
3. Gradient clipping (max_norm=1.0)
4. AdamW (lr=1e-4, weight_decay=1e-5, betas=(0.8, 0.99))
5. ExponentialLR scheduler (gamma=0.998)

### End-to-End Mode (use_decoder=True, active)
1. Load per-chunk dataset (content, f0, mel, audio paths)
2. Fixed-length padding to seq_len=925; `drop_last=True` in DataLoader
3. For each batch:
   - Optimizer zero_grad
   - **D step**: discriminator loss on real audio vs generator audio → backward → clip → optimizer step
   - **G step**: forward pass → (mel, audio) → G loss = L1_mel(mel_pred, mel_gt) + L1_audio(audio_pred, audio_gt) + LS-GAN(fake_scores) + FM(feats_real, feats_fake)
   - Backward → clip (max_norm=1.0) → optimizer step
4. Generator AdamW (lr=1e-4, weight_decay=1e-5, betas=(0.8, 0.99))
5. Discriminator AdamW (lr=5e-5, weight_decay=1e-5, betas=(0.8, 0.99))
6. ExponentialLR scheduler on generator only (gamma=0.998)
7. Checkpoint saves `(G_state, D_state, optim_G, optim_D, scheduler, epoch)`

## Config Format

Each trained voice saves a config JSON:
```json
{
    "name": "voice_name",
    "version": 2,
    "input_size": 768,
    "output_size": 192,
    "sample_rate": 40000,
    "hop_length": 160,
    "f0": true,
    "use_decoder": true,
    "upsample_rates": [8, 5, 4, 2],
    "epoch": 300,
    "pitch_adjustment": 0.0
}
```

## Key Design Decisions

### 1. hop_length=160 for end-to-end mode

**Decision**: Use `hop_length=160` with `upsample_rates=[8,5,4,2]` (product=160) when the decoder is active.

**Rationale**: The decoder's `product(upsample_rates) = 160` sets the stride from mel frames to audio samples. Using `hop_length=160` ensures each mel frame corresponds to exactly one decoder stride step: `T_audio = T_mel × 160`. With 40kHz audio and 3.7s chunks: `925 × 160 = 148000` samples.

The old `hop_length=300` (mel-only) produced `upsample_rates=[1]` with alignment done in the dataset. That config is now superseded.

### 2. Per-chunk dataset (no global concatenation)

**Decision**: Each audio chunk is an independent sample in the dataset.

**Rationale**: Global concatenation of chunks would introduce cross-chunk boundary artifacts and make frame alignment unreliable at boundaries. Per-chunk processing ensures clean, self-contained samples.

### 3. Fixed-length padding with drop_last

**Decision**: Pad all sequences to a fixed `seq_len=925` and use `drop_last=True` in the DataLoader.

**Rationale**: The Intel Arc B70 GPU cannot JIT-compile 2D conv backward shapes from MPD when batch sizes vary. Fixed-length padding ensures every batch has uniform tensor shapes, while `drop_last=True` avoids the irregular final batch. Mel padding uses -100 dB (silence floor).

### 4. Pretrained weight loading with strict=False

**Decision**: Load pretrained f0G48k.pth weights with key lookup (`weight` → `model`) + shape pre-filtering + `strict=False`.

**Rationale**: The checkpoint stores weights under key `"model"` (not `"weight"`). The decoder resblocks, conv_pre, and conv_post are architecturally compatible and transfer directly (229/560 keys). The encoder and flow are incompatible (different `enc_p` structure: fused QKV vs separate Q/K/V convs, LayerNorm vs no LN, different channel counts). Upsample layers have different kernel sizes (24/20 in pretrained vs 16/16 in our config) and are skipped.

Crucially, the pretrained decoder is co-adapted to VITS latent input distribution (posterior encoder + flow + speaker conditioning), not our mel spectrograms. Merged checkpoints show 100% tanh saturation initially. This clears over ~10-20 epochs as the pretrained resblock weights fine-tune to the new input distribution.

### 5. D-before-G training order

**Decision**: Run discriminator step before generator step each iteration.

**Rationale**: Per HiFi-GAN convention and the original paper (Kong et al. 2020), updating the discriminator first ensures the generator always faces an up-to-date critic. This stabilizes adversarial training.

### 6. Test isolation via per-test mocking

**Decision**: Mock heavy dependencies (torch, torchaudio, etc.) only inside individual test methods, not at module level.

**Rationale**: Module-level mocking pollutes the pytest session and prevents other tests from accessing real modules. Per-test scoped mocks with explicit cleanup allow both frame_alignment tests (which need real torch) and CLI tests (which mock torch) to run together in a single `pytest tests/` invocation.

## Device Handling

Automatic device detection via `kokoclone_train.core.rvc_assets.detect_device()`:
```
MPS (Apple Silicon) → CUDA (NVIDIA GPU) → XPU (Intel GPU) → CPU
```
All tensors are moved to the detected device before training/inference.

A device override is supported on the `train`, `train-only`, and `extract`
subcommands via `--device auto|cuda[:N]|xpu[:N]|mps|cpu`. Resolution
precedence: **CLI `--device` > voice `training_config.json` `"device"` key >
root `training_config.json` `"device"` key > `"auto"`** (`run_extract` is a
plain CLI passthrough — no config merge). `detect_device(requested=None)`
keeps the exact auto-detect chain above for `None`/`"auto"`; explicit
requests for unavailable, out-of-range (`cuda:N`/`xpu:N` past the device
count), or unknown devices raise an actionable `ValueError` naming what was
requested, what is available, and how to fix it. Cache clearing via
`empty_cache(device)` dispatches on device type: `cuda`/`xpu` call their
framework cache emptiers, `mps` is guarded with `hasattr` (safe on non-Apple
builds), and `cpu` is a no-op. The resolved device is recorded in the
per-voice `training_hp.json`.

### ROCm 5.7 environment (AMD RX 470) — a second, separate venv

Full training runs on the Polaris/gfx803 RX 470 use a **separate ROCm 5.7
environment** (`.venv-rocm57`, torch 2.3.1 + torchaudio 2.3.1 from the
`rocm5.7` wheel index) that is never mixed with the primary XPU `.venv`
(torch 2.10) — the two torch builds cannot coexist in one environment. Setup
is scripted (`tools/setup_rocm_env.sh`, idempotent, `--no-deps` package
install so torch is never re-resolved from the pytorch-xpu index); runtime
verification is `tools/check_rocm.py`; the full guide is
[`docs/ROCM_RX470_SETUP.md`](ROCM_RX470_SETUP.md).

Because a ROCm build of torch exposes AMD GPUs through the CUDA API, the RX
470 appears as `cuda:0` in that env and the `--device cuda:0` override above
is exactly how AMD runs are pinned. The `--device cuda` `ValueError` from
`detect_device` points at `docs/ROCM_RX470_SETUP.md` for this setup path.
