# Research Guide — Understanding the Pipeline

> This document explains the *why* behind every piece of the kokoclone-train
> pipeline in plain English.  No prior speech-processing knowledge is assumed.
> If you can read Python and know what a neural network is, you can read this.
>
> Each section answers a question you'd naturally ask: "what is a mel frame?",
> "why does HuBERT run at 50 fps?", "what is a discriminator?", etc.  Examples
> use real numbers from our pipeline so you can verify the math yourself.

---

## Table of Contents

1. [Audio as Numbers](#1-audio-as-numbers)
2. [Mel Spectrograms](#2-mel-spectrograms)
3. [HuBERT Content Features](#3-hubert-content-features)
4. [F0 Pitch](#4-f0-pitch)
5. [The Frame Alignment Problem](#5-the-frame-alignment-problem)
6. [The Generator: SynthesizerTrn](#6-the-generator-synthesizertrn)
7. [The HiFi-GAN Decoder](#7-the-hifi-gan-decoder)
8. [Why G40k.pth Failed (the VITS Problem)](#8-why-g40kpth-failed-the-vits-problem)
9. [Discriminators: The Adversarial Referee](#9-discriminators-the-adversarial-referee)
10. [GAN Training: The Cat-and-Mouse Game](#10-gan-training-the-cat-and-mouse-game)
11. [Loss Functions Explained](#11-loss-functions-explained)
12. [Weight Normalisation](#12-weight-normalisation)
13. [Why tanh Saturates](#13-why-tanh-saturates)
14. [Glossary](#14-glossary)

---

## 1. Audio as Numbers

A sound wave is continuous pressure variation in air.  To store it in a
computer, we **sample** it: measure the pressure thousands of times per
second and write each measurement as a number.

```
Time:   0s      0.000025s   0.00005s   0.000075s   ...
Value:  0.031   0.058       -0.012     -0.044      ...
```

- Each number is a **sample**.
- The number of samples per second is the **sample rate**.  Our pipeline
  uses **40,000 samples per second (40 kHz)**.
- A 3.7-second audio chunk = 3.7 × 40,000 = **148,000 samples**.

**Why 40 kHz?**  The Nyquist theorem says you need at least 2× the highest
frequency you want to capture.  Human hearing tops out around 20 kHz, so
40 kHz is just enough.  RVC chose 40 kHz (not the more common 44.1 kHz) for
its own historical reasons — our pipeline inherits this.

**What does a sample value mean?**  It's a float in [-1, 1]:
- `0.0` = silence (no pressure deviation)
- `+1.0` = maximum positive pressure (loudest possible forward push)
- `-1.0` = maximum negative pressure (loudest possible pull-back)

When we say the model outputs "saturated audio with mean=-0.9996", we mean
almost every sample is pinned at -1.0 — the sound card would output a
constant maximum negative voltage.  No speech, just a DC click.

---

## 2. Mel Spectrograms

### 2.1 What is a spectrogram?

A raw waveform (148,000 samples for 3.7s) is hard for a neural network to
work with directly — it's too long and too low-level.  Instead, we convert
it into a **spectrogram**: a 2D grid showing "what frequencies are present
at each moment in time".

```
Frequency (high)
    ↑  |  ░░░░░░░░░░  ░░░        ░░░░░░  |
    |  |  ░░░░░░░░░░  ░░░  ████  ░░░░░░  |
    |  |  ░░░░░░░░░░  ░░░  ████  ░░░░░░  |
    |  |  ████░░░░░░  ████ ████  ████░░  |
    |  |  ████░░░░░░  ████ ████  ████░░  |
    ↓  |  ██████████  ████ ████  ████░░  |
       +--------------------------------→ Time (low)
         0s         1s         2s         3.7s
```

Each **column** is a time step (a "frame").  Each **row** is a frequency
band.  The brightness shows how much energy is at that frequency at that
time.

### 2.2 How do we compute it?

We slide a **window** across the waveform, and for each window position we
compute the **FFT (Fast Fourier Transform)** — a mathematical operation that
decomposes a signal into its constituent frequencies.

- **Window size** (`n_fft`): how many samples go into each FFT.  We use
  `n_fft = 640` (= hop_length × 4).
- **Hop length**: how many samples the window advances each step.  We use
  `hop_length = 160`.
- **Mel bins** (`n_mels`): how many frequency bands we group into.  We use
  `192`.

### 2.3 What is a "mel"?

Human hearing is not linear — we can easily tell the difference between 100
Hz and 200 Hz, but not between 10,000 Hz and 10,100 Hz.  The **mel scale**
compresses frequencies so that equal steps on the mel scale sound equally
far apart to human ears.

```
Linear Hz:    0 ---- 1000 ---- 2000 ---- 5000 ---- 10000 ---- 20000
Mel scale:    0 ---- 1000 ---- 1500 ---- 2000 ----  2500 ----  3000
              (more spread out at low frequencies, compressed at high)
```

We use 192 mel bands because RVC v2 standardised on this for 40 kHz audio.

### 2.4 What is a "mel frame"?

A **mel frame** is one column of the mel spectrogram — it's the frequency
content of the audio during one hop-length-wide window of time.

- At `hop_length = 160` and `sr = 40,000`, each frame covers
  `160 / 40,000 = 0.004 seconds = 4 milliseconds` of audio.
- A 3.7-second chunk produces `3.7 / 0.004 = 925` mel frames.
- Each frame is a 192-dimensional vector (one value per mel band).

So one mel spectrogram = `[925, 192]` — 925 time steps, 192 frequency bands.

### 2.5 Why dB scale?

The raw mel spectrogram contains **power** values, which can range from 0
to millions.  Neural networks work best with small numbers, so we convert
to **decibels (dB)**:

```
mel_db = 10 × log10(power)
```

This compresses the range: power 1,000,000 → 60 dB, power 1 → 0 dB, power
0.0000000001 → -100 dB.  Our training targets have `mean ≈ -37 dB, range
[-100, +37]`.

- **-100 dB** = near silence (the floor we clamp to)
- **0 dB** = power 1.0 (fairly loud)
- **+37 dB** = very loud (a peak)

**Important**: 0 dB is NOT silence!  This is why we pad mel targets with
-100, not 0 — padding with 0 would tell the model "produce loud output
here" when the region is actually silent.

### 2.6 Frame rate (fps)

Just like video has frames per second (fps), mel spectrograms have a frame
rate:

```
frame_rate = sample_rate / hop_length
           = 40,000 / 160
           = 250 fps  (250 frames per second of audio)
```

This means 1 second of audio → 250 mel frames.  3.7 seconds → 925 frames.

---

## 3. HuBERT Content Features

### 3.1 What is HuBERT?

**HuBERT** (Hidden-Unit BERT) is a self-supervised speech model trained on
960 hours of LibriSpeech.  Think of it as a "speech understanding" model —
it listens to audio and produces a compressed representation that captures
*what* is being said (phonemes, speaker characteristics, etc.) without
needing text labels.

We use `facebook/hubert-base-ls960` from Hugging Face.

### 3.2 What does HuBERT output?

HuBERT takes 16 kHz audio as input and produces a sequence of
**768-dimensional feature vectors**, one per frame:

```
Audio (16 kHz, 3.7s = 59,200 samples)
    → HuBERT
    → [185, 768]  (185 frames, each a 768-dim vector)
```

Each 768-dim vector encodes "what's happening in the speech at this
moment" — phoneme identity, coarticulation cues, speaker timbre, etc.

### 3.3 Why does HuBERT run at ~50 fps?

HuBERT processes audio in chunks of 320 samples (called the "stride" or
"downsampling factor").  At 16 kHz input:

```
HuBERT frame rate = 16,000 / 320 = 50 fps
3.7 seconds → 185 frames
```

This is **much slower** than our mel frame rate (250 fps).  This mismatch
is the core of the "frame alignment problem" (see §5).

### 3.4 Why 16 kHz input?

HuBERT was trained on 16 kHz audio.  Our audio is 40 kHz, so we resample
from 40 kHz → 16 kHz before feeding it to HuBERT.  This loses high-
frequency information, but HuBERT's learned representations are about
*speech content* (phonemes, speaker identity), which lives mostly below
8 kHz anyway.

### 3.5 Why 768 dimensions?

This is the size of HuBERT's hidden state — it's a design choice by the
HuBERT authors.  The number 768 comes from the transformer architecture
(12 layers × 64 attention head dim = 768, similar to BERT-base).

---

## 4. F0 Pitch

### 4.1 What is F0?

**F0 (fundamental frequency)** is the lowest frequency in a periodic
waveform — it's what we perceive as "pitch".  When you sing a C4 note,
your vocal cords vibrate at ~262 Hz; that's the F0.

- Male speech: typically 80–180 Hz
- Female speech: typically 150–300 Hz
- Silence: F0 = 0 (unvoiced)

### 4.2 How do we extract it?

We use `pyworld.harvest`, an algorithm that analyses the waveform and
estimates the F0 at regular intervals.  The interval is controlled by
`frame_period` (in milliseconds):

```
frame_period = hop_length / sample_rate × 1000
             = 160 / 40,000 × 1000
             = 4.0 ms
```

So F0 is extracted every 4 ms = 250 times per second.  This matches our
mel frame rate exactly.

### 4.3 What is "coarse F0"?

We don't use the raw F0 frequency directly — we convert it to a **coarse
bin index** using a logarithmic scale:

```python
f0_coarse = round(12 × log2(f0 / 75)) + 1
```

- `75 Hz` is the reference (lowest singable pitch)
- `12 × log2(f0/75)` converts to semitones above 75 Hz
- `+1` shifts so that bin 0 = silence/unvoiced (F0=0)

Examples:
| F0 (Hz) | Semitones | Coarse bin |
|---------|-----------|------------|
| 0       | —         | 0          |
| 75      | 0         | 1          |
| 150     | 12        | 13         |
| 300     | 24        | 25         |
| 600     | 36        | 37         |

We use 256 bins (0–255), covering ~21 octaves — more than enough for any
human voice.  The coarse bin is looked up in an **embedding table**
(`nn.Embedding(256, 256)`) to produce a 256-dim vector the model can use.

---

## 5. The Frame Alignment Problem

### 5.1 The mismatch

We have three signals, each at a different frame rate:

| Signal     | Frame rate | Frames per 3.7s chunk |
|------------|------------|----------------------|
| HuBERT     | 50 fps     | 185                  |
| F0 / Mel   | 250 fps    | 925                  |

The model needs content (HuBERT) and F0 as **inputs**, and produces mel as
the **output**.  But the input frame count (185) doesn't match the output
frame count (925).  How do we bridge this gap?

### 5.2 Our solution: linear interpolation

We resample HuBERT's 185 frames to 925 frames using **linear interpolation**
— the same technique image resizing uses, but in 1D:

```
HuBERT output:  [185, 768]  (185 frames, 768-dim each)

         F.interpolate(mode='linear', size=925)

Resampled:      [925, 768]  (925 frames, 768-dim each)
```

This is like stretching a 185-pixel-wide image to 925 pixels — the values
between original frames are linearly blended.  Each interpolated frame is a
weighted average of its two nearest original frames.

**Why is this OK?**  Speech content changes slowly (phonemes last
~50–100 ms).  HuBERT's 20 ms frame spacing (50 fps) already captures all
the content information; interpolating to 4 ms spacing (250 fps) just
gives the model a finer-grained view of the same information.

### 5.3 The old (broken) approach

Previously, the model tried to handle the upsampling itself using
`upsample_rates = [10, 4, 2, 2]` (product = 160).  But:

- 185 × 160 = 29,600, which is not 925 (the mel frame count)
- The actual ratio is 925/185 ≈ 5.0, not 160

This produced outputs with the wrong temporal dimension, making the loss
meaningless.  We fixed it by doing the interpolation in the dataset
(`upsample_rates=[1]`, identity) so the model always sees aligned inputs
and outputs of the same length.

---

## 6. The Generator: SynthesizerTrn

### 6.1 Overview

The generator takes content features + F0 and produces a mel spectrogram
(+ audio in end-to-end mode).  Here's the full data flow:

```
Content [B, 925, 768]  ──→  Encoder  ──→  [B, 925, 256]
                                              │
F0 [B, 925]  ──→  Embedding  ─────────→  [B, 925, 256]  (added)
                                              │
                                         Flows × 2  ──→  [B, 256, 925]
                                              │
                                         Upsample    ──→  [B, 192, 925]
                                              │
                                         out_conv    ──→  mel [B, 192, 925]
                                              │ (if use_decoder=True)
                                         HiFi-GAN dec ──→  audio [B, 1, 148000]
```

### 6.2 The Encoder (Transformer)

The encoder is a 6-layer transformer (same architecture as BERT) that
processes the 768-dim HuBERT features and projects them to 256 dimensions:

```
Input:  [B, 925, 768]  (HuBERT content features)
  → Linear(768 → 256)
  → Transformer × 6 layers (self-attention + feed-forward)
Output: [B, 925, 256]  (encoded speech representation)
```

Each transformer layer has:
- **Self-attention**: every frame can "look at" every other frame to
  understand context (e.g., "this vowel sound is followed by a consonant,
  which changes how it should be produced")
- **Feed-forward network**: a 2-layer MLP (256 → 768 → 256) that processes
  each frame independently

### 6.3 F0 Embedding

The coarse F0 bin (0–255) is looked up in an embedding table, producing a
256-dim vector per frame.  This vector is **added** to the encoder output:

```
x_enc = encoder_output + emb_f0(f0_bins)
```

This tells the model "produce this pitch at this frame".  The model learns
to combine content ("what to say") with pitch ("how high") to produce the
right mel spectrogram.

### 6.4 Flows (Residual Coupling Blocks)

The flows are **normalising flows** — a technique from probability theory
that learns invertible transformations.  In our pipeline, they serve as a
learned non-linear projection of the encoded features.

```
Input:  [B, 256, 925]
  → ResidualCouplingBlock (4 coupling layers, with permutations between)
  → ResidualCouplingBlock (another 4 coupling layers)
Output: [B, 256, 925]  (same shape, transformed values)
```

Each **coupling layer** splits the 256 channels in half (128 + 128) and
applies an affine transformation to one half conditioned on the other:

```
x = [x_left, x_right]   (128 channels each)
y_right = x_right × scale(x_left) + shift(x_left)
y_left = x_left          (unchanged)
```

The `scale` and `shift` functions are small WaveNet networks (dilated
convolutions with gated activations).  The "residual" part means the
input is added to the output, and "coupling" means the two halves interact.

### 6.5 Upsample (256 → 192 channels)

The flow output has 256 channels, but our mel target has 192 (the number
of mel bands).  We use a single `Conv1d(256, 192, kernel=3, padding=1)`
layer with `stride=1` (identity — no temporal change) to project:

```
[B, 256, 925] → Conv1d(256, 192, k=3, p=1) → [B, 192, 925]
```

Then `out_conv` (another `Conv1d(192, 192, k=3, p=1)`) refines the mel.

---

## 7. The HiFi-GAN Decoder

### 7.1 What it does

The decoder takes a mel spectrogram `[B, 192, 925]` and produces raw audio
`[B, 1, 148000]`.  It "undoes" the spectrogram — turning frequency
information back into a time-domain waveform.

### 7.2 Architecture

```
mel [B, 192, 925]
  │
  ▼
conv_pre:  Conv1d(192 → 512, kernel=7, pad=3)    + LeakyReLU
  │                                                [B, 512, 925]
  ├──→ ups.0: ConvTranspose1d(512→256, k=16, s=8)  + LeakyReLU → [B, 256, 14800]
  │     └──→ ResBlock×3 (kernels 3, 7, 11)         → average the 3 → [B, 256, 14800]
  │
  ├──→ ups.1: ConvTranspose1d(256→128, k=16, s=5)  + LeakyReLU → [B, 128, 74000]
  │     └──→ ResBlock×3                             → [B, 128, 74000]
  │
  ├──→ ups.2: ConvTranspose1d(128→64,  k=4,  s=2)  + LeakyReLU → [B, 64, 148000]
  │     └──→ ResBlock×3                             → [B, 64, 148000]
  │
  └──→ ups.3: ConvTranspose1d(64→32,   k=4,  s=2)  + LeakyReLU → [B, 32, 148000]
        └──→ ResBlock×3                             → [B, 32, 148000]
  │
  ▼
conv_post: Conv1d(32 → 1, kernel=7, pad=3)   → [B, 1, 148004]
  │
  ▼
tanh                                          → audio [B, 1, 148004]
```

### 7.3 Transposed convolutions (upsampling)

A regular `Conv1d` slides a kernel across the input and produces one output
per position — it can shrink or keep the length.  A `ConvTranspose1d` does
the reverse: it "spreads" each input value across multiple output
positions, **increasing** the length.

```
Input:    A B C D         (4 values)
Kernel:   × × ×           (3 weights)
          ↓ ↓ ↓
Output:   aAbBcCdD...     (upsampled)
```

The **stride** controls how far the kernel jumps between input positions,
which determines the upsampling factor:
- `stride=8` → 8× longer output
- `stride=5` → 5× longer
- `stride=2` → 2× longer

Total upsampling: 8 × 5 × 2 × 2 = **160×**.  This is why we set
`hop_length=160` — each mel frame maps to exactly 160 audio samples.

### 7.4 ResBlock (Multi-Receptive Field)

Each upsampling stage is followed by **3 ResBlocks** with different kernel
sizes (3, 7, 11).  Each ResBlock applies 3 dilated convolutions:

```
x ──→ conv(dilation=1) ──→ +x ──→ conv(dilation=1) ──→ +x
  ──→ conv(dilation=3) ──→ +x ──→ conv(dilation=3) ──→ +x
  ──→ conv(dilation=5) ──→ +x ──→ conv(dilation=5) ──→ +x
```

**Why different dilations?**  Dilation controls how far apart the kernel
samples are:

```
dilation=1:  [x x x]          ← sees 3 adjacent samples (receptive field = 3)
dilation=3:  [x . . x . . x]  ← sees samples 3 apart  (receptive field = 7)
dilation=5:  [x . . . . x . . . . x]  ← sees samples 5 apart (receptive field = 11)
```

By combining multiple dilations, each ResBlock has a **wide receptive
field** — it can "see" a long stretch of audio context, not just the
immediate neighbours.  This produces more coherent waveforms.

**Why 3 different kernel sizes (3, 7, 11)?**  This is the **Multi-Receptive
Field (MRF)** from the HiFi-GAN paper.  Different kernel sizes capture
different temporal patterns, and averaging their outputs gives a richer
representation.

### 7.5 Why tanh at the end?

The final `conv_post` layer produces a single-channel output, and `tanh`
squashes it to `[-1, 1]` — the valid range for audio samples.  Without
tanh, the decoder could produce arbitrarily large values, which would clip
to ±1 at the sound card anyway (distortion).  Tanh makes the squashing
**differentiable**, so the model can learn to produce values that stay
within range naturally.

### 7.6 The +4 sample offset

Due to symmetric padding in `conv_pre` (kernel=7, pad=3) and `conv_post`
(kernel=7, pad=3), the output is always 4 samples longer than
`T_mel × 160`:

```
T_audio = T_mel × 160 + 4
```

For 925 mel frames: 925 × 160 + 4 = 148,004 samples (vs the ground truth's
148,000).  We handle this by truncating both predicted and ground-truth
audio to the shorter length before computing losses.

---

## 8. Why G40k.pth Failed (the VITS Problem)

### 8.1 What is VITS?

**VITS** (Variational Inference with adversarial learning for end-to-end
Text-to-Speech) is a complete TTS pipeline that includes:

```
Text → Text Encoder → Posterior Encoder (enc_q) → Flow → Decoder → Audio
                       ↑                                        ↑
                       Mel spectrogram (input)      Speaker embedding (conditioning)
```

The key insight: in VITS, the decoder **does not** take the mel spectrogram
as input.  It takes the **posterior latent** `z` — a compressed
representation produced by the posterior encoder (`enc_q`) from the mel.
The decoder also receives a **speaker embedding** via `dec.cond` (a
`Conv1d(256, 512, 1)` layer).

### 8.2 Why G40k.pth is a VITS checkpoint

G40k.pth is a **full SynthesizerTrn checkpoint** from the RVC project.  Its
549 state-dict keys include:

- `enc_p.*` — content encoder (our pipeline's encoder)
- `enc_q.*` — posterior encoder (we don't have this)
- `flow.*` — normalising flows
- `dec.*` — HiFi-GAN decoder (what we tried to use as a vocoder)
- `emb_g.weight` — speaker embedding table (109 speakers × 256 dims)

The `dec` module was trained **jointly** with `enc_q` and `flow` — it
learned to expect inputs from the specific distribution that `enc_q` and
the flow produce.  It also learned to depend on the speaker embedding via
`dec.cond`.

### 8.3 What happens when you use it standalone

When we extract `dec.*` into a standalone `Generator` and feed it a mel
spectrogram:

1. The mel goes through `conv_pre` → small, reasonable values (mean ≈ 0.02)
2. The first upsampling stage produces mean ≈ 1.1 (growing)
3. Each subsequent stage multiplies the magnitude by ~3–15×
4. After 4 stages: mean ≈ 130 (exploded)
5. `conv_post` maps 130 → -6.76
6. `tanh(-6.76) = -0.9994` ← saturated

The resblock weights amplify because they were trained to operate on
inputs from `enc_q` (which has a specific distribution), not on raw mel
spectrograms.  Without the posterior encoder's "normalising" effect, the
activations drift outside the trained range.

### 8.4 Why conditioning didn't help

We tried passing:
- The flow output from our trained model as conditioning
- The mean speaker embedding from G40k.pth's `emb_g`
- Zero conditioning
- Random conditioning

All produced the **same** saturated output.  This is because the
resblock amplification happens *before* the conditioning has a chance to
shape the output — the conditioning is added after `conv_pre` but the
explosion happens in the upsampling stages that follow.

### 8.5 The conclusion

G40k.pth's `dec` is not a standalone vocoder — it's a decoder *slot* in a
VITS pipeline.  Using it standalone is like taking a car engine out and
trying to use it as a generator: it was designed to work with the rest of
the car's systems, not on its own.

**Our solution**: train our own HiFi-GAN decoder from scratch, jointly
with our generator.  This way the decoder learns to work with *our*
encoder's output distribution, not someone else's posterior encoder.

---

## 9. Discriminators: The Adversarial Referee

### 9.1 Why do we need discriminators?

The mel MSE loss alone produces "blurry" audio — it averages over many
possible waveforms that match the mel, producing a muddy result.  This is
the same reason GAN-based image generators produce sharper images than
MSE-trained ones.

**Discriminators** are neural networks that learn to tell the difference
between real audio (ground truth) and fake audio (generator output).  By
training the generator to "fool" the discriminator, we push it toward
producing realistic-sounding audio.

### 9.2 Multi-Period Discriminator (MPD)

The MPD looks for periodic patterns in the audio.  It "folds" the 1D
waveform into a 2D grid with a specific **period** P:

```
Original:  [s0, s1, s2, s3, s4, s5, s6, s7, ...]
           ↓  fold with period=3
2D grid:   [s0, s1, s2]
           [s3, s4, s5]
           [s6, s7, s8]
           ...
```

Then it applies 2D convolutions to this grid.  Different periods (2, 3, 5,
7, 11) capture different periodic structures — these correspond to
different pitch ranges and harmonic patterns in speech.

Each sub-discriminator is a 4-layer 2D CNN:
```
Conv2d(1→32)  →  Conv2d(32→128)  →  Conv2d(128→512)  →  Conv2d(512→1024)  →  Conv2d(1024→1)
```

Output: a single logit (real vs fake score) per sub-discriminator.

### 9.3 Multi-Scale Discriminator (MSD)

The MSD looks at the audio at different temporal resolutions.  Each sub-
discriminator is a 5-layer 1D CNN with progressively larger channels:

```
Conv1d(1→16)  →  Conv1d(16→64)  →  Conv1d(64→256)  →  Conv1d(256→1024)  →  Conv1d(1024→1024)  →  Conv1d(1024→1)
```

We use 3 sub-discriminators (all identical in our implementation; the
HiFi-GAN paper uses progressive 1×/2×/4× downsampling between scales).

### 9.4 What do they "look for"?

The discriminators learn to detect artifacts that the generator tends to
produce:
- **MPD**: detects periodicity errors (wrong pitch, missing harmonics)
- **MSD**: detects broad-spectrum noise, spectral holes, temporal
  discontinuities

As the generator improves, the discriminators must find subtler flaws,
which in turn pushes the generator further.  This "arms race" is the core
of GAN training.

---

## 10. GAN Training: The Cat-and-Mouse Game

### 10.1 The two players

- **Generator (G)**: our SynthesizerTrn.  Tries to produce realistic audio.
- **Discriminator (D)**: the MPD + MSD.  Tries to tell real from fake.

### 10.2 The training loop (per batch)

```
1. Generator produces fake audio from content + F0
2. Discriminator step:
   - Feed real audio → D should output "real" (target = 1)
   - Feed fake audio (detached) → D should output "fake" (target = 0)
   - Compute D loss, backprop, update D only
3. Generator step:
   - Feed fake audio → G wants D to output "real" (target = 1)
   - Also compute feature matching + L1 + mel MSE
   - Compute G loss, backprop, update G only
```

**Why D first?**  If G updates first, it can "win" before D has a chance to
adapt — G would then have no gradient signal to improve further.  By
updating D first, we ensure D is always a worthy opponent.

**Why detach fake audio for D's step?**  If we don't detach, gradients from
D's loss would flow back into G — we'd accidentally train G to make audio
that's *easy to classify as fake*, the opposite of what we want.

### 10.3 LS-GAN loss

We use **Least Squares GAN** loss (MSE form) instead of the original
GAN's sigmoid + binary cross-entropy:

```
Original GAN:  D tries to minimise -log(D(real)) - log(1 - D(fake))
LS-GAN:       D tries to minimise (D(real) - 1)² + (D(fake) - 0)²
              G tries to minimise (D(fake) - 1)²
```

LS-GAN is more stable because:
- It doesn't use sigmoid (which saturates and kills gradients)
- The penalty grows quadratically (not flatly) as D becomes more confident
- It penalises confident-but-wrong predictions more heavily

---

## 11. Loss Functions Explained

### 11.1 The four losses

```
Total Generator Loss = 45 × Mel_MSE + 1 × Audio_L1 + 2 × Feature_Matching + Adversarial
```

| Loss | Weight | What it does |
|------|--------|-------------|
| Mel MSE | 45 | Ensures predicted mel matches ground truth mel — "are you saying the right sounds?" |
| Audio L1 | 1 | Ensures predicted waveform is close to ground truth waveform — "sample-level accuracy" |
| Feature Matching | 2 | Ensures internal discriminator features match between real and fake — "do they sound similar at multiple levels of detail?" |
| Adversarial | 1 (implicit) | Pushes D to classify fake as real — "is it realistic enough to fool the discriminator?" |

### 11.2 Why mel MSE has weight 45?

The mel MSE is the "anchor" — without it, the generator could produce
realistic-sounding audio that says completely different words than the
input content.  The weight 45 (from the HiFi-GAN paper) makes this term
dominate the total loss, ensuring content fidelity is always the top
priority.

### 11.3 Why audio L1 has weight 1?

Audio L1 is a "helper" term — it keeps the waveform shape close to ground
truth without being too strict (L1 is more forgiving than L2 for
speech, which has lots of small variations).  Weight 1 is small enough
that the adversarial + feature matching losses can still sharpen the
output.

### 11.4 What is feature matching?

Instead of just looking at the discriminator's final "real/fake" verdict,
we also compare the **intermediate layer activations** of D when
processing real vs fake audio:

```
D(real) → [conv1_out, conv2_out, conv3_out, conv4_out, logits]
D(fake) → [conv1_out, conv2_out, conv3_out, conv4_out, logits]

Feature matching loss = Σ |D(real).conv_i_out - D(fake).conv_i_out|
```

This gives the generator a richer signal: not just "are you fooling D?"
but "do your internal representations match real audio at every level of
detail?"  It acts as a regulariser that prevents mode collapse (the
generator producing the same audio for every input).

---

## 12. Weight Normalisation

### 12.1 What is it?

**Weight normalisation** is a training technique that reparameterises a
convolution layer's weight `w` as:

```
w = g × v / ||v||
```

where:
- `v` is an unconstrained direction vector (learned)
- `g` is a scalar magnitude (learned)
- `||v||` is the L2 norm of `v`

This decouples the weight's **direction** from its **magnitude**, which
improves training stability (similar to batch normalisation but without
depending on batch statistics).

### 12.2 Why does it matter for us?

During training, the HiFi-GAN decoder uses weight-norm'd convolutions
(`weight_norm(Conv1d(...))`).  The checkpoint stores the decomposed
parameters `weight_v` and `weight_g` (not the fused `weight`).

At inference time, we call `remove_weight_norm()` which fuses them back:
```
weight = weight_g × weight_v / ||weight_v||
```

This is important because:
- During training, PyTorch recomputes `w = g × v / ||v||` on every forward
  pass (dynamic, allows gradients to flow to both `g` and `v`)
- During inference, we can pre-compute the fused `weight` (static, faster)

### 12.3 What went wrong with G40k.pth?

We initially suspected `remove_weight_norm()` was fusing incorrectly.  We
verified it wasn't by:
1. Manually computing `g × v / ||v||` and comparing — exact match
2. Running inference without `remove_weight_norm()` — identical output
3. Checking `weight_g` values (~1.12, normal) and `weight_v` std (~0.071,
   normal)

The fusion was correct.  The saturation was caused by the VITS joint-
training problem (see §8), not by weight normalisation.

---

## 13. Why tanh Saturates

### 13.1 The tanh function

```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

- `tanh(0) = 0`
- `tanh(1) ≈ 0.76`
- `tanh(3) ≈ 0.995`
- `tanh(6) ≈ 0.99998`
- `tanh(-6) ≈ -0.99998`

For `|x| > 3`, tanh is essentially flat at ±1 — it has "saturated".

### 13.2 What happened with G40k.pth

The decoder's pre-tanh values reached `mean = -6.76`:
- 99.9% of samples had `|x| > 3` → tanh output ≈ ±1
- Mean of tanh output = -0.9996

The audio was a constant -1.0 — a DC signal, not speech.

### 13.3 Why the pre-tanh values exploded

Each resblock stage added residual connections:
```
x = x + leaky_relu(conv(x))
x = x + leaky_relu(conv(x))
```

Even with zero input, the conv biases produce small non-zero outputs
(~0.02).  After 12 resblocks (3 per stage × 4 stages), these biases
accumulate:
- After stage 0: mean ≈ 1.1
- After stage 1: mean ≈ 2.1
- After stage 2: mean ≈ 8.0
- After stage 3: mean ≈ 130

The weights were trained to operate on inputs from VITS's posterior encoder
(which produces values in a specific range).  Our mel inputs are in a
completely different range, so the weights amplify instead of stabilise.

### 13.4 Why our trained decoder won't have this problem

Our decoder is trained from random initialisation on **our** mel
spectrograms.  The weights learn to operate on the actual distribution of
mel values we produce, so they won't amplify outside the trained range.
The smoke test confirmed this: 0% saturation, mean ≈ 0.04.

---

## 14. Glossary

| Term | Meaning |
|------|---------|
| **Sample** | One number in a waveform (e.g., 0.031) |
| **Sample rate** | Samples per second (e.g., 40,000 = 40 kHz) |
| **Spectrogram** | 2D grid of frequency vs time showing energy |
| **Mel scale** | Frequency scale that matches human hearing (192 bins) |
| **Mel frame** | One column of the mel spectrogram (4 ms of audio at our settings) |
| **Hop length** | Samples between successive mel frames (160 in our pipeline) |
| **dB (decibel)** | Logarithmic scale for power: `10 × log10(power)` |
| **HuBERT** | Self-supervised speech model producing 768-dim content features at 50 fps |
| **F0** | Fundamental frequency (pitch), extracted at 250 fps |
| **Coarse F0** | F0 converted to a 0–255 bin index via `12 × log2(f0/75) + 1` |
| **Frame rate (fps)** | Frames per second of a signal (mel=250, HuBERT=50) |
| **F.interpolate** | PyTorch function for resampling (stretching/compressing) sequences |
| **Conv1d** | 1D convolution — slides a kernel across a sequence |
| **ConvTranspose1d** | Transposed convolution — "spreads" input to a longer output (upsampling) |
| **ResBlock** | Residual block — adds input to conv output: `x + f(x)` |
| **MRF** | Multi-Receptive Field — 3 ResBlocks with different kernel sizes, averaged |
| **Dilation** | Spacing between kernel samples (sees further context) |
| **tanh** | Squashes values to [-1, 1]; saturates for \|x\| > 3 |
| **VITS** | Variational Inference with adversarial learning for end-to-end TTS |
| **Posterior encoder (enc_q)** | VITS component that compresses mel into a latent z; we don't have one |
| **Speaker embedding** | 256-dim vector representing speaker identity; we don't use one |
| **GAN** | Generative Adversarial Network — generator vs discriminator |
| **MPD** | Multi-Period Discriminator — folds audio into 2D by period, runs 2D convs |
| **MSD** | Multi-Scale Discriminator — runs 1D convs at multiple resolutions |
| **LS-GAN** | Least-Squares GAN loss — uses MSE instead of sigmoid + BCE |
| **Feature matching** | L1 between intermediate D features of real vs fake |
| **Weight norm** | Reparameterisation `w = g × v / \|v\|` for stable training |
| **Mode collapse** | GAN failure where generator produces the same output for every input |
| **oneDNN** | Intel's deep learning optimisation library; JIT-compiles GPU convolution kernels |
| **Nyquist theorem** | To capture frequency f, you need sample rate ≥ 2f |
| **STFT** | Short-Time Fourier Transform — computes frequency content in sliding windows |
| **n_fft** | Window size for STFT (640 in our pipeline = hop × 4) |