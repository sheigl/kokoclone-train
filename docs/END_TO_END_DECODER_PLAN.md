# End-to-End HiFi-GAN Decoder — Design Document

> **Status**: Training in progress (epoch 202/300). Implementation complete
> across both training and inference sides. Key findings from pretrained
> decoder integration documented below.
>
> **Context**: This document specifies the replacement for the failed
> "extract G40k.pth's dec module as a standalone vocoder" approach described
> in `VOCODER_INVESTIGATION.md`. The goal is to train a HiFi-GAN decoder
> end-to-end with our existing `SynthesizerTrn` generator so the inference
> pipeline owns the entire `content + F0 → mel → audio` chain.

---

## 1. Goal

Replace the external pretrained vocoder with a **HiFi-GAN decoder** that is
trained jointly with our generator. After training, inference becomes a
single forward pass through our model — no G40k.pth dependency.

## 2. Background

### 2.1 Current pipeline (broken at the vocoder step)

```
Audio (40kHz, 3.7s chunks)
    ├→ 16kHz resample → HuBERT → content [T_c≈185, 768]
    ├→ pyworld.harvest → F0 [T_m≈493]
    └→ MelSpectrogram → mel [T_m, 192]   ← dB-normalized

SynthesizerTrn (our trained model):
    content + F0 → flows → upsample → mel_pred [B, 192, T]

Vocoder (G40k.pth dec module):                   ← FAILS HERE
    mel_pred → audio [B, 1, T_audio]
    Result: saturated (mean=-0.9996, std=0.03) for any input
```

### 2.2 Proposed pipeline (end-to-end)

```
SynthesizerTrn (extended):
    content + F0 → flows → upsample → mel_pred [B, 192, T]
                                              ↓ embedded HiFi-GAN decoder
                                          audio_pred [B, 1, T_audio]

Losses (computed against GT audio):
    L1(mel_pred, mel_gt)         ← mel reconstruction (existing)
    L1(audio_pred, audio_gt)     ← waveform reconstruction (new)
    LS-GAN(MPD, MPD)             ← multi-period discriminator (new)
    LS-GAN(MSD, MSD)             ← multi-scale discriminator (new)
    FM(MPD/MSD feats)            ← feature matching (new)
```

## 3. Architecture Changes

### 3.1 HiFi-GAN Decoder (new module)

Added to `SynthesizerTrn` as a `dec` submodule. Matches the topology we
extracted from G40k.pth (see `VOCODER_INVESTIGATION.md` §7) but trained
from scratch on our data.

```python
class HiFiGANDecoder(nn.Module):
    """HiFi-GAN mel→audio decoder.

    Topology (matches G40k.pth for compatibility with existing RVC tooling):
      conv_pre:  Conv1d(n_mels=192       → 512,  k=7, pad=3)
      ups.0:     ConvTr1d(512 → 256, k=16, stride=8)     # ×8
      ups.1:     ConvTr1d(256 → 128, k=16, stride=5)     # ×5
      ups.2:     ConvTr1d(128 → 64,  k=4,  stride=2)     # ×2
      ups.3:     ConvTr1d(64  → 32,  k=4,  stride=2)     # ×2
      resblocks[0..11]: 3 MRF groups (kernels 3,7,11) × 4 stages
                         dilations [[1,3,5]] for all groups
      conv_post: Conv1d(32 → 1, k=7, pad=3, bias=False) → tanh

    Product of upsample strides = 8*5*2*2 = 160.
    With hop_length=160 at sr=40kHz, mel frame rate = 250fps.
    """
```

### 3.2 Frame rate adjustment

**Current**:
```
hop_length=300 at sr=40kHz → mel frame rate = 40000/300 = 133.3 fps
HuBERT at 16kHz stride 320 → 50fps; ratio content/mel = 2.67
```

**New** (to match decoder product=160):
```
hop_length=160 at sr=40kHz → mel frame rate = 40000/160 = 250 fps
HuBERT at 16kHz stride 320 → 50fps; ratio content/mel = 5.0
```

**Implications**:
- 3.7s chunks now produce ~925 mel frames (was 493)
- HuBERT produces ~185 content frames, needed ratio is now 5.0
- `F.interpolate(content, size=len(f0), mode='linear')` handles this fine
- Mel spectrogram extraction (stage 4) must be re-run with hop_length=160
- All existing `4_mel/*.npy` files must be deleted and re-extracted

### 3.3 SynthesizerTrn forward (extended)

```python
def forward(self, c, f0):
    # Encoder + flow (unchanged)
    x_enc = self.enc_p(c, mask=...)
    x_enc = x_enc + self.emb_f0(f0)
    x = x_enc.transpose(1, 2)
    for flow in self.flows:
        x, _ = flow(x, g=None)

    # Upsample flow output → mel channels (unchanged)
    for up_layer in self.ups:
        x = up_layer(x)
    mel = self.out_conv(x)          # [B, 192, T_mel]

    # NEW: HiFi-GAN decoder produces audio from predicted mel
    audio = self.dec(mel)           # [B, 1, T_audio]

    return mel, audio
```

### 3.4 Discriminators

Both discriminators already exist in
`kokoclone_train/core/rvc_models/models.py` (`MultiPeriodDiscriminator`,
`MultiScaleDiscriminator`). They were stub implementations used in earlier
training experiments — they need to be brought up to HiFi-GAN spec:

#### Multi-Period Discriminator (MPD)
Periods: `[2, 3, 5, 7, 11]` (standard HiFi-GAN).
Each disc folds the time axis into `(T // P, P)` and runs 2D convs:
```
Conv2d(1   → 32,  (5,1), (3,1))  LeakyReLU
Conv2d(32  → 128, (5,1), (3,1))  LeakyReLU
Conv2d(128 → 512, (5,1), (3,1))  LeakyReLU
Conv2d(512 → 1024,(5,1), (3,1))    LeakyReLU
Conv2d(1024→ 1,   (3,1), 1)
```

#### Multi-Scale Discriminator (MSD)
Three scales (1×, 2× downsample, 4× downsample). Each scale:
```
Conv1d(1    → 16,   15, 1, pad=7)    LeakyReLU
Conv1d(16   → 64,   41, 4, g=4,  pad=20) LeakyReLU
Conv1d(64   → 256,  41, 4, g=16, pad=20) LeakyReLU
Conv1d(256  → 1024, 41, 4, g=64, pad=20) LeakyReLU
Conv1d(1024 → 1024, 41, 4, g=256,pad=20) LeakyReLU
Conv1d(1024 → 1,    3,  1, pad=1)
```

## 4. Loss Functions

### 4.1 Reconstruction loss (mel)
```python
L_mel = MSELoss(mel_pred, mel_gt)            # existing loss; keep weight 45
```
Mirrors original HiFi-GAN config. MSE preserves the existing training signal
that has already converged on mel prediction.

### 4.2 L1 waveform reconstruction
```python
L_audio = L1Loss(audio_pred, audio_gt)      # weight 1.0
```
Standard in HiFi-GAN.音频 L1 keeps the high-frequency structure pinned
to ground truth; GAN losses alone tend to drift toward plausible-but-wrong
content.

### 4.3 Adversarial loss (LS-GAN)
For each sub-discriminator `D` in MPD and MSD:
```python
# Real
loss_D_real = MSE(D(real).logits, 1.0)
loss_D_fake = MSE(D(fake.detach()).logits, 0.0)
loss_D = loss_D_real + loss_D_fake

# Generator
loss_G_adv = MSE(D(fake).logits, 1.0)
```
LS-GAN uses least-squares form `MSE` (not Sigmoid+BCE) for stable training.

### 4.4 Feature matching loss
For each sub-discriminator, sum L1 between intermediate features of real
and fake:
```python
loss_fm = 0
for d in discriminators:
    real_feats = d(real)[1]    # feature_list
    fake_feats = d(fake)[1]
    loss_fm += sum(L1(rf, ff) for rf, ff in zip(real_feats, fake_feats))
loss_fm *= 2.0                  # HiFi-GAN uses weight 2.0
```

### 4.5 Total loss
```python
# Generator step
loss_G = 45.0 * L_mel + 1.0 * L_audio + 2.0 * loss_fm + loss_G_adv

# Discriminator step (schedule alternation — see §5)
loss_D = sum(loss_D_sub for each sub-discriminator)
```

## 5. Training Loop

For each batch:
1. **Forward generator** → `(mel_pred, audio_pred)`
2. **Discriminator step** (zero_grad D, compute `loss_D`, backward, step)
3. **Generator step** (zero_grad G, compute `loss_G`, backward,
   `clip_grad_norm_(G.parameters(), max_norm=1.0)`, step)
4. `scheduler_G.step()`, `scheduler_D.step()`

This pessimistic ordering (D before G) follows the original HiFi-GAN
implementation. D sees a "fresh" fake each batch and prevents the generator
from winning the cat-and-mouse game too quickly.

### 5.1 Optimizers and schedulers
```python
opt_G = AdamW(net_g.parameters(),    lr=2e-4, betas=(0.8, 0.99), weight_decay=1e-5)
opt_D = AdamW(disc.parameters(),      lr=2e-4, betas=(0.8, 0.99), weight_decay=1e-5)
sch_G = ExponentialLR(opt_G, gamma=0.999)
sch_D = ExponentialLR(opt_D, gamma=0.999)
```
`2e-4` is the HiFi-GAN recommended LR; differs from our earlier mel-only
`1e-4`. `gamma=0.999` is gentler than mel-only `0.998` because adversarial
training runs many more epochs.

### 5.2 Resume semantics
Checkpoints save under keys `{"weight_g", "opt_g", "sch_g", "weight_d",
"opt_d", "sch_d", "epoch", "best_loss"}`. Existing resume logic extended
to load D weights and optimiser state.

## 6. Dataset Changes

### 6.1 New dataset output: audio chunks
`ContentFeatureDataset.__getitem__` returns `(content, f0, mel, audio)`
where `audio` is the raw waveform chunk loaded from `0_gt_wavs/{stem}.wav`.

### 6.2 Audio length alignment
Each 3.7s chunk at 40kHz = 148,000 samples. The decoder produces
`T_mel × 160` audio samples (from `product(ups)=160`), so for 925 mel
frames we get 148,000 samples — exact match by construction.

We must enforce exact alignment in the loader:
```python
T_audio_target = T_mel * 160
audio = audio[:T_audio_target]      # truncate
if len(audio) < T_audio_target:
    audio = F.pad(audio, (0, T_audio_target - len(audio)))
```

## 7. Checkpoint and Inference Changes

### 7.1 Checkpoint
Checkpoint now stores the extended SynthesizerTrn (which includes `dec`)
plus the discriminators:
```python
torch.save({
    "weight_g": net_g.state_dict(),    # includes dec.*
    "opt_g":   opt_G.state_dict(),
    "sch_g":   sch_G.state_dict(),
    "weight_d": disc.state_dict(),
    "opt_d":   opt_D.state_dict(),
    "sch_d":   sch_D.state_dict(),
    "epoch":   epoch,
    "best_loss": best_loss,
}, path)
```

### 7.2 Inference (kokoclone side)
`rvc_engine._infer` no longer loads G40k.pth; instead it uses the
generator's embedded `dec`:
```python
mel, audio = net_g.infer(c, f0=f0_tensor)
# audio has shape [B, 1, T_audio] — already the final waveform
```
The `_load_vocoder` function and `_vocoder_cache` are **removed** entirely.

## 8. Testing Strategy

### 8.1 Unit tests (new)
- `test_decoder_outputs_audio_shape`: zero-mel through `HiFiGANDecoder`
  produces `[B, 1, T_mel×160]` audio.
- `test_decoder_non_saturated`: tanh output `mean ∈ (-0.5, 0.5)` and
  `std > 0.05` for small-magnitude mel inputs (catches the saturation
  regression we spent days debugging on G40k.pth).
- `test_discriminator_shapes`: MPD/MSD forward produces sane logit shapes
  over batched inputs.
- `test_frame_alignment_with_hop_160`: dataset returns aligned
  content/f0/mel/audio with the new hop_length.

### 8.2 Smoke training
Two-epoch end-to-end run on a small subset, verifying:
- Generator + discriminator losses are finite and decreasing
- `audio_pred` `|mean| < 0.3` after one epoch (not saturated)
- No NaN/Inf in grads

### 8.3 Acceptance criterion
After `N` epochs of end-to-end training, inference produces audio where:
- `|mean(wav)| < 0.1`
- `std(wav) > 0.05`
- Audible speech (informal listen)

If after reasonable tuning the model drifts to saturation, we will switch
gears to **option A** (install a standard pretrained HiFi-GAN).

## 9. Rollback Plan

### 9.1 Trigger Conditions

Roll back if **any** of the following are met:

1. **Loss divergence**: Generator loss increases for >50 consecutive epochs
   after initial decrease (indicates G/D imbalance that won't self-correct).
2. **Saturation regression**: `|audio_mean| > 0.5` or `audio_std < 0.01`
   persists for >30 epochs (decoder stuck in tanh saturation).
3. **Quality ceiling**: After 500+ epochs (~130k steps), audio std is still
   <70% of ground truth when tested with real-audio input.
4. **NaN/Inf gradients**: Gradient clipping (`max_norm=1.0`) cannot prevent
   NaN propagation for >10 consecutive batches.
5. **Time budget exceeded**: 2 weeks of training without meaningful improvement
   (audio std not trending toward GT).

### 9.2 Rollback Steps

If triggered:

1. **Preserve artifacts**:
   - Save the latest checkpoint as `{voice}_e2e_last.pth` (not `v1.pth`).
   - Export a short audio sample from the last epoch for comparison.

2. **Switch to mel-only training** (`use_decoder=False`):
   - Re-run preprocessing with `hop_length=300` (mel frame rate 133fps).
   - Train SynthesizerTrn with MSE(mel_pred, mel_gt) only — no decoder.
   - This produces high-quality mel spectrograms (the encoder+flow are
     well-trained from prior runs).

3. **Use standalone pretrained HiFi-GAN vocoder** for inference:
   - Download `jik876/hifi-gan` release checkpoint (original paper, not RVC variant).
   - In `kokoclone/core/rvc_engine.py`, the `_load_vocoder()` path already
     supports this — it loads a standalone `Generator` from G40k.pth.
   - Replace G40k.pth with the standard HiFi-GAN checkpoint.
   - Adjust mel normalization at inference boundary if needed (standard
     HiFi-GAN expects different dB range than our internal mels).

4. **Verify**: Run `convert_audio()` on a known input and confirm:
   - `|mean(wav)| < 0.1`
   - `std(wav) > 0.05`
   - Audible speech (informal listen)

### 9.3 What Is Preserved

- **Decoder class** (`HiFiGANDecoder`): Reusable for future attempts.
- **Discriminators** (MPD/MSD): Standard HiFi-GAN components, always useful.
- **Audio loader**: Dataset returning `(content, f0, mel, audio)` is more
  capable than mel-only; can be used for other loss experiments.
- **Encoder + flow weights**: These are well-trained on mel prediction;
  only the decoder path changes.

### 9.4 What Is Discarded

- Discriminator optimizer state (not needed in mel-only mode).
- Joint G/D checkpoint format (revert to generator-only checkpoints).

## 10. Honest Risks

1. **Training time.** Adversarial training typically requires
   ~hundreds of epochs to converge on small datasets. We have 529 chunks
   (≈30 min of audio). Expect ≥6 hours per attempt on the 32GB GPU.
2. **Mode collapse.** Classic GAN failure. Mitigations: feature
   matching (already in the loss), large discriminator learning rate
   decay, gradient penalty if needed.
3. **Audio quality ceiling.** HiFi-GAN trained on 30 min of one speaker
   will sound like that speaker, but the original kokoro-onnx vocoder
   produces much higher fidelity out of the box. For production voice
   conversion, the standalone pretrained approach may be a better choice;
   for the project's pedagogical goal, end-to-end wins.
4. **Loss weight tuning.** The weights 45/1/2 in `§4.5` are HiFi-GAN
   paper defaults, but they may need adjustment for our small dataset.

## 11. References

- HiFi-GAN: Kong et al. 2020. "HiFi-GAN: Generative Adversarial Networks
  for Efficient and High Fidelity Speech Synthesis". arXiv:2010.05646.
- VITS: Kim et al. 2021. "Conditional Variational Autoencoder with
  Adversarial Learning for End-to-End Text-to-Speech". arXiv:2106.06163.
  — the architecture G40k.pth is built on; clarified why `dec` cannot
  be extracted standalone.
- RVC-Boss: `lj1995/VoiceConversionWebUI`. The original RVC ecosystem;
  confirmed G40k.pth file layout.

---

## 12. Implementation Task Breakdown

| # | Task | Files |
|---|---|---|
| 1 | Add `HiFiGANDecoder` class to training-side `models.py` (or new `vocoder.py`) | `kokoclone_train/core/rvc_models/vocoder.py` (new) |
| 2 | Add `dec` to `SynthesizerTrn` (training side); update `forward()` to return `(mel, audio)` | `kokoclone_train/core/rvc_models/models.py` |
| 2b | Update `infer()` to return `(mel, audio)` on kokoclone side | `kokoclone/core/rvc_models/models.py` |
| 3 | Port HiFi-GAN `MultiPeriodDiscriminator` & `MultiScaleDiscriminator` (upgrade existing stubs) | `kokoclone_train/core/rvc_models/models.py` |
| 4 | Update `hop_length=160` and `n_fft=640` in mel extraction and config | `kokoclone_train/core/rvc_train.py`, config |
| 5 | Update `ContentFeatureDataset` to return `(content, f0, mel, audio)` with aligned audio | `kokoclone_train/core/rvc_train.py` |
| 6 | Update training loop: G/D alternation, full loss combination | `kokoclone_train/core/rvc_train.py` |
| 7 | Update checkpoint save/load to carry both G and D state | same |
| 8 | Update inference `rvc_engine._infer` to use `net_g.dec` directly; remove G40k.pth dependency | `kokoclone/core/rvc_engine.py` |
| 9 | Run 2-epoch smoke training; verify non-saturated output | ✅ Done (epochs 1-100) |
| 10 | Add `tests/test_end_to_end_audio.py` with the unit tests in §8.1 | `tests/` |

## 13. Training Findings

### 13.1 f0G48k.pth pretrained decoder integration (July 2026)

**What we tried**: Load pretrained `f0G48k.pth` decoder weights into our
end-to-end model to accelerate convergence.

**Three cascading bugs discovered**:
1. **Dead URL**: The original HF repo
   (`liujing04/Retrieval-based-Voice-Conversion-WebUI`) returns 404.
   Correct URL: `lj1995/VoiceConversionWebUI/pretrained_v2/f0G48k.pth`.
2. **Wrong state dict key**: Code used `ckpt.get("weight", ckpt)`, but the
   checkpoint uses key `"model"`. Zero weights were loaded silently.
3. **Shape mismatches crash**: Even with `strict=False`,
   `load_state_dict` raises RuntimeError on shape mismatches. The
   pretrained has kernel sizes 24/20 for upsample layers; our config uses
   16/16. Pre-filtering to matching shapes only avoids the crash.

**Result**: 229/560 keys loaded (all decoder resblocks + conv_pre +
conv_post + ups.2/ups.3). Encoder+flow are architecturally incompatible
(different `enc_p` structure, different flow channels) and stay random init.

**Initial inference**: 100% tanh saturation. The pretrained decoder
expects VITS-style latent input (posterior encoder + flow + speaker
conditioning), not our mel spectrograms. This is the exact same
co-adaptation pattern documented in G40k.pth (§7 of
`VOCODER_INVESTIGATION.md`).

**Resolution**: Unlike G40k.pth (frozen), our decoder's pretrained
resblock weights can fine-tune to the mel input distribution. Expected
to de-saturate over ~10-20 epochs of training. D loss dropped from 3.94
→ 2.67 immediately after merge, confirming the pretrained decoder
produces more realistic audio (and the discriminator detects it).

### 13.2 Training Progress (July 2026)

| Metric | Epoch 100 | Epoch 200 |
|--------|-----------|-----------|
| Mel loss | ~22 | ~17 |
| Mel mean rec. | -48 dB | -48 dB |
| G loss | 2055 | 1718 |
| D loss | ~3.97 | ~3.97 (3.94) → 2.67 after merge |
| Audio std | 0.067 | saturated (post-merge) |

**Trend**: Mel loss decreasing at ~0.02/epoch. Target: mel mean within
~few dB of GT (-37 dB). Current convergence rate suggests ~500+ epochs
may be needed for full intelligibility. `LAMBDA_MEL` increased 45→100
to push spectral accuracy; discriminator LR reduced 2e-4→5e-5 to
prevent D from overpowering G.

**Key bottleneck**: The mel→decoder input distribution mismatch. The
decoder was designed for VITS latent vectors; it receives mel
spectrograms instead. The first conv layer (`conv_pre` 192→512) and
resblocks need to adapt to this very different signal. Once fine-tuned,
the HiFi-GAN topology is known to produce high-fidelity audio given
reasonable mel inputs.

### 13.3 XTTS v2 Integration & TTS Distribution Mismatch (July 2026)

**Context**: Tested XTTS v2 (coqui-tts 0.27.5, idiap fork) as the TTS source for
TTS→RVC voice conversion, replacing Kokoro. Required monkey-patch for
transformers 5.x (`isin_mps_friendly` removed), torchaudio 2.8.0+xpu for
compatibility, and unidic dictionary download.

**Test results (Jefferson epoch 600)**:
- Real audio → RVC: std=0.095 (96% of GT), good quality
- XTTS→RVC: std=0.0897 (91% of GT), noticeable degradation — robotic,
  muffled ("cotton balls" effect)
- Kokoro→RVC: std=0.086 — XTTS is marginally better but same class of problem

**Test results (Welch epoch 289)**:
- XTTS→RVC: std=0.0687 (69% of GT) — worse due to less decoder training

**Root cause**: The decoder was trained exclusively on mels derived from
real audio (torchaudio.MelSpectrogram on ground truth waveforms). TTS-
generated audio has different spectral characteristics (cleaner, less
noise, different spectral tilt), producing HuBERT features and F0
contours at different points in the distribution. The encoder+flow are
well-generalized (real audio converts at 96% GT), but the decoder sees
out-of-distribution mels and produces muffled output.

**Corrective actions (July 6, 2026)**:
1. Restore discriminator LR from 5e-5 → 2e-4 (standard HiFi-GAN, matched
   to generator LR) — the earlier D LR reduction compensated for
   LAMBDA_FM=20 being 10x too high; fix both to match literature.
2. Reduce LAMBDA_FM from 20 → 2.0 (HiFi-GAN standard).
3. Keep LAMBDA_AUDIO_L1=200 (primary waveform supervision, has no
   standard counterpart in vanilla HiFi-GAN which uses mel-of-audio).
4. Keep LAMBDA_MEL=100 (internal mel prediction loss, not the same as
   HiFi-GAN's mel-of-audio loss).

**XTTS data augmentation** (deferred pending hyperparameter evaluation):
Pre-generate XTTS audio from training transcripts, extract HuBERT+F0
from synthetic versions, and interleave with real-source samples during
training. Target mels are always from real audio. This would directly
teach the decoder to handle TTS-style inputs at inference time.

**Training horizon note**: Standard HiFi-GAN trains for 2.5M steps.
Jefferson at epoch 600 (264 batches/epoch) = 158,400 steps = 6% of
standard. The decoder is critically undertrained. Correcting
hyperparameters ensures each remaining epoch produces maximum
improvement per step.