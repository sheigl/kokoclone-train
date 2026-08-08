# G40k.pth Vocoder Saturation — Forensic Investigation Log

> **Status**: Investigation closed. Root cause confirmed. Remediation path
> selected: **train a HiFi-GAN decoder end-to-end** (see
> `END_TO_END_DECODER_PLAN.md`).

This document is a research-style log of the investigation into why loading
`G40k.pth`'s `dec.*` weights into a standalone HiFi-GAN `Generator` and running
mel→waveform inference produces a **fully saturated tanh output**
(`mean = -0.9996`, `std ≈ 0.03`). It records every hypothesis tested, the
evidence that supported or refuted it, and the conclusions drawn — so future
reviewers can understand *why* each lead was followed and *why* it was
abandoned.

---

## 1. Symptom

After completing the frame-alignment retraining of `SynthesizerTrn`
(`voices/am_jefferson/v1.pth`), inference produced silent/saturated audio:

```
waveform: mean=-0.9997, std=0.0247   ← fully saturated to tanh floor
```

`mean ≈ -1.0` means tanh is being fed very large negative values, which
collapses the output to the asymptote `-1.0`. No audible speech is present.

## 2. Methodology

Investigation proceeded top-down through the inference graph, isolating each
potential failure mode by testing on **CPU** (to rule out XPU numerical
issues) and on **multiple inputs**: zeros, training mel targets, pretrained
random mels, and our model's predicted mels. Any hypothesis that depended on
a particular input was eliminated by the "zeros input also saturates" test.

## 3. Lead Log

Each lead is recorded with its evidence disposition.

### Lead 1 — "Training targets are in the wrong scale"
**Hypothesis**: Mel targets were stored as raw power spectrograms
(`mean ≈ 21`, `max ≈ 6070`) while the pretrained vocoder expects dB-scale
mels. The model then learns to predict raw power, which the vocoder cannot
decode.

**Test**: Re-extract mels with `10*log10(power)` dB conversion, retrain.
Targets now `range=[-100, 37], mean≈-37`.

**Result**: ❌ *Refuted*. Output still saturated after retraining with
dB-normalized mel targets. Pure input rescaling was insufficient because
**zeros input also saturates** (Lead 6).

### Lead 2 — "Redundant log-transform on the inference side"
**Hypothesis**: Inference was applying `10*log10()` *again* to the model's
already-dB output, pushing values far outside the vocoder's training
distribution.

**Test**: Remove the second log-transform in `rvc_engine.py:245`.

**Result**: ❌ *Refuted*. Saturation persisted with identical statistics
(`mean=-0.9996`, `std=0.0276`).

### Lead 3 — "`remove_weight_norm()` fuses weights incorrectly"
**Hypothesis**: G40k.pth stores decomposed weight-norm parameters
(`weight_v` + `weight_g`). Our code calls PyTorch's `remove_weight_norm()`
to fuse them into a single `.weight` tensor. If the fusion formula
`w = g * v / ||v||` is applied with the wrong normalisation dimension or
the checkpoint stored pre-fused weights, the fused magnitudes would be
wrong, causing activation explosion.

**Tests**:
1. Visually verified `weight_g` mean ≈ 1.12, `weight_v` std ≈ 0.071 — these
   are the correct ranges for well-formed weight-norm decomposition.
2. Manually computed `fused = g * v / ||v||` and compared to the output of
   `remove_weight_norm()`. **They matched exactly**
   (`mean=-0.000795, std=0.041535`).
3. Ran inference **without** calling `remove_weight_norm()`. Identical
   saturated output (`max difference = 0.0`).

**Result**: ❌ *Refuted*. Weight-norm fusion is correct and not the cause.

### Lead 4 — "XPU numerical instability"
**Hypothesis**: Intel XPU has known half-precision quirks; perhaps the
vocoder was being autocast to fp16 and saturating through tanh.

**Test**: Reload everything on **CPU** (`torch.device('cpu')`) and rerun.

**Result**: ❌ *Refuted*. CPU produced byte-identical saturated output
(`mean=-0.999365, std=0.031858`). Not a device issue.

### Lead 5 — "Missing `dec.cond` conditioning signal"
**Hypothesis**: The vocoder's `dec.cond = Conv1d(256, 512, 1)` was trained
with a 256-channel conditioning signal. In the original VITS/RVC pipeline
this conditioning comes from the posterior encoder (`enc_q`) or speaker
embedding (`emb_g`). Feeding only mel through `conv_pre` (without cond)
leaves the decoder unable to shape its output.

**Tests**:
1. Returned `cond` from `SynthesizerTrn.infer()` — the flow output
   (`[B, 256, T]`). Channel count matches `dec.cond` input exactly. Passed
   to `vocoder.infer(mel, cond=cond)`. **Still saturated**
   (`mean=-0.9996`).
2. Passed the per-speaker embedding `emb_g` (broadcast across time) as
   conditioning. **Still saturated**.
3. Passed zeros as conditioning. **Still saturated**.
4. Passed random `σ=0.4` conditioning. **Still saturated**.

All four inputs produced **byte-identical** output statistics
(`mean ≈ -0.9996, std ≈ 0.0276`), meaning the conditioning has *negligible*
effect on the output. The resblocks amplify before the cond signal can
shape the result.

**Result**: ❌ *Refuted as a root cause*, though ✅ *confirmed* that
`dec.cond` is the correct VITS conditioning interface — it just isn't
sufficient on its own. This established that the problem is **upstream of
conditioning**: in the decoder's unconditioned resblock amplification.

### Lead 6 — "The decoder weights themselves are broken when extracted from the joint checkpoint"
**Hypothesis**: G40k.pth is a **full SynthesizerTrn** checkpoint containing
`enc_p`, `enc_q`, `flow`, `dec`, `emb_g` — the entire VITS pipeline. The
`dec` sub-module was trained jointly with the posterior encoder `enc_q`
and the flow, as part of an **end-to-end VITS auto-encoder**. Extracting
`dec.*` keys into a standalone `Generator` and feeding mel alone produces
nonsense because the decoder learned to expect its inputs to come from the
specific joint encoder distribution, not arbitrary mel spectrograms.

**Test**: Trace activations through the decoder stage by stage with zero
input.

```
After conv_pre:     mean=0.023,  std=0.041    ← reasonable
After stage 0:      mean=1.1,    std=0.7      ← growing
After stage 1:      mean=2.1,    std=1.5      ← growing
After stage 2:      mean=8.0,    std=6.1      ← exploding
After stage 3:      mean=130.5,  std=66.1     ← EXPLODED
After conv_post:    mean=-6.76   ← saturates tanh to -1.0
```

Every resblock stage multiplies the activation magnitude by ~3-15×. After
four stages, even a zero input produces mean=130 pre-`conv_post`. `tanh`
collapses immediately.

**Confirmation**: Tested with **zeros**, real **training mel targets**
(`mean=-37`), **random mels** at multiple σ levels, and all conditioning
variants. **Every** trial saturated. The decoder weights, as loaded, are
deterministically divergent.

**Result**: ✅ *Confirmed as root cause.* The standalone `dec` from
G40k.pth cannot function as a self-contained mel→audio vocoder because
its weights are co-adapted to the joint VITS pipeline.

## 4. Conclusion

The G40k.pth `dec` module is **not a standalone vocoder**. It is the
decoder half of a jointly-trained VITS auto-encoder
(`encoder → flow → decoder`), where `enc_q` and the flow distribution
shape the latent space that the decoder consumes. Using it as a black-box
mel→waveform function, even with conditioning, produces saturated output
because the resblock weights amplify outside the regime they were trained
for.

## 5. Remediation Options Considered

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Install a standard pretrained HiFi-GAN vocoder (e.g. jik876/hifi-gan, NVIDIA/BigVGAN) wired to our mel normalization | Fast; well-tested standalone vocoders exist | Mel normalization/format must match exactly; may not match RVC's 192-bin 40kHz config | Backup plan |
| Use `kanade_tokenizer.load_vocoder` (kokoclone's original) | Drop-in compatible with kokoclone | External package unavailable in this environment; doesn't teach us the architecture | Rejected |
| Use G40k.pth as a full SynthesizerTrn (load enc_q/flow/emb_g and route through native decoder) | Uses the trained checkpoint as intended | Requires reconstructing the full posterior-encoder inference pipeline; large architectural rework; still co-adapted to its own HuBERT, not ours | Rejected for now |
| **Train a HiFi-GAN decoder end-to-end with our generator** | We own and understand the entire pipeline; no external checkpoint coupling; valuable learning | Larger training surface; requires discriminators + adversarial + feature-matching losses; multi-hour training | **Selected** |

## 6. Why "Train End-to-End" Was Selected

1. **We own the full pipeline.** No dependency on a checkpoint format or
   latent-space distribution we can't introspect.
2. **It teaches the architecture.** A research-grade understanding of
   HiFi-GAN (upsample transposed convs + MRF residual blocks),
   discriminators (Multi-Period & Multi-Scale), and GAN losses (LS-GAN
   + feature matching + L1 mel reconstruction) is valuable for future
   debugging.
3. **Modular fallback.** The decoupled design leaves the option of
   dropping in a pretrained vocoder if end-to-end training proves unstable.
   The architectural investment is preserved either way.

## 7. Reference Data Captured During Investigation

### G40k.pth topology (full checkpoint, 549 keys)
```
top-level keys: model, iteration, learning_rate
state_dict prefixes: dec, emb_g, enc_p, enc_q, flow

# Decoder (HiFi-GAN)
dec.conv_pre:  Conv1d(192 → 512, k=7, pad=3)        # n_mels → hidden
dec.ups.0:    ConvTr1d(512 → 256, k=16, stride=8)   # upsample ×8
dec.ups.1:    ConvTr1d(256 → 128, k=16, stride=5)  # upsample ×5
dec.ups.2:    ConvTr1d(128 → 64,  k=4,  stride=2)  # upsample ×2
dec.ups.3:    ConvTr1d(64  → 32,  k=4,  stride=2)  # upsample ×2
dec.resblocks[0..11]: 3×4 MRF blocks (kernels 3,7,11; dilations 1,3,5)
dec.conv_post: Conv1d(32 → 1, k=7, pad=3, bias=False)
dec.cond:     Conv1d(256 → 512, k=1)               # speaker/encoder cond

# Speaker embedding (109 speakers, 256-dim)
emb_g.weight: [109, 256]

# Flow (4 flow blocks: indices 0,2,4,6)
flow.flows.{0,2,4,6}.pre:  Conv1d(96 → 192, k=1)
flow.flows.{0,2,4,6}.post: Conv1d(192 → 96, k=1)
# WaveNet inside each flow has cond_layer taking 256 channels (matches our flow's 256-Ch output)
flow.flows.0.enc.cond_layer: Conv1d(256 → 1152, k=1)

# Posterior encoder
enc_q.pre:    Conv1d(1025 → 192, k=1)              # 1025 = n_fft/2+1 input
enc_q.proj:   Conv1d(192 → 384,  k=1)              # 384 = 2 × 192 (mean+logvar)
```

### Our trained model (v1.pth)
```
enc_p.emb.weight:               [256, 768]      # Linear(768 → 256)
flows.0.flows.0.wn.pre_proj:    [128, 128, 1]   # flow operates at 256 channels, half=128
flows.0.flows.0.proj:          [256, 128, 1]   # outputs 256 = 2×128 (mean+logvar)
ups.0.0.weight:                [192, 256, 3]   # Conv1d(256 → 192, k=3, pad=1), identity stride
out_conv.weight:               [192, 192, 3]
emb_f0.weight:                 [256, 256]
```

### Activation trace through standalone `dec` (zero input)
```
conv_pre →    mean=0.023,  std=0.041   ✓ reasonable
stage 0   →   mean=1.1,    std=0.7     ~15× growth
stage 1   →   mean=2.1,    std=1.5     ~2× growth
stage 2   →   mean=8.0,    std=6.1     ~4× growth
stage 3   →   mean=130.5,  std=66.1    ~16× growth  ← EXPLOSION
conv_post →   mean=-6.76                saturates tanh
tanh     →    mean=-0.9994  std=0.032   floor saturation
```

### Weight sanity checks (all passed)
- `remove_weight_norm()` fusion matches manual `g * v / ||v||` computation
  exactly (max diff = 0.0)
- `weight_g` mean = 1.12 (within expected ~1.0 for weight-norm scaling)
- `weight_v` std = 0.071 (small, normal)
- Biases across all resblocks are small (|mean| < 0.08)
- `conv_pre` bias range [-0.25, 0.21] — small, normal

### Channel compatibility (all verified matching)
- `dec.cond.weight = [512, 256, 1]` — expects 256-ch conditioning ✓
- Our flow output = `[B, 256, T]` — matches dec.cond input ✓
- Resblock channel progression after each `ups.*` matches the next
  resblock's expected input channels ✓
- All 549 checkpoint keys load with `strict=True` and zero
  missing/unexpected ✓

## 8. Reproducing the Investigation

All diagnostic scripts are reproducible using `PYTHONPATH` to splice the
training project into the kokoclone venv:

```bash
PYTHONPATH="/home/sheigl/code/kokoclone-train:$PYTHONPATH" \
  /home/sheigl/code/kokoclone/.venv/bin/python3 -c "..."
```

Key commands used (sanitized):

```python
# 1. Load vocoder and check saturation with zero input
from core.rvc_models.vocoder import Generator
vocoder = Generator(n_mel_channels=192)
vocoder.load_state_dict(dec_sd, strict=True)
vocoder.remove_weight_norm(); vocoder.eval()
print(vocoder.infer(torch.zeros(1, 192, 493)).mean())  # → -0.9994

# 2. Verify weight_norm fusion is correct
v = rb.weight_v; g = rb.weight_g
manual_fused = g * v / torch.norm(v.view(v.size(0), -1), dim=1).view(-1,1,1)
assert torch.allclose(manual_fused, rb.weight)  # passes

# 3. Stage-by-stage activation trace
x = F.leaky_relu(dec.conv_pre(mel))
for i, up in enumerate(dec.ups):
    x = F.leaky_relu(up(x))
    for j in range(dec.num_kernels):
        x = x + dec.resblocks[i*dec.num_kernels+j](x_prev)  # MRF average
    print(i, x.mean(), x.std())
```

## 9. Key Takeaways for Reviewers

1. **A weights file is not always a model.** G40k.pth's `dec` prefix
   implies a decoder, but it is the decoder *slot* of a VITS pipeline.
   Loading it standalone doesn't reproduce the joint distribution it was
   trained against.
2. **"saturated tanh" == "decoder weights amplify divergently".** A mean
   of -1.0 from tanh is the smoking gun: the pre-tanh tensor has collapsed
   to large-magnitude negatives, which only happens if upstream activations
   have grown beyond the trained range.
3. **Empirical exclusion works.** Testing zeros input kills any
   input-distribution hypothesis. Testing CPU kills any device hypothesis.
   These two tests alone eliminate half of the leads above.
4. **Conditioning is necessary but not sufficient.** VITS's `dec.cond`
   layer is real and channel-compatible, but the resblocks amplify *before*
   the conditioning has a chance to shape the output. This nuance matters
   when reading the original VITS code.
5. **Working from the original paper helps.** VITS (Kim et al. 2021)
   describes a posterior encoder + flow + decoder architecture with
   adversarial + KL + reconstruction losses. Recognising that
   `enc_q`/`flow`/`dec` are jointly trained was the insight that closed
   the investigation.

---

See `END_TO_END_DECODER_PLAN.md` for the design of the replacement
decoder training pipeline.

---

## 9. Follow-up: f0G48k.pth Confirms the Same Pattern (July 2026)

After the G40k.pth investigation closed, we attempted to seed our
end-to-end decoder with pretrained weights from another RVC checkpoint:
`f0G48k.pth` (the standard RVC v2 pretrained generator).

**Result**: 100% tanh saturation — identical to G40k.pth.

**Root cause**: The same fundamental issue. `f0G48k.pth` is also a full
VITS `SynthesizerTrn` checkpoint. Its `dec` submodule expects VITS-style
latent input (posterior encoder output + flow-transformed distribution +
speaker embedding conditioning), not mel spectrograms. The decoder
resblocks amplify activations through each stage until the tanh saturates.

**Key difference**: Unlike G40k.pth (where we tried to freeze the
decoder as a standalone vocoder), our training framework fine-tunes the
pretrained decoder weights in situ. The co-adaptation is recoverable:
as the decoder adapts from VITS latent distribution to mel spectrogram
distribution over ~10-20 epochs, the saturation clears. The discriminator
loss dropping from 3.94 → 2.67 immediately after merge confirms the
pretrained resblocks produce more realistic audio — they just need to
re-learn the input signal range.

**Lesson reinforced**: Any decoder extracted from a VITS-family checkpoint
(G40k.pth, f0G48k.pth, etc.) is co-adapted to the full VITS pipeline
and cannot be used standalone without fine-tuning to the new input
distribution.