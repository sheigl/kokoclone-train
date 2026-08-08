# Changelog

All notable changes to kokoclone-train will be documented in this file.

## [Unreleased]

### August 8, 2026 — `pytorch-vulkan` backend shipped: kokoclone-train trains on the RX 470 via Vulkan

The RX 470 (Polaris) cannot run ROCm compute (PCH PCIe slot lacks atomic ops;
KFD rejects the card). We built a **generic native Vulkan `PrivateUse1`
PyTorch backend** in `~/code/pytorch-vulkan` (package `pytorch_vulkan`,
device `"vulkan"`, explicit `pytorch_vulkan.init()`) and integrated it into
this project.

- `detect_device("vulkan")` + `_vulkan_available()` lazy probe +
  `_available_summary` + `empty_cache(vulkan)` in `rvc_assets.py`
- `--device vulkan` on train/train-only/extract
- `torch.load` uses `map_location="cpu"`; checkpoint saves cast state dicts
  to CPU (torch.save cannot `.cpu()` vulkan tensors via storage.set_)
- **Verified: full SynthesizerTrn (29.7M params + MPD/MSD) trains on the
  RX 470 via Vulkan** on the real am_jefferson dataset (forward+backward+Adam),
  checkpoint save/load works, fp16 pretrained checkpoint loads.
- Backend op coverage: GEMM/Linear, Conv1d/2d/Transpose1d/2d, LayerNorm,
  softmax, embedding, elementwise, losses, view ops — forward+backward with
  CPU-parity + gradcheck. 82 backend tests green.
- **Performance (honest): vulkan ~0.17x CPU on the full model** (806ms vs
  135ms/step) — correctness-first kernels aren't competitive with AVX-512
  on this 2016 card. The "faster than CPU" speed bar is NOT met; closing
  the gap needs a kernel-optimization pass (see the plan doc).
- Full context: `~/code/pytorch-vulkan/docs/VULKAN_BACKEND_PLAN.md`.

### August 8, 2026 — `pytorch-vulkan` backend (planned)

The RX 470 cannot run ROCm compute on this machine (PCH PCIe slot lacks
atomic ops; KFD rejects the card: `PCI rejects atomics 730<0`). The card
does work as a Vulkan device via RADV. Decision: build a **generic native
Vulkan `PrivateUse1` PyTorch backend** in a new repo
`~/code/pytorch-vulkan` (`pytorch_vulkan` package, device string `"vulkan"`,
explicit `pytorch_vulkan.init()`) and consume it from this project for
training on the RX 470 via `--device vulkan`.

- New repo scaffolded + docs written (`docs/VULKAN_BACKEND_PLAN.md`,
  `AGENTS.md`, `README.md`) — the plan doc doubles as a disconnected-session
  handoff with a "Resume here" section.
- Implementation in progress (T1: scaffold + device plumbing).
- Definition of done: generic transformer trains on `"vulkan"`; kokoclone-train
  trains `am_jefferson` via `--device vulkan` on the RX 470 (steps complete,
  checkpoint saves, sane losses); every op CPU-parity + `gradcheck`; step-time
  reported vs CPU and B580.
- Full context: `docs/VULKAN_BACKEND_PLAN.md` in the pytorch-vulkan repo.

### August 6, 2026 — setup_rocm_env.sh `--no-project` fix

- `tools/setup_rocm_env.sh` was failing at the torch install with "no solution
  found": `uv pip install` run inside the project reads `pyproject.toml`'s
  `[[tool.uv.index]]` (pytorch-xpu), and since that index comes first uv treats
  it as authoritative and never falls through to the `--index-url …/rocm5.7`
  passed on the command line. Fixed by passing `--no-project` to all three
  `uv pip install` commands so the project config is ignored and only the
  rocm5.7 index / PyPI are used. `docs/ROCM_RX470_SETUP.md` and `AGENTS.md`
  updated to match (the pip fallback branches are unchanged — pip does not read
  `[[tool.uv.index]]`).

### August 6, 2026 — ROCm 5.7 Environment Setup (AMD RX 470)

Guides and tooling for running full training jobs on the AMD RX 470
(Polaris/gfx803) from a separate ROCm 5.7 environment, without touching the
Intel XPU `.venv` or `pyproject.toml`.

- `tools/setup_rocm_env.sh` — creates `.venv-rocm57` (torch 2.3.1 +
  torchaudio 2.3.1 from `https://download.pytorch.org/whl/rocm5.7`, PyPI deps
  pinned for torch 2.3, package installed with `--no-deps`). Prefers `uv venv`
  + `uv pip install` (never `uv sync`), falls back to `python3 -m venv` + pip;
  idempotent (recreate-or-skip prompt); never touches `.venv`.
- `tools/check_rocm.py` — runtime verification: prints torch version,
  `torch.cuda.is_available()`, device count, device names (expects
  `AMD Radeon RX 470` at `cuda:0`), and `detect_device("auto")`; exits 1 with
  actionable hints on failure.
- `docs/ROCM_RX470_SETUP.md` — full guide: why ROCm 5.7 (Polaris dropped in
  6.x/7.x), amdgpu already loaded (no dkms rebuild), best-effort 5.7 userspace
  on Ubuntu 24.04, two-environments constraint, exact commands, verify +
  `--device cuda:0` workflow, known limitations.
- README gains a "ROCm / AMD RX 470" pointer under Setup; `.gitignore` ignores
  `.venv-rocm57/`.
- Deliberately no `pyproject.toml` change — the XPU build stays primary and the
  ROCm env is not resolvable by `uv sync`. `docs/ROCM_RX470_SETUP.md` is now
  the real document referenced by the `--device cuda` error message (was a
  forward reference from story 001).

### August 6, 2026 — Device Selection (`--device`) in the Training Pipeline

Users with multiple accelerators (AMD RX 470 via ROCm, Intel Arc B580s via
XPU) can now choose which device the pipeline runs on instead of relying on
the fixed auto-detect order (MPS → CUDA → XPU → CPU).

- `detect_device(requested=None)` in `rvc_assets.py` accepts `"auto"`,
  `"cuda"`, `"cuda:N"`, `"xpu"`, `"xpu:N"`, `"mps"`, `"cpu"` (case/whitespace
  normalized). `None`/`"auto"` behaves exactly as before; unavailable or
  out-of-range devices raise an actionable `ValueError` (e.g. pointing to
  `docs/ROCM_RX470_SETUP.md` for the AMD/ROCm case). `empty_cache()` now
  handles `mps` (guarded) and treats `cpu` as a no-op.
- New `--device` CLI flag on `train`, `train-only`, and `extract` (argparse
  default `None` so the merged config decides when unset; effective default
  `"auto"`). Threaded through `train_voice`, `run_train_only` (both resolve
  `arg > training_config["device"] > "auto"`) and `run_extract` (plain CLI
  passthrough).
- `training_config.json` gains `"device": "auto"` (overridable per voice);
  new `get_device(config)` helper in `training_config.py`. Precedence:
  CLI `--device` > voice config > root config > `"auto"`.
- The resolved device is recorded in the saved per-voice `training_hp.json`.
- New tests: `tests/test_detect_device.py` (23 tests covering the auto-detect
  chain, explicit `cuda`/`xpu`/`mps`/`cpu` requests, indexed variants, error
  messages, and `empty_cache()` behavior) and 12 new CLI tests in
  `tests/test_cli_train.py` for `--device` parsing/precedence. 99/99 tests pass.

> **Note**: The `--device cuda` error message references
> `docs/ROCM_RX470_SETUP.md` (AMD/ROCm setup guide). That document is a forward
> reference — it does not exist yet and is delivered by a separate workstream.
> **RESOLVED** — delivered in the "ROCm 5.7 Environment Setup" entry above.

### July 9, 2026 — Parallel F0 and Mel Extraction

**Problem**: F0 extraction (pyworld/harvest) and mel spectrogram extraction ran sequentially over WAV files. Each file is independent so the CPU was underutilized on multi-core machines.

**Solution**: Both `_extract_f0_data` and `_extract_mel_targets` now use `concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count())` to process all WAV files in parallel. Worker functions (`_extract_f0_single`, `_extract_mel_single`) are module-level so multiprocessing can pickle them.

#### Changed
- **`kokoclone_train/core/rvc_train.py`**:
  - Added `concurrent.futures` and `os` imports
  - Added `_extract_f0_single()` — module-level worker for single-file F0 extraction, returns `(stem, success, error_msg)` tuple
  - Added `_extract_mel_single()` — module-level worker for single-file mel extraction, same return pattern
  - Refactored `_extract_f0_data()` to use `ProcessPoolExecutor` with `as_completed` + `tqdm` progress bar
  - Refactored `_extract_mel_targets()` similarly
  - HuBERT extraction (`_extract_features`) remains sequential (GPU-bound, would serialize across processes)

### June 30, 2026 — Frame Alignment Fix (Hybrid A+C)

**Problem**: The original `upsample_rates=[10, 4, 2, 2]` (product=160) caused the model to produce outputs with a temporal dimension ~59× larger than the mel spectrogram targets, making MSE loss meaningless. HuBERT content features arrive at ~50fps while F0/mel frames are at ~133fps — a ratio of ~2.67, not 160.

**Solution**: Move frame alignment from the model upsampling layers to the dataset layer via linear interpolation. Model upsampling is now identity (`upsample_rates=[1]`).

#### Changed
- **`kokoclone_train/core/rvc_models/models.py`**:
  - `upsample_rates` default changed from `[10, 4, 2, 2]` to `[1]` (identity)
  - When stride `u=1`, use regular `Conv1d(kernel=3, padding=1)` instead of `ConvTranspose1d` — this preserves the temporal dimension exactly (no shrink or expand)
  - Upsample channel progression now starts at flow output channels (256) and ends at spec_channels (192). With `n_ups=1`, there's a single `Conv1d(256→192)` layer

- **`kokoclone_train/core/rvc_train.py`**:
  - Rewrote `ContentFeatureDataset` as a per-chunk dataset with on-the-fly content resampling
  - Content features are interpolated from HuBERT frame rate (~50fps) to F0/mel frame rate (~133fps) using `F.interpolate(..., mode='linear', align_corners=False)`
  - Restructured training loop for per-chunk loading (each ~3.7s audio chunk is a separate sample)

#### Fixed
- **`tests/test_cli_train.py`**: Moved torch mocking from module-level to per-test scoped `_mock_heavy_deps()`/`_unmock_heavy_deps()` helpers, preventing pytest session pollution and allowing both test suites to run together

#### Added
- **`tests/conftest.py`**: Simplified to only handle test ordering — frame_alignment tests execute first (before any torch mocking)
- Test suite now includes 10 frame alignment verification tests covering defaults, model shapes, interpolation math, dataset behavior, and config format

#### Why This Matters
Model outputs now exactly match mel target shapes: input `[B, T=493, C=768]` → output `[B, 192, T=493]`. MSE loss is computed on properly aligned frames, enabling meaningful training. Pretrained weights still load with `strict=False` (encoder/flow transfer; upsample layers are trivial identity convs).

### June 30, 2026 — End-to-End HiFi-GAN Decoder

**Problem**: The mel-only inference path relied on loading `G40k.pth`'s `dec.*` weights as a standalone mel→audio vocoder. This produced saturated `tanh` output (`mean=-0.9996`, `std≈0.03`) for *any* input, including zeros. Forensic investigation (see `docs/VOCODER_INVESTIGATION.md`) confirmed the root cause: G40k.pth is a full SynthesizerTrn/VITS checkpoint whose `dec` is jointly trained with its posterior encoder (`enc_q`), flow, and speaker embedding — extracting it as a standalone vocoder is not valid.

**Solution**: Train a HiFi-GAN decoder end-to-end as an embedded `dec` submodule of `SynthesizerTrn`. Full design rationale, architecture, loss recipe, and rollback plan in `docs/END_TO_END_DECODER_PLAN.md`.

#### Added
- **`kokoclone_train/core/rvc_models/vocoder.py`**: New module with `HiFiGANDecoder` and `ResBlock` (MRF residual block). Topology matches G40k.pth's `dec` layout for future compatibility: `upsample_rates=[8,5,2,2]` (product=160), `upsample_kernel_sizes=[16,16,4,4]`, 12 resblocks (3 kernels × 4 stages).
- **`kokoclone/core/rvc_models/vocoder_e2e.py`**: Mirror copy of the above for the kokoclone inference side. The legacy `vocoder.py` is preserved as reference material only.
- `use_decoder` parameter on `SynthesizerTrn.__init__` (training + inference side). When `True`, the model carries `self.dec = HiFiGANDecoder(...)` and `forward()` returns `(mel, audio)`; when `False` (legacy) it returns `(mel, cond)` for backward compatibility with the superseded `am_jefferson/v1.pth`.
- `use_decoder` flag on `train_voice()` (default `False`). When `True`, switches:
  - `hop_length = 160` (so mel frame rate = 250 fps matches decoder's `product(upsample_rates) = 160`)
  - Dataset returns 4-tuple `(content, f0, mel, audio)` with audio loaded from `0_gt_wavs/`
  - Training loop runs the adversarial G/D alternation
- Adversarial training loop in `_train_model` implementing HiFi-GAN V1 loss recipe:
  - Mel MSE ×45
  - Audio L1 ×1
  - LS-GAN adversarial (MPD + MSD, both sub-discriminators process real and fake)
  - Feature matching ×2 (L1 between per-layer features of real vs fake)
  - D step before G step (HiFi-GAN convention to keep D fresh)
  - `AdamW(lr=2e-4, betas=(0.8, 0.99), weight_decay=1e-5)`, `ExponentialLR(gamma=0.999)`
- New checkpoint format: `{"weight_g", "optimizer_g", "scheduler_g", "weight_d_mpd", "weight_d_msd", "optimizer_d", "scheduler_d", "epoch", "best_loss", "use_decoder", "hop_length"}`. Resume logic handles both new and legacy `{"weight": ...}` envelopes.
- `build_rvc_model_from_ckpt` (kokoclone side) infers `use_decoder` from both explicit config flag and presence of `dec.*` keys in state dict.
- `rvc_engine._infer` dispatches end-to-end vs legacy paths based on `net_g.use_decoder`.

#### Changed
- `DiscriminatorS` and `DiscriminatorP` activation changed from `torch.relu` to `F.leaky_relu(x, 0.1)` (HiFi-GAN paper spec).
- `DiscriminatorP.forward`: fixed bug where `T` was not updated after `F.pad(y, (0, pad))`, causing `y.view()` to compute the wrong shape for inputs whose length was not already a multiple of `period`.
- `ContentFeatureDataset.__getitem__` now optionally loads audio via `soundfile.read` and pads/truncates to `T_mel × audio_upsample` samples so the L1 + adversarial losses can compute elementwise differences.
- `_extract_f0_data` and `_extract_mel_targets` now take `hop_length` as a parameter (was hardcoded to 300).
- `SynthesizerTrn.forward` now returns `(mel, secondary)` where `secondary` is `audio` in end-to-end mode or `cond` in legacy mode. Previously returned a single tensor on training side and `(mel, cond)` on inference side — now both sides match.

#### Documentation
- **`docs/VOCODER_INVESTIGATION.md`** (312 lines): research-style forensic log of the G40k.pth saturation issue. 6 numbered leads each with hypothesis → test → evidence → refuted/confirmed disposition.
- **`docs/END_TO_END_DECODER_PLAN.md`** (363 lines): design document for the end-to-end replacement. Architecture, losses, optimizers, training schedule, rollback plan, honest risks, 10-task implementation breakdown.
- **`AGENTS.md`**: added "READ THESE DOCS FIRST" callout at top, "Research Log & Design Docs" table, and three Architecture Invariants (no G40k.pth standalone; inference owns its decoder; hop_length=160 for end-to-end).

#### Verification
- All 31 existing tests still pass (`pytest tests/`).
- Smoke test: 3 iterations of end-to-end training on CPU shows losses decreasing (G: 61→53→48, D: 8.1→7.0→6.1) and no NaN/Inf in model parameters.
- Smoke test: legacy `am_jefferson/v1.pth` still loads cleanly (0 missing/unexpected keys) and runs in mel-only mode — backward compat preserved.
- Smoke test: end-to-end inference with a fake checkpoint produces non-saturated audio (0% saturation, mean=0.04, std=0.02).

#### Why This Matters
Inference no longer depends on the G40k.pth vocoder, whose joint VITS training made it unusable standalone. The trained `SynthesizerTrn` now owns its entire `content + F0 → mel → audio` pipeline. The architectural investment is modular: if end-to-end training proves unstable in practice, the `HiFiGANDecoder` class and discriminators remain reusable for a future "install a standard pretrained HiFi-GAN" fallback (rollback plan in `END_TO_END_DECODER_PLAN.md` §9).

### July 2, 2026 — Training Bug Fixes & Hyperparameter Tuning

#### Fixed
- **oneDNN GPU conv JIT error**: Intel Arc B70 GPU cannot JIT-compile certain 2D conv backward shapes produced by MPD on variable-length batches. Fixed via `drop_last=True` in DataLoader + fixed-length padding (`seq_len=925`) with mel padded at -100 dB (silence floor).
- **Checkpoint cleanup sort bug**: `sorted()` on `epoch_*.pth` filenames uses lexicographic order, causing `epoch_100` to sort before `epoch_20` and get deleted first. Fixed via `_sort_checkpoints` helper (line 86-97 of `rvc_train.py`) that extracts the epoch number as an integer sort key.

#### Changed
- `LAMBDA_MEL` increased from 45 to 100 to push spectral accuracy over adversarial noise.
- Discriminator LR reduced from 2e-4 to 5e-5 to prevent D from overpowering G.
- Training extended from 100 to 300 epochs. Properly resumed from epoch 100 with reconstructed checkpoint.

### July 3, 2026 — Pretrained f0G48k.pth Download Fix

**Problem**: The pretrained `f0G48k.pth` was never loaded during training due to three cascading bugs:
1. The download URL pointed to a deleted HF repo (`liujing04/Retrieval-based-Voice-Conversion-WebUI` → 404)
2. The state dict key lookup used `ckpt.get("weight", ckpt)`, but the checkpoint stores weights under key `"model"` — so the entire checkpoint dict (keys: `['model', 'iteration', 'learning_rate']`) was passed to `load_state_dict`, matching zero parameters
3. Even after fixing (1) and (2), `load_state_dict` crashes on shape mismatches in `dec.ups.0.weight_v` and `dec.ups.1.weight_v` (kernel 24/20 in pretrained vs 16/16 in our config) even with `strict=False`

Every training run to date used 100% random init for both encoder+flow and decoder.

#### Changed
- **`kokoclone_train/core/rvc_assets.py:65`**: Download URL updated from dead `liujing04/Retrieval-based-Voice-Conversion-WebUI` to live `lj1995/VoiceConversionWebUI/pretrained_v2/f0G48k.pth`.
- **`kokoclone_train/core/rvc_train.py:801`**: State dict lookup changed from `ckpt.get("weight", ckpt)` to `ckpt.get("weight", ckpt.get("model", ckpt))` — tries `weight` first (legacy format), then `model` (current format), then falls back to raw dict.
- **`kokoclone_train/core/rvc_train.py:800-809`**: Pretrained init now pre-filters the state dict to only keys with matching shapes between model and checkpoint, skipping kernel size mismatches (decoder upsample layers). Logs loaded/skipped counts.

#### Added
- 229 decoder keys (resblocks, conv_pre, conv_post, ups.2, ups.3) from `f0G48k.pth` merged into epoch 200 checkpoint via `tools/merge_pretrained.py`.
- Training resumed from merged checkpoint at epoch 201; D loss dropped from 3.94→2.67 (better discrimination) as pretrained decoder produces more realistic audio.

#### Impact
Encoder+flow are still 100% random init (architecturally incompatible with f0G48k.pth — different `enc_p` structure, different flow channels). The pretrained decoder is co-adapted to VITS latent input, not mel spectrograms — initial inference shows 100% saturation, expected to clear as decoder fine-tunes over ~10-20 epochs.

### July 6, 2026 — XTTS v2 Integration & Decoder Bottleneck Analysis

**What**: Tested XTTS v2 (coqui-tts 0.27.5, idiap fork) as the TTS source engine for the TTS→RVC pipeline, replacing Kokoro.

**Installation**: Required `coqui-tts`, compatible `torchaudio` (downgraded from 2.11→2.8 to match torch 2.8 XPU), monkey-patch for transformers 5.x (`isin_mps_friendly` removed), and `unidic` dictionary download for Japanese phonemizer.

**XTTS v2 → RVC Jefferson test**:
- XTTS generates 24000Hz audio from text + reference speaker wav
- RVC converts to target voice via full pipeline (HuBERT → F0 → mel → decoder)
- Output: 9.5s @ 40000Hz, std=0.0897 (91% of GT 0.099), 0% saturation
- Quality: noticeably better than Kokoro→RVC (0.086), but still robotic/muffled

**XTTS v2 → RVC Welch test**:
- Output: 9.87s, std=0.0687 (69% of GT), 0% saturation
- Comparable to earlier Kokoro→Welch test (0.067)
- Worse than Jefferson due to fewer training epochs (289 vs 600)

**Root cause confirmed**: The decoder was trained exclusively on real audio mels. TTS-generated audio produces HuBERT features and mel spectrograms with different spectral statistics, causing the decoder to produce muffled output. The model converts real audio to 96% of GT std (Jefferson epoch 600), proving the encoder+flow are well-trained — the bottleneck is exclusively the decoder's distribution mismatch for TTS inputs.

**Literature research on hyperparameters** (web search, July 2026):
| Setting | Our value | Standard | Source |
|---|---|---|---|
| Generator LR | 2e-4 | 2e-4 ✓ | jik876/hifi-gan, coqui VITS |
| Discriminator LR | 5e-5 | 2e-4 | kan-bayashi, jik876 |
| Betas | (0.8, 0.99) | (0.5, 0.9) recommended | kan-bayashi/ParallelWaveGAN |
| Mel loss weight | 100 | 45 | jik876/hifi-gan |
| Audio L1 weight | 200 | N/A (different arch) | N/A — our mel+decoder is unique |
| Feature matching | 20 | 2.0 | jik876/hifi-gan |
| Scheduler | ExponentialLR(0.999) | MultiStepLR preferred | kan-bayashi |
| Training steps | ~158k (Jefferson) | 2,500,000 | jik876/hifi-gan |

**Key insight**: 2.5M steps is the standard HiFi-GAN training horizon. We're at 6% of that. The discriminator LR was reduced to 5e-5 (from 2e-4) in an earlier fix that may have been compensating for the too-high LAMBDA_FM=20. Fixing both to match literature should improve per-epoch decoder improvement rate.

**Decision**: Continue end-to-end training with corrected hyperparameters:
- Restore D LR to 2e-4 (match generator)
- Reduce LAMBDA_FM from 20 to 2 (match HiFi-GAN standard)
- Keep LAMBDA_AUDIO_L1=200 (primary waveform supervision)
- Keep LAMBDA_MEL=100 (internal mel prediction loss)

**XTTS data augmentation concept** (deferred): Pre-generate XTTS audio for all training transcripts, extract HuBERT+F0 from synthetic versions, and train with a mix of real-source and TTS-source samples. This would teach the decoder to handle TTS-style inputs directly. The real target mels are always used. Deferred until hyperparameter fix is evaluated first.
