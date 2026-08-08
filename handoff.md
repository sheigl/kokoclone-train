# KokoClone Train — Project Handoff

## Project Goal

Build an OpenAI-compatible TTS API server (Auralis) using XTTS v2 + RVC voice
conversion, with **paired XTTS data augmentation** pipelined into training so
the RVC decoder learns to handle TTS-style inputs.

## Architecture

```
audio (3.7s chunk)
  ├→ HuBERT → content [~185, 768] @50fps
  ├→ pyworld → F0 [~493, 1] @133fps (hop=300) or [~925, 1] @250fps (hop=160)
  └→ MelSpectrogram → mel [~493, 192] or [~925, 192]

SynthesizerTrn (our model):
  content + F0 → enc_p (Transformer×6) → flows → upsample → mel_pred
                                                               ↓
                                                      HiFiGANDecoder (dec)
                                                               ↓
                                                      audio_pred
```

Two modes (must not be mixed):
- **Mel-only** (`use_decoder=False`): hop_length=300, MSE(mel_pred, mel_gt)
- **End-to-end** (`use_decoder=True`): hop_length=160, adversarial + L1 + mel MSE

## Current Status (July 13, 2026)

### Running Now
- **Jefferson training** (GPU 0, B580, `train_jefferson.log`): Preprocessing 2×30min segments (segment_000 + segment_001, ~1058 chunks). Feature extraction in progress. No XTTS augmentation yet.
- **Welch training** (GPU 2, B580, `train_welch.log`): Preprocessing single 462MB WAV (~806 chunks). Feature extraction in progress. No XTTS augmentation yet.
- **`--force` flag removed** from all CLI subcommands to prevent accidental `training_temp/` destruction.

### Done (July 13)
- **`--force` flag permanently removed from CLI**: Kept accidentally clearing `training_temp/` one too many times. All 4 subcommands (`train`, `preprocess`, `extract`, `train-only`): removed argument definition, function parameter, and all force-related logic (rmtree blocks, `if not force` guards). To re-run a completed step, manually delete `training_temp/{name}/state.json` or the relevant subdirectory. Updated error messages to stop referencing `--force`.
- **Audio source reset**: Training restarted from scratch after `--force` cleared all checkpoints and features. Jefferson now uses only 2 segments (60 min total) instead of all 37 (18.4h). Welch uses its single WAV as before.
- **XTTS augmentation switched to built-in speakers**: Previously used target's own audio as XTTS clone reference. This produced near-identical speaker features (Welch got a perfect clone from its 12s reference). Now uses XTTS's 58 built-in LibriTTS voices (e.g. Claribel Dervla, Daisy Studious, Damien Black) — each sample picks a random voice via `speaker=` param. The VC model now learns to map foreign voices → target speaker.
- **`auralis/tts_engine.py`**: Added `get_builtin_speakers()` method + `speaker` parameter to `generate()`/`generate_to_file()`.
- **`auralis/training/augment.py`**: Added `builtin_speakers` list param to `generate_xtts_from_transcripts`, `generate_xtts_pool`, `build_xtts_augmentation`, and `_fallback_legacy_generation`. Random speaker selected per sample.
- **`tools/build_xtts_aug.py`**: Replaced `speaker_wav` fallback chain with `tts.get_builtin_speakers()`. No longer needs any reference wav file.
- **XTTS pools regenerated** (July 13 16:30-17:45): Both voices with 250 built-in-speaker samples each. (Since cleared by --force rmtree; needs regeneration.)

### Previously Done
- **XTTS augmentation now hard-fails**: `--xtts-augment` raises `ValueError` if XTTS feature dirs don't exist or are empty (was silently falling back to no augmentation).
- **`build_xtts_aug.py` added**: New tool in `tools/` to generate XTTS features.
- **`auralis/tts_engine.py` monkey-patch fixed**: `isin_mps_friendly` patch now applied before `from TTS.api import TTS` (was after, causing ImportError with transformers 5.x).
- **Unidic installed** for XTTS Japanese phonemizer dependency.
- **Checkpoint saving active**: `checkpoint_interval=50`. Both voices have epoch 50, 100, 150, 200, 250, ..., 500 checkpoints in `training_temp/{name}/`.
- **Transcription completed** for both voices: Jefferson 19,511/19,511 (8,756 new + 10,755 skipped), Welch 806/806 (224 new + 582 skipped)
- **Jefferson source audio corrected (July 10)**: Full 18.4h audiobook (37×30min segments) was used instead of intended ~30 min. Kept first ~45 min (segments 0–1, 794 chunks). Cleaned out segments 2–36.
- **Parallel F0/Mel extraction**: `concurrent.futures.ProcessPoolExecutor` with spawn context. 32 new tests, all 63 pass.
- **Parallel transcription**: `multiprocessing.Pool` (not ThreadPoolExecutor). `--concurrency` flag.
- **New CLI subcommands**: `preprocess`, `extract`, `train-only` — each with state tracking.
- **Import bug fixed**: `run_train_only` lazy import.
- **Preprocessing completed**: Welch 806 chunks split from 4.7GB WAV.
- **Deep research**: Validated ≤30% synthetic ratio, flow-based architecture requirement.
- **torchcodec==0.14.0+cpu installed**.
- **Jefferson audio split**: 37×30min WAV segments.

### Audio Source Files
- **Jefferson**: 2 segments from `/home/sheigl/code/kokoclone-train/training/am_jefferson/segment_000.wav` + `segment_001.wav` (60 min total, ~1058 chunks). Copied to `voices/am_jefferson/audio/`.
- **Welch**: `/home/sheigl/code/kokoclone-train/training/am_welch/am_welch.wav` (single long file, ~806 chunks). Copied to `voices/am_welch/audio/`.

## Decisions Made by User

The following questions were asked by the previous AI. The user's answers are
recorded below. **These are final decisions unless the user changes their mind.**

### 1. Whisper: Which model and deployment?
**AI asked:** `large-v3` local (CPU) or OpenAI API?
**User answered:** "Is there an API we can setup on another machine? I have a
server that has a much stronger CPU. Then we could make an API call to that
server."

**Decision:** Whisper `large-v3` deployed as a FastAPI server — currently
running on the same machine (localhost:8100) for convenience. Can be moved to
the remote machine by installing `whisper-server` package on the remote and
pointing the client to its IP.

**Current setup (July 9):** 3-upstream load-balanced architecture via nginx:
- **nginx proxy** listens on port 8101 with `least_conn` balancing
- **Upstream 1**: `127.0.0.1:7100` (local whisper-server, this machine)
- **Upstream 2**: `192.168.6.18:8101` (server.home)
- **Upstream 3**: `192.168.6.17:8101` (desktop2.home)
- All 3 run `large-v3` model on GPU
- Each server processes one request at a time (single GPU), so load balancing multiplies throughput linearly
- Config: `/etc/nginx/sites-available/unified-paths.conf`

### 2. Betas: Change to (0.5, 0.9) or keep (0.8, 0.99)?
**Decision: Keep `(0.8, 0.99)`.** Confirmed July 7.

### 3. Scheduler: Switch to MultiStepLR or keep ExponentialLR?
**Decision: Keep `ExponentialLR(gamma=0.999)`.** Confirmed July 7.

### 4. XTTS speaker reference: target speaker or neutral voice?

**Original decision (July 7, reversed July 13):** Use target speaker's own
voice as XTTS clone reference. Rationale: inference-time TTS → RVC pipeline
uses target as clone reference, so augmentation should match.

**Why reversed (July 13):** When the target's own audio was used as the XTTS
clone reference (especially with a clean 12s reference like Welch's
`am_welch.wav`), XTTS produced a near-perfect clone of the target speaker.
The HuBERT features extracted from that audio were nearly identical to real
target features — meaning the model was training on the same speaker
distribution it already sees, defeating the purpose of data augmentation.

**New decision (July 13):** Use XTTS's **58 built-in LibriTTS speaker
embeddings** (e.g. Claribel Dervla, Daisy Studious, Damien Black, Craig
Gutsy, Kazuhiko Atallah). Each sample picks a random voice via the
`speaker=` param, so the model learns to map diverse foreign voices →
target speaker. This matches the actual inference scenario where RVC
converts arbitrary speakers.

**Implementation:** `build_xtts_aug.py` calls `tts.get_builtin_speakers()`
and passes the list to `build_xtts_augmentation()`, which picks a random
speaker per sample. No reference wav needed. Both pools regenerated with
250 samples each (July 13).

### 5. Diversity Injection Parameters
**Decision:** Apply controlled diversity to XTTS-generated audio before feature
extraction to prevent TTS artifact memorization. Research confirmed that naive
augmentation degrades performance — flow-based models need stochastic diversity.

| Augmentation | Parameter | Probability |
|---|---|---|
| Pitch shift | ±2 semitones (phase vocoder) | 60% |
| Duration perturbation | ±10% (time-stretch) | 40% |
| Background noise | SNR 20-40dB (Gaussian) | 30% |

**Mixing ratio:** `--aug-ratio 0.3` (≤30% synthetic data max per research).

## Rollback Plan Summary

Full details in `docs/END_TO_END_DECODER_PLAN.md` §9.  Triggers:

| Condition | Threshold |
|---|---|
| Loss divergence | G loss increases >50 consecutive epochs after initial decrease |
| Saturation | \|audio_mean\| > 0.5 or audio_std < 0.01 for >30 epochs |
| Quality ceiling | After 500+ epochs, audio std still <70% of GT |
| NaN gradients | Gradient clipping can't prevent NaNs for >10 consecutive batches |
| Time budget | 2 weeks without meaningful improvement |

**Rollback path**: Switch to mel-only training (`use_decoder=False`,
`hop_length=300`) + standalone pretrained HiFi-GAN vocoder from
`jik876/hifi-gan` releases. The `_load_vocoder()` path in
kokoclone's `rvc_engine.py` already supports this.

**Preserved**: Decoder class, discriminators (MPD/MSD), audio loader,
encoder + flow weights.  **Discarded**: D optimizer state, joint G/D
checkpoint format.

## Quick Start — Training Commands

**IMPORTANT:** Run all training from a **terminal session**, not
through the bash tool. The tool's 120s timeout kills processes.

**IMPORTANT:** Intel GPU oneDNN JIT bug — `batch_size=1` crashes with
`internal error at jit.cpp:161` on backward convolution. Always use
`--batch-size 2` (or omit for config default = 2).

### Start Training (XTTS Augmented, End-to-End)

**⚠️ Always use `--resume`** when restarting — otherwise training starts from
Epoch 1 and ignores checkpoints.

```bash
# Fresh start (no prior state):
# Jefferson on GPU 0 (B580)
ONEAPI_DEVICE_SELECTOR="level_zero:0" nohup uv run kokoclone-train train \
  --name am_jefferson --audio-dir voices/am_jefferson/audio \
  --use-decoder --batch-size 2 --epochs 1000 > train_jefferson.log 2>&1 &

# Welch on GPU 2 (B580) — GPU 1 (Pro B70) unused
ONEAPI_DEVICE_SELECTOR="level_zero:2" nohup uv run kokoclone-train train \
  --name am_welch --audio-dir voices/am_welch/audio \
  --use-decoder --batch-size 2 --epochs 1000 > train_welch.log 2>&1 &
```

```bash
# Resume from checkpoint:
ONEAPI_DEVICE_SELECTOR="level_zero:0" nohup uv run kokoclone-train train \
  --name am_jefferson --audio-dir voices/am_jefferson/audio \
  --use-decoder --batch-size 2 --epochs 1000 --resume > train_jefferson.log 2>&1 &
```

### Pipeline Overview — Four Subcommands

The pipeline has 4 subcommands that can be run separately or together:

| Step | Command | What it does | Dependencies |
|------|---------|-------------|--------------|
| 1. Preprocess | `preprocess` | Splits source audio into 3.7s WAV chunks + 16kHz resamples, normalizes | Source audio in `training/<voice>/` |
| 2. Extract | `extract` | Extracts HuBERT content features, F0 (pyworld), mel spectrograms | Preprocessed chunks (step 1) |
| 3. Transcribe | `transcribe` | Runs Whisper on 16kHz chunks for transcripts | Preprocessed chunks (step 1) |
| 4. Train | `train-only` | Trains the model using extracted features | Extract + Transcribe (steps 2–3) |

**State tracking** (`training_temp/<voice>/state.json`): Each step checks whether prior
steps are completed and skips rework. Use `--resume` to continue from latest
checkpoint. To force re-run a step, manually delete `training_temp/{name}/state.json`.

### Run Full Pipeline (All Steps)

```bash
kokoclone-train train --name <voice> --audio-dir training/<voice> \
  --xtts-augment --aug-ratio 0.3 --batch-size 2
```

This runs preprocess → extract → transcribe → train sequentially.

### Run Steps Individually

```bash
# Step 1: Preprocess (split source audio into chunks)
kokoclone-train preprocess --name <voice> --audio-dir training/<voice>

# Step 2: Extract features (HuBERT on GPU, F0/mel parallel on CPU)
kokoclone-train extract --name <voice>

# Step 3: Transcribe chunks via Whisper (load-balanced, resume-safe)
kokoclone-train transcribe --name <voice> \
  --server-url http://localhost:8101/ --language en --concurrency 3

# Step 4: Train only (uses existing features, skips steps 1–3)
kokoclone-train train-only --name <voice> \
  --xtts-augment --aug-ratio 0.3 --batch-size 2
```

**Use `train-only` for restarts** — skips preprocess/extract/transcribe.
If training was interrupted, just re-run the same command; it auto-resumes
from the latest checkpoint.

### Transcribe

```bash
kokoclone-train transcribe --name <voice> \
  --server-url http://localhost:8101/ --language en --concurrency 3
```

### Whisper Server — Load-Balanced Architecture (Current)

3 servers behind an nginx proxy on port 8101. nginx config (`/etc/nginx/sites-available/unified-paths.conf`):

```nginx
upstream whisper {
    least_conn;
    server 127.0.0.1:7100;
    server 192.168.6.18:8101;  # server.home
    server 192.168.6.17:8101;  # desktop2.home
    keepalive_requests 1;
}

server {
    listen 8101 default_server;
    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;

    location /health {
        proxy_pass http://whisper/health;
    }
    location /transcribe {
        proxy_pass http://whisper/transcribe;
        client_max_body_size 100m;
    }
}
```

Use `least_conn` (not round-robin) so faster servers get more requests. Point client at nginx proxy:

```bash
kokoclone-train transcribe --name <voice> --server-url http://localhost:8101/ --concurrency 3
```

Set `--concurrency` to match number of upstream servers (3). Eeach server handles one request at a time (single GPU), so throughput scales linearly with server count.

### Start a Whisper Server on a New Machine

```bash
cd whisper_server/
uv pip install -e .
whisper-server --host 0.0.0.0 --port 8101  # use 8101 to match config
```

### Auralis Server — OpenAI-Compatible Inference

`auralis/server.py` provides an OpenAI-compatible `/v1/audio/speech` endpoint.
Once deps installed, start with:

```bash
cd ../auralis/
uvicorn auralis.server:app --host 0.0.0.0 --port 8000
```

## What Needs To Happen (Pending Work)

### Always use `--resume` on restart

The checkpoint interval saves to `training_temp/{name}/{name}_epoch_N.pth`,
but the training loop **does not auto-detect checkpoints** — you must pass
`--resume` to pick up from the last saved epoch:

```bash
# Correct — resumes from latest checkpoint
kokoclone-train train --name am_jefferson --audio-dir voices/am_jefferson/audio \
  --use-decoder --batch-size 2 --epochs 1000 --resume

# Wrong — starts from Epoch 1
kokoclone-train train --name am_jefferson --audio-dir voices/am_jefferson/audio \
  --use-decoder --batch-size 2 --epochs 1000
```

### 1. Let preprocessing + feature extraction complete

Both voices started from scratch at ~21:50 July 13. First time through:
- Preprocess (split audio into ~3.7s chunks): ~2-5 min
- HuBERT feature extraction (GPU): ~5-10 min
- F0 extraction (CPU, parallel): ~2-5 min
- Mel extraction (CPU, parallel): ~2-5 min
- **Training start**: ~22:15 July 13

Estimated ~1058 chunks (Jefferson, 2×30min) and ~806 chunks (Welch, single 462MB WAV).

### 2. Generate XTTS augmentation data (after training starts)

The old XTTS pools were cleared by --force. When training is running, regenerate:

```bash
# From kokoclone-train dir (or auralis dir — see build_xtts_aug.py docs)
cd ~/code/auralis
VIRTUAL_ENV=.venv uv run python tools/build_xtts_aug.py \
  --name am_jefferson --voices-dir ../kokoclone-train/voices \
  --num-samples 250 --transcripts-dir ../kokoclone-train/voices/am_jefferson/transcripts
```

Then stop training and restart with `--xtts-augment --resume`:
```bash
kill <PID>
ONEAPI_DEVICE_SELECTOR="level_zero:0" nohup uv run kokoclone-train train \
  --name am_jefferson --audio-dir voices/am_jefferson/audio \
  --use-decoder --batch-size 2 --epochs 1000 \
  --xtts-augment --aug-ratio 0.3 --resume > train_jefferson.log 2>&1 &
```

### 3. Monitor Training

Check logs:
```bash
tail -f train_jefferson.log
tail -f train_welch.log
```

### 4. Test first checkpoint

Checkpoints save every 50 epochs. First one at epoch 50 (~6h from training start). Evaluate:
```bash
PYTHONPATH="/home/sheigl/code/kokoclone-train:$PYTHONPATH" python tools/test_checkpoint.py \
  --ckpt training_temp/am_jefferson/am_jefferson_epoch_50.pth \
  --audio training_temp/am_jefferson/0_gt_wavs/0_0.wav
```

### 5. Adjust Training Config (if needed)

If quality plateaus, adjust HPs (lr, scheduler gamma, etc.) in `voices/{name}/training_config.json`.

## Research Findings — XTTS Augmentation for Voice Conversion

**Completed July 8.** Deep research across arXiv, GitHub (RVC-Project, Applio),
and Coqui XTTS papers. Key findings:

### Summary Table

| Question | Answer |
|---|---|
| Can TTS audio augment VC training? | Yes, but only with careful controls |
| What mixing ratio? | ≤30% synthetic data maximum |
| Self-cloning the target speaker? | Theoretically promising, practically risky without diversity controls |
| Best architecture for augmentation? | Flow-based models (not deterministic) — our model ✓ |
| Inference-time TTS→RVC post-processing? | **Avoid** — consistently degrades quality |
| Minimum real data needed? | 15–30+ minutes recommended — Jefferson: ~45 min ✓, Welch: ~45 min ✓ |

### Key Findings

1. **Naive augmentation degrades performance.** Simply generating more TTS data
   does not improve VC results. Synthetic data lacks stochastic diversity in
   phonetics, prosody, timbre, and environmental acoustics.

2. **Diversity injection is mandatory.** Without explicit controls (noise
   injection, pitch variation, duration perturbation), the model learns
   TTS-specific spectral biases rather than authentic vocal characteristics.
   Implemented: pitch ±2st (60%), duration ±10% (40%), noise SNR 20-40dB (30%).

3. **Flow-based architectures are essential.** Our `ResidualCouplingBlock × 2`
   is the correct choice — it learns underlying probability distributions
   rather than memorizing deterministic mappings.

4. **Self-cloning validated with caveats.** Using target speaker as XTTS clone
   reference expands phonetic coverage without cross-speaker timbral contamination,
   but requires strict mixing ratios (≤30%) and held-out validation on real audio.

5. **Community practices confirm ≤30% synthetic ratio.** RVC/Applio practitioners
   consistently report that exceeding 30% synthetic data causes overfitting and
   quality degradation. Our `--aug-ratio 0.3` aligns with this.

### Search Strategy Used
- arXiv: speech processing, TTS, voice conversion papers
- GitHub: RVC-Project, Applio, Fish Speech issues/discussions
- Coqui XTTS v2 paper and follow-up augmentation work

## Key Files

| File | Purpose |
|------|---------|
| **kokoclone-train/** (training pipeline — standalone) ||
| `kokoclone_train/core/rvc_train.py` | Training loop, dataset, HPs, LR fix, torchaudio bugfix |
| `kokoclone_train/core/training_config.py` | Per-voice config loader + helpers |
| `kokoclone_train/core/transcribe_client.py` | Whisper transcription client |
| `kokoclone_train/__main__.py` | CLI entry point (train, watch, transcribe) |
| `training_config.json` | Project-level default training config |
| `kokoclone_train/core/rvc_models/models.py` | SynthesizerTrn, discriminators |
| `kokoclone_train/core/hubert_manager.py` | HuBERT feature extraction |
| `kokoclone_train/__main__.py` | CLI entry point (train, watch, transcribe, preprocess, extract, train-only) |
| `train_jefferson.log` | Jefferson training log (current run) |
| `train_welch.log` | Welch training log (current run) |
| **whisper_server/** (FastAPI whisper server) ||
| `whisper_server/server.py` | FastAPI app, `/transcribe` endpoint |
| **/etc/nginx/** (reverse proxy) ||
| `/etc/nginx/sites-available/unified-paths.conf` | nginx config with `upstream whisper` block, 3 backends |
| **kokoclone/** (inference engine — used by auralis) ||
| `core/rvc_engine.py` | Inference (`_infer`, `_load_vocoder`) |
| **auralis/** (TTS server) ||
| `auralis/training/augment.py` | Transcript-based XTTS generation, diversity injection, feature extraction |
| `auralis/training/train.py` | Auralis training launcher (calls subprocess) |
| `auralis/tts_engine.py` | XTTS v2 wrapper |
| `auralis/server.py` | FastAPI server (OpenAI-compatible `/v1/audio/speech`) |
| **docs/** ||
| `docs/END_TO_END_DECODER_PLAN.md` | Full decoder design doc + rollback plan |
| `docs/VOCODER_INVESTIGATION.md` | Why G40k.pth can't be used standalone |
| `docs/CHANGELOG.md` | All changes with dates |
| `AGENTS.md` | Coding standards, architecture invariants |
| `handoff.md` | This file |

## Key Metrics

- **Jefferson dataset**: ~1058 chunks from 2×30min WAV segments (segment_000 + segment_001)
- **Welch dataset**: ~806 chunks from single 462MB WAV
- **Training time**: ~1.07s/batch (expected — from previous runs)
- **Batches/epoch**: ~529 (Jefferson, est.), ~403 (Welch)
- **Per‑epoch time**: ~9min (Jefferson), ~7min (Welch)
- **Current epoch**: Preprocessing in progress (fresh start July 13 ~21:50)
- **Best losses**: N/A (fresh start)
- **XTTS augmentation**: Not yet regenerated. 250 samples per voice using 58 built-in XTTS LibriTTS speakers (random per sample).
- **Checkpoints**: `training_temp/{name}/{name}_epoch_{N}.pth` (~114MB each, saved every 50 epochs). None yet.
- **Final model**: Written to `voices/{name}/v1.pth` on completion
- **Loss weights**: LAMBDA_MEL=100, LAMBDA_AUDIO_L1=200, LAMBDA_FM=2
- **Scheduler**: ExponentialLR(gamma=0.999)
- **Betas**: (0.8, 0.99)
- **Grad clip**: max_norm=1.0
- **G/D LR**: 2e-4 each
- **Whisper servers**: 3× large-v3 behind nginx proxy on port 8101
- **Nginx config**: `/etc/nginx/sites-available/unified-paths.conf` — upstream on 127.0.0.1:7100, 192.168.6.18:8101, 192.168.6.17:8101
- **GPU assignment**: GPU 0 (B580) = Jefferson, GPU 2 (B580) = Welch. GPU 1 (Pro B70, 12GB) unused.

## Known Bugs & Sharp Edges

1. **Intel GPU oneDNN JIT bug — batch_size=1 crashes**: `--batch-size 1`
   triggers `internal error at jit.cpp:161` in `onednn_verbose` during
   backward convolution primitive creation (convolution `jit:ir` backend).
   The JIT compiler has a bug with `mb1` special-cased backward convolutions
   at specific spatial dimensions (e.g., `ih1495oh499kh5sh3`). Workaround:
   always use `--batch-size 2` (or omit — config default = 2).

2. **torchcodec required for non-WAV audio**: torchaudio ≥2.10 uses
   torchcodec backend for MP3/FLAC. Pre-convert to WAV or install torchcodec
   (`torchcodec==0.14.0+cpu` installed in both `.venv`s).

3. **WAV 4GB limit**: Single WAV files >4GB produce broken output.
   Split large audio into <2GB segments with ffmpeg.

4. **Process killed on bash tool timeout**: Training preprocessing takes
   2-8 minutes. Launch from a terminal, not through the tool.

5. **Large MP3 segfault**: `librosa.load()` on >400MB MP3 segfaults.
   Use `--start`/`--end` to trim, or convert to WAV first.

6. **`--force` flag removed (July 13)**: Accidentally cleared `training_temp/` multiple times. To re-run a completed pipeline step, delete `training_temp/{name}/state.json` or the relevant subdirectory manually. There is no CLI flag for this anymore.

7. **`run_train_only` import missing** (Fixed July 10): Added lazy import
   `from kokoclone_train.core.rvc_train import run_train_only` in
   `__main__.py`. Without it, `train-only` subcommand crashes with
   `NameError: name 'run_train_only' is not defined`.

8. **Diversity injection required for XTTS augmentation**: Without controlled
   variation (noise, pitch shift, duration perturbation), the VC model will
   memorize TTS-specific artifacts. See §Research Findings.

## Transcribe CLI Usage

```bash
kokoclone-train transcribe \
  --name am_jefferson \
  --server-url http://localhost:8101/ \
  --language en \
  --concurrency 3
```

Behavior:
- Scans `training_temp/{name}/0_gt_wavs/*.wav`
- Skips chunks with existing `.txt` counterpart (resume-safe)
- Saves per-chunk `.txt` file with raw transcript text in both
  `training_temp/{name}/0_gt_wavs/` (consumed by training) and
  `voices/{name}/transcripts/` (permanent backup, survives `--force`)
- Writes combined `full_text.jsonl` in `training_temp/{name}/`
- Retries failed chunks up to 3 times with exponential backoff
- **Parallel requests**: Uses `multiprocessing.Pool` (not `ThreadPoolExecutor`) — each worker is a separate process to avoid GIL contention on `urllib` socket reads. Set `--concurrency` to match number of nginx upstream whisper servers (3 in current setup). Default = CPU count if not specified.

## Quick Test Command

```bash
cd ~/code/kokoclone-train/
uv run pytest tests/ -x -q
```

## Checkpoint Testing Tools

Located in `tools/` in the kokoclone-train repo:

| Tool | What it does | Usage |
|------|-------------|-------|
| `tools/test_checkpoint.py` | Full end-to-end: loads checkpoint, extracts HuBERT+F0 from a real audio file, runs inference, writes `test_epoch_N.wav`, prints saturation diagnostics | `--ckpt <path> --audio <path>` |
| `tools/verify_model.py` | Quick mel-only sanity: loads checkpoint, runs random input, checks shape/std/mean | `--name <voice>` or `--checkpoint <path>` |
| `tools/build_xtts_aug.py` | Generate XTTS augmentation pool from transcripts: runs XTTS v2 on each transcript, applies diversity injection, extracts HuBERT+F0 features | Run from `auralis/` dir via `VIRTUAL_ENV=.venv uv run ...` |

All tools import from `kokoclone_train.*` and work on CPU (no GPU conflict).

## Recent Fixes (July 12-13)

1. **`--force` flag permanently removed (July 13)**: Accidentally cleared `training_temp/` one too many times. Removed from all subcommands (`train`, `preprocess`, `extract`, `train-only`). All `shutil.rmtree` blocks, `if not force` guards, and docstring references removed. To re-run a completed step: `rm training_temp/{name}/state.json` (or delete the relevant subdirectory). To start fresh: `rm -rf training_temp/{name}`.
2. **XTTS augmentation switched to built-in speakers (July 13)**: Was using target's own reference wav → XTTS clone of target → features nearly identical to real training data. Now uses 58 built-in LibriTTS speakers (random per sample). Modified `tts_engine.py`, `augment.py`, `build_xtts_aug.py`.
3. **`auralis/tts_engine.py`** (July 12): Added `get_builtin_speakers()` method + `speaker` parameter to `generate()`/`generate_to_file()`.
4. **`build_xtts_aug.py`** (July 13): Replaced `speaker_wav` fallback chain with `tts.get_builtin_speakers()` — no longer needs any reference wav.
5. **`--xtts-augment` now hard-fails** (July 12): Raises `ValueError` if XTTS dirs missing. No more silent no-op.
6. **Monkey-patch order fix** (July 12): `isin_mps_friendly` patched before `from TTS.api import TTS`.
7. **Unidic dictionary installed** (July 12): For XTTS Japanese phonemizer.
8. **torchcodec installed** (July 12): For PyTorch ≥2.9 audio IO.
