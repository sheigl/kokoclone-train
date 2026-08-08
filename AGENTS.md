# KokoClone Train — Agent Context & Coding Standards

> ## ⚠️ READ THESE DOCS FIRST
>
> Before touching the vocoder pipeline, the inference engine, the training
> loop, or the loss recipe, **read the research log and design doc**. They
> capture decisions that would otherwise look arbitrary in the code alone.
>
> | File | Topic | Why read it |
> |---|---|---|
> | `docs/RESEARCH_GUIDE.md` | Plain-English explanation of every concept in the pipeline (mel frames, HuBERT, F0, HiFi-GAN, GANs, VITS, tanh saturation, etc.) | **Read this first if you're new to speech processing.** No prior knowledge assumed — explains the *why* behind every architectural decision with real numbers and diagrams. |
> | `docs/ARCHITECTURE.md` | Overall pipeline and tensor shapes | First-stop reference for the data flow. |
> | `docs/CHANGELOG.md` | Chronological change log | When did X change? |
> | `docs/VOCODER_INVESTIGATION.md` | Forensic log of why `G40k.pth`'s `dec` cannot be used as a standalone mel→audio vocoder | Read before any work that assumes G40k.pth is a usable vocoder. Documents every refuted hypothesis (mel scale, weight_norm fusion, XPU numerics, missing conditioning) AND the confirmed root cause (VITS joint-training), with reproduction steps. |
> | `docs/END_TO_END_DECODER_PLAN.md` | Design of the replacement: train a HiFi-GAN decoder end-to-end with our generator | Read before extending `SynthesizerTrn` with a `dec` submodule or touching the training loss/loop. Contains the full architecture, loss recipe, optimizer choices, rollback plan, and a 10-task implementation breakdown. |
> | `docs/ROCM_RX470_SETUP.md` | How to set up the separate ROCm 5.7 env (`.venv-rocm57`, torch 2.3.1) for full training on the AMD RX 470 (Polaris/gfx803) | Read before touching device selection or running training on the AMD card. Polaris is unsupported in ROCm 6.x/7.x — 5.7 is the last supported release; never `uv sync` in that env (pyproject maps torch → pytorch-xpu index). |
>
> The machine has multiple accelerators: Intel Arc B580s (XPU, primary env
> `.venv`) **and** an AMD RX 470 8GB (ROCm, separate env `.venv-rocm57`).
> Device selection is explicit via `--device` (see "Device Selection" below).

> **Architecture invariants are enforced at the bottom of this file
> (§ "Architecture Invariants").** Check them before any merge.
>
> The "Architecture Summary" below describes the **mel-only training
> state** (hop_length=300, `upsample_rates=[1]`).  That config produced
> `voices/am_jefferson/v1.pth`, which is now superseded — see
> `docs/END_TO_END_DECODER_PLAN.md` for the in-progress end-to-end
> decoder direction (hop_length=160, joint GAN training, embedded
> `HiFiGANDecoder`).  Do not mix the two configs.

## Project Overview

RVC v2 voice conversion training pipeline. Trains `SynthesizerTrn` generator models that convert HuBERT content features + F0 pitch into mel spectrograms for voice cloning.

### Architecture Summary (Post Frame Alignment Fix)

```
Audio (40kHz, 3.7s chunks)
    ├──→ 16kHz resample → HuBERT → content [T_c≈185, 768] @ ~50fps
    ├──→ pyworld.harvest → F0 [T_m≈493] @ ~133fps  
    └──→ torchaudio.MelSpectrogram → mel [T_m≈493, 192] @ ~133fps

Dataset __getitem__:
    content resampled: [185, 768] → F.interpolate → [493, 768] (matches F0/mel)
    
Model forward:
    input:   [B, T=493, C=768] + f0 [B, T=493]
    encoder: Linear(768→256) → Transformer × 6 layers → [B, 493, 256]
    flows:   ResidualCouplingBlock × 2 on [B, 256, 493]
    upsample: Conv1d(256→192, kernel=3, pad=1) → [B, 192, 493]
    output:  mel spectrogram [B, 192, T=493] ← matches target exactly
    
Loss: MSE(pred_mel, gt_mel) on aligned frames
```

### Key Frame Rates
- **HuBERT**: `facebook/hubert-base-ls960`, stride=320 at 16kHz → ~50 fps
- **F0/Mel**: pyworld/torchaudio at 40kHz, hop_length=300 → ~133.3 fps  
- **Ratio**: content→mel = 133.3/50 ≈ 2.67 (NOT 160)
- **upsample_rates**: `[1]` — identity; alignment done in dataset via interpolation

## Code Standards

### Backend / PyTorch

#### Tensor Shapes Convention
- Dataset outputs: `[B, T, C]` for content features, `[B, T]` for F0, `[B, T, n_mels]` for mel targets
- Model internal (flows/conv): `[B, C, T]` — transpose at boundaries
- Model output: `[B, spec_channels=192, T]`

#### Device Handling
```python
from kokoclone_train.core.rvc_assets import detect_device
device = detect_device()               # Auto: MPS → CUDA → XPU → CPU
device = detect_device("cuda:0")       # Explicit, validated
tensor.to(device)
```
`detect_device(requested=None)` accepts `"auto" | "cuda" | "cuda:N" | "xpu" |
"xpu:N" | "mps" | "cpu"` (case/whitespace normalized). `None`/`"auto"` keeps
the legacy chain exactly. Unavailable / out-of-range / unknown requests raise
an actionable `ValueError` (requested vs available vs fix; the `--device cuda`
message on a non-ROCm build points at `docs/ROCM_RX470_SETUP.md`).
`empty_cache(device)` dispatches `cuda`/`xpu`, guarded `mps`, `cpu` no-op.

Preference: resolve a device string once (CLI flag or config) and thread the
`torch.device` through the pipeline rather than calling `detect_device()`
per-stage. See "Device Selection" below for precedence.

#### Gradient Clipping
Always use `clip_grad_norm_` with max_norm=1.0:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### Optimizer Defaults
All optimizer/scheduler/loss parameters are loaded from `training_config.json`
(project root) + per-voice override (`voices/{name}/training_config.json`).
Use the helpers in `kokoclone_train.core.training_config`:
```python
from kokoclone_train.core.training_config import (
    get_lr_g, get_lr_d, get_betas, get_weight_decay,
    get_scheduler_gamma, get_lambda_mel, get_lambda_audio_l1, get_lambda_fm,
)
cfg = resolve_training_config(voice_name="am_jefferson", voices_dir=Path("voices"))
optimizer_g = torch.optim.AdamW(
    net_g.parameters(), lr=get_lr_g(cfg), weight_decay=get_weight_decay(cfg),
    betas=get_betas(cfg),
)
scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
    optimizer_g, gamma=get_scheduler_gamma(cfg),
)
```
Hardcoded defaults (when no config file exists): `lr=1e-4` (mel-only) or
`2e-4` (end-to-end), `betas=(0.8, 0.99)`, `gamma=0.998` (mel-only) or
`0.999` (e2e).

### Dataset Pattern (Post-Fix)

Per-chunk dataset with on-the-fly resampling:
```python
class ContentFeatureDataset(Dataset):
    def __init__(self, chunks: list[dict], seq_len: int = 512):
        self.chunks = chunks  # [{"content": path, "f0": path, "mel": path}, ...]
    
    def __getitem__(self, idx):
        content = np.load(self.chunks[idx]["content"])   # [T_c, 768] at ~50fps
        f0 = np.load(self.chunks[idx]["f0"])             # [T_m] at ~133fps
        
        # Resample content to mel frame rate
        if len(content) != len(f0):
            c_torch = torch.FloatTensor(content).T.unsqueeze(0)  # [1, 768, T_c]
            content = F.interpolate(c_torch, size=len(f0), mode='linear', 
                                    align_corners=False).squeeze(0).transpose(0, 1)
        
        return (content[:self.seq_len], f0[:self.seq_len], mel[:self.seq_len])
```

### Model Building Pattern
```python
from kokoclone_train.core.rvc_models import build_rvc_model

config = {
    "sample_rate": 40000,
    "hop_length": 300,
    "input_size": 768,
    "output_size": 192,
    "f0": True,
}
model = build_rvc_model(config).to(device)

# Load pretrained with strict=False — f0G48k.pth uses key "model", not "weight"
ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
sd = ckpt.get("weight", ckpt.get("model", ckpt))
# Pre-filter to skip shape mismatches (e.g. decoder upsample kernels)
model_sd = model.state_dict()
filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
missing, unexpected = model.load_state_dict(filtered, strict=False)
```

### Logging Pattern
```python
from kokoclone_train.core.rvc_train import _log
_log(callback, f"Epoch {epoch}/{epochs} — loss={avg_loss:.4f}", progress)
```

## Testing Standards

### Framework: pytest (via `pytest>=8.0.0`)

Run with: `pytest tests/` or `uv run pytest tests/`

### Test Patterns

#### Mocking Heavy Dependencies
Mock torch, torchaudio, librosa, faiss **inside test methods only** (not at module level) to avoid polluting the pytest session. Use helper functions with explicit cleanup:
```python
def _mock_heavy_deps():
    """Inject mocks for heavy ML dependencies into sys.modules."""
    from unittest.mock import MagicMock
    import sys

    torch_mock = MagicMock()
    torch_mock.__path__ = []
    sys.modules["torch"] = torch_mock

    for mod in ("torchaudio", "librosa", "faiss", "tqdm"):
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()


def _unmock_heavy_deps():
    """Remove mocked modules from sys.modules."""
    import sys
    to_remove = [k for k in sys.modules if k == "torch" or k.startswith("torch.")
                 or k in ("torchaudio", "librosa", "faiss", "tqdm")]
    for mod_name in to_remove:
        del sys.modules[mod_name]


# Usage inside test methods:
def test_something():
    _mock_heavy_deps()
    try:
        # ... test code that needs mocked deps ...
        pass
    finally:
        _unmock_heavy_deps()
```

#### Frame Alignment Test (Critical)
```python
def test_model_output_matches_mel_target_shape():
    """Model output temporal dimension must match mel target exactly."""
    from kokoclone_train.core.rvc_models.models import SynthesizerTrn
    
    model = SynthesizerTrn(
        spec_channels=192, input_size=768, upsample_rates=[1], f0=True,
    )
    
    B, T, C = 2, 493, 768
    content = torch.randn(B, T, C)
    f0 = torch.randint(1, 255, (B, T))
    
    output = model(content, f0=f0)
    
    assert output.shape == (B, 192, T), f"Expected [{B}, 192, {T}], got {output.shape}"
```

#### Content Resampling Test
```python
def test_content_resample_to_mel_frames():
    """Verify F.interpolate correctly resamples content features."""
    import torch
    from torch.nn import functional as F
    
    # Simulate HuBERT output at ~50fps for 3.7s audio
    t_content = 185
    t_mel = 493
    content = torch.randn(t_content, 768)
    
    # Resample to mel frame rate
    c_torch = content.T.unsqueeze(0)  # [1, 768, T_c]
    resampled = F.interpolate(c_torch, size=t_mel, mode='linear', 
                              align_corners=False).squeeze(0).transpose(0, 1)
    
    assert resampled.shape == (t_mel, 768), f"Expected ({t_mel}, 768), got {resampled.shape}"
```

#### Per-Chunk Dataset Test
```python
def test_per_chunk_dataset_shapes():
    """Dataset returns aligned content/F0/mel with matching temporal dimensions."""
    import tempfile, numpy as np
    from pathlib import Path
    
    # Create mock chunk files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Content at HuBERT frame rate (~50fps)
        content = np.random.randn(185, 768).astype(np.float32)
        np.save(tmpdir / "content.npy", content)
        
        # F0 at mel frame rate (~133fps)  
        f0 = np.random.randint(1, 255, size=493).astype(np.int64)
        np.save(tmpdir / "f0.npy", f0)
        
        # Mel targets at same frame rate as F0
        mel = np.random.randn(493, 192).astype(np.float32)
        np.save(tmpdir / "mel.npy", mel)
        
        chunks = [{"content": str(tmpdir / "content.npy"), 
                   "f0": str(tmpdir / "f0.npy"),
                   "mel": str(tmpdir / "mel.npy")}]
        
        dataset = ContentFeatureDataset(chunks=chunks, seq_len=512)
        sample = dataset[0]
        
        c, f, m = sample
        assert c.shape[0] == f.shape[0], f"Content/F0 length mismatch: {c.shape} vs {f.shape}"
        assert c.shape[0] == m.shape[0], f"Content/Mel length mismatch: {c.shape} vs {m.shape}"
        assert c.shape[1] == 768, f"Content dim should be 768, got {c.shape[1]}"
        assert m.shape[1] == 192, f"Mel dim should be 192, got {m.shape[1]}"
```

## Dependencies

- **Core**: torch>=2.10, torchaudio>=2.10, numpy>=1.26, scipy>=1.11, librosa>=0.10
- **ML**: transformers>=4.40 (HuBERT), faiss-cpu>=1.7 (indexing)
- **Audio**: pyworld>=0.3.4 (F0 extraction), soundfile>=0.13
- **Testing**: pytest>=8.0

## File Structure

```
kokoclone_train/
├── __main__.py          # CLI entry point
├── training_config.json # Project-level default training config
└── core/
    ├── rvc_models/
    │   ├── models.py     # SynthesizerTrn, Discriminators, build_rvc_model
    │   ├── attentions.py # Transformer Encoder (MultiHeadAttention, etc.)
    │   └── modules.py    # ResidualCouplingBlock, WaveNet flows
    ├── rvc_train.py      # Training pipeline: dataset, training loop, FAISS
    ├── rvc_assets.py     # Model downloads, device detection
    ├── hubert_manager.py # HuBERT feature extraction wrapper
    ├── training_config.py  # Per-voice config loader + HP helpers
    └── training_watcher.py  # Watch directory for new training jobs

tests/
├── conftest.py                    # Pytest config: sorts frame_alignment tests first
├── test_frame_alignment.py        # Frame alignment fix verification (10 tests, needs real torch)
├── test_parallel_extraction.py    # Parallel F0/mel extraction (32 tests)
├── test_detect_device.py          # detect_device override + validation (23 tests, mocked torch)
└── test_cli_train.py              # CLI arg parsing & integration (34 tests, mocks torch per-test)

tools/
├── setup_rocm_env.sh              # Create .venv-rocm57 (torch 2.3.1 + rocm5.7) for the RX 470
├── check_rocm.py                  # Runtime verification that the RX 470 shows up as cuda:0
├── merge_pretrained.py            # Merge pretrained decoder weights into checkpoints
└── build_xtts_aug.py              # Generate XTTS augmentation features
```

## Config Format (Saved per Voice)

```json
{
    "name": "voice_name",
    "version": 2,
    "input_size": 768,
    "output_size": 192,
    "sample_rate": 40000,
    "hop_length": 300,
    "f0": true,
    "upsample_rates": [1],
    "epoch": 100,
    "pitch_adjustment": 0.0
}
```

## Training Config (Per-Voice)

In addition to `config.json` (model architecture saved per voice), each voice
can have a `training_config.json` in `voices/{name}/` that controls training
hyperparameters.  Merge order: project defaults → voice-specific → CLI flags.
See `kokoclone_train/core/training_config.py` for the full API.

After training completes, `voices/{name}/training_hp.json` is saved with the
exact HPs used (for reproducibility), including the resolved `"device"`.

## Device Selection (August 6, 2026)

The pipeline supports an explicit device override for multi-accelerator boxes
(AMD RX 470 via ROCm + Intel Arc B580s via XPU):

- **CLI flag**: `--device auto|cuda[:N]|xpu[:N]|mps|cpu` on `train`,
  `train-only`, and `extract`. argparse default is `None` so an unset flag
  lets the merged config decide; effective default is `"auto"`.
- **Config key**: `"device"` in `training_config.json` (root default:
  `"auto"`) and per-voice `voices/{name}/training_config.json`.
- **Precedence**: CLI `--device` > voice config > root config > `"auto"`.
  `extract` is a plain CLI passthrough (no config merge).
- **Resolution**: `train_voice`/`run_train_only` call
  `get_device(training_config)` (`kokoclone_train.core.training_config`) to
  fill an absent flag; `_extract_features`/`_train_model` resolve the string
  via `detect_device(device)` once and thread the `torch.device` through
  (HuBERT, `.to(device)`, `map_location`, `pin_memory`).
- **Recorded**: the resolved device string is saved in `training_hp.json`.
- **ROCm note**: the RX 470 (Polaris/gfx803) only works through the ROCm
  build of torch, which exposes it as `cuda:0`. That build lives in the
  separate `.venv-rocm57` env — see "ROCm 5.7 environment" below.
- **Vulkan note (August 8, 2026)**: the RX 470 **cannot** run ROCm compute
  on this machine — its PCH PCIe slot lacks atomic ops, so KFD hard-rejects
  the card (`kfd: skipped device 1002:67df, PCI rejects atomics 730<0`); no
  software workaround exists (ROCm issues #26/#100/#1086). However the card
  works as a **Vulkan device via RADV**. A generic native Vulkan `PrivateUse1`
  PyTorch backend (`pytorch_vulkan`, device `"vulkan"`) is being built in
  `~/code/pytorch-vulkan` and will be wired into `detect_device` so training
  can run on the RX 470 via `--device vulkan`. Full plan + handoff:
  `~/code/pytorch-vulkan/docs/VULKAN_BACKEND_PLAN.md`.

## Known Architectural Decisions

1. **upsample_rates=[1]**: Content features are pre-resampled to mel frame rate in the dataset. Model upsampling is identity. This was changed from [10,4,2,2]=160 which produced 59x oversized outputs.
2. **Per-chunk dataset**: Each audio chunk (~3.7s) is a separate sample. No global concatenation to avoid cross-chunk alignment issues.
3. **Linear interpolation for content resampling**: `F.interpolate(..., mode='linear', align_corners=False)` maps HuBERT's ~50fps features to F0/mel's ~133fps. Standard practice in speech processing.
4. **Pretrained weight loading with strict=False**: Encoder and flow weights do NOT transfer from f0G48k.pth — our `enc_p` uses a different architecture (fused QKV, LayerNorm, FFN convs) vs the original RVC encoder (separate Q/K/V convs, relative position embeddings, no LN). The decoder resblocks DO transfer, but the upsample layers have different kernel sizes (24/20 in pretrained vs 16/16 in our config) and are skipped. Pre-filter state dict to matching shapes before calling `load_state_dict`.
5. **Upsample layer channel progression**: `ups_channels` always starts at `inter_channels=256` (flow output) and ends at `spec_channels=192`. With `n_ups=1`, there's a single Conv1d(256→192, kernel=3, stride=1, padding=1) — NOT a ConvTranspose1d which would shrink the temporal dimension by 1.
6. **Test isolation**: Both test suites run together (`pytest tests/`). `tests/conftest.py` sorts items so frame_alignment tests execute first (before any torch mocking). `test_cli_train.py` mocks torch only inside specific test methods, never at module level.
7. **Large MP3 segfault workaround**: `librosa.load()` on large MP3 files (>400MB) causes a segfault during decoding. Always use `--start` and `--end` to trim audio before loading: `kokoclone-train train --name foo --audio-dir bar --force --start 300 --end 2100`. Without trimming, the process crashes with exit code 139 (SIGSEGV) during preprocessing.

## New Flags (July 6, 2026 — Auralis Integration)

- **`--xtts-augment` CLI flag**: Added to `kokoclone-train train` subparser.
  When set, `_train_model()` looks for XTTS-generated features in
  `temp_dir/xtts_aug/{content,f0}/` and passes them to
  `ContentFeatureDataset` as `xtts_chunks`. The dataset then randomly swaps
  source features with XTTS features at `aug_ratio` probability (default 0.3),
  keeping the real mel/audio as targets. This teaches the model to map
  TTS-style inputs to the target speaker's voice.
- **`--aug-ratio` CLI flag**: Float (default 0.3) controlling the fraction
  of batches that use XTTS source features.
- **`train_voice()` and `_train_model()`** both accept `xtts_augment: bool`
  and `aug_ratio: float` parameters, wired through from CLI → `train_voice`
  → `_train_model` → `ContentFeatureDataset`.

## Auralis Package

- `/home/sheigl/code/auralis/` is a FastAPI server wrapping XTTS v2 + RVC
  voice conversion. Added `kokoclone` and `kokoclone-train` to its
  `[project.dependencies]` (they were only in `[tool.uv.sources]`, so uv
  never resolved them).
- All modules import cleanly. Lazy imports for kokoclone/kokoclone-train
  (inside function bodies) mean the server starts without either installed.
- XTTS data augmentation: `auralis/training/augment.py` generates an XTTS
  audio pool, extracts HuBERT content + F0 features, and stores them in
  `temp_dir/xtts_aug/{content,f0}/`. `build_xtts_augmentation()` is the
  single-call entry point.
- Training trigger: `auralis/training/train.py` runs `kokoclone-train` as a
  subprocess, passing `--xtts-augment` when augmentation is enabled.

## ROCm 5.7 environment (AMD RX 470)

Full training runs on the AMD RX 470 (Polaris/gfx803) happen in a **separate
venv** `.venv-rocm57` (torch 2.3.1 + torchaudio 2.3.1 from
`https://download.pytorch.org/whl/rocm5.7`), never mixed with the XPU `.venv`
(torch 2.10). Setup: `tools/setup_rocm_env.sh`; runtime check:
`tools/check_rocm.py`; guide: `docs/ROCM_RX470_SETUP.md`.

- **Why 5.7**: ROCm 6.x/7.x dropped Polaris/gfx803; 5.7 is the last release
  with first-class support (the TheRock community project has no gfx803
  roadmap either). `amdgpu` kernel driver already supports Polaris — no
  `amdgpu-dkms` rebuild needed.
- **Never `uv sync` in `.venv-rocm57`** — `[tool.uv.sources]` in pyproject
  maps torch/torchaudio to the pytorch-xpu index and would replace the ROCm
  build. The setup script passes `--no-project` to every `uv pip install` so
  the project's `[[tool.uv.index]]` (pytorch-xpu) is ignored and only the
  `--index-url …/rocm5.7` / PyPI are used; the package installs itself with
  `-e . --no-deps`. Deps are pinned for torch 2.3 (`numpy<2`,
  `transformers>=4.40,<4.48`, `huggingface-hub>=0.23,<1.0`, `scipy<1.14`).
- **Usage**: `source .venv-rocm57/bin/activate && python tools/check_rocm.py`
  (expect `cuda:0 -> AMD Radeon RX 470`), then
  `kokoclone-train train --name <voice> --audio-dir <dir> --device cuda:0 …`.
  In that env the auto chain also reaches the card (CUDA comes before XPU).
- **pyproject.toml is intentionally NOT modified** — the XPU build stays the
  primary environment; the ROCm env is not resolvable by `uv sync`.

## Recent Changes (August 2026)

- **Parallel F0/Mel extraction** (July 9): `_extract_f0_data` and `_extract_mel_targets` now use `concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count())` for near-linear speedup on multi-core machines. Worker functions `_extract_f0_single` and `_extract_mel_single` are module-level so multiprocessing can pickle them. HuBERT extraction remains sequential (GPU-bound).
- **Frame alignment fix**: Rewrote upsample channel progression to handle `upsample_rates=[1]` correctly. First upsample layer now accepts flow output channels (256) instead of spec_channels (192).
- **Conv1d for identity stride**: When `u=1`, use regular Conv1d(kernel=3, padding=1) instead of ConvTranspose1d to preserve temporal dimension exactly.
- **Test suite**: Added `tests/test_frame_alignment.py` with 10 tests covering defaults, model shapes, interpolation math, dataset behavior, and config format.
- **Test isolation fix**: Refactored `test_cli_train.py` to mock torch only inside test methods (not at module level). Both suites now run together via `pytest tests/` — conftest sorts frame_alignment first.
- **Vocoder investigation closed**: Loaded `G40k.pth`'s `dec` as a standalone HiFi-GAN vocoder; produces saturated `tanh` output (mean≈-1.0) for ANY input — including zeros, real mel targets, and zero conditioning. Root cause: G40k.pth is a full VITS SynthesizerTrn checkpoint whose decoder is co-adapted to its posterior encoder (`enc_q`) + flow distribution; cannot be extracted standalone. Investigation log with reproduction steps in `docs/VOCODER_INVESTIGATION.md`.
- **End-to-end decoder direction selected**: Will train a HiFi-GAN decoder jointly with our generator (adversarial + feature-matching + mel/audio L1 losses). Full design — architecture, losses, optimizers, training schedule, rollback plan, and 10-task breakdown — in `docs/END_TO_END_DECODER_PLAN.md`. Implementation in progress; existing `am_jefferson/v1.pth` is now superseded.

- **Pretrained f0G48k.pth download fix**: The public pretrained URL was dead (`liujing04/Retrieval-based-Voice-Conversion-WebUI` → 404). Repointed to `lj1995/VoiceConversionWebUI/pretrained_v2/f0G48k.pth`. Additionally, the state dict key loader used `ckpt.get("weight", ckpt)` but the checkpoint uses `"model"` — fixed to `ckpt.get("weight", ckpt.get("model", ckpt))`. Shape mismatches in `dec.ups.0/.1` (kernel size 24/20 vs 16/16) crash `load_state_dict` even with `strict=False` — fixed via pre-filter to matching shapes only.
- **229 decoder keys merged from f0G48k.pth**: Resblocks, conv_pre, conv_post, ups.2/ups.3 seeded via `tools/merge_pretrained.py`. Encoder+flow are architecturally incompatible (different `enc_p`, different flow channels) and remain from 200 epochs of training. Merged checkpoint shows 100% saturation initially — decoder needs to fine-tune from VITS latent input distribution to mel input distribution (~10-20 epochs).

- **XTTS v2 integrated for TTS source** (July 6): Replaced Kokoro with XTTS v2 (coqui-tts 0.27.5, idiap fork) as the TTS engine. Generates 24kHz audio from text + reference speaker wav. RVC converts at 40kHz. Quality is noticeably better than Kokoro but still limited by decoder training.

### August 6, 2026 — Device Selection (`--device`) + ROCm 5.7 env

- **Device selection shipped**: `detect_device(requested=None)` accepts
  `auto | cuda[:N] | xpu[:N] | mps | cpu` with actionable validation;
  `--device` flag on `train`/`train-only`/`extract`; `"device"` config key
  (CLI > voice > root > auto); device threaded through all stages and
  recorded in `training_hp.json`; `empty_cache` handles mps/cpu. New tests:
  `tests/test_detect_device.py` (23) + `--device` cases in `test_cli_train.py`
  (34 total). Suite: 99/99 green.
- **ROCm 5.7 env for the RX 470 shipped**: `tools/setup_rocm_env.sh` builds
  `.venv-rocm57` (torch 2.3.1 + rocm5.7 index, never `uv sync`), the package
  installs `-e . --no-deps`; `tools/check_rocm.py` verifies the card appears
  as `cuda:0`; full guide in `docs/ROCM_RX470_SETUP.md`. XPU `.venv` and
  `pyproject.toml` untouched.
- **Hyperparameter research** (July 6): Web search confirmed literature values. D LR was 5e-5 (should be 2e-4). LAMBDA_FM was 20 (should be 2). These are being corrected to make each epoch more effective.
- **Training horizon**: 2.5M steps is standard HiFi-GAN. We're at 158k (Jefferson). Decoder bottleneck is expected to improve with more epochs + corrected HPs.
- **XTTS data augmentation implemented** (July 6): Pre-generate synthetic training samples from TTS audio to teach decoder TTS-style inputs. CLI flags `--xtts-augment` and `--aug-ratio` added. `train_voice()`/`_train_model()` accept the parameters and wire through to `ContentFeatureDataset`.

## Architecture Invariants (do not regress)

1. **No `G40k.pth` as standalone vocoder.** Loading `dec.*` from
   `G40k.pth` into a standalone `Generator` produces saturated tanh
   output for *any* mel input, including zeros (verified; see
   `docs/VOCODER_INVESTIGATION.md`). If you find a code path that does
   this, it is a regression — do not "fix" it by passing conditioning
   (verified to not help).
2. **Inference owns its decoder.** Going forward the trained
   `SynthesizerTrn` carries its own `HiFiGANDecoder` as the `dec`
   submodule; there is no separate `_load_vocoder` step. The
   `rvc_models/vocoder.py` `Generator` class in the kokoclone repo is
   now legacy reference material only, not the inference vocoder.
3. **hop_length=160 when training with the HiFi-GAN decoder.** The
   decoder's `product(upsample_rates) = 160`, so `sr/hop = 40000/160 =
   250 fps`. The mel-only training (hop_length=300, `upsample_rates=[1]`)
   configs produced the now-superseded `am_jefferson/v1.pth` (still on
   disk but unusable end-to-end). Do not mix the two configs.

