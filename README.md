# KokoClone Train

RVC v2 voice conversion training pipeline. Trains `SynthesizerTrn` generator models that convert HuBERT content features + F0 pitch into mel spectrograms for voice cloning.

## Quick Start

### Prerequisites

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone and enter the project
cd kokoclone-train

# Install dependencies (CPU)
uv sync

# Or install with GPU support
uv sync --extra gpu

# Or install with Intel XPU support
uv sync --extra xpu
```

### ROCm / AMD RX 470

Have an AMD Polaris GPU (e.g. RX 470)? Use the separate ROCm 5.7 environment
(`.venv-rocm57`, torch 2.3.1) for full training runs on the AMD card while
keeping the XPU `.venv` intact: `bash tools/setup_rocm_env.sh`, then verify
with `python tools/check_rocm.py`. See
[`docs/ROCM_RX470_SETUP.md`](docs/ROCM_RX470_SETUP.md) for the full guide.

### Run Tests

```bash
uv run pytest tests/
```

### Train a Voice Model

```bash
kokoclone-train train \
  --name <voice_name> \
  --audio-dir <path_to_audio_chunks> \
  --epochs 100 \
  --batch-size 4 \
  --device auto          # auto | cuda[:N] | xpu[:N] | mps | cpu
```

`--device` is also available on the `train-only` and `extract` subcommands.
When unset, the merged `training_config.json` `"device"` key is used
(CLI > voice config > root config > `auto`). Run `kokoclone-train train
--help` for details.

See `kokoclone-train --help` for all available options.

## Key Features

- **Frame-aligned training**: Content features are interpolated to match F0/mel frame rates in the dataset layer, ensuring exact temporal alignment between model output and mel targets
- **Per-chunk processing**: Each audio chunk (~3.7s) is a separate sample — no cross-chunk alignment issues
- **Pretrained weight transfer**: Loads pretrained RVC v2 encoder/flow weights with `strict=False` for fast convergence
- **Multi-accelerator support**: Automatic device detection (MPS → CUDA → XPU → CPU) with an explicit `--device` override for `cuda[:N]`, `xpu[:N]`, `mps`, or `cpu`
- **FAISS indexing**: Built-in feature indexing for efficient nearest-neighbor search during inference

## Project Structure

```
kokoclone_train/
├── __main__.py          # CLI entry point (train, train-only, extract, watch, ...)
├── training_config.json # Project-level default training config
└── core/
    ├── rvc_models/
    │   ├── models.py     # SynthesizerTrn, Discriminators, build_rvc_model
    │   ├── attentions.py # Transformer Encoder layers
    │   └── modules.py    # ResidualCouplingBlock, WaveNet flows
    ├── rvc_train.py      # Training pipeline: dataset, training loop, FAISS
    ├── rvc_assets.py     # Model downloads, device detection (detect_device)
    ├── hubert_manager.py # HuBERT feature extraction wrapper
    ├── training_config.py  # Per-voice config loader + HP helpers
    └── training_watcher.py  # Watch directory for new training jobs

tests/
├── conftest.py               # Pytest config: test ordering
├── test_frame_alignment.py   # Frame alignment verification (10 tests)
├── test_parallel_extraction.py  # Parallel F0/mel extraction (32 tests)
├── test_detect_device.py     # Device detection/validation (23 tests)
└── test_cli_train.py         # CLI arg parsing & integration (34 tests)
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >= 2.10 | Deep learning framework |
| torchaudio | >= 2.10 | Audio I/O and mel spectrograms |
| transformers | >= 4.40 | HuBERT feature extraction |
| faiss-cpu | >= 1.7 | Feature indexing |
| librosa | >= 0.10 | Audio processing utilities |
| pyworld | >= 0.3.4 | F0 pitch extraction |
| numpy | >= 1.26 | Numerical operations |
| scipy | >= 1.11 | Signal processing |

## License

See project repository for license details.
