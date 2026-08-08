# Project Context

> **Populated by the Document agent.** Keep in sync with `README.md` and `docs/`.

## What this is

**kokoclone-train** is an RVC v2 (Retrieval-based Voice Conversion) training
pipeline. It trains `SynthesizerTrn` generator models that convert HuBERT
content features + F0 pitch into mel spectrograms — and, in end-to-end mode,
raw audio via an embedded HiFi-GAN decoder (`use_decoder=True`).

## Repo layout (top level)

- `kokoclone_train/` — the Python package (CLI in `__main__.py`, core in `core/`)
- `docs/` — design + research documentation (see README table in `AGENTS.md`)
- `tests/` — pytest suite (99 tests)
- `training_config.json` — project-level default training config
- `voices/` — trained voices (config, checkpoints, `training_hp.json`)
- `tools/merge_pretrained.py` — checkpoint merge helper
- `tools/setup_rocm_env.sh` + `tools/check_rocm.py` — ROCm 5.7 env setup/verify (AMD RX 470)
- `training_temp/` — staging area for preprocessing/feature extraction

## Current direction (July–August 2026)

- **End-to-end decoder training is the active path** (hop_length=160,
  `upsample_rates=[8,5,4,2]`, joint GAN training, embedded `HiFiGANDecoder`).
  The older mel-only config (hop_length=300, `upsample_rates=[1]`) produced
  `voices/am_jefferson/v1.pth`, which is superseded. Do not mix the two configs.
- **Device selection (`--device`)** shipped August 6, 2026 — see
  `docs/CHANGELOG.md` and the "Device Handling" section of `docs/ARCHITECTURE.md`.
- **ROCm 5.7 env for the AMD RX 470** shipped August 6, 2026 — a separate
  `.venv-rocm57` (torch 2.3.1) sibling of the XPU `.venv` (torch 2.10); never
  mixed. See `docs/ROCM_RX470_SETUP.md` and the ROCm section of
  `.opencode/context/architecture.md`.

## Key docs to read first

| File | Topic |
|---|---|
| `docs/RESEARCH_GUIDE.md` | Plain-English explanation of the whole pipeline |
| `docs/ARCHITECTURE.md` | Data flow, tensor shapes, design decisions |
| `docs/END_TO_END_DECODER_PLAN.md` | The active training direction |
| `docs/VOCODER_INVESTIGATION.md` | Why G40k.pth cannot be a standalone vocoder |
| `docs/ROCM_RX470_SETUP.md` | ROCm 5.7 env for full training on the AMD RX 470 |
