# Implementation State

> **Populated by the Document agent.** Mirrors `docs/CHANGELOG.md`; update after
> every completed feature so the next agent starts from current state.

## Latest completed feature: ROCm 5.7 env setup (AMD RX 470) — August 6, 2026

Guide + tooling for full training runs on the AMD RX 470 (Polaris/gfx803)
from a separate ROCm 5.7 environment. The XPU `.venv` and `pyproject.toml`
are untouched (XPU build stays primary).

- `tools/setup_rocm_env.sh` — creates `.venv-rocm57` (torch==2.3.1 +
  torchaudio==2.3.1 from `--index-url https://download.pytorch.org/whl/rocm5.7`;
  PyPI deps pinned for torch 2.3: `numpy<2`, `scipy>=1.11,<1.14`,
  `huggingface-hub>=0.23,<1.0`, `transformers>=4.40,<4.48`, …; package installed
  `-e . --no-deps`). Prefers `uv venv`+`uv pip install` (never `uv sync` — that
  would apply `[tool.uv.sources]` → pytorch-xpu index; every `uv pip install`
  passes `--no-project` so the project's `[[tool.uv.index]]` (pytorch-xpu) is
  ignored and only the rocm5.7 index / PyPI are used); falls back to
  `python3 -m venv`+pip; dies with a helpful message if neither uv nor python3.
  Idempotent: existing `.venv-rocm57` → prompt recreate-or-skip; never touches
  `.venv`. Prints next steps (activate, `python tools/check_rocm.py`, train with
  `--device cuda:0`).
- `tools/check_rocm.py` — lazy `import torch` inside `main()`; prints torch
  version, `torch.cuda.is_available()`, `device_count()`, device names (expects
  `AMD Radeon RX 470` at `cuda:0`), and `detect_device("auto")`; exit 1 with
  actionable hints on failure.
- `docs/ROCM_RX470_SETUP.md` — full guide (why 5.7, amdgpu loaded/no dkms,
  best-effort 24.04 userspace + HSA_OVERRIDE_GFX_VERSION fallback, two-env
  constraint, exact commands, verify output, `--device cuda:0` workflow, known
  limitations). Resolves the forward reference from story 001's `--device cuda`
  error message.
- README "Setup" gains a ROCm pointer bullet; `.gitignore` adds `.venv-rocm57/`.
- Verified: `bash -n`, `py_compile`, `pytest tests/` → 99 passed. Script NOT run
  (would download ~2GB wheels); check_rocm NOT run in the XPU env.

## Previous feature: device selection (`--device`) — August 6, 2026

- `kokoclone_train/core/rvc_assets.py`
  - `detect_device(requested=None)` — accepts `auto | cuda[:N] | xpu[:N] |
    mps | cpu` (case/whitespace normalized). No-arg/`"auto"` keeps the legacy
    MPS → CUDA → XPU → CPU chain. Unavailable/out-of-range/unknown requests
    raise actionable `ValueError` (mentions requested, available, and how to
    fix; `--device cuda` on a non-ROCm build points to `docs/ROCM_RX470_SETUP.md`,
    a **forward reference** delivered by another workstream).
  - `empty_cache(device)` — `cuda`/`xpu` dispatch, `mps` guarded via `hasattr`
    (safe on non-Apple builds), `cpu` no-op.
- `kokoclone_train/core/training_config.py` — new `get_device(config)` helper
  (falls back to `"auto"`).
- `training_config.json` (root) — new `"device": "auto"` key.
- `kokoclone_train/__main__.py` — `--device` flag on `train`, `train-only`,
  `extract` (default `None` → merged config decides).
- `kokoclone_train/core/rvc_train.py` — `device` threaded through
  `train_voice` / `run_train_only` (resolve `arg > training_config["device"] >
  "auto"`) and `run_extract` (CLI-only passthrough); `_extract_features` and
  `_train_model` call `detect_device(resolved)`; resolved device saved to
  `training_hp.json`.

## Recent prior state (July 2026)

- Parallel F0/mel extraction via `ProcessPoolExecutor` (July 9).
- XTTS v2 TTS engine + `--xtts-augment` / `--aug-ratio` flags (July 6).
- Pretrained f0G48k.pth download + state-dict key fixes (July 3).
- End-to-end HiFi-GAN decoder (`use_decoder`, adversarial loop) — active path.
- Frame alignment fix (`upsample_rates=[1]` mel-only era, now superseded).

## Known limitations / deferred

- Pretrained encoder/flow do not transfer from f0G48k.pth (architecture
  mismatch); decoder fine-tunes from VITS latent distribution (~10-20 epochs).
