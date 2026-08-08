# Testing Context

> **Populated by the Document agent.** Run the whole suite with `pytest tests/`
> (or `uv run pytest tests/`).

## Status: 99/99 tests pass (August 6, 2026)

| File | Count | Scope |
|---|---|---|
| `tests/test_frame_alignment.py` | 10 | Model output ↔ mel target shapes, interpolation, per-chunk dataset |
| `tests/test_parallel_extraction.py` | 32 | Parallel F0/mel workers, hop_length, formulas, callbacks |
| `tests/test_detect_device.py` | 23 | Auto chain, explicit cuda/xpu/mps/cpu, indexed variants, error messages, `empty_cache()` |
| `tests/test_cli_train.py` | 34 | CLI parsing, `--device` flag parsing/precedence, `training_hp.json` records device |
| **Total** | **99** | |

## Device-selection coverage (Story 1)

`tests/test_detect_device.py` (new) covers:
- No-arg / `"auto"` regression (MPS → CUDA → XPU → CPU ordering).
- Explicit `cpu`, `cuda`, `cuda:N`, `xpu`, `xpu:N`, `mps`.
- Actionable `ValueError`s: CUDA unavailable (mentions ROCm +
  `ROCM_RX470_SETUP.md`), index out of range, XPU missing/unavailable, MPS
  unavailable, unknown device, bad index (`cuda:abc`).
- Case/whitespace normalization (`"  CUDA:0 "` → `cuda:0`).
- `empty_cache()`: cpu no-op, mps called, mps guarded (no `empty_cache` attr),
  cuda, xpu.

`tests/test_cli_train.py` adds: `--device` parsing on all three subcommands,
unset → `None`, `auto` accepted, CLI > config precedence for `train` and
`train-only`, `extract` passthrough, config honored when flag absent,
`get_device` fallback, `training_hp.json` records device.

## Story 2 (ROCm env) verification — non-pytest

Story 2 (`tools/setup_rocm_env.sh`, `tools/check_rocm.py`,
`docs/ROCM_RX470_SETUP.md`) adds **no pytest tests**. Acceptance was:

- Script syntax: `bash -n tools/setup_rocm_env.sh`; `py_compile
  tools/check_rocm.py`.
- Docs review: `docs/ROCM_RX470_SETUP.md` setup commands and the expected
  `check_rocm.py` output block match the script and the script's own printed
  output format.
- Real-GPU verification is **user-box manual**: CI cannot download ~2GB ROCm
  wheels or see the GPU. On the AMD box, run `bash tools/setup_rocm_env.sh`
  then `python tools/check_rocm.py` and confirm `cuda:0 -> AMD Radeon RX 470`
  and `detect_device('auto') -> cuda`.
- Full suite still 99/99 (`pytest tests/`).

## Conventions

- `tests/conftest.py` sorts frame_alignment tests to run first.
- Mock heavy deps (`torch`, `torchaudio`, ...) **inside test methods only** via
  `_mock_heavy_deps()` / `_unmock_heavy_deps()`.
- `test_detect_device.py` patches attributes on the already-imported
  `rvc_assets.torch` (module-level `import torch` cannot be sys.modules-mocked).
