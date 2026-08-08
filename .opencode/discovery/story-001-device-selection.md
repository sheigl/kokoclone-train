# Story: Device selection option in the training pipeline

## User Story
As a user with multiple accelerators (AMD RX 470 via ROCm, Intel Arc B580s via XPU),
I want to choose which device the training/extraction pipeline runs on
(`--device cuda:0` / `--device xpu:0` / `--device cpu`), so that I can point full
training runs at a specific GPU without editing code or relying on auto-detect order.

## Context

The pipeline auto-detects a device via `kokoclone_train.core.rvc_assets.detect_device()`
(MPS → CUDA → XPU → CPU) with no override. Auto-detect is order-fixed and cannot select
a specific accelerator in a multi-GPU box. The user has an AMD RX 470 (Polaris/gfx803)
usable only through the ROCm 5.7 build of torch (which exposes the GPU via the `torch.cuda`
API) alongside two Intel Arc B580s (exposed via `torch.xpu`). The two torch builds cannot
coexist in one environment; the ROCm build lives in a separate venv. In that venv,
`--device cuda:0` (or auto-detect) must select the RX 470; in the XPU venv,
`--device xpu:0` must select an Arc.

Current call sites (all auto-detect, no override):
- `kokoclone_train/core/rvc_train.py:627-629` — `_extract_features()` (HuBERT extraction) calls `detect_device()` and passes the result to `HuBERTManager.get(hubert_path, device=device)`.
- `kokoclone_train/core/rvc_train.py:890-892` — `_train_model()` calls `detect_device()`; resolved device is used for `map_location` and `.to(device)`.
- `kokoclone_train/core/hubert_manager.py:96-101` — `HuBERTManager.get()` falls back to `detect_device()` when no device is passed.
- `kokoclone_train/core/rvc_train.py:1009` — `pin_memory=device.type in ("cuda", "xpu")` (must stay).

CLI (`kokoclone_train/__main__.py`): `train` and `train-only` already merge a training
config via `resolve_training_config(voice_name, voices_dir, default_path, cli_overrides)`
(defaults → voices/{name}/training_config.json → CLI overrides). The `extract` subcommand
does NOT use a merged config today — it reads `use_decoder` straight from the voice config
file, so `--device` there is a plain CLI passthrough.

`_extract_features` is a sequential loop (no multiprocessing); the F0/mel extractors use
`ProcessPoolExecutor` but are CPU-only and do not take a device. So no pickling concern for
a device param. `DataLoader(num_workers=0)`.

## Acceptance Criteria

- [ ] `detect_device(requested: str | None = None) -> torch.device` accepts the values
      `"auto"`, `"cuda"`, `"cuda:N"`, `"xpu"`, `"xpu:N"`, `"mps"`, `"cpu"` (case/whitespace
      normalized). `None`/`"auto"` behaves exactly like today's `detect_device()`.
- [ ] Requesting a device that does not exist raises `ValueError` with an actionable message
      that states what was requested, what is actually available, and how to fix it (e.g. for
      `cuda:0` with no CUDA build: "install the ROCm build of torch to use the AMD GPU, see
      docs/ROCM_RX470_SETUP.md, or use --device auto"). Index out of range (e.g. `cuda:2` when
      `torch.cuda.device_count() == 1`) is also rejected with the count in the message.
- [ ] `"cpu"` always succeeds; `"mps"` succeeds only when `torch.backends.mps.is_available()`;
      `"xpu"/"xpu:N"` succeeds only when `hasattr(torch, "xpu")` and `torch.xpu.is_available()`
      (and index < `torch.xpu.device_count()` when that exists).
- [ ] `empty_cache()` gains handling for `mps` (guarded `torch.mps.empty_cache()`) and treats
      `cpu` as a no-op; `cuda`/`xpu` unchanged.
- [ ] `--device` CLI flag added to `train`, `train-only`, and `extract` subcommands with
      documented choices and effective default `"auto"`. Implementation note: argparse default
      should be `None` so that an unset flag lets the merged config decide; `get_device()` falls
      back to `"auto"`.
- [ ] `"device"` key added to the project-level `training_config.json` (`"device": "auto"`),
      supported as a per-voice override in `voices/{name}/training_config.json`, and overridden
      by the CLI flag. A `get_device(config) -> str` helper is added to `training_config.py`.
- [ ] Resolved device is threaded through: `train_voice(device=...)` → `_extract_features(..., device=...)`
      and `_train_model(..., device=...)`; `run_train_only(device=...)` → `_train_model`;
      `run_extract(device=...)` → `_extract_features`. All bare `detect_device()` calls at the
      three call sites are replaced by `detect_device(resolved)`. `HuBERTManager.get()` keeps
      its existing fallback (unchanged behavior for callers that pass nothing).
- [ ] Backward compatibility: `detect_device()` with no argument, and all existing public
      function signatures called with no `device` kwarg, behave exactly as before (device
      auto-detect). `pin_memory` logic stays `device.type in ("cuda", "xpu")`.
- [ ] Tests (mocked torch per the AGENTS.md `_mock_heavy_deps`/`_unmock_heavy_deps` pattern;
      no GPU-dependent e2e tests):
  - Unit tests for `detect_device` override + validation (auto/no-arg regression, cpu, each
    available branch, each unavailable branch raising with the right message, index-out-of-range).
  - `test_cli_train.py` updated: `--device` parses on `train`/`train-only`/`extract`; CLI value
    overrides config; config `"device"` is honored when the flag is absent; `"auto"` fallback.
    Prefer importing the real `_build_parser()` from `kokoclone_train.__main__` inside a mocked
    context (rather than duplicating the parser inline) — note the current tests reimplement the
    parser; the new tests should exercise the real one.
- [ ] The resolved device is recorded in the saved `training_hp.json` (reproducibility), matching
      the existing pattern of persisting the exact hyperparameters used.

## Technical Plan

**Plan file**: `plans/story-001-device-selection-plan.md`

The technical design, architecture decisions, and task breakdown for this story
are in the companion plan file linked above.

## Dependencies
- None (code-side change; story 002 (ROCm env) is complementary and independent)

## Priority: High

## Notes
- Auralis (separate repo at `/home/sheigl/code/auralis`) runs `kokoclone-train` as a subprocess
  from `auralis/training/train.py`. The plan must specify adding `--device {device}` to that
  subprocess command, but the primary deliverable is in kokoclone-train; the auralis change is
  documented (not required to be implemented by this story).
- The `extract` subcommand has no `--config`/merged-config plumbing today; do not add config
  merging for it in this story — plain CLI passthrough of `--device` to `run_extract` is enough.
- `training_config.json` default file and the `voices/am_jefferson/training_config.json`
  per-voice file both use a `_comment` key which `_deep_merge` already skips; add `"device"`
  alongside the other top-level keys.
- When the ROCm venv (story 002) is active, `torch.cuda.is_available()` is True and
  `--device cuda:0` / auto selects the RX 470. When the XPU venv is active, `--device xpu:0`
  selects an Arc. The two never coexist in one env (verified constraint).
