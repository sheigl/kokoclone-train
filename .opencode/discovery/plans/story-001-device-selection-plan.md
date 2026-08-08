# Technical Plan: Device selection option in the training pipeline

**Story**: `story-001-device-selection.md`
**Status**: Ready for implementation
**Priority**: High

## 1. Architecture & Design Decisions

### 1.1 `detect_device(requested: str | None = None)` in `rvc_assets.py`

Signature change (backward-compatible — no-arg call behaves exactly as today):

```python
_VALID_DEVICES = ("auto", "cuda", "xpu", "mps", "cpu")  # "cuda:N"/"xpu:N" parsed separately

def detect_device(requested: str | None = None) -> torch.device:
    """Detect or validate the compute device.

    requested: "auto" (default) | "cuda" | "cuda:N" | "xpu" | "xpu:N" | "mps" | "cpu".
    None/"auto" → MPS → CUDA → XPU → CPU (current behavior).
    Raises ValueError with an actionable message for unavailable/invalid requests.
    """
    req = (requested or "auto").strip().lower()

    if req in ("auto", None, ""):
        return _auto_detect()

    # Parse optional ":N" index suffix
    base, _, idx_str = req.partition(":")
    idx = None
    if idx_str:
        if not idx_str.isdigit():
            raise ValueError(...)   # bad index
        idx = int(idx_str)

    if base == "cpu":
        return torch.device("cpu")

    if base == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(
                "Requested device 'mps' but this torch build has no MPS backend "
                "(MPS is Apple-Silicon only). Available: {_available_summary()}. "
                "Use --device auto or --device cpu."
            )
        return torch.device("mps")

    if base == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "Requested device 'cuda:{idx}' but no CUDA device is available in this "
                "torch build. Available: {_available_summary()}. If you intend to use an "
                "AMD GPU (e.g. RX 470), install the ROCm build of torch — see "
                "docs/ROCM_RX470_SETUP.md — or use --device auto."
            )
        n = torch.cuda.device_count()
        if idx is not None and idx >= n:
            raise ValueError(f"Requested 'cuda:{idx}' but this machine has {n} CUDA device(s).")
        return torch.device(f"cuda:{idx}" if idx is not None else "cuda")

    if base == "xpu":
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise ValueError(
                "Requested device 'xpu' but no Intel XPU device is available in this torch "
                "build. Available: {_available_summary()}. Use the XPU build of torch "
                "(pytorch-xpu index) or use --device auto."
            )
        n = torch.xpu.device_count() if hasattr(torch.xpu, "device_count") else 1
        if idx is not None and idx >= n:
            raise ValueError(f"Requested 'xpu:{idx}' but this machine has {n} XPU device(s).")
        return torch.device(f"xpu:{idx}" if idx is not None else "xpu")

    raise ValueError(
        f"Unknown device '{requested}'. Valid values: auto, cuda[:N], xpu[:N], mps, cpu. "
        f"Available now: {_available_summary()}."
    )

def _auto_detect() -> torch.device:
    """Current MPS → CUDA → XPU → CPU chain (extracted, unchanged semantics)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")

def _available_summary() -> str:
    """Human-readable list of currently available devices for error messages."""
    parts = []
    if torch.backends.mps.is_available():
        parts.append("mps")
    if torch.cuda.is_available():
        parts.append(f"cuda:{torch.cuda.device_count()} device(s)")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        parts.append("xpu")
    parts.append("cpu")
    return ", ".join(parts) if parts else "none"
```

**Design decisions:**

- **`None`/`"auto"` → exact current behavior.** The original chain is extracted verbatim into `_auto_detect()` so the no-arg path is byte-for-byte identical in behavior (MPS → CUDA → XPU → CPU).
- **`"cuda"` without index returns `torch.device("cuda")`** (torch resolves to the current device, normally `cuda:0`). `"cuda:N"` returns an explicit indexed device. Same for xpu.
- **Validation order**: index-format check happens before availability checks so garbage like `cuda:abc` produces a clear message rather than a cryptic one.
- **Error messages are actionable**: they state (a) what was requested, (b) what is actually available, (c) the fix (point to `docs/ROCM_RX470_SETUP.md` for the AMD/ROCm case, or the XPU index for the Intel case), (d) the `--device auto` escape hatch.
- **No exceptions swallowed**: a `ValueError` at CLI level surfaces as a clean `Error: ...` exit (the `train`/`extract`/`train-only` handlers already wrap calls in `try/except Exception` and print `Error: {e}`, returning 1).

### 1.2 `empty_cache` extension

```python
def empty_cache(device: torch.device) -> None:
    """Clear framework-specific cache for the given device."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "mps" and hasattr(torch.backends, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    # cpu: no-op
```

- `torch.mps.empty_cache()` exists on macOS builds (torch>=1.13); guard with `hasattr` so other builds never crash.
- `cpu` remains a no-op. Unknown device types remain a silent no-op (defensive, unchanged behavior).

### 1.3 CLI flag: `--device` on `train`, `train-only`, `extract`

- **argparse `default=None`, not `"auto"`.** This is deliberate: `resolve_training_config()` filters `None` values out of `cli_overrides` (`{k: v for k, v in cli_overrides.items() if v is not None}`), so an unset flag lets the merged config decide; `get_device()` falls back to `"auto"`. The *effective* default is `"auto"` (documented in help text). If argparse default were `"auto"`, a user who set `"device"` in `training_config.json` would be silently overridden by the flag's default — violating the CLI-wins-unless-set rule used by every other flag.
- **Help text** (same on all three subcommands):

```
--device {auto,cuda,cpu,xpu,mps}
                        Device to run on: auto (default, MPS→CUDA→XPU→CPU),
                        cuda[:N] (NVIDIA/AMD via ROCm), xpu[:N] (Intel Arc),
                        mps, cpu.
```

Use `choices=None` (free-form string) so `cuda:0` and `xpu:1` pass through argparse; validation happens in `detect_device()`, which gives better error messages than argparse choices would.

- **`train` / `train-only`**: add `"device": args.device` to the existing `cli_overrides` dict, then pass `device=args.device` (None when unset) to `train_voice` / `run_train_only`. The merge chain handles precedence: CLI (not None) > voice config > root config > `"auto"`.
- **`extract`**: no merged-config plumbing exists (by design — see story notes). Add the flag and pass `device=args.device` straight to `run_extract(..., device=args.device)`.

### 1.4 Config plumbing

- **`training_config.py`**: add

```python
def get_device(config: dict) -> str:
    return str(config.get("device", "auto"))
```

- **Root `training_config.json`**: add `"device": "auto",` as a top-level key (next to `epochs`, `use_decoder`, etc.).
- **`voices/am_jefferson/training_config.json`**: optionally add `"device": "auto"` with a comment (not required — default falls through). Leave untouched to prove fallback works; the Implement agent may add it for documentation value.
- **`training_hp.json`**: after training completes, the existing block persists resolved HPs (`get_lr_g`, `get_lr_d`, ...). Add `"device": str(resolved_device)` to that dict so each run records exactly which device it trained on.

### 1.5 Function signature changes in `rvc_train.py`

All additions use `device: str | None = None` (a *string override*, resolved inside each function via `detect_device(device)`). Rationale: keeps the public API string-based (matches the CLI/config string values), and each function remains self-contained with the bare `detect_device()` call simply becoming `detect_device(device)` — a one-line change per call site.

- `_extract_features(wavs16k_dir, feature_dir, hubert_path, callback, device: str | None = None)` — line 627-629 becomes `device = detect_device(device)`; `HuBERTManager.get(hubert_path, device=device)` unchanged.
- `_train_model(..., device: str | None = None)` — line 890-892 becomes `device = detect_device(device)`. All downstream uses (`map_location`, `.to(device)`, `pin_memory`) inherit the resolved `torch.device` automatically. **`pin_memory` stays `device.type in ("cuda", "xpu")`** — unchanged.
- `train_voice(..., device: str | None = None)` — resolve once in the config-resolution block:
  ```python
  resolved_device = device if device is not None else get_device(training_config or {})
  ```
  Pass `device=resolved_device` to `_extract_features(...)` (line 389) and `_train_model(...)` (line 437).
- `run_train_only(..., device: str | None = None)` — same resolution (`device if device is not None else get_device(training_config or {})`), passed to `_train_model(...)` (line 1623).
- `run_extract(..., device: str | None = None)` — pass `device=device` to `_extract_features(...)` (line 1531). No config merge for extract (per story decision).

**`HuBERTManager.get()`** stays untouched — its `detect_device()` fallback only fires when no device is passed, and all in-repo callers now pass one.

### 1.6 Multiprocessing / pickling — no concern

- `_extract_features` is a sequential loop over `wav_files` — no `ProcessPoolExecutor`, so a device param is trivially safe.
- `_extract_f0_single` / `_extract_mel_single` (ProcessPoolExecutor workers) are CPU-only and take no device. No change.
- `DataLoader(num_workers=0)` — no worker pickling.

### 1.7 Auralis integration note (separate repo — documented, not implemented)

`/home/sheigl/code/auralis/training/train.py` runs `kokoclone-train` as a subprocess. The plan specifies the following change (to be applied in the auralis repo, out of scope here):

```python
# when building the subprocess command list:
cmd = ["kokoclone-train", "train", "--name", name, "--audio-dir", audio_dir]
if device := config.get("device"):   # "auto" or explicit
    cmd += ["--device", device]
```

If auralis has no device setting, it should simply omit the flag (defaults to auto). Primary deliverable is the kokoclone-train side; auralis change is a follow-up.

## 2. Files to Modify

| File | Change |
|---|---|
| `kokoclone_train/core/rvc_assets.py` | `detect_device(requested)` + `_auto_detect()` + `_available_summary()`; `empty_cache` mps/cpu handling |
| `kokoclone_train/core/rvc_train.py` | `device: str \| None = None` on `_extract_features`, `_train_model`, `train_voice`, `run_train_only`, `run_extract`; resolve + pass through at the 3 call sites; `training_hp.json` gains `"device"` |
| `kokoclone_train/__main__.py` | `--device` on `train`, `train-only`, `extract`; add to `cli_overrides` (train/train-only); pass to `run_extract` |
| `kokoclone_train/core/training_config.py` | `get_device(config)` helper |
| `training_config.json` (root) | `"device": "auto"` key |
| `tests/test_cli_train.py` | `--device` parsing + precedence tests |
| `tests/test_detect_device.py` | NEW — unit tests for `detect_device` override/validation (mocked torch) |
| `docs/ARCHITECTURE.md` (optional) | Update "Device Handling" section to mention the override |
| `auralis/training/train.py` (separate repo) | Documented only: `--device` subprocess passthrough |

## 3. Task Breakdown

1. **`detect_device` override + `empty_cache`** in `rvc_assets.py`. Verify: `pytest tests/test_detect_device.py` (new).
2. **`get_device` helper** in `training_config.py` + `"device": "auto"` in root `training_config.json`. Verify: import test.
3. **CLI flags** on `train`, `train-only`, `extract` + `cli_overrides` wiring. Verify: new CLI parsing tests.
4. **Signature + call-site threading** in `rvc_train.py` (`_extract_features`, `_train_model`, `train_voice`, `run_train_only`, `run_extract`). Verify: full `pytest tests/` (existing tests exercise all these paths with patched sub-functions).
5. **`training_hp.json`** records `"device"`. Verify: manual/targeted test asserting key presence.
6. **Tests**: `test_detect_device.py` + `test_cli_train.py` updates. Verify: `pytest tests/` green (conftest ordering intact).
7. **Docs**: ARCHITECTURE.md "Device Handling" note. Verify: markdown renders.

## 4. Testing Strategy

### 4.1 `tests/test_detect_device.py` (new, mocked torch)

Pattern (per AGENTS.md `_mock_heavy_deps`/`_unmock_heavy_deps`). **Critical import-caching subtlety**: `rvc_assets.py` does `import torch` at module level. If another test already imported `rvc_assets`, its `torch` reference is the *real* torch and sys.modules mocking won't affect it. Robust approach: use `patch.object` on the already-imported module's attributes:

```python
import sys
from unittest.mock import MagicMock, patch

import kokoclone_train.core.rvc_assets as rvc_assets


def _mock_heavy_deps():
    """Inject mocks for heavy ML dependencies into sys.modules."""
    torch_mock = MagicMock()
    torch_mock.__path__ = []
    sys.modules["torch"] = torch_mock
    # NOTE: rvc_assets.torch is bound at import time; if the module was already
    # imported with the real torch, patch.object(rvc_assets.torch, ...) is used instead.


def _unmock_heavy_deps():
    to_remove = [k for k in sys.modules if k == "torch" or k.startswith("torch.")]
    for mod_name in to_remove:
        del sys.modules[mod_name]
```

Two viable mocking styles (pick one consistently; `patch.object` is preferred because it works regardless of import order):

1. **`patch.object(rvc_assets.torch.backends.mps, "is_available", return_value=False)`** etc. — with a MagicMock torch, sub-attributes auto-generate, so `rvc_assets.torch.cuda.is_available` is a MagicMock that can be configured. This is the most robust.
2. **sys.modules injection before import** — only reliable in a fresh test process; conftest ordering can't guarantee `rvc_assets` isn't imported earlier.

Test cases:

| Test | Setup | Expectation |
|---|---|---|
| `test_detect_auto_no_arg` | mps=False, cuda=False, xpu not available | `cpu` |
| `test_detect_auto_mps_first` | mps=True | `mps` (chain order preserved) |
| `test_detect_auto_cuda` | mps=False, cuda=True | `cuda` |
| `test_detect_auto_xpu` | mps=False, cuda=False, xpu available | `xpu` |
| `test_detect_cpu` | any | `cpu` |
| `test_detect_cuda_available` | cuda=True, device_count=1 | `cuda` / `cuda:0` |
| `test_detect_cuda_indexed` | cuda=True, device_count=2 | `cuda:1` |
| `test_detect_cuda_unavailable_raises` | cuda=False | `ValueError`, message contains "cuda", "ROCm", "ROCM_RX470_SETUP.md", "auto" |
| `test_detect_cuda_index_oob_raises` | cuda=True, device_count=1, request `cuda:2` | `ValueError` with "1" (count) in message |
| `test_detect_xpu_available` | xpu available | `xpu` / `xpu:0` |
| `test_detect_xpu_unavailable_raises` | no `xpu` attr | `ValueError` with "xpu" in message |
| `test_detect_mps_available` | mps=True | `mps` |
| `test_detect_mps_unavailable_raises` | mps=False | `ValueError` |
| `test_detect_unknown_raises` | `"foo"` | `ValueError` listing valid values |
| `test_detect_case_whitespace` | `"  CUDA:0 "` | `cuda:0` |
| `test_empty_cache_cpu_noop` | cpu device | no exception; `torch.cuda.empty_cache` not called |
| `test_empty_cache_mps` | mps device, `torch.mps.empty_cache` exists | called |

**Important**: `torch.cuda.device_count()` is only called *after* `is_available()` is True (guard order in the implementation), so tests never stub `device_count` when `is_available()` is False. MagicMock auto-attributes make `torch.xpu.is_available()` etc. trivially configurable. When `hasattr(torch, "xpu")` must be False, use a plain `object()` torch-like with only the needed attributes, or `patch.object` with `spec`.

### 4.2 `tests/test_cli_train.py` updates

Current tests reimplement the parser inline. New tests should exercise the **real** `_build_parser()` from `kokoclone_train.__main__`. Import caveat: `__main__` does `from kokoclone_train.core.rvc_train import train_voice` at module level, and `rvc_train.py` does `import torch` at module level — so `_mock_heavy_deps()` must run **before** importing `__main__`, inside the test method, wrapped in `try/finally` with `_unmock_heavy_deps()`.

```python
def test_train_device_flag_parsed(self):
    _mock_heavy_deps()
    try:
        from kokoclone_train.__main__ import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["train", "--name", "v", "--audio-dir", "/tmp/a", "--device", "cuda:0"])
        assert args.device == "cuda:0"
        args2 = parser.parse_args(["train", "--name", "v", "--audio-dir", "/tmp/a"])
        assert args2.device is None  # unset → None → config decides → "auto"
    finally:
        _unmock_heavy_deps()
```

Cases:
- `--device cuda:0` parses on `train`, `train-only`, `extract`.
- Unset `--device` → `None` (proves config authority is preserved).
- CLI override beats config: extend `TestTrainCommandIntegration` to patch `resolve_training_config` or assert `cli_overrides` content — simplest: patch `kokoclone_train.__main__.resolve_training_config` with a spy returning a fixed dict, call `_main`-equivalent logic, assert it was called with `cli_overrides["device"] == "cuda:0"`. (Keep to the existing style of the file: patch `train_voice` + assert kwargs, or spy on the config resolver.)
- Config honored when flag absent: unit-test `get_device({"device": "xpu:1"}) == "xpu:1"` and `get_device({}) == "auto"`.
- Existing 21 tests must still pass unchanged (flag is additive).

### 4.3 What NOT to test

- No real-GPU e2e (would require the actual ROCm/XPU hardware).
- No test that calls `detect_device("cuda:0")` against a real CUDA device — all device probing is mocked.

## 5. Edge Cases

- **`cuda:2` with 1 device** → clear `ValueError` with the count.
- **`xpu` without `torch.xpu.device_count`** → default count 1, guarded.
- **`mps` on Linux** → clear `ValueError` (MPS is macOS-only).
- **Unknown string / bad index (`cuda:abc`)** → `ValueError` listing valid values.
- **`empty_cache` on mps/cpu** → guarded, no-op, never raises.
- **Backward compat**: no-arg `detect_device()` identical; `HuBERTManager.get(device=None)` still auto-detects; all existing callers unchanged; `pin_memory` semantics unchanged.
- **Extract path stays simple**: `--device` on extract is CLI-only (no config file merging) — consistent with how `--step`/`--use-decoder` work there today.
- **`training_hp.json`** gains the device string — old saved files are unaffected (new key only).

## 6. Risks & Rollback

- **Import-caching in tests**: mitigated by `patch.object` on the already-imported `rvc_assets` module (style 1 in §4.1). If flakiness appears, isolate `test_detect_device.py` via a fresh subprocess or conftest ordering entry (group 0 is fine — detect_device tests don't need real torch, but running them before the torch-mocking CLI tests avoids any interaction; simplest: leave them in group 1 and rely on `patch.object`).
- **Regression of existing CLI tests**: all changes are additive; the parser signature and `cli_overrides` dict only gain a key. Run `pytest tests/` after each task.
- **Rollback**: every change is a small, revertable edit; no schema migrations, no data changes. Worst case reverts cleanly to auto-detect behavior (all `device` params default to `None` → `"auto"`).

## 7. Verification Commands

```bash
uv run pytest tests/                     # full suite (conftest ordering preserved)
uv run pytest tests/test_detect_device.py -v   # new unit tests
uv run pytest tests/test_cli_train.py -v       # CLI + precedence tests
kokoclone-train train --name test --audio-dir /tmp/a --device bogus   # manual: clean ValueError exit code 1
```
