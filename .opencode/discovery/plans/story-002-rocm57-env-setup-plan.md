# Technical Plan: ROCm 5.7 environment setup for the RX 470 (guide + tooling)

**Story**: `story-002-rocm57-env-setup.md`
**Status**: Ready for implementation
**Priority**: High

## 1. Architecture & Design Decisions

### 1.1 Venv strategy

- **Location**: `.venv-rocm57` at the repo root (sibling of the existing `.venv`). Rationale: obvious, git-ignore-able, and impossible to confuse with the XPU env. (Alternative `venvs/rocm57` is fine if preferred; pick one and be consistent in docs.)
- **Never touch `.venv`** — the script only creates/uses `.venv-rocm57`.
- **Never use `uv sync`** — `uv sync` resolves `[tool.uv.sources]` from `pyproject.toml`, which maps `torch`/`torchaudio` to the `pytorch-xpu` index and would clobber the ROCm install. Instead, the script uses `uv venv` (pure venv creation, no project resolution) followed by `uv pip install` with explicit per-package index flags. `uv pip` operates on a venv directly and does NOT read `[tool.uv.sources]` (that is a `uv sync`/project-level behavior) — this is the key mechanism that keeps the ROCm env isolated.
- If `uv` is unavailable, fall back to `python3 -m venv` + `pip install` with the same index flags (document both; the script should prefer uv and `die` with a helpful message if neither exists).
- **Python version**: project requires `>=3.12`. torch 2.3.1 ships cp38–cp312 wheels on the rocm5.7 index, so Python 3.12 works. The script should detect the active interpreter and print a warning if `< 3.12`.

### 1.2 Install strategy (exact command shape)

```bash
# 1. Create isolated venv (NO project resolution — safe)
uv venv .venv-rocm57 --python 3.12

# 2. Activate for the remainder of the script
source .venv-rocm57/bin/activate

# 3. torch/torchaudio from the ROCm index ONLY (never the xpu index)
uv pip install \
    torch==2.3.1 \
    torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/rocm5.7

# 4. Remaining runtime deps from PyPI (torch-2.3-compatible pins)
uv pip install \
    "numpy<2" \
    "scipy>=1.11,<1.14" \
    "soundfile>=0.13" \
    "pyworld>=0.3.4" \
    "faiss-cpu>=1.7" \
    "tqdm>=4.65" \
    "watchdog>=4.0" \
    "ninja>=1.13" \
    "huggingface-hub>=0.23,<1.0" \
    "transformers>=4.40,<4.48"

# 5. The package itself WITHOUT re-resolving torch from the XPU index
uv pip install -e . --no-deps

# 6. Verify
python tools/check_rocm.py
```

**Version rationale (torch-2.3 compatibility):**

| Package | Pin | Why |
|---|---|---|
| `torch` | `==2.3.1` | Last line with official rocm5.7 wheels supporting gfx803 |
| `torchaudio` | `==2.3.1` | Matches torch 2.3.1 |
| `numpy` | `<2` | torch 2.3.x built against numpy 1.x; numpy 2 breaks binary ABI |
| `transformers` | `>=4.40,<4.48` | 4.44–4.47 era works with torch 2.3; >=4.48 trends toward newer torch requirements. Exact upper bound to be confirmed at implement time against transformers' `torch` requirement |
| `huggingface-hub` | `>=0.23,<1.0` | transformers 4.4x requires `hub>=0.23` but predates hub 1.x API changes |
| `scipy` | `>=1.11,<1.14` | pyworld 0.3.4 compatibility window |
| `torchvision` | **omitted** | Declared in pyproject but never imported by kokoclone_train (verified via grep). Optional parity pin: `torchvision==0.18.1` from the rocm5.7 index — not required |
| `faiss-cpu`, `soundfile`, `pyworld`, `tqdm`, `watchdog`, `ninja` | as in pyproject | CPU-only, no torch ABI coupling |

### 1.3 `tools/setup_rocm_env.sh` structure

```
#!/usr/bin/env bash
set -euo pipefail

Sections:
1. Header/usage — prints purpose, the two-environments warning (XPU vs ROCm never coexist),
   and the "this is best-effort on Ubuntu 24.04" caveat.
2. Pre-flight checks:
   - `command -v uv` or `command -v python3` (die with instructions if neither)
   - `python3 --version` >= 3.12 check (warn only)
   - `[ -d .venv-rocm57 ]` → prompt to reuse or remove (idempotency)
3. Venv creation: `uv venv .venv-rocm57 --python 3.12` (or `python3 -m venv` fallback)
4. Activate venv (`source .venv-rocm57/bin/activate`)
5. Install torch/torchaudio: rocm5.7 index ONLY — with an explicit comment:
   "# NOTE: never run 'uv sync' in this env — pyproject [tool.uv.sources] maps torch
    # to the pytorch-xpu index."
6. Install deps (§1.2 step 4)
7. Install package: `uv pip install -e . --no-deps`
8. Print next steps: activate env, run `python tools/check_rocm.py`, then
   `kokoclone-train train --name <voice> --audio-dir <dir> --device cuda:0 ...`
```

**Idempotency**: if `.venv-rocm57` already exists, the script asks whether to recreate (`rm -rf` + recreate) or skip creation and just re-verify. Never deletes `.venv`.

### 1.4 `tools/check_rocm.py`

```python
"""Runtime verification for the ROCm 5.7 environment (AMD RX 470 / gfx803)."""

import sys

def main() -> int:
    import torch  # imported here so a broken env prints a clean message

    print(f"torch version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[FAIL] CUDA is not available.")
        print("The RX 470 must appear as a CUDA device when using the ROCm build of torch.")
        print("Check: correct venv active? (expect .venv-rocm57)  `which python`")
        print("       torch installed from the rocm5.7 index?  `pip show torch`")
        return 1

    n = torch.cuda.device_count()
    print(f"torch.cuda.device_count(): {n}")
    for i in range(n):
        print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
    if n == 0:
        print("[FAIL] device_count() is 0 despite is_available()==True.")
        return 1

    # Tie-in with story-001 device selection: what the pipeline would pick.
    from kokoclone_train.core.rvc_assets import detect_device
    d = detect_device("auto")
    print(f"kokoclone-train detect_device('auto') -> {d}")
    if d.type != "cuda":
        print(f"[WARN] auto-detect returns '{d}' (expected cuda). Use --device cuda:0 explicitly.")
        return 1
    print("[OK] RX 470 exposed as a CUDA device; kokoclone-train will use it.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- Prints torch version, `is_available()`, device count, per-device names (expecting `AMD Radeon RX 470`), and the pipeline's `detect_device("auto")` result (proves story-001 wiring routes to the AMD card).
- Fails fast (exit 1) with actionable messages when CUDA is missing.
- Imports torch lazily inside `main()` so `py_compile`/imports in CI never require torch.

### 1.5 Documentation: `docs/ROCM_RX470_SETUP.md` (+ README pointer)

Sections:
1. **Why ROCm 5.7** — Polaris/gfx803 unsupported in ROCm 6.x/7.x (verified compatibility matrix); 5.7 is the last supported release; TheRock community project has no gfx803 roadmap.
2. **Kernel**: `amdgpu` already supports Polaris and is loaded on this box — **no amdgpu-dkms rebuild needed**. Verify with `lsmod | grep amdgpu` / `rocminfo`.
3. **System-level ROCm 5.7 userspace (best-effort on Ubuntu 24.04)**: ROCm 5.7 officially targets 20.04/22.04. On 24.04 use AMD's `rocm-install-linux-5.7.sh` (or the 5.7 apt repo) with the caveat that some components may need manual resolution; `HSA_OVERRIDE_GFX_VERSION=8.0.3` is **not** required for Polaris on 5.7 (gfx803 is a first-class citizen there) but note it as the classic fallback trick if a newer component is accidentally pulled in. Keep this section honest: Python-side verification (`check_rocm.py`) is the source of truth.
4. **Two environments, never mixed** — `.venv` (XPU, torch 2.10) vs `.venv-rocm57` (ROCm, torch 2.3). `uv sync` in the ROCm env is forbidden (pyproject maps torch to the xpu index). Show `pip show torch` as the "which env am I in" check.
5. **Exact setup commands** (mirroring the script, documented for hand-rolling).
6. **Verify** — `python tools/check_rocm.py` expected output (cuda:0 → AMD Radeon RX 470).
7. **Workflow** — activate `.venv-rocm57`; `kokoclone-train train --name <voice> --audio-dir <dir> --device cuda:0 ...` (flag added by story 001); note `--use-decoder` full runs on the 8GB card benefit from the small `--batch-size` (2 is the per-voice default for am_jefferson).
8. **Known limitations** — torch 2.3 (older feature set than 2.10); no torch.xpu in this env; do not install flash-attn here.

README: add a short "ROCm / AMD RX 470" bullet under Setup pointing to `docs/ROCM_RX470_SETUP.md`.

## 2. Files to Create / Modify

| File | Action |
|---|---|
| `tools/setup_rocm_env.sh` | NEW — env creation + install + verification guide output |
| `tools/check_rocm.py` | NEW — runtime verification snippet |
| `docs/ROCM_RX470_SETUP.md` | NEW — full guide |
| `README.md` | EDIT — pointer bullet under Setup |
| `.gitignore` | EDIT — add `.venv-rocm57/` (verify `.venv/` is already ignored; if not, add both) |
| `pyproject.toml` | **NO CHANGE** — must stay XPU-primary |

## 3. Task Breakdown

1. **`tools/setup_rocm_env.sh`** — write with sections per §1.3. Verify: `bash -n tools/setup_rocm_env.sh` (and `shellcheck tools/setup_rocm_env.sh` if installed).
2. **`tools/check_rocm.py`** — write per §1.4. Verify: `python3 -m py_compile tools/check_rocm.py`; also confirm it imports cleanly *without* torch in the current XPU env? — no: it imports torch inside main(), so compile-check only; do NOT execute it in the XPU env (it would report xpu env state, not fail).
3. **`docs/ROCM_RX470_SETUP.md`** — write per §1.5. Verify: prose review; commands are copy-paste-consistent with the script.
4. **README + .gitignore** edits. Verify: `git status` shows the intended files only.
5. **Test-suite integrity** — `pytest tests/` still green (the new files must not be collected as tests; `testpaths=["tests"]` guarantees this — `tools/` is outside). No new pytest files required.

## 4. Testing Strategy

**Automated (CI-safe, no GPU):**
- `bash -n tools/setup_rocm_env.sh` (syntax).
- `shellcheck` if available (warn-only if not installed).
- `python3 -m py_compile tools/check_rocm.py`.
- `pytest tests/` green (unchanged suite).
- Optional tiny test `tests/test_tools_syntax.py` asserting both checks above via `subprocess` — only if it adds value; keep it torch-free. (Recommended: skip — CI noise for marginal value; the Implement agent may add it.)

**Manual (user's box — out of scope for automated tests):**
- Run `tools/setup_rocm_env.sh` on the Ubuntu 24.04 machine.
- `source .venv-rocm57/bin/activate && python tools/check_rocm.py` → expect `cuda:0 -> AMD Radeon RX 470`, `detect_device('auto') -> cuda`.
- `pip show torch` shows 2.3.1 from the rocm5.7 index.
- Full training smoke: `kokoclone-train train --name test --audio-dir <dir> --epochs 1 --device cuda:0`.
- Confirmed: the guide explicitly documents that CI cannot verify GPU training; acceptance = script syntax + docs + runtime check script.

## 5. Edge Cases

- **`uv` not installed** → die with `python3 -m venv` fallback instructions (script auto-falls back).
- **`.venv-rocm57` already exists** → prompt to recreate or skip (idempotent re-run).
- **numpy 2.x accidentally present** → `"numpy<2"` pin prevents it; check_rocm prints a clear failure otherwise.
- **Python < 3.12** → warning; torch 2.3.1 cp312 wheels exist, cp311 also exists (script notes the venv python should be >=3.11; 3.12 preferred to match project `requires-python`).
- **`huggingface-hub` 1.x vs transformers 4.4x conflict** → pinned `<1.0`; pip would otherwise resolve the latest hub and may break transformers' hub calls.
- **User ran `uv sync` inside the ROCm venv by mistake** → docs call this out prominently; check_rocm's `torch.__version__`/index check would reveal torch 2.10-xpu installed.
- **System ROCm missing/failed on 24.04** → check_rocm fails with the "which venv / which torch index" hints; docs explain best-effort userspace install; Python env script itself does NOT install system ROCm (kept out of scope — user runs AMD's installer separately).
- **Never touches `.venv`** → script guards: only operates on `.venv-rocm57`.

## 6. Risks & Rollback

- **cp312 wheel availability for torch 2.3.1 rocm5.7**: exists (torch 2.3.1 shipped cp38-cp312). If a future regression appears, fall back to `--python 3.11` (torch 2.3.1 cp311 wheels definitely exist).
- **AMD userspace on 24.04**: known best-effort; docs are honest, check_rocm is the gate. Rollback for the user = uninstall nothing; the Python env is fully disposable (`rm -rf .venv-rocm57`).
- **XPU env untouched**: all changes are additive files in `tools/` + docs + .gitignore. `pyproject.toml` intentionally not modified. Worst case rollback = delete the 3 new files + 1 README line.

## 7. Verification Commands

```bash
bash -n tools/setup_rocm_env.sh
shellcheck tools/setup_rocm_env.sh        # if installed
python3 -m py_compile tools/check_rocm.py
uv run pytest tests/                      # must stay green
# Manual (user's box):
# bash tools/setup_rocm_env.sh
# source .venv-rocm57/bin/activate && python tools/check_rocm.py
```
