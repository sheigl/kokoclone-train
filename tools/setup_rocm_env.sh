#!/usr/bin/env bash
#
# kokoclone-train — ROCm 5.7 environment setup (AMD RX 470 / gfx803)
#
# Creates a SEPARATE Python environment (`.venv-rocm57`) with torch 2.3.1 built
# for ROCm 5.7 so the RX 470 shows up as a CUDA device (cuda:0) for full
# training runs, WITHOUT touching the existing Intel XPU `.venv`.
#
# This is best-effort on Ubuntu 24.04 (ROCm 5.7 officially targets 20.04/22.04).
# The amdgpu kernel driver for Polaris is already loaded — no dkms rebuild is
# needed. See docs/ROCM_RX470_SETUP.md for the full guide.
set -euo pipefail

# --- Locate repo root (script lives in tools/) ---------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

VENV_DIR=".venv-rocm57"
ROCM_INDEX="https://download.pytorch.org/whl/rocm5.7"

# =============================================================================
# Section 1 — header / two-environments warning
# =============================================================================
cat <<'EOF'
======================================================================
  kokoclone-train — ROCm 5.7 environment setup (AMD RX 470 / gfx803)
======================================================================

  This script creates a SEPARATE Python environment (.venv-rocm57) with
  torch 2.3.1 built for ROCm 5.7, so the RX 470 is exposed as a CUDA
  device (cuda:0) for full training runs.

  IMPORTANT — two environments, never mixed:
    * .venv        = Intel XPU build (torch 2.10, pytorch-xpu index)
    * .venv-rocm57 = AMD ROCm build (torch 2.3.1, rocm5.7 index)
  The two torch builds cannot coexist in one environment. This script
  ONLY creates/uses .venv-rocm57 and NEVER touches .venv.

  NOTE: best-effort on Ubuntu 24.04 — ROCm 5.7 officially targets
  20.04/22.04. The amdgpu kernel driver for Polaris is already loaded
  (no dkms rebuild needed). See docs/ROCM_RX470_SETUP.md.
EOF

# =============================================================================
# Section 2 — pre-flight checks
# =============================================================================
HAVE_UV=0
if command -v uv >/dev/null 2>&1; then
    HAVE_UV=1
fi

if [ "${HAVE_UV}" -eq 0 ] && ! command -v python3 >/dev/null 2>&1; then
    cat <<'EOF'
[ERROR] Neither 'uv' nor 'python3' was found on PATH.
        Install uv (https://docs.astral.sh/uv/) or python3 (>= 3.12),
        then re-run this script.
EOF
    exit 1
fi

# Warn (not fail) if the default interpreter is older than 3.12; the venv is
# what matters and uv can fetch a managed Python 3.12 if needed.
if [ "${HAVE_UV}" -eq 0 ] && command -v python3 >/dev/null 2>&1; then
    if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
        PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "unknown")"
        echo "  [WARN] default 'python3' is ${PY_VER}; project requires >= 3.12."
        echo "         torch 2.3.1 ships cp312 wheels; consider installing python3.12."
    fi
fi

# Idempotency: if the venv already exists, let the user recreate or keep it.
# This script NEVER deletes .venv — only .venv-rocm57 is ever touched.
SKIP_INSTALL=0
if [ -d "${VENV_DIR}" ]; then
    echo ""
    echo "  An existing '${VENV_DIR}' was found."
    read -r -p "  Recreate it (wipes the env) and reinstall? [y/N] " answer || true
    case "${answer,,}" in
        y | yes)
            echo "  Removing '${VENV_DIR}' and recreating..."
            rm -rf "${VENV_DIR}"
            ;;
        *)
            echo "  Keeping '${VENV_DIR}'. Skipping install; re-verifying instead."
            SKIP_INSTALL=1
            ;;
    esac
fi

# =============================================================================
# Section 3 — create the venv (pure venv creation; NO project resolution)
# =============================================================================
if [ "${SKIP_INSTALL}" -eq 1 ]; then
    echo ""
    echo "  (install skipped — using existing ${VENV_DIR})"
else
    echo ""
    echo "=== Creating ${VENV_DIR} ==="
    if [ "${HAVE_UV}" -eq 1 ]; then
        # 'uv venv' creates an isolated environment without reading
        # pyproject.toml — safe. Prefer Python 3.12 (matches the project's
        # requires-python); torch 2.3.1 has cp312 wheels on the rocm5.7 index.
        if ! uv venv "${VENV_DIR}" --python 3.12 --seed; then
            echo "  [WARN] uv could not create a Python 3.12 venv; using uv's default Python."
            uv venv "${VENV_DIR}" --seed
        fi
    else
        # pip fallback (uv unavailable). On Debian/Ubuntu this needs the
        # python3-venv package:  sudo apt install python3.12-venv
        python3 -m venv "${VENV_DIR}"
    fi

    # Ubuntu 24.04 sets PEP 668 EXTERNALLY-MANAGED in system Python, which
    # gets copied into the venv and blocks 'pip install'. Remove it so pip
    # works normally inside our isolated environment.
    rm -f "${VENV_DIR}/EXTERNALLY-MANAGED"
fi

# Post-creation Python sanity check (existing venv too).
VENV_PY="${VENV_DIR}/bin/python"
if [ -x "${VENV_PY}" ]; then
    if ! "${VENV_PY}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
        PY_VER="$("${VENV_PY}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "unknown")"
        echo "  [WARN] ${VENV_DIR} uses Python ${PY_VER}; project requires >= 3.12."
        echo "         torch 2.3.1 also ships cp311 wheels, but 3.12 is preferred."
    fi
else
    echo "  [ERROR] ${VENV_DIR} was created but ${VENV_PY} is missing."
    exit 1
fi

# =============================================================================
# Section 4 — activate the venv for the rest of the script
# =============================================================================
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "  Activated ${VENV_DIR} (python: $("${VENV_PY}" --version 2>&1))"

# =============================================================================
# Section 5 — install torch/torchaudio from the rocm5.7 index ONLY
# =============================================================================
if [ "${SKIP_INSTALL}" -eq 1 ]; then
    echo ""
    echo "=== Skipping installs (keeping existing ${VENV_DIR}) ==="
else
    echo ""
    echo "=== Installing torch==2.3.1 + torchaudio==2.3.1 (rocm5.7 index) ==="
    echo "  NOTE: never run 'uv sync' in this env — pyproject [tool.uv.sources]"
    echo "  maps torch to the pytorch-xpu index and would clobber this ROCm"
    echo "  install. We use plain 'pip install' (not 'uv pip') below so it does"
    echo "  NOT read the project's [[tool.uv.index]] / [tool.uv.sources] config."
    # Check if uv supports --no-project (added in uv 0.5.x); fall back to pip if not.
    HAVE_UV_NO_PROJECT=0
    if [ "${HAVE_UV}" -eq 1 ] && uv pip install --help 2>&1 | grep -q '\-\-no-project'; then
        HAVE_UV_NO_PROJECT=1
    fi

    _pip_cmd() {
        # Use 'uv pip install --no-project' when available (faster), else call
        # the venv's python -m pip directly to bypass PEP 668 system checks.
        if [ "${HAVE_UV_NO_PROJECT}" -eq 1 ]; then
            uv pip install --no-project "$@"
        else
            "${VENV_PY}" -m pip install --disable-pip-version-check "$@"
        fi
    }

    _pip_cmd torch==2.3.1 torchaudio==2.3.1 --index-url "${ROCM_INDEX}"

    # =========================================================================
    # Section 6 — remaining runtime deps from PyPI (torch-2.3-compatible pins)
    # =========================================================================
    echo ""
    echo "=== Installing runtime dependencies (PyPI, torch-2.3-compatible) ==="
    PYPI_DEPS=(
        "numpy<2"
        "scipy>=1.11,<1.14"
        "soundfile>=0.13"
        "pyworld>=0.3.4"
        "faiss-cpu>=1.7"
        "tqdm>=4.65"
        "watchdog>=4.0"
        "ninja>=1.13"
        "huggingface-hub>=0.23,<1.0"
        "transformers>=4.40,<4.48"
    )
    _pip_cmd "${PYPI_DEPS[@]}"

    # =========================================================================
    # Section 7 — the package itself, WITHOUT re-resolving torch
    # =========================================================================
    # --no-deps keeps 'uv sync'/pip from re-resolving torch from the
    # pytorch-xpu index; only the kokoclone-train entry point is installed.
    echo ""
    echo "=== Installing kokoclone-train (--no-deps) ==="
    _pip_cmd -e . --no-deps
fi

# =============================================================================
# Section 8 — next steps
# =============================================================================
cat <<EOF

======================================================================
  Setup finished.
======================================================================

Next steps:

  1. Activate the ROCm env:
       source .venv-rocm57/bin/activate

  2. Verify the RX 470 is exposed as a CUDA device:
       python tools/check_rocm.py
     Expected output:
       torch version: 2.3.1+rocm5.7
       torch.cuda.is_available(): True
       torch.cuda.device_count(): 1
         cuda:0 -> AMD Radeon RX 470
       kokoclone-train detect_device('auto') -> cuda
       [OK] RX 470 exposed as a CUDA device; kokoclone-train will use it.

  3. Train a voice on the AMD card:
       kokoclone-train train --name <voice> --audio-dir <dir> \\
         --device cuda:0 [--use-decoder --epochs N]

     Full runs on the 8 GB card benefit from a small batch size
     (2 is the default in training_config.json).

Reminders:
  * Never run 'uv sync' inside .venv-rocm57 — it would reinstall the XPU
    torch build (pyproject maps torch to the pytorch-xpu index).
  * 'pip show torch' shows which env/torch build is active.
  * System-level ROCm 5.7 userspace (best-effort on Ubuntu 24.04) is a
    separate step — see docs/ROCM_RX470_SETUP.md.
  * The XPU .venv is untouched.
EOF
