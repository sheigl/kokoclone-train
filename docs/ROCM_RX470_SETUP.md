# ROCm 5.7 / AMD RX 470 Setup Guide

Run full kokoclone-train training jobs on an AMD RX 470 (Polaris / gfx803)
from a **separate ROCm 5.7 environment** (`.venv-rocm57`), without disturbing
the Intel XPU environment (`.venv`).

> This is a guide you run on your own machine. CI cannot verify real GPU
> training — acceptance for this workstream is: script is syntactically valid,
> the runtime check script is correct, and this document is accurate.

---

## 1. Why ROCm 5.7

- **ROCm 6.x and 7.x dropped support for Polaris/gfx803.** The oldest
  officially supported generation in ROCm 6+ is gfx900 (Vega). The TheRock
  community project (ROCm for consumer AMD GPUs) has no gfx803 on its roadmap.
- **ROCm 5.7 is the last release with first-class gfx803 support.**
- The matching PyTorch wheels are **torch 2.3.1 + torchaudio 2.3.1** from
  `https://download.pytorch.org/whl/rocm5.7`.
- A ROCm build of torch exposes AMD GPUs through the **CUDA API**:
  `torch.cuda.is_available()` → `True`, device `cuda:0`. This is exactly what
  story-001's `--device cuda:0` flag expects.

## 2. Kernel: amdgpu already loaded — no dkms rebuild

The `amdgpu` kernel driver has supported Polaris for years and is **already
loaded** on this box (kernel `7.0.0-28-generic`, Ubuntu 24.04):

```bash
lsmod | grep amdgpu   # → amdgpu, gfx803, ... present
rocminfo              # → lists the gfx803 agent (if ROCm userspace is installed)
```

The RX 470 sits at PCI `0a:00.0`. **No `amdgpu-dkms` rebuild is needed.**
Only the ROCm *userspace* is required (next section).

## 3. System-level ROCm 5.7 userspace (best-effort on Ubuntu 24.04)

ROCm 5.7 officially targets Ubuntu 20.04/22.04. On 24.04 the userspace install
is **best-effort** — some components may need manual resolution. Two options:

1. **AMD's installer script** (recommended start):
   ```bash
   # Download rocm-install-linux-5.7.sh from AMD, then:
   sudo ./rocm-install-linux-5.7.sh --usecase rocm --no-verify
   ```
2. **The 5.7 apt repo** (`https://repo.radeon.com/rocm/apt/5.7/`), following
   AMD's repo setup steps for the 5.7 series.

Caveats / notes:

- `HSA_OVERRIDE_GFX_VERSION=8.0.3` is **NOT required** for Polaris on ROCm 5.7 —
  gfx803 is a first-class citizen there. It is the classic fallback trick
  (force-report the gfx803 GPU as gfx900) if a *newer* component accidentally
  gets pulled in; you only need it if something complains about gfx803.
- The Python-side check (`tools/check_rocm.py`) is the **source of truth** —
  if the RX 470 shows up as `cuda:0`, the userspace is good enough regardless
  of which installer path you used.
- This repo deliberately does **not** script the system-level install; run
  AMD's installer separately.

## 4. Two environments, never mixed

| Env | torch | Where it comes from | Use |
|---|---|---|---|
| `.venv` | 2.10 (XPU) | `uv sync` + pytorch-xpu index | primary dev / Intel Arc |
| `.venv-rocm57` | 2.3.1 (ROCm 5.7) | `uv pip install --no-project --index-url …/rocm5.7` | full training on the RX 470 |

- The two torch builds **cannot coexist in one environment**.
- **Never run `uv sync` inside `.venv-rocm57`.** `pyproject.toml`'s
  `[tool.uv.sources]` maps `torch`/`torchaudio` to the pytorch-xpu index and
  would replace the ROCm build. `uv pip install` also reads the project's
  `[[tool.uv.index]]` — that is where the pytorch-xpu index is defined — so
  the setup script passes `--no-project` to every `uv pip install` command.
  This tells uv to ignore the project config entirely, making the
  `--index-url …/rocm5.7` (or the default PyPI) the only source considered.
- "Which env am I in?" check:
  ```bash
  pip show torch        # Version + the rocm5.7/xpu index it came from
  which python          # expect .../.venv-rocm57/bin/python or .../.venv/bin/python
  ```

## 5. Exact setup commands

The one-shot way is the script (creates everything, idempotent):

```bash
bash tools/setup_rocm_env.sh
```

Hand-rolled, this is what the script does:

```bash
# 1. Create isolated venv (NO project resolution — safe)
uv venv .venv-rocm57 --python 3.12      # or: python3 -m venv .venv-rocm57

# 2. Activate
source .venv-rocm57/bin/activate

# 3. torch/torchaudio from the ROCm index ONLY (never the xpu index)
#    --no-project: ignore pyproject.toml's [[tool.uv.index]] (pytorch-xpu) so
#    the rocm5.7 index below is the only source considered for torch.
uv pip install --no-project torch==2.3.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/rocm5.7

# 4. Remaining runtime deps from PyPI (torch-2.3-compatible pins)
uv pip install --no-project \
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
uv pip install --no-project -e . --no-deps

# 6. Verify
python tools/check_rocm.py
```

Pin rationale (torch-2.3 compatibility):

- `numpy<2` — torch 2.3.x was built against numpy 1.x; numpy 2 breaks the binary ABI.
- `transformers>=4.40,<4.48` — the 4.44–4.47 era works with torch 2.3; newer
  releases trend toward newer torch requirements.
- `huggingface-hub>=0.23,<1.0` — transformers 4.4x requires `hub>=0.23` but
  predates the hub 1.x API changes.
- `scipy>=1.11,<1.14` — pyworld 0.3.4 compatibility window.
- `torchvision` is omitted — declared in `pyproject.toml` but never imported
  by the code. (Optional parity pin: `torchvision==0.18.1` from the rocm5.7 index.)

If `uv` is missing, the script falls back to `python3 -m venv` + `pip` with the
same index flags (it dies with a helpful message only if *neither* exists).
The pip fallback does **not** pass `--no-project` (pip would error on that
flag) — pip ignores `pyproject.toml`'s `[[tool.uv.index]]`, so
`pip install --index-url …/rocm5.7` already uses only the rocm5.7 index.

## 6. Verify

```bash
source .venv-rocm57/bin/activate
python tools/check_rocm.py
```

Expected output:

```
torch version: 2.3.1+rocm5.7
torch.cuda.is_available(): True
torch.cuda.device_count(): 1
  cuda:0 -> AMD Radeon RX 470
kokoclone-train detect_device('auto') -> cuda

[OK] RX 470 exposed as a CUDA device; kokoclone-train will use it.
```

The last line proves story-001's device selection will route to the AMD card.
If it fails, the script prints actionable hints (wrong venv, wrong torch index,
missing system ROCm).

## 7. Workflow — train on the RX 470

```bash
source .venv-rocm57/bin/activate
kokoclone-train train \
  --name <voice> \
  --audio-dir <path_to_audio_chunks> \
  --device cuda:0 \
  [--use-decoder] [--epochs N]
```

- `--device cuda:0` (added by story 001) pins the run to the RX 470. `--device
  auto` would also reach it in this env (CUDA comes before XPU in the auto
  chain), but explicit is better when a machine has both AMD and Intel GPUs.
- Full end-to-end runs (`--use-decoder`, hop_length=160, ~2.5M-step horizon)
  on the 8 GB card benefit from a small batch size — pass `--batch-size 2`
  explicitly. (The root `training_config.json` leaves `batch_size` as `null`,
  i.e. auto, which `_train_model` resolves to 2–8 based on dataset size.)
- `pip show torch` after a training session confirms you're in the ROCm env.

## 8. Known limitations

- **torch 2.3.1 is old** — a smaller feature set than the 2.10 used in the XPU
  env. Pin-compatible deps are listed in §5.
- **No `torch.xpu` in this env** — the ROCm build has no Intel XPU backend;
  that's the point of keeping `.venv` separate.
- **Do not install flash-attn here** — it is built for the primary env and is
  not compatible with torch 2.3 / ROCm 5.7.
- ROCm 5.7 on Ubuntu 24.04 is best-effort (§3); if the userspace install
  fights you, fall back to the `HSA_OVERRIDE_GFX_VERSION=8.0.3` trick or use
  the XPU env for the parts that don't need the AMD card.
- Rollback is trivial: the Python env is fully disposable —
  `rm -rf .venv-rocm57`. Nothing in this guide touches `.venv` or
  `pyproject.toml`.
