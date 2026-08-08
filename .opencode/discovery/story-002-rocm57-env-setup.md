# Story: ROCm 5.7 environment setup for the RX 470 (guide + tooling)

## User Story
As a user with an AMD RX 470 (Polaris/gfx803) alongside Intel Arc GPUs, I want a
scripted, documented way to create a separate ROCm 5.7 Python environment
(torch 2.3 + rocm5.7) and verify the RX 470 appears as `cuda:0`, so that I can run
full training runs on the AMD card without disturbing the existing Intel XPU environment.

## Context

Verified research (do NOT re-verify):
- ROCm 6.x/7.x do not support Polaris/gfx803; the oldest supported is gfx900+; the TheRock
  community project has no gfx803 on its roadmap.
- ROCm 5.7 is the last officially-supported release for Polaris.
- Matching PyTorch wheels: torch 2.3.x + rocm5.7 from `https://download.pytorch.org/whl/rocm5.7`
  (`pip install torch==2.3.1 torchaudio --index-url https://download.pytorch.org/whl/rocm5.7`).
- A ROCm torch build exposes AMD GPUs through the `torch.cuda` API (`torch.cuda.is_available()`
  → True, device `cuda:0`).
- The two torch builds (XPU and ROCm) CANNOT coexist in one environment. User accepted this and
  chose: code-side device selection (story 001) + a separate ROCm 5.7 venv for full training runs.

Environment: Ubuntu 24.04, kernel 7.0.0-28-generic, both `amdgpu` and `xe` kernel modules
loaded. The `amdgpu` kernel driver already supports Polaris — no amdgpu-dkms rebuild should be
needed. ROCm 5.7 officially targets Ubuntu 20.04/22.04; on 24.04 the userspace install is
best-effort and must be documented as such.

Constraint: the main `pyproject.toml` pins `torch>=2.10` and maps torch/torchaudio to the
`pytorch-xpu` index via `[tool.uv.sources]`. The ROCm env MUST NOT use `uv sync` or the
pytorch-xpu index, or it would pull the wrong torch build. Also note `torchvision` is declared
in pyproject but never imported by the code — it can be omitted from the ROCm env (or pinned to
the matching rocm5.7 wheel for parity). torch 2.3.x requires `numpy<2` (it was built against
numpy 1.x) — the script must pin numpy accordingly.

## Acceptance Criteria

- [ ] `tools/setup_rocm_env.sh` exists, is executable, and passes `bash -n` (syntax check);
      it is also `shellcheck`-clean if shellcheck is available.
- [ ] The script creates a SEPARATE venv (e.g. `.venv-rocm57` or `venvs/rocm57`) and never
      modifies the existing XPU `.venv`.
- [ ] The script installs `torch==2.3.1` and `torchaudio==2.3.1` from
      `https://download.pytorch.org/whl/rocm5.7` (NOT the pytorch-xpu index). It does NOT run
      `uv sync` (which would apply `[tool.uv.sources]`); it installs the project package itself
      with `--no-deps` (e.g. `uv pip install -e . --no-deps` or equivalent) so the `kokoclone-train`
      entry point works without re-resolving torch from the XPU index.
- [ ] The script installs the remaining runtime deps from PyPI with torch-2.3-compatible
      versions: `numpy<2`, `scipy`, `soundfile`, `pyworld`, `faiss-cpu`, `tqdm`, `watchdog`,
      `ninja`, `huggingface-hub` (version compatible with the pinned transformers), and
      `transformers` in a torch-2.3-compatible range (e.g. `transformers>=4.40,<4.48` — exact
      range documented in the plan).
- [ ] `tools/check_rocm.py` exists and, when run inside the ROCm venv, prints: torch version,
      `torch.cuda.is_available()`, `torch.cuda.device_count()`, each device name (expecting
      "AMD Radeon RX 470" for the card), and the result of the pipeline's own
      `detect_device("auto")` — so the user can confirm the RX 470 shows up as `cuda:0` and that
      story-001 device selection will route to it.
- [ ] Documentation is added (a `docs/ROCM_RX470_SETUP.md` plus a short README pointer, or a
      README section) covering:
  - Why ROCm 5.7 (Polaris unsupported in 6.x/7.x; last supported release).
  - That `amdgpu` is already loaded (no dkms rebuild); ROCm 5.7 userspace install is best-effort
    on Ubuntu 24.04 (official AMD `rocm-install-linux-5.7.sh` / repo steps, with known caveats).
  - The two-environments constraint (XPU venv vs ROCm venv — never in one env).
  - The exact commands: create venv, install torch/torchaudio rocm5.7, install deps, install the
    package `--no-deps`, run `tools/check_rocm.py`.
  - The end-to-end workflow: activate the ROCm venv and run
    `kokoclone-train train --name <voice> --audio-dir <dir> --device cuda:0 ...`.
- [ ] The guide explicitly states this is a guide the user runs on their own box: CI cannot
      verify real GPU training. Acceptance for this story = script syntactically valid + documented
      + runtime check script included and correct. (Running the script against a real GPU is out of
      scope for automated tests.)

## Technical Plan

**Plan file**: `plans/story-002-rocm57-env-setup-plan.md`

The technical design, architecture decisions, and task breakdown for this story
are in the companion plan file linked above.

## Dependencies
- Story 001 (recommended ordering: 001 first, so `--device cuda:0` exists by the time the user
  follows the guide; functionally independent)

## Priority: High

## Notes
- Pairing: in the ROCm venv, story-001's `--device cuda:0` (or plain auto-detect) routes the
  pipeline to the RX 470; `pin_memory` already covers `cuda`.
- Do NOT add ROCm deps to the main `pyproject.toml` — the XPU build is the primary environment;
  the ROCm env is a separate, scripted environment that must not be resolvable by `uv sync`.
- If a synthetic CI check is desired, the script can be `bash -n`-checked and
  `tools/check_rocm.py` can be py_compile'd, but nothing in CI can confirm the GPU responds.
