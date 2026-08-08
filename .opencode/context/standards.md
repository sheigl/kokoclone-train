# Standards & Conventions

> **Populated by the Document agent.** Seeded from `AGENTS.md` (source of truth).

## Tensor shape convention

- Dataset outputs: `[B, T, C]` content, `[B, T]` F0, `[B, T, n_mels]` mel.
- Model internal (flows/conv): `[B, C, T]` — transpose at boundaries.
- Model output: `[B, spec_channels=192, T]` (mel); `[B, 1, T×160]` (audio).

## Device handling

```python
from kokoclone_train.core.rvc_assets import detect_device
device = detect_device()            # MPS → CUDA → XPU → CPU
device = detect_device("cuda:0")    # explicit, validates availability
tensor.to(device)
```

Prefer threading a resolved device string/`torch.device` from the CLI/config
rather than calling `detect_device()` per-stage. `training_config.get_device()`
resolves the `"device"` config key (default `"auto"`).

The ROCm env (`.venv-rocm57`, torch 2.3.1) is a separate sibling of `.venv`
(torch 2.10 XPU). **Never run `uv sync` inside it** — `[tool.uv.sources]`
would reinstall the pytorch-xpu torch build. `uv pip install` also reads the
project's `[[tool.uv.index]]` (pytorch-xpu), so setup passes `--no-project`
to every `uv pip install` to ignore the project config entirely and use only
the rocm5.7 index / PyPI. In the ROCm env the RX 470 is exposed as `cuda:0`;
use `--device cuda:0` to pin AMD runs (see `docs/ROCM_RX470_SETUP.md`).

## Training loop

- Gradient clipping: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`.
- Optimizer/scheduler/loss HPs come from `training_config.json` (root) merged
  with `voices/{name}/training_config.json`, via `kokoclone_train.core.training_config`
  helpers (`get_lr_g`, `get_lr_d`, `get_betas`, `get_lambda_mel`, ...).
- End-to-end mode runs D step before G step (HiFi-GAN convention).

## Logging

```python
from kokoclone_train.core.rvc_train import _log
_log(callback, f"Epoch {epoch}/{epochs} — loss={avg_loss:.4f}", progress)
```

## Other conventions

- Module-level functions for `ProcessPoolExecutor` workers (picklable).
- Pretrained weight loads use key fallback `ckpt.get("weight", ckpt.get("model", ckpt))`
  + shape pre-filtering + `strict=False`.
- Per-voice reproducibility: save resolved HPs to `voices/{name}/training_hp.json`.
