# Discovery Index

## Milestone: AMD RX 470 (ROCm) + Device Selection

Goal: enable full training runs on the AMD RX 470 via a code-side device-selection
option (`--device`) and a separate ROCm 5.7 environment (torch 2.3 rocm5.7), without
disturbing the Intel XPU primary environment. See `docs/ROCM_RX470_SETUP.md` (created
by story 002) and `docs/ARCHITECTURE.md` §Device Handling for background.

| # | Story | File | Plan | Priority | Dependencies | Status |
|---|-------|------|------|----------|--------------|--------|
| 1 | Device selection option in the training pipeline | story-001-device-selection.md | plans/story-001-device-selection-plan.md | High | none | ✅ |
| 2 | ROCm 5.7 environment setup for the RX 470 (guide + tooling) | story-002-rocm57-env-setup.md | plans/story-002-rocm57-env-setup-plan.md | High | 1 (recommended order; functionally independent) | ✅ |

## Status Legend

- ⏳ Planned (stories + plans written, not yet implemented)
- 🔨 In Progress (Implement agent working)
- ✅ Done (implemented and verified)
