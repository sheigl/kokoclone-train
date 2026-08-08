#!/usr/bin/env python3
"""Runtime verification for the ROCm 5.7 environment (AMD RX 470 / gfx803).

Run inside the ROCm venv (``.venv-rocm57``):

    python tools/check_rocm.py

Prints the torch version, whether the RX 470 is exposed through the CUDA API,
the number of devices, each device's name, and what the pipeline's own device
selection (``detect_device``) would pick — so you can confirm the card shows
up as ``cuda:0``. Exits non-zero with actionable hints when the environment
is not usable.
"""

import sys


def main() -> int:
    # Imported here (not at module top) so a broken env prints a clean message
    # instead of a traceback, and so this file can be compile/import checked in
    # CI without torch installed.
    import torch

    print(f"torch version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print()
        print("[FAIL] CUDA is not available.")
        print("The RX 470 must appear as a CUDA device when using the ROCm build of torch.")
        print("Check:")
        print("  - correct venv active? (expect .venv-rocm57)  `which python`")
        print("  - torch installed from the rocm5.7 index?      `pip show torch`")
        print("  - system ROCm userspace installed?             docs/ROCM_RX470_SETUP.md")
        return 1

    n = torch.cuda.device_count()
    print(f"torch.cuda.device_count(): {n}")
    for i in range(n):
        print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
    if n == 0:
        print()
        print("[FAIL] device_count() is 0 despite is_available()==True.")
        print("The ROCm build sees the runtime but no GPU. Check the card is")
        print("visible to the system (`lsmod | grep amdgpu`, `rocminfo`).")
        return 1

    # Tie-in with story-001 device selection: what the pipeline would pick.
    from kokoclone_train.core.rvc_assets import detect_device

    d = detect_device("auto")
    print(f"kokoclone-train detect_device('auto') -> {d}")
    if d.type != "cuda":
        print()
        print(f"[WARN] auto-detect returns '{d}' (expected cuda). Use --device cuda:0 explicitly.")
        return 1

    print()
    print("[OK] RX 470 exposed as a CUDA device; kokoclone-train will use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
