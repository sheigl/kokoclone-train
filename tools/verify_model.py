"""Quick inference check — verify a trained checkpoint produces non-silent mel output."""

import argparse
import json
from pathlib import Path

import torch

from kokoclone_train.core.rvc_assets import detect_device
from kokoclone_train.core.rvc_models.models import build_rvc_model


def main():
    parser = argparse.ArgumentParser(description="Verify RVC checkpoint produces valid output")
    parser.add_argument("--name", required=True, help="Voice name (looks in voices/{name}/)")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Explicit checkpoint path. Defaults to voices/{name}/v1.pth or latest epoch checkpoint.",
    )
    args = parser.parse_args()

    device = detect_device()
    voice_dir = Path("voices") / args.name

    # Load config
    config_path = voice_dir / "config.json"
    if not config_path.exists():
        print(f"[FAIL] No config at {config_path}")
        return 1

    with open(config_path) as f:
        config = json.load(f)

    print(f"[INFO] Config: input_size={config['input_size']}, output_size={config['output_size']}, "
          f"sample_rate={config['sample_rate']}, hop_length={config['hop_length']}, f0={config['f0']}")

    # Build model
    model = build_rvc_model(config).to(device)
    print(f"[INFO] Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Load checkpoint
    ckpt_path = Path(args.checkpoint) if args.checkpoint else voice_dir / "v1.pth"
    if not ckpt_path.exists():
        # Try latest epoch checkpoint
        temp_dir = Path("training_temp") / args.name
        epochs = sorted(temp_dir.glob(f"{args.name}_epoch_*.pth"), key=lambda p: int(p.stem.split("_epoch_")[1]))
        if epochs:
            ckpt_path = epochs[-1]
            print(f"[INFO] Using latest epoch checkpoint: {ckpt_path}")
        else:
            print(f"[FAIL] No checkpoint found at {voice_dir / 'v1.pth'} or training_temp/")
            return 1

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("weight", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:3]}...")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")

    model.eval()

    # Run inference with synthetic input
    T = 493  # typical chunk length at ~133fps for 3.7s audio
    B = 1
    C = config["input_size"]

    content = torch.randn(B, T, C, device=device)
    f0 = torch.randint(1, 255, (B, T), device=device) if config.get("f0") else None

    with torch.no_grad():
        output = model.infer(content, f0=f0)

    # Validate output
    print(f"\n--- Output Analysis ---")
    print(f"Shape: {output.shape} (expected [{B}, {config['output_size']}, {T}])")

    shape_ok = list(output.shape) == [B, config["output_size"], T]
    mean_val = output.abs().mean().item()
    std_val = output.std().item()
    max_val = output.abs().max().item()
    nonzero_pct = (output.abs() > 1e-6).float().mean().item() * 100

    print(f"Mean abs:     {mean_val:.4f}")
    print(f"Std:          {std_val:.4f}")
    print(f"Max abs:      {max_val:.4f}")
    print(f"Non-zero %:   {nonzero_pct:.1f}%")

    checks = [
        ("Shape matches", shape_ok),
        ("Mean > 0.001", mean_val > 0.001),
        ("Std > 0.001", std_val > 0.001),
        ("Non-zero > 95%", nonzero_pct > 95),
    ]

    print()
    all_pass = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {label}")

    if all_pass:
        print("\n[OK] Model produces valid mel spectrogram output.")
        return 0
    else:
        print("\n[FAIL] Model output is suspicious — may need retraining or architecture fix.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
