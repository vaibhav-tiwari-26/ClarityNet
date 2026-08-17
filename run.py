"""ClarityNet entry script.

Usage:
    python run.py <input-dir> <output-dir>

Two-stage restoration (Poisson-Gaussian denoising + 2x sub-pixel CNN
super-resolution). Runs fully offline; uses CUDA when available and
falls back to CPU automatically.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from models.model_defs import build_stage1, build_stage2


def _to_grayscale(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3):
        arr = arr[..., 0] if arr.shape[-1] == 1 else np.mean(arr[..., :3], axis=-1)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Cannot reduce input of shape {arr.shape} to 2D grayscale")
    arr = np.asarray(arr, dtype=np.float32)
    if float(np.nanmax(arr)) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _clean(t):
    t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.clamp(t, 0.0, 1.0)


def _load(model, path, name):
    if not os.path.isfile(path):
        print(f"Warning: {name} weights not found ({path}); using random init")
        return
    try:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        print(f"Loaded {name} weights from {path}")
    except Exception as exc:
        print(f"Warning: could not load {name} weights ({exc}); using random init")


def restore(x_np, stage1, stage2, device):
    img = _to_grayscale(x_np)
    x = torch.from_numpy(img).view(1, 1, *img.shape).to(device)

    with torch.no_grad():
        denoised = _clean(stage1(x))
        h, w = denoised.shape[-2:]
        if h % 2 or w % 2:
            denoised = F.pad(denoised, (0, w % 2, 0, h % 2))
        up = _clean(stage2(denoised))

    return up.squeeze().cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="ClarityNet image restoration")
    parser.add_argument("input_dir", help="Directory containing .npy input images")
    parser.add_argument("output_dir", help="Directory where restored .npy files are written")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"Input directory does not exist: {args.input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    stage1 = build_stage1().to(device).eval()
    stage2 = build_stage2().to(device).eval()

    _load(stage1, os.path.join(models_dir, "stage1_denoiser.pth"), "Stage 1")
    _load(stage2, os.path.join(models_dir, "stage2_sr.pth"), "Stage 2")

    os.makedirs(args.output_dir, exist_ok=True)

    inputs = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".npy"))
    if not inputs:
        sys.exit(f"No .npy files found in {args.input_dir}")

    print(f"Found {len(inputs)} input file(s). Restoring...")
    for name in inputs:
        in_path = os.path.join(args.input_dir, name)
        out_path = os.path.join(args.output_dir, name)
        arr = np.load(in_path)
        out = restore(arr, stage1, stage2, device)
        np.save(out_path, out)
        print(f"  {name}: {arr.shape} -> {out.shape}  saved to {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
