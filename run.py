"""
ClarityNet — KLA Problem Statement: AI-Based Restoration of Degraded Images

Entry point required by the submission spec.

Usage:
    python run.py <input-dir> <output-dir>

Behavior:
    - Reads every .npy file from <input-dir>.
    - Creates <output-dir> if it does not already exist.
    - Runs each input through the two-stage restoration pipeline
      (Stage 1: Poisson-Gaussian denoiser, Stage 2: sub-pixel CNN SR).
    - Writes one restored .npy file per input file, using the same
      filename, into <output-dir>.
    - Output arrays are grayscale, shape (H, W), float32, values
      clipped to [0, 1], with no NaN/Inf values.
    - Runs fully offline: no internet access, API keys, external
      downloads, or manual configuration required. Uses GPU
      automatically if available, otherwise falls back to CPU.

NOTE ON MODEL WEIGHTS:
    The checkpoints shipped in models/ (stage1_denoiser.pth,
    stage2_sr.pth) are RANDOMLY INITIALIZED placeholders. They verify
    that the full I/O pipeline (reading .npy, running both stages,
    writing valid .npy outputs of the correct shape/range) is
    spec-compliant end-to-end. They have not been trained yet, so
    restoration quality is not representative of the final model.
    Swap in trained weights at the same paths once training completes
    — no other code changes are required.
"""

import os
import sys
import glob

import numpy as np
import torch

from models.model_defs import build_pipeline

STAGE1_WEIGHTS = os.path.join(os.path.dirname(__file__), "models", "stage1_denoiser.pth")
STAGE2_WEIGHTS = os.path.join(os.path.dirname(__file__), "models", "stage2_sr.pth")
UPSCALE_FACTOR = 2


def load_pipeline(device):
    denoiser, sr = build_pipeline(device=device, upscale_factor=UPSCALE_FACTOR)

    denoiser.load_state_dict(torch.load(STAGE1_WEIGHTS, map_location=device))
    sr.load_state_dict(torch.load(STAGE2_WEIGHTS, map_location=device))

    denoiser.eval()
    sr.eval()
    return denoiser, sr


def to_model_input(arr):
    """
    Normalize an arbitrary (H,W) or (H,W,1) input array into a
    (1, 1, H, W) float32 tensor in [0, 1] for the network.
    """
    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim != 2:
        raise ValueError(f"Expected (H,W) or (H,W,1) array, got shape {arr.shape}")

    # Defensive normalization: if values look like they're in [0, 255],
    # rescale to [0, 1]. Otherwise assume already in [0, 1].
    if arr.max() > 1.0 + 1e-6:
        arr = arr / 255.0

    arr = np.clip(arr, 0.0, 1.0)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return tensor


def restore_one(denoiser, sr, arr, device):
    tensor = to_model_input(arr).to(device)

    with torch.no_grad():
        denoised = denoiser(tensor)
        denoised = torch.clamp(denoised, 0.0, 1.0)
        restored = sr(denoised)
        restored = torch.clamp(restored, 0.0, 1.0)

    out = restored.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

    # Final safety pass: no NaN/Inf, values strictly within [0, 1]
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    out = np.clip(out, 0.0, 1.0)
    return out


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir, output_dir = sys.argv[1], sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    denoiser, sr = load_pipeline(device)

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    if not input_files:
        print(f"No .npy files found in {input_dir}")
        sys.exit(0)

    print(f"Found {len(input_files)} input file(s). Restoring...")

    for path in input_files:
        filename = os.path.basename(path)
        arr = np.load(path)

        restored = restore_one(denoiser, sr, arr, device)

        out_path = os.path.join(output_dir, filename)
        np.save(out_path, restored)
        print(f"  {filename}: {arr.shape} -> {restored.shape}  saved to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
