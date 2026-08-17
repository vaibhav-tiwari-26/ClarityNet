"""ClarityNet submission entry script.

Usage:
    python run.py <input-dir> <output-dir>

Reads every .npy image in <input-dir>, restores it with a classical
(no-GPU, offline) pipeline, and writes one restored .npy file per input
into <output-dir> (created if missing). Outputs are grayscale arrays of
shape (H, W) or (H, W, 1) with float values in [0, 1] and no NaN/Inf.
"""

import argparse
import os
import sys
import numpy as np
import cv2
from skimage.restoration import denoise_wavelet
from scipy.ndimage import median_filter, gaussian_filter


def _to_grayscale(arr):
    """Return a 2D float array in [0, 1] from any grayscale/RGB input."""
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = np.mean(arr[..., :3], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Could not reduce input of shape {arr.shape} to 2D grayscale.")
    arr = np.asarray(arr, dtype=np.float64)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _normalize(arr):
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi - lo > 1e-12:
        arr = (arr - lo) / (hi - lo)
    return np.clip(arr, 0.0, 1.0)


def restore(arr):
    """Classical restoration: denoise + deblur + contrast stretch."""
    img = _to_grayscale(arr)

    med = median_filter(img, size=3)

    denoised = denoise_wavelet(
        med,
        wavelet="db4",
        mode="soft",
        rescale_sigma=True,
        multichannel=False,
    )

    kernel = cv2.getGaussianKernel(5, 1.2)
    kernel = kernel @ kernel.T
    deblurred = cv2.filter2D(denoised, -1, kernel)

    result = _normalize(deblurred)

    p_low, p_high = np.percentile(result, (2, 98))
    if p_high > p_low + 1e-12:
        result = np.clip((result - p_low) / (p_high - p_low), 0.0, 1.0)

    result = gaussian_filter(result, sigma=0.5)
    return _normalize(result)


def main():
    parser = argparse.ArgumentParser(description="ClarityNet image restoration")
    parser.add_argument("input_dir", help="Directory containing .npy input images")
    parser.add_argument("output_dir", help="Directory where restored .npy files are written")
    args = parser.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_dir

    if not os.path.isdir(in_dir):
        sys.exit(f"Input directory does not exist: {in_dir}")

    os.makedirs(out_dir, exist_ok=True)

    inputs = sorted(f for f in os.listdir(in_dir) if f.endswith(".npy"))
    if not inputs:
        sys.exit(f"No .npy files found in {in_dir}")

    for name in inputs:
        in_path = os.path.join(in_dir, name)
        out_path = os.path.join(out_dir, name)
        arr = np.load(in_path)
        restored = restore(arr)
        np.save(out_path, restored)
        print(f"Restored: {name} -> {out_path}")

    print(f"Done. {len(inputs)} file(s) written to {out_dir}")


if __name__ == "__main__":
    main()