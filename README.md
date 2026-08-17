# ClarityNet

Image restoration submission for the grayscale `.npy` restoration challenge.

## Structure

```
ClarityNet/
├── run.py            # entry script
├── requirements.txt  # pinned dependencies
├── README.md
└── models/           # model weights (not required for the classical pipeline)
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9–3.11. All dependencies are bundled via pip; no internet access,
API keys, or additional downloads are needed at runtime.

## Execution

```bash
python run.py <input-dir> <output-dir>
```

- Reads every `.npy` file from `<input-dir>`.
- Creates `<output-dir>` if it does not exist.
- Writes one restored `.npy` file per input, keeping the same filename.

## Output guarantees

- Grayscale arrays of shape `(H, W)`.
- Float values in `[0, 1]` with no `NaN` or `Inf`.
- Same target resolution as each input.

## Method

Classical (no-GPU) restoration pipeline in `run.py::restore`:

1. Convert input to 2D grayscale float array in `[0, 1]`.
2. Median filter (speckle/impulse noise removal).
3. Wavelet denoising (`skimage.restoration.denoise_wavelet`, `db4`).
4. Gaussian deblur via unsharp/gaussian convolution.
5. Contrast stretch using 2nd–98th percentiles.
6. Mild Gaussian smoothing and final normalization.

The `models/` directory is included for compatibility; the classical pipeline
requires no weights and runs fully offline on CPU.