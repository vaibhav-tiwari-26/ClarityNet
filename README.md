# ClarityNet — KLA Problem Statement: AI-Based Restoration of Degraded Images

Physics-informed, two-stage restoration pipeline (Poisson-Gaussian denoising
+ sub-pixel CNN super-resolution) for semiconductor inspection imagery.

**Team:** ClarityNet — Kamal Sharma (Lead), Hariharan R, Vaibhav Tiwari | VIT Chennai

## Folder Structure

```
ClarityNet/
├── run.py                     # Entry point
├── requirements.txt
├── README.md
└── models/
    ├── model_defs.py          # Stage 1 (U-Net denoiser) + Stage 2 (ESPCN SR) architectures
    ├── stage1_denoiser.pth    # Stage 1 weights
    └── stage2_sr.pth          # Stage 2 weights
```

## Setup

Requires Python 3.9+ and an NVIDIA GPU (CUDA) for accelerated inference;
falls back to CPU automatically if no GPU is available. No internet
access, API keys, or manual configuration are required at run time.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run.py <input-dir> <output-dir>
```

- `<input-dir>`: folder containing `.npy` files (grayscale arrays,
  shape `(H, W)` or `(H, W, 1)`, values in `[0, 1]` or `[0, 255]`).
- `<output-dir>`: created automatically if it does not exist.

For every `<name>.npy` in `<input-dir>`, the script writes a restored
`<name>.npy` to `<output-dir>` — same filename, grayscale array of
shape `(H, W)`, `float32`, values clipped to `[0, 1]`, with no NaN or
Inf values. The output resolution is `2x` the input resolution
(the super-resolution stage's upscale factor).

### Example

```bash
python run.py ./sample_inputs ./sample_outputs
```

```
Using device: cuda
Found 12 input file(s). Restoring...
  img_001.npy: (128, 128) -> (256, 256)  saved to ./sample_outputs/img_001.npy
  ...
Done.
```

## Pipeline

1. **Stage 1 — Physics-informed denoising:** U-Net with residual +
   attention blocks, trained to remove Poisson (shot) + Gaussian
   (read) noise matching real inspection sensor behavior.
2. **Stage 2 — Detail-preserving super-resolution:** compact
   sub-pixel convolution (ESPCN-style) network, upscaling by 2x
   while preserving defect-relevant edges.

Full technical writeup: see `docs/ClarityNet_Round1_Submission.pdf`
in the project's main repository.

## Model Weights

Trained checkpoints are shipped in `models/`:

- `stage1_denoiser.pth` — Stage 1 U-Net denoiser weights.
- `stage2_sr.pth` — Stage 2 ESPCN super-resolution weights.

`run.py` loads both at startup with no internet access, API keys, or
manual configuration required.
