# AI-Based Restoration of Degraded Images for Semiconductor Inspection

SEMICON India Hackathon 2026 — KLA Track, Phase 1 submission. Team **NAADAN PARINDE**, Chandigarh University.

Restores noisy, low-resolution semiconductor inspection images (`NoisyLR`) back to clean images at the
original ground-truth resolution, using a lightweight residual U-Net on top of a bicubic upscale.

**Pipeline:** `NoisyLR image` → bicubic upscale (2x) → `SmallResidualUNet` residual correction → clip to
`[0, 1]` → restored output, same filename as input.

## Problem

Semiconductor inspection images are degraded by two combined effects, applied in an undisclosed order:
speckle noise plus additive Gaussian noise, and downsampling (a 256x256 ground-truth image shrunk to
128x128, for example). Given only the degraded image, the task is to produce a restored image at the
original ground-truth resolution, scored on PSNR, SSIM, LPIPS, and end-to-end runtime against a hidden
ground truth.

## Approach

Bicubic upscaling handles resizing deterministically, so the network only has to learn denoising and
detail correction on top of it, rather than resizing and restoration jointly. This keeps the model small,
fast to train, and easy to explain.

- **Model:** `SmallResidualUNet`, a lightweight residual U-Net. Input is the bicubic-upscaled grayscale
  `NoisyLR` image; the network predicts a residual correction rather than the full image from scratch;
  output is clipped to `[0, 1]`.
- **Loss:** L1 pixel loss
- **Optimizer:** AdamW
- **Epochs / batch size:** 50 / 4
- **Augmentation:** horizontal flip, vertical flip, 90-degree rotations
- **Checkpoint selection:** best validation PSNR
- **Train/validation split:** automatic 90/10, fixed seed 42
- **Hardware:** MacBook Pro, Apple M5 Pro, 24 GB RAM, trained on Apple MPS

## Requirements

- Python 3.10+
- PyTorch
- See `requirements.txt` for the full pinned list

## Installation

```
git clone https://github.com/rounaksharma0123/KLA-Hackathon-Submission.git && cd KLA-Hackathon-Submission
```

```
python3 -m venv venv && source venv/bin/activate
```

```
python3 -m pip install -r requirements.txt
```

## Dataset layout

```
data/
  train/
    GT/          # clean ground-truth images
    NoisyLR/     # matching degraded images, same filenames as GT
  NoisyLR/       # public or hidden test images, degraded only, no GT
```

Supported formats: `.npy`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`. The train/validation split is
created automatically from `data/train/` (90/10, fixed seed 42) — no separate validation folder is
required, and the same split logic in `src/data.py` is what evaluation reuses, so metrics below are always
computed on data the model never trained on.

## Training

```
python3 train.py --epochs 50 --batch-size 4
```

Logs per-epoch loss and validation PSNR, and saves the best checkpoint to `checkpoints/best_model.pt`.

## Inference

`inference.py` is standalone: it takes an input directory and an output directory, and runs without any
manual source-code edits.

```
python3 inference.py --checkpoint checkpoints/best_model.pt --input-dir data/NoisyLR --output-dir outputs/restored --scale 2 --save-png
```

Bicubic-only baseline, no trained weights:

```
python3 inference.py --input-dir data/NoisyLR --output-dir outputs/bicubic --scale 2 --save-png
```

If input is already full resolution, use `--scale 1` instead of `--scale 2`.

| Flag | Required | Description |
|---|---|---|
| `--checkpoint` | No | Path to trained weights; omit for the bicubic-only baseline |
| `--input-dir` | Yes | Folder of degraded images to restore |
| `--output-dir` | Yes | Folder where restored images are written |
| `--scale` | Yes | `2` for the trained setup (128x128 → 256x256), `1` if input is already full resolution |
| `--save-png` | No | Also save a `.png` preview alongside the raw `.npy` output |

Output images keep the same base filename as the input, are clipped to `[0, 1]` before saving, run on
GPU/MPS with automatic CPU fallback, and the script prints total end-to-end runtime (read, preprocess,
model, postprocess, save).

## Evaluation

<!-- TODO: confirm this matches the actual CLI in src/metrics.py / src/visualize.py -->
```
python3 src/metrics.py
python3 src/visualize.py
```

`src/metrics.py` computes PSNR, SSIM, and LPIPS for both `outputs/bicubic` and `outputs/restored` against
the held-out validation ground truth, and writes `results/metrics.json`. `src/visualize.py` generates
before/after comparison grids into `results/samples/`.

## Project structure

```
repository/
  README.md
  requirements.txt
  train.py
  inference.py
  configs/
    default.yaml
  src/
    __init__.py
    data.py          # dataset loading, pairing, normalization, augmentation
    model.py          # SmallResidualUNet architecture
    metrics.py         # PSNR / SSIM / LPIPS evaluation, baseline vs. final
    visualize.py        # before/after visual grid generation
  checkpoints/
    best_model.pt     # final trained weights
  outputs/
    bicubic/           # baseline outputs
    restored/           # final model outputs
  results/
    metrics.json         # PSNR / SSIM / LPIPS, baseline vs. final
    samples/              # before/after visual grids, success + failure cases
```

## Results

Best validation PSNR during training (used for checkpoint selection): **25.49 dB**.

Full evaluation on the held-out validation split (90/10, seed 42, never seen during training):

| Metric | Bicubic Baseline | Final Restored Model | Improvement |
|---|---|---|---|
| PSNR (higher is better) | 22.93 dB | **28.28 dB** | +5.35 dB |
| SSIM (higher is better) | 0.5372 | **0.7597** | +0.2225 |
| LPIPS (lower is better) | 0.4655 | **0.3972** | −0.0683 |

The model improves on the bicubic baseline across all three metrics — a >5 dB PSNR gain, a substantial
structural-similarity improvement, and a lower (better) perceptual distance.

The validation set is an automatic 90/10 split (seed 42) never used during training. The public test set
of 400 `NoisyLR` images (128x128 → 256x256) was processed end to end without errors. Two successful
restorations and one representative failure case are in `results/samples/`.

## Limitations

- **High-frequency texture smoothing:** while effective at suppressing severe speckle and Gaussian noise,
  the model can be overly aggressive on fine, complex texture patterns, softening detail that a human
  inspector might still want to see — visible in the failure case in `results/samples/`.
- Trained for only 50 epochs on a single hardware setup, under hackathon time pressure — not
  extensively hyperparameter-tuned.
- Only L1 loss was used, no perceptual or adversarial term, which likely explains why the LPIPS
  improvement is smaller in relative terms than the PSNR/SSIM gains.
- Augmentation limited to flips and 90-degree rotations; no synthetic re-degradation was explored.
- Trained for a fixed 2x resolution gap; behaviour at other scale factors is untested.
- No external pretrained models or public datasets were used — training relied solely on the official
  KLA-provided pairs, which limits generalization compared to larger-scale pretraining.

## Next steps

- Add SSIM or edge-aware loss terms to reduce high-frequency over-smoothing and improve LPIPS.
- Extend training and run a proper hyperparameter sweep.
- Benchmark against a second architecture for comparison.
- Optimize inference throughput (batching, mixed precision) for the H100 evaluation benchmark.

## External resources

None used. All code — model, training loop, and data loader — was written by the team specifically for
this challenge, using only the official KLA-provided dataset.

## License

Submitted for SEMICON India Hackathon 2026, KLA Track, Phase 1 evaluation.

---

Team NAADAN PARINDE — SEMICON India Hackathon 2026, KLA Track
