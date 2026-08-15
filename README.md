# AI-Based Restoration of Degraded Images for Semiconductor Inspection

**SEMICON India Hackathon 2026 — KLA Track (Phase 1 Submission)**  
**Team Name:** NAADAN PARINDE  
**Problem Statement:** Restoration of low-resolution, noisy semiconductor defect inspection images to clean, high-resolution ground truth.

---

## 📌 Executive Summary

Semiconductor inspection images require clean, sharp details; however, noise and downsampling can hide critical defects. The degraded input images provided in this track contain a mix of speckle noise, Gaussian noise, and downsampling.

This repository provides an end-to-end, lightweight AI restoration pipeline designed to take these degraded inputs and output cleaner, sharper images at the expected ground-truth resolution. The pipeline is optimized for image quality (PSNR, SSIM, LPIPS) and fast end-to-end inference runtime.

---

## 🚀 Key Features

* **Lightweight Restoration Engine:** Compact CNN/U-Net architecture optimized for minimal runtime overhead and GPU/CPU portability.
* **Deterministic Input/Output Contract:** Strictly preserves source filenames across `.npy` raw feature arrays and `.png` visual inspection maps.
* **Dynamic Resolution Scaling:** Supports arbitrary test batch sizes and automated spatial upscaling (2x factor baseline).
* **Zero Hardcoded Dependencies:** Fully parameterized via CLI flags and YAML configuration files for frictionless judge evaluation.
* **Evaluation Metric Ready:** Built-in support for standardized PSNR, SSIM, and LPIPS benchmarking.

---

## 📁 Project Architecture & Structure
KLA-Hackathon-Submission/
|-- configs/
|   -- default.yaml          # Hyperparameters, batch sizes, and relative paths |-- checkpoints/ |   -- best_model.pt         # Pretrained model weights checkpoint
|-- src/
|   |-- init.py           # Package module initializer
|   |-- data.py               # Dataset loader, paired transforms & tensor normalization
|   -- model.py              # Restoration network architecture definition |-- outputs/                  # Automated target directory for evaluation outputs |   |-- bicubic/              # Baseline upscaled comparison outputs |   -- restored/             # Model-restored final predictions (.npy and .png)
|-- train.py                  # Training pipeline with validation tracking
|-- inference.py              # Standalone evaluation & inference CLI tool
|-- requirements.txt          # Minimal pinned environment dependencies
`-- README.md                 # Project documentation and reproduction guide

---

## ⚙️ System Requirements

* **Python:** 3.10 or 3.11 (recommended)
* **Hardware:** CUDA-capable GPU (optional, automatic fallback to CPU)
* **Memory:** ~2 GB RAM / VRAM minimum for standard inspection batches

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/rounaksharma0123/KLA-Hackathon-Submission.git](https://github.com/rounaksharma0123/KLA-Hackathon-Submission.git)
cd KLA-Hackathon-Submission
```

2. Environment Setup
Windows (PowerShell / Command Prompt): ```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt ```

macOS / Linux: ```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt ```

🧪 Inference & Evaluation Guide
The standalone inference script processes raw input folders containing degraded images, runs end-to-end tensor normalization, executes the model forward pass, and persists restored predictions.

Standard Inference Command ```
python3 inference.py --checkpoint checkpoints/best_model.pt --input-dir /path/to/NoisyLR --output-dir outputs/restored --scale 2 --save-png ```

Argument Reference
--checkpoint: Path to trained weight file (checkpoints/best_model.pt).

--input-dir: Absolute or relative path to folder containing degraded input images (NoisyLR).

--output-dir: Destination path where restored images will be written.

--scale: Spatial upscaling factor (Default: 2).

--save-png: Optional flag to output visual .png previews alongside .npy arrays.

Input/Output Contract
Input Directory: Contains arbitrary degraded grayscale images (e.g., sample_001.png or sample_001.npy).

Output Directory: Predictions are written retaining the exact original filenames.

Value Range: Output arrays are clipped to [0.0, 1.0] range for exact evaluation scoring.

🏋️ Training Reproduction
To retrain or fine-tune the restoration model on paired ground-truth inspection datasets: ```
python3 train.py --epochs 50 --batch-size 4 ```

Hyperparameters such as base channels, learning rate, and validation split fraction can be customized directly in configs/default.yaml.

📊 Technical Approach & Method
Preprocessing & Tensor Normalization: Robust range scaling handling float/integer arrays safely without assuming prior [0, 1] bounding.

Spatial Alignment & Upscaling: Bicubic / feature-space pre-upsampling mapping degraded inputs into target ground-truth coordinate space.

Deep Residual Denoising: Multiscale feature extraction penalizing structural and pixel-wise deviations via combined L1 and perceptual/SSIM objective formulations.

Post-Processing: Range clipping [0, 1] to prevent artifact overflow prior to metric computation.

⚠️ Known Constraints & Edge Cases
Strict Grayscale Modality: The network processes single-channel defect maps; multi-channel RGB inputs are converted or treated as single-channel inspection frames.

Extreme Downsampling Degradation: In severe subsampling scenarios, extreme sub-surface structural textures may rely on prior regularized smoothing.

👥 Team NAADAN PARINDE — Work AllocationMemberPrimary ResponsibilitiesDeliverablesMember 1Repository Architecture, Packaging, Dependency Isolation, CLI ContractPublic Repo, README.md, requirements.txt, Packaging CheckMember 2Dataset Profiling, Train/Val Split Formulation, Quantitative MetricsPaired Split Validation, PSNR/SSIM/LPIPS Baseline BenchmarksMember 3Neural Architecture Design, Loss Formulations, Training & Checkpointsmodel.py, train.py, Model Checkpoints (best_model.pt)Member 4Solution Presentation, Technical Documentation, Video Walkthrough8–9 Slide Presentation Deck, 5-Minute Solution Demo Video

📜 License & Acknowledgments
Developed for academic and competitive evaluation under SEMICON India Hackathon 2026 — KLA Track.
