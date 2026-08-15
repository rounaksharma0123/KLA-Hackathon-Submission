\# SEMICON India Hackathon 2026 - KLA Track Phase 1



\*\*Team Name:\*\* NAADAN PARINDE

\*\*Problem Statement:\*\* AI-Based Restoration of Degraded Images for Semiconductor Inspection.



\## Overview

This repository contains a lightweight AI restoration pipeline designed to reconstruct clean, high-resolution semiconductor inspection images from degraded (noisy, low-resolution) inputs. 



\## Repository Structure

\* `src/`: Contains core modules for data loading (`data.py`) and model architecture (`model.py`).

\* `configs/`: Contains configuration files (`default.yaml`).

\* `checkpoints/`: Stores the trained model weights (`best\_model.pt`).

\* `train.py`: Script used to train the model.

\* `inference.py`: Standalone script for executing end-to-end inference on a directory of images.



\## Setup Instructions

1\. Clone this repository to your local machine.

2\. Ensure you have Python 3 installed.

3\. Install the required dependencies:

&#x20;  pip install -r requirements.txt



\## Input/Output Contract

\* \*\*Input:\*\* The script expects a directory containing degraded grayscale images (`NoisyLR`).

\* \*\*Output:\*\* The model processes these inputs and writes the restored images to the designated output folder.

\* \*\*Filenames:\*\* Output files will retain the exact same filenames as their corresponding input files.

\* \*\*Formats:\*\* The script saves `.npy` files for exact value scoring and `.png` previews when the `--save-png` flag is utilized.



\## Inference Command

To run the model on a folder of degraded images, execute the following command from the root directory:

python3 inference.py --checkpoint checkpoints/best\_model.pt --input-dir /path/to/NoisyLR --output-dir outputs/restored --scale 2 --save-png



\## Training Command

To reproduce the model training, use the following command:

python3 train.py --epochs 50 --batch-size 4

