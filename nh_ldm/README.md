# Neuro-Hybrid Latent Diffusion Model (NH-LDM)

This project implements a system to reconstruct high-resolution images from fMRI brain activity data using a dual-stream architecture (Semantic + Structural) and Stable Diffusion v1.4.

## Project Structure

```
nh_ldm/
├── config.yaml          # Configuration file
├── requirements.txt     # Python dependencies
└── project/
    ├── data/            # Data loading and preprocessing
    ├── models/          # Ridge regression and decoder models
    ├── evaluation/      # Metrics and visualization
    ├── train.py         # Training script
    └── inference.py     # Reconstruction pipeline
```

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data:**
    - Place the Natural Scenes Dataset (NSD) files in the directory specified in `config.yaml` (default: `./data/nsd`).

3.  **Configuration:**
    - Adjust hyperparameters and paths in `config.yaml` as needed.

## Usage

### Training
```bash
python -m project.train
```

### Inference / Reconstruction
```bash
python -m project.inference
```
