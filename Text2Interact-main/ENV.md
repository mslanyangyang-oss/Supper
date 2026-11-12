# Environment Setup

## Prerequisites

- [conda](https://docs.conda.io/) (miniconda or anaconda)
- An NVIDIA GPU with a CUDA 11.7-compatible driver (the pinned PyTorch build is `+cu117`)

## Setup

```bash
# 1. Create the environment (Python 3.8)
conda create -n text2interact python=3.8 -y

# 2. Install ffmpeg via conda — REQUIRED for .mp4 rendering, install before pip
conda install -n text2interact -c conda-forge ffmpeg -y

# 3. Install the Python dependencies
conda run -n text2interact pip install -r requirements.txt
```

## Run inference

```bash
conda activate text2interact
python tools/infer.py
```

Outputs are written to `results/` as `.mp4` files.

## Notes

- Prompts are read from `custom_prompts.txt` (one per line).
- The checkpoint path is set in `configs/model-finetune-0.35.yaml`
  (`CHECKPOINT: checkpoints/finetune-0.35.ckpt`); place the checkpoint there.
- CLIP weights are downloaded automatically on first run.
