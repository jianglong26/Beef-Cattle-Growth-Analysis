# Beef Cattle Detection and Growth Curve Analysis

A reorganized GitHub-ready project for beef cattle monitoring from aerial images.

## Overview

This repository organizes the research workflow into a clean, upload-ready structure:

1. Detect standing cattle with YOLOv11.
2. Pass standing-cattle bounding boxes to SAM 2.1 for instance segmentation.
3. Post-process masks and extract per-instance morphology features (for example width/length ratio).
4. Analyze growth trajectories and support recommended selling-date decisions.

## Related Paper

DOI: [10.1016/j.compag.2026.111559](https://doi.org/10.1016/j.compag.2026.111559)

Recommended badge:

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.compag.2026.111559-blue.svg)](https://doi.org/10.1016/j.compag.2026.111559)

##Checkpoints 
https://huggingface.co/Arvin26/Cattle-detection-model

## Repository Structure

```text
.
├─ src/
│  ├─ training/
│  ├─ inference/
│  ├─ evaluation/
│  ├─ visualization/
│  ├─ configs/
│  └─ utils/
├─ scripts/
│  └─ data_preparation/
├─ experiments/
├─ docs/
├─ data/                # dataset placeholder (not committed)
├─ checkpoints/         # checkpoint placeholder (not committed)
├─ models/              # model artifact placeholder (not committed)
├─ pretrained_models/   # pretrained model placeholder (not committed)
├─ outputs/             # run outputs placeholder (not committed)
└─ roi_visualization/
```

## Key Script Mapping

- Training: `src/training/train.py`
- Inference: `src/inference/inference_and_statistics.py`, `src/inference/video_inference.py`
- Evaluation: `src/evaluation/evaluation.py`, `src/evaluation/evaluate_best_model_newdata.py`, `src/evaluation/evaluate_best_model_testset_nogtbox.py`
- Visualization: `src/visualization/training_results_visualization.py`
- Data preparation tools: `scripts/data_preparation/*.py`
- Experimental analyses: `experiments/*.py`

## Quick Start

### 1) Create environment

```bash
conda create -n cattle python=3.10 -y
conda activate cattle
pip install -r requirements.txt
```

### 2) Prepare data and models

- Put your dataset according to your own storage policy (for example outside this repository), and configure paths in files under `src/configs/`.
- Put checkpoints and pretrained weights in your preferred local locations.

### 3) Run core scripts

```bash
python src/training/train.py
python src/inference/inference_and_statistics.py
python src/evaluation/evaluation.py
```

## Data and Weight Policy

This repository is code-first and publication-ready. Large assets are intentionally excluded from version control:

- raw datasets
- training checkpoints
- heavy model binaries
- generated outputs and logs

See placeholder README files in `data/`, `checkpoints/`, `models/`, `pretrained_models/`, and `outputs/`.
