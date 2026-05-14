# container-navsim-ego-mlp <!-- omit in toc -->

🌐 **Language**: 🇺🇸 [English](README.md) | 🇯🇵 [日本語](README.ja.md)

A container setup for training and evaluating [NAVSIM](https://github.com/autonomousvision/navsim) (Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking) on SageMaker AI ML Pipeline.

For an overview of NAVSIM itself (Pseudo-Simulation, PDM Score, etc.) and the container design approach in this project, see the [NAVSIM Autonomous Driving Simulation Guide](../../docs/navsim-guide.md).

- [What this container implements](#what-this-container-implements)
- [Directory structure](#directory-structure)
- [Dataset](#dataset)
- [Quick start](#quick-start)
- [Training parameters](#training-parameters)
- [Evaluation metrics](#evaluation-metrics)
- [Customization tips](#customization-tips)
- [References](#references)

## What this container implements

Of the multiple baseline agents provided by NAVSIM, this container runs EgoStatusMLP on SageMaker.

EgoStatusMLP is a lightweight MLP model that predicts the future 4-second trajectory using only ego state (velocity, acceleration, driving command) — without any sensor input like cameras or LiDAR. It serves as a baseline that indicates the upper bound of "how far you can go without sensors."

The training and evaluation flow is as follows.

```
Dataset preparation (prepare_dataset.sh)
  ↓ Download navsim mini split → Extract features → Upload to S3
SageMaker Training Job (train.py)
  ↓ Train EgoStatusMLP with L1 loss → Save model.pth to S3
SageMaker Processing Job (evaluate.py)
  ↓ Run inference on test data → Compute ADE / FDE / PDM Score
Output evaluation.json to S3
```

## Directory structure

Each file has the following role.

| File | Role |
|------|------|
| `Dockerfile` | Lightweight BYOC image based on PyTorch DLC (Python 3.11). The navsim devkit is not installed (reason below) |
| `train.py` | Training script for EgoStatusMLP. Runs as the entry_point of a SageMaker Training Job |
| `evaluate.py` | Evaluation script for the trained model. Runs as a SageMaker Processing Job |
| `requirements.txt` | Additional Python packages installed into the container |
| `scripts/prepare_dataset.sh` | One-shot script that clones the navsim repository, downloads the mini split, extracts features, balances the dataset, and uploads to S3 |
| `scripts/extract_features.py` | Converts EgoStatus features and trajectory targets to npz format using navsim's SceneLoader |
| `scripts/balance_dataset.py` | Fully equalizes the command distribution (LEFT / FORWARD / RIGHT) in the training data |
| `data/README.md` | Data format description |

`train.py` and `evaluate.py` are not copied into the Dockerfile; instead, they are injected into the container via the SageMaker SDK's `entry_point` + `source_dir` mechanism. Therefore, you do not need to rebuild the container when changing these scripts.

## Dataset

NAVSIM uses the [nuPlan](https://www.nuscenes.org/nuplan) / [OpenScene](https://github.com/OpenDriveLab/OpenScene) datasets. This container uses the minimal mini split (about 5 GB).

Review the dataset [license](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) in advance.

`prepare_dataset.sh` automatically creates a Python 3.9 conda environment (`navsim-py39`), installs the navsim devkit, and extracts features. This is a workaround to avoid dependency conflicts with the SageMaker AI Notebook (Python 3.11).

1. Create conda environment `navsim-py39` (Python 3.9) — skipped on subsequent runs
2. Install navsim devkit + nuplan-devkit
3. Download the mini split data
4. Extract EgoStatus features (velocity, acceleration, driving command) and trajectory targets using navsim's SceneLoader
5. Balance the command distribution (LEFT / FORWARD / RIGHT) with `balance_dataset.py`
6. Split into train/test in npz format and upload to the S3 dataset bucket

### About dataset balancing

The OpenScene dataset has an imbalanced distribution where FORWARD commands dominate. Training on this directly produces a model with weak response to LEFT / RIGHT commands. `balance_dataset.py` downsamples other classes to match the smallest class, adjusting the model to respond evenly to all commands.

- **Strategy**: `equal` (fully equalize to match the smallest class)
- **Exclude**: `UNKNOWN` command (Index 3)
- **Backup**: Original data is preserved in `train_data_original.npz`
- **Skip condition**: Skipped when the imbalance ratio (max/min) is below 2.0x

If you run the pipeline without executing `prepare_dataset.sh`, `train.py` falls back to dummy data (for verification purposes). Training with real data requires running `prepare_dataset.sh`. For details, see [NAVSIM Guide - Dataset preparation](../../docs/navsim-guide.md#dataset-preparation).

Estimated execution time:

| Step | Time |
|------|------|
| conda environment setup | ~5 min |
| OpenScene mini split download | ~40 min |
| Feature extraction | ~10 min |
| Dataset balancing | ~1 min |
| S3 upload | ~4 min |
| **Total** | **~60 min** |

## Quick start

### 1. Prepare the dataset

```bash
./pipelines/container-navsim-ego-mlp/scripts/prepare_dataset.sh
```

You will be asked to accept the license when running the script.

### 2. Build the container and run the pipeline

```bash
# Build the container and push to ECR
./pipelines/scripts/02-build-and-push-container.sh -c container-navsim-ego-mlp

# Run the pipeline end-to-end (data upload → container build → pipeline execution → wait for completion)
./pipelines/scripts/run-pipeline.sh -c container-navsim-ego-mlp --skip-upload
```

### 3. Run from a notebook

Open `notebooks/navsim-ego-mlp-pipeline.ipynb` in JupyterLab and run the cells in order. It covers everything from data preparation to training, evaluation, and result inspection.

## Training parameters

The following hyperparameters are available in `train.py`. Pass them via the `hyperparameters` argument of the SageMaker Estimator.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 64 | Mini-batch size |
| `--learning-rate` | 0.001 | Learning rate for the Adam optimizer |
| `--hidden-dim` | 128 | Dimension of MLP hidden layers. Larger values increase model capacity but risk overfitting |
| `--num-poses` | 8 | Number of future trajectory poses to predict. Determined by `time-horizon` / pose interval (0.5 sec) |
| `--time-horizon` | 4.0 | Prediction time horizon in seconds. NAVSIM's standard setting is 4 seconds |

## Evaluation metrics

`evaluate.py` computes simplified metrics aligned with PDM Score. A full PDM Score computation requires nuPlan maps and scene information, so this focuses on trajectory prediction accuracy.

| Metric | Description |
|--------|-------------|
| `pdm_score` | A simple overall score (0-1) based on ADE. Lower ADE results in a higher score |
| `ade` | Average Displacement Error. Mean L2 distance (m) between predicted and ground truth positions across all timesteps |
| `fde` | Final Displacement Error. L2 distance (m) at the final timestep. Indicates long-term prediction accuracy |
| `heading_error` | Mean absolute error (rad) between predicted and ground truth headings |
| `miss_rate` | Fraction of samples where FDE > 2.0 m. Indicates the frequency of large prediction errors |

## Customization tips

EgoStatusMLP is a minimal baseline that does not use sensors. For higher-accuracy models, consider the following.

- NAVSIM's [TransfuserAgent](https://github.com/autonomousvision/navsim/blob/main/navsim/agents/transfuser/) is an implementation example of a sensor agent that uses cameras + LiDAR. You can adapt this container by replacing the model definition in `train.py` and enabling cameras / LiDAR in `SensorConfig`
- The instance type `ml.c7i.xlarge` (CPU) is automatically selected. GPU is not required for an 8-dimensional input MLP
- To use NAVSIM v2 (with Pseudo-Simulation support), change the clone branch in the Dockerfile to `main` and adjust dependencies accordingly

## References

- [NAVSIM GitHub repository](https://github.com/autonomousvision/navsim)
- [NAVSIM Paper (NeurIPS 2024)](https://arxiv.org/abs/2406.15349) - Original paper for NAVSIM v1
- [Pseudo-Simulation Paper (CoRL 2025)](https://arxiv.org/abs/2506.04218) - Pseudo-Simulation method for NAVSIM v2
- [nuPlan dataset](https://www.nuscenes.org/nuplan)
- [ML/AI Model Development Guide](../../docs/model-development-guide.md) - Overall container design in this project
