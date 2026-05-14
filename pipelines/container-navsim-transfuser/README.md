# container-navsim-transfuser <!-- omit in toc -->

🌐 **Language**: 🇺🇸 [English](README.md) | 🇯🇵 [日本語](README.ja.md)

A BYOC container setup for training and evaluating the [NAVSIM](https://github.com/autonomousvision/navsim) Transfuser architecture on SageMaker AI ML Pipeline.

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

Of the multiple baseline agents provided by NAVSIM, this container runs Transfuser on SageMaker.

Transfuser is a model that uses a pretrained ResNet-34 from `timm` as a backbone and fuses camera and LiDAR features through multi-scale fusion with a GPT-style Transformer. A Transformer Decoder jointly learns trajectory prediction, object detection, and BEV semantic segmentation.

Specifying `--latent true` enables Latent Transfuser (LTF) mode, which predicts trajectories from camera images alone without LiDAR input.

The training and evaluation flow is as follows.

```
Dataset preparation (prepare_dataset.sh)
  ↓ Download navsim mini split → Extract features → Upload to S3
SageMaker Training Job (train.py)
  ↓ Train Transfuser → Save model.pth to S3
SageMaker Processing Job (evaluate.py)
  ↓ Run inference on test data → Compute ADE / FDE / PDM Score
Output evaluation.json to S3
```

## Directory structure

Each file has the following role.

| File | Role |
|------|------|
| `Dockerfile` | BYOC image based on PyTorch DLC (Python 3.11). The navsim devkit is not installed |
| `train.py` | Training script for Transfuser. Runs as the entry_point of a SageMaker Training Job |
| `evaluate.py` | Evaluation script for the trained model. Runs as a SageMaker Processing Job |
| `transfuser_config.py` | TransfuserConfig |
| `transfuser_backbone.py` | TransfuserBackbone: timm ResNet-34 + GPT fusion |
| `transfuser_model.py` | TransfuserModel: Backbone + Decoder + Heads |
| `transfuser_loss.py` | Loss functions: trajectory + detection + BEV semantic |
| `requirements.txt` | Additional Python packages installed into the container |
| `scripts/prepare_dataset.sh` | One-shot script that clones the navsim repository, downloads the mini split, extracts features, balances the dataset, and uploads to S3 |
| `scripts/extract_features.py` | Converts cameras, LiDAR BEV, EgoStatus, and GT trajectories to pt format using navsim's SceneLoader |
| `scripts/balance_dataset.py` | Fully equalizes the command distribution (LEFT / FORWARD / RIGHT) in the training data |

`train.py` and `evaluate.py` are not copied into the Dockerfile; instead, they are injected into the container via the SageMaker SDK's `entry_point` + `source_dir` mechanism. Therefore, you do not need to rebuild the container when changing these scripts.

## Dataset

NAVSIM uses the [nuPlan](https://www.nuscenes.org/nuplan) / [OpenScene](https://github.com/OpenDriveLab/OpenScene) datasets. This container uses the mini split (about 20 GB) including camera images and LiDAR data.

Review the dataset [license](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) in advance.

`prepare_dataset.sh` automatically creates a Python 3.9 conda environment (`navsim-py39`), installs the navsim devkit, and extracts features. This is a workaround to avoid dependency conflicts with the SageMaker AI Notebook (Python 3.11).

1. Create conda environment `navsim-py39` (Python 3.9) — skipped on subsequent runs
2. Install navsim devkit + nuplan-devkit
3. Download the mini split data (including camera images + LiDAR)
4. Extract camera images, LiDAR BEV, EgoStatus, and GT trajectories using navsim's SceneLoader
5. Balance the command distribution (LEFT / FORWARD / RIGHT) with `balance_dataset.py`
6. Split into train/test in pt format and upload to the S3 dataset bucket

### About dataset balancing

The OpenScene dataset has an imbalanced distribution where FORWARD commands dominate. Training on this directly produces a model with weak response to LEFT / RIGHT commands. `balance_dataset.py` downsamples other classes to match the smallest class, adjusting the model to respond evenly to all commands.

- **Strategy**: `equal` (fully equalize to match the smallest class)
- **Exclude**: `UNKNOWN` command (Index 3)
- **Backup**: Original data is preserved in `train_original/`
- **Skip condition**: Skipped when the imbalance ratio (max/min) is below 2.0x

If you run the pipeline without executing `prepare_dataset.sh`, `train.py` falls back to dummy data (for verification purposes). Training with real data requires running `prepare_dataset.sh`. For details, see [NAVSIM Guide - Dataset preparation](../../docs/navsim-guide.md#dataset-preparation).

Estimated execution time. Transfuser takes substantially longer than EgoStatusMLP because it includes camera images and LiDAR data.

| Step | Time |
|------|------|
| conda environment setup | ~5 min |
| OpenScene mini split download | ~40 min |
| Map download | ~1 min |
| Feature extraction | ~85 min |
| Dataset balancing | ~5 min |
| S3 upload | ~5 min |
| **Total** | **~140 min** |

### Sample count and training time

With the default setup (`mini` split + balancing), the training set is reduced to about 381 samples. You have the following options depending on your use case.

| Configuration | Samples | Training time (ml.g6.4xlarge, 30 epochs) | Use case |
|---------------|---------|------------------------------------------|----------|
| **mini split + balancing** (default) | ~381 | ~10 min | Smoke testing, prototyping |
| mini split (no balancing) | ~2,892 | ~90 min | When you want to avoid reducing data |
| trainval split + balancing | ~38,000 | ~16 hours (estimated) | Production, higher-accuracy models |

The trainval training time is a linear extrapolation from the mini split measurement and has not been directly validated. Actual time depends on GPU utilization, data loader throughput, and storage I/O.

To use the `trainval` split, change the `extract_features.py` invocation inside `prepare_dataset.sh` to `--split trainval`. The download size and disk usage grow significantly, so review the disk space estimate in the [NAVSIM Guide - Dataset preparation](../../docs/navsim-guide.md#dataset-preparation) first.

## Quick start

### 1. Prepare the dataset

```bash
./pipelines/container-navsim-transfuser/scripts/prepare_dataset.sh
```

You will be asked to accept the license when running the script.

### 2. Build the container and run the pipeline

```bash
# Build the container and push to ECR
./pipelines/scripts/02-build-and-push-container.sh -c container-navsim-transfuser

# Run the pipeline end-to-end (data upload → container build → pipeline execution → wait for completion)
./pipelines/scripts/run-pipeline.sh -c container-navsim-transfuser --skip-upload
```

### 3. Run from a notebook

Open `notebooks/navsim-transfuser-pipeline.ipynb` in JupyterLab and run the cells in order. It covers everything from data preparation to training, evaluation, and result inspection.

## Training parameters

The following hyperparameters are available in `train.py`. Pass them via the `hyperparameters` argument of the SageMaker Estimator.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 30 | Number of training epochs |
| `--batch-size` | 16 | Mini-batch size |
| `--learning-rate` | 0.0003 | Learning rate |
| `--latent` | `false` | Set `true` for LTF mode (no LiDAR, cameras only) |

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

Transfuser is a sensor agent that uses cameras + LiDAR. For customization, consider the following.

- Switching to LTF mode with `--latent true` predicts trajectories from camera images alone without LiDAR. Dataset preparation becomes lighter since LiDAR data is not required
- The instance type `ml.g6.4xlarge` (GPU) is automatically selected. A GPU is required for the CNN processing of cameras + LiDAR, and 64 GB RAM is needed due to the large data size
- To use NAVSIM v2 (with Pseudo-Simulation support), change the clone branch in the Dockerfile to `main` and adjust dependencies accordingly

## References

- [NAVSIM GitHub repository](https://github.com/autonomousvision/navsim)
- [NAVSIM Paper (NeurIPS 2024)](https://arxiv.org/abs/2406.15349) - Original paper for NAVSIM v1
- [Pseudo-Simulation Paper (CoRL 2025)](https://arxiv.org/abs/2506.04218) - Pseudo-Simulation method for NAVSIM v2
- [nuPlan dataset](https://www.nuscenes.org/nuplan)
- [ML/AI Model Development Guide](../../docs/model-development-guide.md) - Overall container design in this project
