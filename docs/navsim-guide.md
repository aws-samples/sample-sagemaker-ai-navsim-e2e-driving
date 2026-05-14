# NAVSIM Autonomous Driving Simulation Guide <!-- omit in toc -->

🌐 **Language**: 🇺🇸 [English](navsim-guide.md) | 🇯🇵 [日本語](navsim-guide.ja.md)

A comprehensive guide for using [NAVSIM](https://github.com/autonomousvision/navsim) with the SageMaker AI ML Pipeline. Covers the NAVSIM overview, baseline agent comparison, and implementation patterns with the SageMaker Python SDK.

- [What is NAVSIM](#what-is-navsim)
  - [Evaluation Challenges and Pseudo-Simulation](#evaluation-challenges-and-pseudo-simulation)
  - [PDM Score](#pdm-score)
  - [Dataset](#dataset)
  - [Versions](#versions)
- [Baseline Agents](#baseline-agents)
  - [Overview and Comparison](#overview-and-comparison)
  - [ConstantVelocityAgent](#constantvelocityagent)
  - [EgoStatusMLPAgent](#egostatusmlpagent)
  - [TransfuserAgent](#transfuseragent)
  - [Latent TransfuserAgent (LTF)](#latent-transfuseragent-ltf)
- [SageMaker AI Implementation Patterns](#sagemaker-ai-implementation-patterns)
  - [Container Design Policy](#container-design-policy)
  - [Official NAVSIM Code Migration Approach](#official-navsim-code-migration-approach)
  - [Why PyTorch DLC](#why-pytorch-dlc)
  - [EgoStatusMLP Implementation Example](#egostatusmlp-implementation-example)
  - [Switching to Latent Transfuser](#switching-to-latent-transfuser)
- [Container Configuration in This Project](#container-configuration-in-this-project)
  - [Instance Types and Performance](#instance-types-and-performance)
- [Dataset Preparation](#dataset-preparation)
  - [Prerequisites](#prerequisites)
  - [Disk Space Requirements](#disk-space-requirements)
  - [Execution Steps](#execution-steps)
  - [Automatic Data-Pipeline Mapping](#automatic-data-pipeline-mapping)
  - [Output Data Format](#output-data-format)
  - [About Dummy Data](#about-dummy-data)
- [References](#references)

## What is NAVSIM

NAVSIM (Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking) is a framework for evaluating End-to-End driving models for autonomous vehicles. It was developed as a joint research effort by the University of Tübingen, NVIDIA Research, Robert Bosch GmbH, and others.

### Evaluation Challenges and Pseudo-Simulation

There are two major approaches to evaluating autonomous driving models, each with its own challenges.

| Approach | Features | Challenges |
|----------|----------|------------|
| Open-loop evaluation | Predicts against recorded data and compares with GT. Fast and scalable | Cannot evaluate error accumulation or recovery behavior. Cannot measure impact when predicted trajectory deviates from GT |
| Closed-loop evaluation | Runs the model in a simulator. Closer to reality | High computational cost. Requires model access (inference API). Scalability challenges |

NAVSIM proposes Pseudo-Simulation to bridge this gap. By adding synthetic observations to real data, it achieves evaluation accuracy close to closed-loop while maintaining open-loop efficiency.

Key features of Pseudo-Simulation:

- Approximately 6x faster than closed-loop
- Can evaluate using model predictions only (no inference API access required)
- No sequential or interactive processing needed, suitable for large-scale leaderboard operations
- 143 teams and 463 entries participated in the CVPR 2024 competition

### PDM Score

NAVSIM's evaluation metric is the PDM Score (Predictive Driver Model Score). It comprehensively scores the following elements:

- Collision avoidance
- Drivable area compliance
- Comfort
- Progress
- Time to collision

v2 extends this to the Extended PDM Score (EPDMS) with additional metrics and penalties.

### Dataset

NAVSIM uses the [nuPlan](https://www.nuscenes.org/nuplan) / [OpenScene](https://github.com/OpenDriveLab/OpenScene) datasets.

Available data splits:

| Split | Purpose | Size |
|-------|---------|------|
| mini | Minimal configuration for development and debugging | ~5 GB |
| trainval | Full dataset for training and validation | ~100 GB |
| test | Testing (leaderboard evaluation) | ~50 GB |
| navtrain | Subset of trainval (training subset) | ~20 GB |

Please review the dataset [license](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE) beforehand.

### Versions

NAVSIM has two major versions.

| Version | Branch | Paper | Key Features |
|---------|--------|-------|-------------|
| v1.1 | [v1.1](https://github.com/autonomousvision/navsim/tree/v1.1) | [NeurIPS 2024](https://arxiv.org/abs/2406.15349) | Initial version. Open-loop evaluation + PDM Score |
| v2.x | [main](https://github.com/autonomousvision/navsim) | [CoRL 2025](https://arxiv.org/abs/2506.04218) | Pseudo-Simulation, Extended PDM Score, reactive traffic agents |

Both versions require Python 3.9 (nuplan-devkit constraint).

## Baseline Agents

### Overview and Comparison

NAVSIM officially provides four baseline agents.

| Agent | Sensor Input | Training | GPU Required | Overview |
|-------|-------------|----------|-------------|----------|
| ConstantVelocity | None | None | No | Constant velocity straight-line motion. Simplest rule-based baseline |
| EgoStatusMLP | Velocity, acceleration, driving command | Yes | No | Lightweight MLP showing the upper bound without sensors |
| Transfuser | Camera (3 front stitched) + LiDAR BEV | Yes | Yes | CNN + Transformer fusing image and LiDAR |
| Latent Transfuser (LTF) | Camera (3 front stitched) only | Yes | Yes | Replaces Transfuser's LiDAR with positional encoding |

Pre-trained checkpoints are available on [Hugging Face](https://huggingface.co/autonomousvision/navsim_baselines).

### ConstantVelocityAgent

The simplest baseline. Outputs a trajectory that maintains the current velocity and heading in a straight line. No training required. Used for understanding the `AbstractAgent` interface and analyzing scenes where PDM Score tends to be high.

Input/output:

- Input: Ego velocity (ego_velocity)
- Output: Constant velocity straight-line trajectory (4 seconds, 8 poses at 0.5-second intervals)

### EgoStatusMLPAgent

A "blind" baseline that predicts trajectories using only the ego vehicle's state, without any camera or LiDAR input. Shows the upper bound of performance achievable through kinematic state extrapolation alone.

Input/output:

- Input (8 dimensions): velocity_x, velocity_y, accel_x, accel_y, cmd_left, cmd_straight, cmd_right, cmd_unknown
- Output: Future trajectory (8 poses × 3 dimensions: x, y, heading)
- Loss function: L1 Loss (MAE)
- Architecture: 4-layer MLP (8 → hidden_dim → hidden_dim → hidden_dim → num_poses × 3)

### TransfuserAgent

A sensor agent that fuses camera images and LiDAR BEV (Bird's Eye View). Based on the Transfuser backbone from [CARLA Garage](https://github.com/autonomousvision/carla_garage).

Input/output:

- Camera input: 3 front cameras (cam_l0, cam_f0, cam_r0) stitched into a 1024×256 wide-angle image
- LiDAR input: Point cloud converted to a 256×256 BEV histogram
- EgoStatus input: Driving command + velocity + acceleration
- Output: Future trajectory + object detection (DETR style) + BEV segmentation

Architecture features:

- Camera branch: ResNet-34 for image feature extraction
- LiDAR branch: ResNet-34 for BEV histogram feature extraction
- Fusion: Multiple Transformer layers progressively fuse camera and LiDAR features
- Auxiliary tasks: BEV semantic segmentation + DETR-style object detection (Hungarian matching)

### Latent TransfuserAgent (LTF)

A variant of Transfuser that replaces LiDAR input with positional encoding. Works in environments where LiDAR is unavailable, making it more flexible.

The only difference from Transfuser:

- Set `TransfuserConfig.latent = True`
- LiDAR data loading is skipped, and positional encoding is used instead

On the CARLA leaderboard, it shows the highest performance among image-only methods.

## SageMaker AI Implementation Patterns

### Container Design Policy

navsim v1.1 / v2 pins older versions such as Python 3.9, `numpy==1.23.4`, and `torch==2.0.1`. This is due to nuplan-devkit constraints that conflict with PyTorch DLC dependencies (Python 3.11 + numpy 2.x + torch 2.5.x).

This project addresses the issue with the following approach:

- Feature extraction (using navsim's SceneLoader) runs on SageMaker AI Notebook in an automatically created conda Python 3.9 environment (`prepare_dataset.sh`)
- SageMaker Training Job / Processing Job containers do not install navsim devkit and use lightweight PyTorch DLC-based images
- Only preprocessed data (npz / pt format) is passed to training/evaluation containers
- This enables fast builds and leverages the latest PyTorch optimizations

### Official NAVSIM Code Migration Approach

The official NAVSIM Transfuser model definition files (`transfuser_backbone.py`, etc.) do not depend on the entire navsim devkit, but they have top-level imports of the `nuplan` package. Since `nuplan-devkit` pins Python 3.9 and conflicts with the DLC, the official code cannot be used as-is.

This project copies the official code and replaces `nuplan` / `navsim` imports with local references or self-defined equivalents. The architecture (model structure, weight initialization, loss function logic) is not modified at all.

| File | Changes |
|------|---------|
| `transfuser_backbone.py` | `from navsim...` → `from transfuser_config` (1 import line) |
| `transfuser_model.py` | Import changes + self-defined `StateSE2Index` / `BoundingBox2DIndex` |
| `transfuser_config.py` | Remove `nuplan` imports, replace BEV class definitions with integers |
| `transfuser_loss.py` | `from navsim...` → local references (2 import lines) |

### Why PyTorch DLC

Both EgoStatusMLP and Transfuser are implemented in PyTorch (nn.Module, DataLoader, optim, etc.). The reasons for using AWS PyTorch DLC (Deep Learning Container) as the base image:

- PyTorch, torchvision, numpy, and CUDA runtime are pre-installed, eliminating large installations like `pip install torch`
- SageMaker Training Toolkit is built in, so `entry_point` / `source_dir` script injection works out of the box
- AWS continuously applies security patches, eliminating the need to manage base images yourself
- Distributed training (Data Parallel / Model Parallel) support is built in

All containers use the GPU version DLC (`pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker`). The GPU version also works on CPU instances (it simply includes additional CUDA libraries), so Dockerfiles are unified to share the build cache.

### EgoStatusMLP Implementation Example

EgoStatusMLP takes only an 8-dimensional vector as input, so features are saved in npz format and loaded in the container.

```python
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

estimator = Estimator(
    image_uri=ecr_image_uri,
    entry_point="train.py",
    source_dir="pipelines/container-navsim-ego-mlp",
    role=role_arn,
    instance_count=1,
    instance_type="ml.c7i.xlarge",  # CPU is sufficient
    output_path=model_output_uri,
    hyperparameters={
        "epochs": 50,
        "batch-size": 64,
        "learning-rate": 0.001,
        "hidden-dim": 128,
        "num-poses": 8,
    },
)

estimator.fit(
    # input_mode: "FastFile" = on-demand streaming from S3 / "File" = full download before training
    inputs={"train": TrainingInput(s3_data=train_data_uri, input_mode="FastFile")},
    wait=True,
)
```

See `pipelines/container-navsim-ego-mlp/README.md` for details.

### Switching to Latent Transfuser

Transfuser and Latent Transfuser (LTF) run in the same container (`container-navsim-transfuser`). Switch using the `latent` hyperparameter.

```python
# Transfuser (camera + LiDAR)
hyperparameters={"latent": "false", ...}

# Latent Transfuser (camera only)
hyperparameters={"latent": "true", ...}
```

In `train.py`, the `--latent` argument is received via argparse and applied during model initialization. When `latent=true`, LiDAR feature loading is skipped and positional encoding is used instead.

## Container Configuration in This Project

NAVSIM-related containers:

| Container | Agent | Sensors | Instance |
|-----------|-------|---------|----------|
| `container-navsim-ego-mlp` | EgoStatusMLP | None (velocity, acceleration, command) | CPU (`ml.c7i.xlarge`) |
| `container-navsim-transfuser` | Transfuser / LTF | Camera + LiDAR (or camera only) | GPU (`ml.g6.4xlarge`) |

### Instance Types and Performance

Instance type selection is based on the implementation bottleneck of each agent.

**EgoStatusMLP** (`ml.c7i.xlarge`): Data is very small 8-dimensional vector npz, and the MLP is lightweight. DataLoader `num_workers` is default (0 = main thread only), and DataParallel is not used. The bottleneck is pure computation, but since computation is minimal, CPU is sufficient.

**Transfuser** (`ml.g6.4xlarge`): CNN + Transformer forward/backward is GPU-bound. DataLoader uses `num_workers=2` for parallel data loading. `ml.g6.4xlarge` has 1x L4 GPU + 16 vCPUs + 64GB RAM, providing sufficient memory and parallel data loading performance. To increase GPU performance, `ml.g6.12xlarge` (4x L4) + DataParallel implementation, or `ml.p4d.24xlarge` (A100) etc. would be needed.


## Dataset Preparation

NAVSIM datasets are based on nuPlan / OpenScene and require navsim devkit (Python 3.9) for feature extraction. Since this conflicts with SageMaker AI Notebook (Python 3.11) dependencies, `prepare_dataset.sh` automatically creates a conda Python 3.9 environment to extract features.

### Prerequisites

Run in an environment that meets the following conditions:

- conda is installed (pre-installed on SageMaker AI Notebook)
- AWS CLI is configured
- 500 GB disk space recommended (mini split sensor data ~151 GB + conda environment + temporary extraction space totaling ~210 GB)
- Agreement to the [nuPlan dataset license](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE)

### Disk Space Requirements

Required space when using the mini split:

| Purpose | Size |
|---------|------|
| conda environment (navsim-py39) | ~5 GB |
| Download (tgz compressed) | ~50 GB |
| Extracted (logs + sensor data) | ~152 GB |
| Extracted features (pt / npz) | A few MB to several GB |
| **Total** | **~210 GB** |

### Execution Steps

Run the following from the JupyterLab terminal:

```bash
# For EgoStatusMLP (numerical data only, a few MB after extraction)
./pipelines/container-navsim-ego-mlp/scripts/prepare_dataset.sh

# For Transfuser (camera + LiDAR + EgoStatus + auxiliary targets, several GB after extraction)
./pipelines/container-navsim-transfuser/scripts/prepare_dataset.sh

```

Each script automatically performs:

1. Creates conda environment `navsim-py39` (Python 3.9) — skipped on subsequent runs
2. Installs navsim devkit + nuplan-devkit
3. Downloads mini split dataset (logs ~1 GB + sensor data ~151 GB)
4. Extracts features with `extract_features.py`
5. Balances command distribution with `balance_dataset.py`
6. Uploads to S3 dataset bucket

The conda environment is shared between EgoStatusMLP and Transfuser, so subsequent scripts skip environment creation.

### Dataset Balancing

The OpenScene dataset has an imbalanced command distribution where FORWARD dominates. Training on this directly produces a model that barely responds to LEFT / RIGHT commands. `prepare_dataset.sh` runs `balance_dataset.py` after feature extraction to equalize the LEFT / FORWARD / RIGHT distribution. Applied to both EgoStatusMLP and Transfuser.

- **Strategy**: Downsample larger classes to match the smallest class (`--strategy equal`)
- **Exclude**: `UNKNOWN` command (Index 3) is excluded from balancing (`--exclude-unknown`)
- **Backup**:
  - EgoStatusMLP: Original data preserved as `train_data_original.npz`, balanced `train_data.npz` uploaded to S3
  - Transfuser: Original data preserved in `train_original/`, balanced data uploaded as `train/` to S3
- **Skip condition**: Skipped when the imbalance ratio (max/min) is below 2.0x

### Automatic Data-Pipeline Mapping

Data uploaded by `prepare_dataset.sh` is automatically used when the pipeline runs. The mechanism works as follows:

1. `prepare_dataset.sh` uploads data to S3 using the container name as a prefix
2. `03-create-and-run-pipeline.py` generates the S3 path from the same container name
3. SageMaker mounts the data to `/opt/ml/input/data/train/` inside the container
4. `train.py` reads the data via the `SM_CHANNEL_TRAIN` environment variable

```
s3://{project}-dataset-{account}-{region}/
  ├── container-navsim-ego-mlp/
  │   ├── train/   ← uploaded by prepare_dataset.sh
  │   └── test/
  └── container-navsim-transfuser/
      ├── train/   ← automatically referenced at pipeline execution
      └── test/
```

After data preparation, run the pipeline with `--skip-upload`:

```bash
./pipelines/scripts/run-pipeline.sh -c container-navsim-transfuser --skip-upload
```

### Output Data Format

Extracted data format varies by model.

| Model | Format | Content | Size Estimate |
|-------|--------|---------|---------------|
| EgoStatusMLP | npz | `features` [N, 8] + `targets` [N, 8, 3] | A few MB |
| Transfuser | pt | `camera` [3, 256, 1024] + `lidar` [1, 256, 256] + `status` [8] + `trajectory` [8, 3] | Several GB |

### About Dummy Data

If you run the Pipeline without executing `prepare_dataset.sh`, the fallback feature in `train.py` trains with dummy data (randomly generated). This can be used for Pipeline operation verification, but model prediction accuracy should not be expected. Running `prepare_dataset.sh` is required for training with real data.

## References

- [NAVSIM GitHub Repository](https://github.com/autonomousvision/navsim)
- [NAVSIM Paper (NeurIPS 2024)](https://arxiv.org/abs/2406.15349) - v1 original paper
- [Pseudo-Simulation Paper (CoRL 2025)](https://arxiv.org/abs/2506.04218) - v2 Pseudo-Simulation method
- [Transfuser Paper](https://arxiv.org/abs/2205.15997) - Transfuser architecture original paper
- [NAVSIM Baseline Checkpoints (Hugging Face)](https://huggingface.co/autonomousvision/navsim_baselines)
- [nuPlan Dataset](https://www.nuscenes.org/nuplan)
- [ML/AI Model Development Guide](model-development-guide.ja.md) - Overall container configuration for this project
