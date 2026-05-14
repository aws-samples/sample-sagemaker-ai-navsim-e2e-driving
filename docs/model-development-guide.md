# ML/AI Model Development Guide <!-- omit in toc -->

🌐 **Language**: 🇺🇸 [English](model-development-guide.md) | 🇯🇵 [日本語](model-development-guide.ja.md)

This document summarizes the container configuration of this project, custom container templates for various scenarios, and the design of model groups.

Refer to this document when considering model changes or adapting to new use cases. For common specifications of SageMaker containers (Training Toolkit, input/output paths, etc.), see the [SageMaker Python SDK Guide](sagemaker-python-sdk-guide.md).

- [Container Types](#container-types)
- [Project Configuration](#project-configuration)
  - [About the Model Architecture](#about-the-model-architecture)
  - [PyTorch DLC Based](#pytorch-dlc-based)
  - [PyTorch DLC Based BYOC](#pytorch-dlc-based-byoc)
  - [NAVSIM Containers](#navsim-containers)
- [Model Group Design](#model-group-design)
  - [Design Principles](#design-principles)
  - [Application in This Project](#application-in-this-project)
  - [Recommended Naming Convention](#recommended-naming-convention)
- [Reference Links](#reference-links)

## Container Types

In SageMaker AI, each step of the ML workflow runs on a container. There are three ways to provide containers, and you choose one based on model complexity and customization needs.

| Provisioning Method | Overview | Customization Scope | Container Management | Suitable Cases |
|---------|------|----------------|------------|-------------|
| Built-in Algorithm | Managed algorithms provided by SageMaker AI | Hyperparameters only | Not required (fully managed) | When you want to quickly try standard ML tasks |
| Prebuilt DLC | AWS Deep Learning Containers (PyTorch, TensorFlow, etc.) | Training script can be swapped | Not required (SDK selects image automatically) | When using major frameworks as is |
| Custom Container (BYOC) | Build any environment with your own Dockerfile | Fully flexible (build from Dockerfile) | Self-managed (push to ECR) | When you have unique libraries or complex dependencies |

For details, see the following AWS documentation.

- [Docker containers for training and deploying models](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html)
- [Built-in algorithms and pretrained models](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)
- [Prebuilt Docker images for deep learning](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html)
- [Adapting your own Docker container](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers-adapt-your-own.html)

## Project Configuration

This project provides multiple containers for different purposes.

| Container | Train | Evaluate | Model | GPU |
|---------|-------|----------|-------|-----|
| `container-pytorch-dlc` | AWS managed (`PyTorch`) | AWS managed (`PyTorchProcessor`) | SimpleClassifier | Supported |
| `container-pytorch-dlc-byoc` | BYOC (PyTorch DLC) | BYOC (PyTorch DLC) | SimpleClassifier | Supported |
| `container-navsim-ego-mlp` | BYOC | BYOC | EgoStatusMLP | Not required |
| `container-navsim-transfuser` | BYOC | BYOC | Transfuser / LTF | Required |

`container-pytorch-dlc` uses AWS managed containers for both Train and Evaluate, so building a Dockerfile and pushing images to ECR are not required, enabling a fast development cycle. Additional packages can be listed in `requirements.txt` inside `source_dir`, and the managed container automatically runs `pip install` at startup.

`container-pytorch-dlc-byoc` uses the same PyTorch DLC as its base but builds a BYOC image. Dependencies are baked into the Dockerfile, so job startup is faster and does not depend on the network. Choose this when you need libraries not in the DLC, or when you want to train and evaluate in exactly the same environment.

`container-navsim-ego-mlp` / `container-navsim-transfuser` are BYOC containers for training and evaluating NAVSIM autonomous driving models. They require running `prepare_dataset.sh` beforehand to extract features from the OpenScene dataset. See each container's README for details.

The PyTorch containers (`container-pytorch-dlc` / `container-pytorch-dlc-byoc`) include sample datasets in their `data/` subdirectory.

| File | Rows | Content |
|---------|------|------|
| `data/train.csv` | 800 rows | Training data (4 features `f1,f2,f3,f4` + target `target`, 3-class classification) |
| `data/test.csv` | 200 rows | Test data (same format) |

Sample data is automatically uploaded to S3 when `deploy.sh` is executed. If you modify the data, re-upload it using `01-upload-dataset.sh`.

```bash
# Default (data for container-pytorch-dlc)
./pipelines/scripts/01-upload-dataset.sh

# Data for NAVSIM Transfuser container
```

In your actual project, replace the files under `data/` with your own dataset. The `train.py` and `evaluate.py` scripts are designed to read the last column of the CSV as the target, so if you change the column structure, update the scripts accordingly.

Switch containers using the `-c` option of `02-build-and-push-container.sh`.

```bash
# Default (NAVSIM Transfuser)
./pipelines/scripts/02-build-and-push-container.sh

# PyTorch DLC
./pipelines/scripts/02-build-and-push-container.sh -c container-pytorch-dlc
```

### About the Model Architecture

The models used in this project are as follows.

| Model | Overview | Container |
|-------|----------|-----------|
| SimpleClassifier | 3-layer MLP classification model | PyTorch DLC Based / PyTorch DLC Based BYOC |
| EgoStatusMLP | 4-layer MLP trajectory prediction model | NAVSIM Containers (`container-navsim-ego-mlp`) |
| Transfuser | Multimodal trajectory prediction with ResNet-34 + GPT-style Transformer | NAVSIM Containers (`container-navsim-transfuser`) |

All containers share the same `train.py` / `evaluate.py` code regardless of the container provisioning method (DLC / BYOC). In a real project, `SimpleClassifier` is expected to be replaced by pretrained models such as ResNet or Transformer. In that case, add additional libraries to `requirements.txt` (for DLC) or `Dockerfile` (for BYOC).

### PyTorch DLC Based

A configuration that uses AWS managed PyTorch DLC containers. Building a Dockerfile or pushing to ECR is not required. Additional packages are managed via `requirements.txt`. Container-related files are located in `pipelines/container-pytorch-dlc/`.

The model is `SimpleClassifier` (3-layer MLP: input dim → 64 → 32 → number of classes), using only standard PyTorch modules such as `nn.Linear` and `nn.ReLU`. No external weight downloads are required.

For available framework versions, refer to [Available Deep Learning Containers Images](https://github.com/aws/deep-learning-containers/blob/master/available_images.md).

### PyTorch DLC Based BYOC

Uses the same PyTorch DLC as its base image, but builds a custom image with a Dockerfile and pushes it to ECR. Dependencies are baked into the Dockerfile, so job startup is faster and does not depend on the network. Container-related files are located in `pipelines/container-pytorch-dlc-byoc/`.

The model is the same `SimpleClassifier` as `container-pytorch-dlc`.

> ⚠️ The PyTorch DLC image includes GPU libraries such as CUDA and is over 10 GB in size. Downloading during the first build takes time, so ensure sufficient disk space (20 GB or more recommended) and network bandwidth. If the managed container is sufficient, we recommend using `container-pytorch-dlc`.

```bash
# Build & run
./pipelines/scripts/run-pipeline.sh -c container-pytorch-dlc-byoc
```

### NAVSIM Containers

BYOC containers for training and evaluating NAVSIM End-to-End driving models on SageMaker. Running `prepare_dataset.sh` beforehand is required to extract features from the OpenScene dataset.

| Container | Agent | Sensor | Instance |
|---------|------------|---------|------------|
| `container-navsim-ego-mlp` | EgoStatusMLP | None (velocity, acceleration, command) | CPU (`ml.c7i.xlarge`) |
| `container-navsim-transfuser` | Transfuser / LTF | Camera + LiDAR (or camera-only) | GPU (`ml.g6.4xlarge`) |

`EgoStatusMLP` is a lightweight 4-layer MLP that predicts future trajectories from the ego vehicle's state (velocity, acceleration, driving command). It uses only standard PyTorch modules and requires no external weight downloads.

`Transfuser` uses a `timm` pretrained ResNet-34 as its backbone with a GPT-style Transformer for multi-scale fusion of camera and LiDAR features. Pretrained weights are downloaded on the first training run.

For details on each container, refer to the following.

- [container-navsim-ego-mlp/README.md](../pipelines/container-navsim-ego-mlp/README.md)
- [container-navsim-transfuser/README.md](../pipelines/container-navsim-transfuser/README.md)
- [NAVSIM Autonomous Driving Simulation Guide](navsim-guide.md)

## Model Group Design

In MLflow Model Registry and SageMaker Model Registry, models are managed in groups (model package groups). A group is the unit that bundles the version history of a model, and one group corresponds to "the version history of models that solve the same problem."

### Design Principles

The most important principle in model group design is: **1 group = 1 problem × 1 architecture**.

Conditions where groups should be separated are as follows.

- Model architectures differ (e.g., RandomForest vs neural network)
- Inference containers differ (e.g., scikit-learn vs PyTorch)
- Comparison/evaluation contexts differ (e.g., when the meaning of accuracy metrics changes)

Conditions where groups should NOT be separated are as follows.

- Hyperparameter differences within the same architecture
- Retraining the same model (data updates)
- Minor improvements to the same model

### Application in This Project

This project uses multiple PyTorch and NAVSIM containers, each managed as a separate model group.

| Container | Model Group Name | Reason |
|---------|---------------|------|
| `container-pytorch-dlc` | `{project}-pytorch` | SimpleClassifier (PyTorch), CPU training |
| `container-pytorch-dlc-byoc` | `{project}-pytorch-byoc` | SimpleClassifier (PyTorch), trained with BYOC image |
| `container-navsim-ego-mlp` | `{project}-navsim-ego-mlp` | EgoStatusMLP (NAVSIM), CPU inference |
| `container-navsim-transfuser` | `{project}-navsim-transfuser` | Transfuser (NAVSIM), GPU inference |

Reasons for using separate groups per container are as follows.

- The architectures are fundamentally different
- Since the inference containers differ, comparing versions within the same group would not be meaningful
- Accuracy metric trends differ, and comparing versions would cause confusion

### Recommended Naming Convention

We recommend a model group name format of `{project}-{task}-{framework}`.

```
sagemaker-ai-ml-pipeline-pytorch
sagemaker-ai-ml-pipeline-pytorch-byoc
sagemaker-ai-ml-pipeline-navsim-transfuser
```

In `03-create-and-run-pipeline.py`, `MODEL_GROUP_NAME` is automatically set based on the value of `--container-dir`.

```bash
# PyTorch container → MODEL_GROUP_NAME = sagemaker-ai-ml-pipeline-pytorch
python pipelines/scripts/03-create-and-run-pipeline.py \
    --container-dir pipelines/container-pytorch-dlc ...

# NAVSIM Transfuser → MODEL_GROUP_NAME = sagemaker-ai-ml-pipeline-navsim-transfuser
python pipelines/scripts/03-create-and-run-pipeline.py \
    --container-dir pipelines/container-pytorch-dlc ...

# PyTorch BYOC container → MODEL_GROUP_NAME = sagemaker-ai-ml-pipeline-pytorch-byoc
python pipelines/scripts/03-create-and-run-pipeline.py \
    --container-dir pipelines/container-pytorch-dlc-byoc ...
```

## Reference Links

A list of AWS documentation referenced in this document.

- [Docker containers for training and deploying models](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html) - Overview of container usage in SageMaker AI
- [SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html) - Details on model groups and version management
- [Pre-built SageMaker AI Docker images](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers-prebuilt.html) - List and selection of prebuilt images
- [Prebuilt SageMaker AI Docker images for deep learning](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html) - DLC details and framework-specific usage
- [Adapting your own Docker container to work with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers-adapt-your-own.html) - Requirements for building custom containers
- [Available Deep Learning Containers Images (GitHub)](https://github.com/aws/deep-learning-containers/blob/master/available_images.md) - List of DLC image URIs
- [SageMaker Training Toolkit (GitHub)](https://github.com/aws/sagemaker-training-toolkit) - Training Toolkit specifications and usage
- [SageMaker AI Training Storage](https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage.html) - Input/output path conventions for Training Jobs
- [Built-in algorithms and pretrained models](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html) - List and usage of built-in algorithms
