# Notebooks <!-- omit in toc -->

🌐 **Language**: 🇺🇸 [English](README.md) | 🇯🇵 [日本語](README.ja.md)

This directory contains Jupyter Notebooks that correspond to each phase of the ML workflow.

- [Notebook list](#notebook-list)
- [Container and notebook mapping](#container-and-notebook-mapping)
- [Runtime environment](#runtime-environment)
- [Output location](#output-location)

## Notebook list

Each notebook covers an end-to-end flow: data inspection → local training → local evaluation → SageMaker Job → Pipeline.

| Notebook | Purpose | Local execution | SageMaker Job / Pipeline |
|----------|---------|-----------------|--------------------------|
| `pytorch-pipeline.ipynb` | Training, evaluation, and pipeline for the PyTorch DLC model | Section 3-5 | Section 7-9 |
| `pytorch-byoc-pipeline.ipynb` | Training, evaluation, and pipeline for the PyTorch BYOC model | Section 3-5 | Section 7-11 |
| `navsim-ego-mlp-pipeline.ipynb` | Training and evaluation for NAVSIM EgoStatusMLP | - | Data preparation → Training Job → Processing Job |
| `navsim-transfuser-pipeline.ipynb` | Training and evaluation for NAVSIM Transfuser / LTF | - | Data preparation → Training Job (GPU) → Processing Job |
| `carla-transfuser-demo.ipynb` | CARLA simulation demo | Drive and record video with CARLA + TransFuser (3 cameras + LiDAR) | - |

## Container and notebook mapping

Choose the notebook based on the container you plan to use.

| Container | `CONTAINER_DIR` | Notebook | Model format | Recommended instance | ECR push |
|-----------|-----------------|----------|--------------|----------------------|----------|
| PyTorch DLC (managed) | `../pipelines/container-pytorch-dlc` | `pytorch-pipeline.ipynb` | `model.pth` | `ml.c7i.xlarge` (CPU) | Not required |
| PyTorch DLC BYOC | `../pipelines/container-pytorch-dlc-byoc` | `pytorch-byoc-pipeline.ipynb` | `model.pth` | `ml.c7i.xlarge` (CPU) | Required |
| NAVSIM EgoStatusMLP | `../pipelines/container-navsim-ego-mlp` | `navsim-ego-mlp-pipeline.ipynb` | `model.pth` | `ml.c7i.xlarge` (CPU) | Required |
| NAVSIM Transfuser (official-compliant) | `../pipelines/container-navsim-transfuser` | `navsim-transfuser-pipeline.ipynb` | `model.pth` | `ml.g6.4xlarge` (GPU) | Required |

## Runtime environment

All notebooks are intended to run in JupyterLab on a SageMaker AI Notebook instance. Use the `conda_python3` kernel.

| Notebook | Kernel | Notes |
|----------|--------|-------|
| `pytorch-pipeline.ipynb` | `conda_python3` | |
| `pytorch-byoc-pipeline.ipynb` | `conda_python3` | |
| `navsim-ego-mlp-pipeline.ipynb` | `conda_python3` | |
| `navsim-transfuser-pipeline.ipynb` | `conda_python3` | |
| `carla-transfuser-demo.ipynb` | `conda_python3` | Requires a GPU instance (ml.g4dn.2xlarge or larger) |

> ⚠️ The `navsim-py39` kernel is reserved for feature extraction in `prepare_dataset.sh` (because navsim devkit requires Python 3.9). Pipeline execution requires the SageMaker SDK, so always use `conda_python3`.

When you open a notebook in JupyterLab, the working directory is `notebooks/`, the directory containing the notebook.

```
~/SageMaker/{project-name}/
├── notebooks/                 ← Current directory (JupyterLab)
├── pipelines/                 ← Container / scripts
└── ...
```

For this reason, local paths inside the notebooks are written as relative paths from `notebooks/`. For example, `CONTAINER_DIR = "../pipelines/container-pytorch-dlc"` uses `../` to reference the `pipelines/` directory. Training data is read directly from S3, so there is no dependency on local file paths.

## Output location

Local execution output is saved under `notebooks/output/`. This directory is included in `.gitignore` and is not committed to the repository.

```
notebooks/output/
├── model/           # Trained models saved from local training
└── evaluation/      # Evaluation results saved from local evaluation (evaluation.json)
```
