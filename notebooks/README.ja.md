# Notebooks <!-- omit in toc -->

🌐 **Language**: 🇺🇸 [English](README.md) | 🇯🇵 [日本語](README.ja.md)

ML ワークフローの各フェーズに対応した Jupyter Notebook を格納しています。

- [Notebook 一覧](#notebook-一覧)
- [コンテナとノートブックの対応](#コンテナとノートブックの対応)
- [実行環境](#実行環境)
- [出力先](#出力先)

## Notebook 一覧

各 Notebook はデータ確認 → ローカル学習 → ローカル評価 → SageMaker Job → Pipeline の一気通貫フローに対応しています。

| Notebook | 用途 | ローカル実行 | SageMaker Job / Pipeline |
|----------|------|------------|-------------------------|
| `pytorch-pipeline.ipynb` | PyTorch DLC モデルの学習・評価・Pipeline | Section 3-5 | Section 7-9 |
| `pytorch-byoc-pipeline.ipynb` | PyTorch BYOC モデルの学習・評価・Pipeline | Section 3-5 | Section 7-11 |
| `navsim-ego-mlp-pipeline.ipynb` | NAVSIM EgoStatusMLP の学習・評価 | - | データ準備 → Training Job → Processing Job |
| `navsim-transfuser-pipeline.ipynb` | NAVSIM Transfuser / LTF の学習・評価 | - | データ準備 → Training Job (GPU) → Processing Job |
| `carla-transfuser-demo.ipynb` | CARLA シミュレーションデモ | CARLA + TransFuser で走行・動画録画 (3 カメラ + LiDAR) | - |

## コンテナとノートブックの対応

使用するコンテナに合わせてノートブックを選択してください。

| コンテナ | `CONTAINER_DIR` | Notebook | モデル形式 | 推奨インスタンス | ECR プッシュ |
|---------|----------------|----------|-----------|----------------|-------------|
| PyTorch DLC (マネージド) | `../pipelines/container-pytorch-dlc` | `pytorch-pipeline.ipynb` | `model.pth` | `ml.c7i.xlarge` (CPU) | 不要 |
| PyTorch DLC BYOC | `../pipelines/container-pytorch-dlc-byoc` | `pytorch-byoc-pipeline.ipynb` | `model.pth` | `ml.c7i.xlarge` (CPU) | 必要 |
| NAVSIM EgoStatusMLP | `../pipelines/container-navsim-ego-mlp` | `navsim-ego-mlp-pipeline.ipynb` | `model.pth` | `ml.c7i.xlarge` (CPU) | 必要 |
| NAVSIM Transfuser (公式準拠) | `../pipelines/container-navsim-transfuser` | `navsim-transfuser-pipeline.ipynb` | `model.pth` | `ml.g6.4xlarge` (GPU) | 必要 |

## 実行環境

すべての Notebook は SageMaker AI Notebook 上の JupyterLab で実行する前提です。カーネルは `conda_python3` を使用してください。

| Notebook | カーネル | 備考 |
|----------|---------|------|
| `pytorch-pipeline.ipynb` | `conda_python3` | |
| `pytorch-byoc-pipeline.ipynb` | `conda_python3` | |
| `navsim-ego-mlp-pipeline.ipynb` | `conda_python3` | |
| `navsim-transfuser-pipeline.ipynb` | `conda_python3` | |
| `carla-transfuser-demo.ipynb` | `conda_python3` | GPU インスタンス (ml.g4dn.2xlarge 以上) が必要 |

> ⚠️ `navsim-py39` カーネルは `prepare_dataset.sh` の特徴量抽出専用です (navsim devkit が Python 3.9 を要求するため)。Pipeline の実行には SageMaker SDK が必要なので、必ず `conda_python3` を使用してください。

JupyterLab でノートブックを開くと、カレントディレクトリはノートブックが配置されている `notebooks/` ディレクトリになります。

```
~/SageMaker/{project-name}/
├── notebooks/                 ← カレントディレクトリ (JupyterLab)
├── pipelines/                 ← コンテナ・スクリプト
└── ...
```

そのため、Notebook 内のローカルパスは `notebooks/` からの相対パスで記述しています。例えば `CONTAINER_DIR = "../pipelines/container-pytorch-dlc"` のように `../` を使って `pipelines/` ディレクトリを参照します。学習データは S3 から直接読み込むため、ローカルのデータファイルパスに依存しません。

## 出力先

ローカル実行時の出力は `notebooks/output/` に保存されます。このディレクトリは `.gitignore` に含まれているため、リポジトリにはコミットされません。

```
notebooks/output/
├── model/           # ローカル学習で出力する学習済みモデル
└── evaluation/      # ローカル評価で出力する評価結果 (evaluation.json)
```
