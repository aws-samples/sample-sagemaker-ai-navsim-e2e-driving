# ML/AI モデル開発ガイド <!-- omit in toc -->

🌐 **Language**: 🇺🇸 [English](model-development-guide.md) | 🇯🇵 [日本語](model-development-guide.ja.md)

本ドキュメントは、本プロジェクトのコンテナ構成、シナリオ別のカスタムコンテナテンプレート、およびモデルグループの設計についてまとめています。

モデルの変更や新しいユースケースへの対応を検討する際に参照してください。SageMaker コンテナの共通仕様 (Training Toolkit、入出力パス等) については [SageMaker Python SDK ガイド](sagemaker-python-sdk-guide.ja.md) を参照してください。

- [コンテナの種類](#コンテナの種類)
- [本プロジェクトの構成](#本プロジェクトの構成)
  - [モデルアーキテクチャについて](#モデルアーキテクチャについて)
  - [PyTorch DLC ベース](#pytorch-dlc-ベース)
  - [PyTorch DLC ベース BYOC](#pytorch-dlc-ベース-byoc)
  - [NAVSIM コンテナ](#navsim-コンテナ)
- [モデルグループの設計](#モデルグループの設計)
  - [設計原則](#設計原則)
  - [本プロジェクトでの適用](#本プロジェクトでの適用)
  - [推奨命名規則](#推奨命名規則)
- [参考リンク](#参考リンク)

## コンテナの種類

SageMaker AI では、ML ワークフローの各ステップをコンテナ上で実行します。コンテナの提供方法は 3 種類あり、モデルの複雑さやカスタマイズの必要性に応じて選択します。

| 提供方法 | 概要 | カスタマイズ範囲 | コンテナ管理 | 適したケース |
|---------|------|----------------|------------|-------------|
| ビルトインアルゴリズム | SageMaker AI が提供するマネージドアルゴリズム | ハイパーパラメータのみ | 不要 (フルマネージド) | 標準的な ML タスクをすぐに試したい場合 |
| プリビルト DLC | AWS Deep Learning Containers (PyTorch、TensorFlow 等) | 学習スクリプトを差し替え可能 | 不要 (SDK がイメージを自動選択) | 主要フレームワークをそのまま使う場合 |
| カスタムコンテナ (BYOC) | 自作の Dockerfile で任意の環境を構築 | すべて自由 (Dockerfile から構築) | 自前で管理 (ECR にプッシュ) | 独自のライブラリや複雑な依存関係がある場合 |

詳細は以下の AWS ドキュメントを参照してください。

- [Docker containers for training and deploying models](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html)
- [Built-in algorithms and pretrained models](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html)
- [Prebuilt Docker images for deep learning](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html)
- [Adapting your own Docker container](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers-adapt-your-own.html)

## 本プロジェクトの構成

本プロジェクトでは用途に応じて複数のコンテナを用意しています。

| コンテナ | Train | Evaluate | モデル | GPU |
|---------|-------|----------|-------|-----|
| `container-pytorch-dlc` | AWS マネージド (`PyTorch`) | AWS マネージド (`PyTorchProcessor`) | SimpleClassifier | 対応 |
| `container-pytorch-dlc-byoc` | BYOC (PyTorch DLC) | BYOC (PyTorch DLC) | SimpleClassifier | 対応 |
| `container-navsim-ego-mlp` | BYOC | BYOC | EgoStatusMLP | 不要 |
| `container-navsim-transfuser` | BYOC | BYOC | Transfuser / LTF | 必要 |

`container-pytorch-dlc` は AWS マネージドコンテナを使用するため、Dockerfile のビルドや ECR へのプッシュが不要で、開発サイクルが速いのが特徴です。追加パッケージは `source_dir` 内の `requirements.txt` に記載すると、コンテナ起動時に自動で `pip install` されます。

`container-pytorch-dlc-byoc` は同じ PyTorch DLC をベースにした BYOC イメージを使用します。依存パッケージは Dockerfile に焼き込むため、ジョブ起動が速く、ネットワークに依存しません。DLC にないライブラリが必要な場合や、完全に同じ環境で学習・評価を行いたい場合に選択してください。

`container-navsim-ego-mlp` / `container-navsim-transfuser` は NAVSIM の自動運転モデルを学習・評価する BYOC コンテナです。事前に `prepare_dataset.sh` で OpenScene データセットから特徴量を抽出する必要があります。詳細は各コンテナの README を参照してください。

PyTorch コンテナ (`container-pytorch-dlc` / `container-pytorch-dlc-byoc`) には `data/` サブディレクトリにサンプルデータセットが含まれています。

| ファイル | 行数 | 内容 |
|---------|------|------|
| `data/train.csv` | 800 行 | 学習データ (4 特徴量 `f1,f2,f3,f4` + ターゲット `target`、3 クラス分類) |
| `data/test.csv` | 200 行 | テストデータ (同形式) |

サンプルデータは `deploy.sh` の実行時に S3 に自動アップロードされます。データを変更した場合は `01-upload-dataset.sh` で再アップロードしてください。

```bash
# デフォルト (container-pytorch-dlc のデータ)
./pipelines/scripts/01-upload-dataset.sh

# NAVSIM Transfuser コンテナのデータ
```

実際のプロジェクトでは、`data/` 内のファイルを自分のデータセットに差し替えてください。`train.py` と `evaluate.py` が CSV の最終カラムをターゲットとして読み込む設計になっているため、カラム構成を変更する場合はスクリプトも合わせて修正してください。

`02-build-and-push-container.sh` の `-c` オプションでコンテナを切り替えます。

```bash
# デフォルト (NAVSIM Transfuser)
./pipelines/scripts/02-build-and-push-container.sh

# PyTorch DLC
./pipelines/scripts/02-build-and-push-container.sh -c container-pytorch-dlc
```

### モデルアーキテクチャについて

本プロジェクトで使用するモデルは以下の通りです。

| モデル | 概要 | コンテナ |
|-------|------|---------|
| SimpleClassifier | 3 層 MLP による分類モデル | PyTorch DLC ベース / PyTorch DLC ベース BYOC |
| EgoStatusMLP | 4 層 MLP による軌跡予測モデル | NAVSIM コンテナ (`container-navsim-ego-mlp`) |
| Transfuser | ResNet-34 + GPT-style Transformer によるマルチモーダル軌跡予測モデル | NAVSIM コンテナ (`container-navsim-transfuser`) |

すべてのコンテナで `train.py` / `evaluate.py` のコードはコンテナの提供方法 (DLC / BYOC) に依存しません。実際のプロジェクトでは、`SimpleClassifier` を ResNet や Transformer 等の事前学習済みモデルに置き換えることを想定しています。その場合は `requirements.txt` (DLC の場合) または `Dockerfile` (BYOC の場合) に追加ライブラリを記載してください。

### PyTorch DLC ベース

AWS マネージドの PyTorch DLC コンテナを使用する構成です。Dockerfile のビルドや ECR へのプッシュは不要です。追加パッケージは `requirements.txt` で管理します。コンテナ関連のファイルは `pipelines/container-pytorch-dlc/` に配置されています。

モデルは `SimpleClassifier` (3 層 MLP: 入力次元 → 64 → 32 → クラス数) で、`nn.Linear`、`nn.ReLU` 等の PyTorch 標準モジュールのみを使用しています。外部からの重みダウンロードは不要です。

利用可能なフレームワークバージョンは [Available Deep Learning Containers Images](https://github.com/aws/deep-learning-containers/blob/master/available_images.md) で確認してください。

### PyTorch DLC ベース BYOC

同じ PyTorch DLC をベースイメージとして使いますが、Dockerfile でカスタムイメージをビルドし、ECR にプッシュして使用する構成です。依存パッケージは Dockerfile に焼き込むため、ジョブ起動が速く、ネットワークに依存しません。コンテナ関連のファイルは `pipelines/container-pytorch-dlc-byoc/` に配置されています。

モデルは `container-pytorch-dlc` と同じ `SimpleClassifier` です。

> ⚠️ PyTorch DLC イメージは CUDA 等の GPU ライブラリを含むため、サイズが 10 GB 以上になります。初回ビルド時のダウンロードに時間がかかるため、十分なディスク空き容量 (20 GB 以上推奨) とネットワーク帯域を確保してください。マネージドコンテナで十分な場合は `container-pytorch-dlc` の利用を推奨します。

```bash
# ビルド & 実行
./pipelines/scripts/run-pipeline.sh -c container-pytorch-dlc-byoc
```

### NAVSIM コンテナ

NAVSIM の End-to-End 運転モデルを SageMaker 上で学習・評価する BYOC コンテナです。事前に `prepare_dataset.sh` で OpenScene データセットから特徴量を抽出する必要があります。

| コンテナ | エージェント | センサー | インスタンス |
|---------|------------|---------|------------|
| `container-navsim-ego-mlp` | EgoStatusMLP | なし (速度・加速度・コマンド) | CPU (`ml.c7i.xlarge`) |
| `container-navsim-transfuser` | Transfuser / LTF | カメラ + LiDAR (または カメラのみ) | GPU (`ml.g6.4xlarge`) |

`EgoStatusMLP` は 4 層 MLP で自車の状態 (速度・加速度・走行コマンド) から将来軌跡を予測する軽量モデルです。PyTorch 標準モジュールのみで構成されており、外部からの重みダウンロードは不要です。

`Transfuser` は `timm` の事前学習済み ResNet-34 をバックボーンに使用し、GPT-style Transformer でカメラ画像と LiDAR の特徴量を multi-scale fusion するモデルです。初回学習時に事前学習済み重みのダウンロードが発生します。

各コンテナの詳細は以下を参照してください。

- [container-navsim-ego-mlp/README.ja.md](../pipelines/container-navsim-ego-mlp/README.ja.md)
- [container-navsim-transfuser/README.ja.md](../pipelines/container-navsim-transfuser/README.ja.md)
- [NAVSIM 自動運転シミュレーション ガイド](navsim-guide.ja.md)

## モデルグループの設計

MLflow Model Registry および SageMaker Model Registry では、モデルをグループ (モデルパッケージグループ) 単位で管理します。グループはモデルのバージョン履歴を束ねる単位であり、「同じ問題を解くモデルのバージョン履歴」が 1 グループに対応します。

### 設計原則

モデルグループの設計で最も重要な原則は、**1 グループ = 1 つの問題 × 1 つのアーキテクチャ**です。

グループを分けるべき条件は以下の通りです。

- モデルアーキテクチャが異なる (例: RandomForest vs ニューラルネットワーク)
- 推論コンテナが異なる (例: scikit-learn vs PyTorch)
- 比較・評価の文脈が異なる (例: 精度指標の意味が変わる場合)

グループを分けるべきでない条件は以下の通りです。

- 同じアーキテクチャのハイパーパラメータ違い
- 同じモデルの再学習 (データ更新)
- 同じモデルの軽微な改善

### 本プロジェクトでの適用

本プロジェクトでは PyTorch と NAVSIM の複数のコンテナを使用しており、それぞれ別のモデルグループとして管理します。

| コンテナ | モデルグループ名 | 理由 |
|---------|---------------|------|
| `container-pytorch-dlc` | `{project}-pytorch` | SimpleClassifier (PyTorch)、CPU 学習 |
| `container-pytorch-dlc-byoc` | `{project}-pytorch-byoc` | SimpleClassifier (PyTorch)、BYOC イメージで学習 |
| `container-navsim-ego-mlp` | `{project}-navsim-ego-mlp` | EgoStatusMLP (NAVSIM)、CPU 推論 |
| `container-navsim-transfuser` | `{project}-navsim-transfuser` | Transfuser (NAVSIM)、GPU 推論 |

コンテナごとに別グループにする理由は以下の通りです。

- アーキテクチャが根本的に異なる
- 推論コンテナが異なるため、同一グループ内でバージョン比較しても意味がない
- 精度指標の傾向が異なり、バージョン間の比較が混乱を招く

### 推奨命名規則

モデルグループ名は `{project}-{task}-{framework}` の形式を推奨します。

```
sagemaker-ai-ml-pipeline-pytorch
sagemaker-ai-ml-pipeline-pytorch-byoc
sagemaker-ai-ml-pipeline-navsim-transfuser
```

`03-create-and-run-pipeline.py` では `--container-dir` の値に応じて `MODEL_GROUP_NAME` を自動設定します。

```bash
# PyTorch コンテナ → MODEL_GROUP_NAME = sagemaker-ai-ml-pipeline-pytorch
python pipelines/scripts/03-create-and-run-pipeline.py \
    --container-dir pipelines/container-pytorch-dlc ...

# NAVSIM Transfuser → MODEL_GROUP_NAME = sagemaker-ai-ml-pipeline-navsim-transfuser
python pipelines/scripts/03-create-and-run-pipeline.py \
    --container-dir pipelines/container-pytorch-dlc ...

# PyTorch BYOC コンテナ → MODEL_GROUP_NAME = sagemaker-ai-ml-pipeline-pytorch-byoc
python pipelines/scripts/03-create-and-run-pipeline.py \
    --container-dir pipelines/container-pytorch-dlc-byoc ...
```

## 参考リンク

本ドキュメントで参照している AWS ドキュメントの一覧です。

- [Docker containers for training and deploying models](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers.html) - SageMaker AI でのコンテナ利用の概要
- [SageMaker Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html) - モデルグループとバージョン管理の詳細
- [Pre-built SageMaker AI Docker images](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers-prebuilt.html) - プリビルトイメージの一覧と選択方法
- [Prebuilt SageMaker AI Docker images for deep learning](https://docs.aws.amazon.com/sagemaker/latest/dg/pre-built-containers-frameworks-deep-learning.html) - DLC の詳細とフレームワーク別の利用方法
- [Adapting your own Docker container to work with SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/docker-containers-adapt-your-own.html) - カスタムコンテナの構築要件
- [Available Deep Learning Containers Images (GitHub)](https://github.com/aws/deep-learning-containers/blob/master/available_images.md) - DLC のイメージ URI 一覧
- [SageMaker Training Toolkit (GitHub)](https://github.com/aws/sagemaker-training-toolkit) - Training Toolkit の仕様と使い方
- [SageMaker AI Training Storage](https://docs.aws.amazon.com/sagemaker/latest/dg/model-train-storage.html) - Training Job の入出力パス規約
- [Built-in algorithms and pretrained models](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html) - ビルトインアルゴリズムの一覧と使い方
