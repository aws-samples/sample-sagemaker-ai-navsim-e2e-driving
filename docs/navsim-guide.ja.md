# NAVSIM 自動運転シミュレーション ガイド <!-- omit in toc -->

🌐 **Language**: 🇺🇸 [English](navsim-guide.md) | 🇯🇵 [日本語](navsim-guide.ja.md)

[NAVSIM](https://github.com/autonomousvision/navsim) を SageMaker AI ML Pipeline 上で活用するための包括的なガイドです。NAVSIM の概要、ベースラインエージェントの比較、SageMaker Python SDK での実装パターンを解説します。

- [NAVSIM とは](#navsim-とは)
  - [評価手法の課題と Pseudo-Simulation](#評価手法の課題と-pseudo-simulation)
  - [PDM Score](#pdm-score)
  - [データセット](#データセット)
  - [バージョン](#バージョン)
- [ベースラインエージェント](#ベースラインエージェント)
  - [一覧と比較](#一覧と比較)
  - [ConstantVelocityAgent](#constantvelocityagent)
  - [EgoStatusMLPAgent](#egostatusmlpagent)
  - [TransfuserAgent](#transfuseragent)
  - [Latent TransfuserAgent (LTF)](#latent-transfuseragent-ltf)
- [SageMaker AI での実装パターン](#sagemaker-ai-での実装パターン)
  - [コンテナ設計の方針](#コンテナ設計の方針)
  - [NAVSIM 公式コードの移植方針](#navsim-公式コードの移植方針)
  - [なぜ PyTorch DLC を使うのか](#なぜ-pytorch-dlc-を使うのか)
  - [EgoStatusMLP の実装例](#egostatusmlp-の実装例)
  - [Latent Transfuser への切り替え](#latent-transfuser-への切り替え)
- [本プロジェクトのコンテナ構成](#本プロジェクトのコンテナ構成)
  - [インスタンスタイプとパフォーマンス](#インスタンスタイプとパフォーマンス)
- [データセット準備](#データセット準備)
  - [前提条件](#前提条件)
  - [ディスク容量の目安](#ディスク容量の目安)
  - [実行手順](#実行手順)
  - [データと Pipeline の自動紐づけ](#データと-pipeline-の自動紐づけ)
  - [出力データ形式](#出力データ形式)
  - [ダミーデータについて](#ダミーデータについて)
- [参考リンク](#参考リンク)

## NAVSIM とは

NAVSIM (Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking) は、自動運転車の End-to-End 運転モデルを評価するためのフレームワークです。University of Tübingen、NVIDIA Research、Robert Bosch GmbH 等の共同研究として開発されています。

### 評価手法の課題と Pseudo-Simulation

自動運転モデルの評価手法には大きく 2 つのアプローチがあり、それぞれに課題があります。

| アプローチ | 特徴 | 課題 |
|-----------|------|------|
| Open-loop 評価 | 記録済みデータに対して予測を行い、GT と比較する。高速で大規模に実行可能 | エラー蓄積や回復動作を評価できない。予測軌跡が GT から逸脱した場合の影響を測定できない |
| Closed-loop 評価 | シミュレータ内でモデルを実際に走行させる。現実に近い評価が可能 | 計算コストが高い。モデルへのアクセス (推論 API) が必要。スケーラビリティに課題がある |

NAVSIM はこの 2 つのギャップを埋める Pseudo-Simulation を提案しています。実データに合成観測を加えることで、Open-loop の効率性を保ちつつ Closed-loop に近い評価精度を実現します。

Pseudo-Simulation の主な特徴は以下の通りです。

- Closed-loop 比で約 6 倍高速に動作する
- モデル予測のみで評価できる (推論 API へのアクセスが不要)
- 逐次的・対話的な処理が不要なため、大規模なリーダーボード運用に適している
- CVPR 2024 のコンペティションでは 143 チーム・463 エントリーが参加した

### PDM Score

NAVSIM の評価指標は PDM Score (Predictive Driver Model Score) です。以下の要素を総合的にスコア化します。

- 衝突回避 (collision avoidance)
- 走行可能領域の遵守 (drivable area compliance)
- 快適性 (comfort)
- 進行度 (progress)
- 車線逸脱 (time to collision)

v2 では Extended PDM Score (EPDMS) に拡張され、より多くのメトリクスとペナルティが追加されています。

### データセット

NAVSIM は [nuPlan](https://www.nuscenes.org/nuplan) / [OpenScene](https://github.com/OpenDriveLab/OpenScene) データセットを使用します。

利用可能なデータ分割は以下の通りです。

| Split | 用途 | サイズ |
|-------|------|--------|
| mini | 開発・デバッグ用の最小構成 | 約 5 GB |
| trainval | 学習・検証用のフルデータセット | 約 100 GB |
| test | テスト用 (リーダーボード評価) | 約 50 GB |
| navtrain | trainval の一部 (学習用サブセット) | 約 20 GB |

データセットの[ライセンス](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE)を事前に確認してください。

### バージョン

NAVSIM には 2 つのメジャーバージョンがあります。

| バージョン | ブランチ | 論文 | 主な特徴 |
|-----------|---------|------|---------|
| v1.1 | [v1.1](https://github.com/autonomousvision/navsim/tree/v1.1) | [NeurIPS 2024](https://arxiv.org/abs/2406.15349) | 初版。Open-loop 評価 + PDM Score |
| v2.x | [main](https://github.com/autonomousvision/navsim) | [CoRL 2025](https://arxiv.org/abs/2506.04218) | Pseudo-Simulation、Extended PDM Score、リアクティブ交通エージェント |

両バージョンとも Python 3.9 を前提としています (nuplan-devkit の制約)。

## ベースラインエージェント

### 一覧と比較

NAVSIM が公式に提供するベースラインエージェントは以下の 4 つです。

| エージェント | センサー入力 | 学習 | GPU 必須 | 概要 |
|-------------|------------|------|---------|------|
| ConstantVelocity | なし | なし | 不要 | 等速直線運動。ルールベースの最もシンプルなベースライン |
| EgoStatusMLP | 速度・加速度・走行コマンド | あり | 不要 | センサーなしの上限を示す軽量 MLP (Multi-Layer Perceptron) |
| Transfuser | カメラ (前方 3 台合成) + LiDAR BEV | あり | 必要 | CNN + Transformer で画像と LiDAR を融合 |
| Latent Transfuser (LTF) | カメラ (前方 3 台合成) のみ | あり | 必要 | Transfuser の LiDAR を位置エンコーディングで代替 |

学習済みチェックポイントは [Hugging Face](https://huggingface.co/autonomousvision/navsim_baselines) で公開されています。

### ConstantVelocityAgent

最もシンプルなベースラインです。現在の速度と heading を維持して直進する軌跡を出力します。学習は不要で、`AbstractAgent` インターフェースの理解や、PDM Score が高くなりやすいシーンの分析に使用します。

入出力は以下の通りです。

- 入力: 自車速度 (ego_velocity)
- 出力: 等速直線運動の軌跡 (4 秒間、0.5 秒間隔 × 8 ポーズ)

### EgoStatusMLPAgent

カメラや LiDAR を一切使わず、自車の状態のみから軌跡を予測する「ブラインド」ベースラインです。自車の運動学的状態の外挿だけでどこまで性能が出るかの上限を示します。

入出力は以下の通りです。

- 入力 (8 次元): velocity_x, velocity_y, accel_x, accel_y, cmd_left, cmd_straight, cmd_right, cmd_unknown
- 出力: 将来軌跡 (8 ポーズ × 3 次元: x, y, heading)
- 損失関数: L1 Loss (MAE)
- アーキテクチャ: 4 層 MLP (8 → hidden_dim → hidden_dim → hidden_dim → num_poses × 3)

### TransfuserAgent

カメラ画像と LiDAR の BEV (Bird's Eye View) を融合するセンサーエージェントです。[CARLA Garage](https://github.com/autonomousvision/carla_garage) の Transfuser バックボーンをベースにしています。

入出力は以下の通りです。

- カメラ入力: 前方 3 台 (cam_l0, cam_f0, cam_r0) をスティッチして 1024×256 の広角画像を生成
- LiDAR 入力: 点群を 256×256 の BEV ヒストグラムに変換
- EgoStatus 入力: 走行コマンド + 速度 + 加速度
- 出力: 将来軌跡 + 物体検出 (DETR スタイル) + BEV セグメンテーション

アーキテクチャの特徴は以下の通りです。

- カメラブランチ: ResNet-34 で画像特徴量を抽出
- LiDAR ブランチ: ResNet-34 で BEV ヒストグラムの特徴量を抽出
- 融合: 複数の Transformer レイヤーでカメラと LiDAR の特徴量を段階的に融合
- 補助タスク: BEV セマンティックセグメンテーション + DETR スタイルの物体検出 (Hungarian matching)

### Latent TransfuserAgent (LTF)

Transfuser の LiDAR 入力を位置エンコーディング (positional encoding) で代替したバリアントです。LiDAR が利用できない環境でも動作するため、より柔軟です。

Transfuser との違いは以下の 1 点のみです。

- `TransfuserConfig.latent = True` に設定する
- LiDAR データの読み込みがスキップされ、代わりに位置エンコーディングが使用される

CARLA リーダーボードでは、画像のみの手法の中で最も高い性能を示しています。

## SageMaker AI での実装パターン

### コンテナ設計の方針

navsim v1.1 / v2 は Python 3.9 + `numpy==1.23.4` + `torch==2.0.1` 等の古いバージョンをピン留めしています。これは nuplan-devkit の制約によるもので、PyTorch DLC (Python 3.11 + numpy 2.x + torch 2.5.x) と依存関係が競合します。

この問題に対して、本プロジェクトでは以下の方針を採用しています。

- 特徴量抽出 (navsim の SceneLoader を使用) は SageMaker AI Notebook 上で conda の Python 3.9 環境を自動作成して実行する (`prepare_dataset.sh`)
- SageMaker Training Job / Processing Job のコンテナには navsim devkit をインストールせず、PyTorch DLC ベースの軽量イメージを使用する
- 学習・評価コンテナには前処理済みデータ (npz / pt 形式) だけを渡す
- これにより、ビルド速度と最新の PyTorch 最適化を活用できる

### NAVSIM 公式コードの移植方針

NAVSIM 公式の Transfuser モデル定義ファイル (`transfuser_backbone.py` 等) は、navsim devkit 全体に依存しているわけではありませんが、トップレベルの import に `nuplan` パッケージへの依存があります。`nuplan-devkit` を pip install すると Python 3.9 制約により DLC と競合するため、公式コードをそのまま使うことはできません。

本プロジェクトでは、公式コードをコピーし、`nuplan` / `navsim` の import をローカル参照または自前定義に置き換える方式を採用しています。アーキテクチャ (モデル構造、重みの初期化、損失関数のロジック) は一切変更していません。

具体的な変更箇所は以下の通りです。

| ファイル | 変更内容 |
|---------|---------|
| `transfuser_backbone.py` | `from navsim...` → `from transfuser_config` (import 1 行) |
| `transfuser_model.py` | import 変更 + `StateSE2Index` / `BoundingBox2DIndex` の自前定義 |
| `transfuser_config.py` | `nuplan` の import を削除、BEV クラス定義を整数に変更 |
| `transfuser_loss.py` | `from navsim...` → ローカル参照 (import 2 行) |

### なぜ PyTorch DLC を使うのか

EgoStatusMLP / Transfuser ともに PyTorch (nn.Module, DataLoader, optim 等) でモデルを実装しているため、PyTorch が必要です。ベースイメージとして AWS の PyTorch DLC (Deep Learning Container) を採用している理由は以下の通りです。

- PyTorch、torchvision、numpy、CUDA ランタイムがプリインストール済みで、`pip install torch` のような大きなインストールが不要
- SageMaker Training Toolkit が組み込まれており、`entry_point` / `source_dir` によるスクリプト注入がそのまま動作する
- AWS が継続的にセキュリティパッチを適用しており、自前でベースイメージを管理する必要がない
- 分散学習 (Data Parallel / Model Parallel) のサポートが組み込まれている

全コンテナで GPU 版 DLC (`pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker`) を使用しています。GPU 版は CPU インスタンスでも問題なく動作するため (CUDA ライブラリが追加で入っているだけ)、Dockerfile を統一してビルドキャッシュを共有しています。

### EgoStatusMLP の実装例

本プロジェクトの EgoStatusMLP は、NAVSIM 公式の `EgoStatusMLPAgent` と同一のアーキテクチャ (4 層 MLP: `Linear(8→hidden) → ReLU → Linear(hidden→hidden) → ReLU → Linear(hidden→hidden) → ReLU → Linear(hidden→num_poses*3)`) を採用しています。`navsim` パッケージへの依存を排除し、SageMaker Training Job で独立して実行できるよう再実装したものです。

EgoStatusMLP は入力が 8 次元のベクトルのみなので、npz 形式で特徴量を保存し、コンテナ内で読み込みます。

```python
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

estimator = Estimator(
    image_uri=ecr_image_uri,
    entry_point="train.py",
    source_dir="pipelines/container-navsim-ego-mlp",
    role=role_arn,
    instance_count=1,
    instance_type="ml.c7i.xlarge",  # CPU で十分
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
    # input_mode: "FastFile" = S3 からオンデマンドストリーミング / "File" = 全量ダウンロード後に学習開始
    inputs={"train": TrainingInput(s3_data=train_data_uri, input_mode="FastFile")},
    wait=True,
)
```

詳細は `pipelines/container-navsim-ego-mlp/README.ja.md` を参照してください。

### Latent Transfuser への切り替え

Transfuser と Latent Transfuser (LTF) は同じコンテナ (`container-navsim-transfuser`) で動作します。切り替えはハイパーパラメータ `latent` で行います。

```python
# Transfuser (カメラ + LiDAR)
hyperparameters={"latent": "false", ...}

# Latent Transfuser (カメラのみ)
hyperparameters={"latent": "true", ...}
```

`train.py` 内では `--latent` 引数を argparse で受け取り、モデルの初期化時に反映します。`latent=true` の場合、LiDAR 特徴量の読み込みがスキップされ、代わりに位置エンコーディングが使用されます。

## 本プロジェクトのコンテナ構成

NAVSIM 関連のコンテナは以下の通りです。

| コンテナ | エージェント | センサー | インスタンス |
|---------|------------|---------|------------|
| `container-navsim-ego-mlp` | EgoStatusMLP | なし (速度・加速度・コマンド) | CPU (`ml.c7i.xlarge`) |
| `container-navsim-transfuser` | Transfuser / LTF | カメラ + LiDAR (または カメラのみ) | GPU (`ml.g6.4xlarge`) |

### インスタンスタイプとパフォーマンス

インスタンスタイプの選択は、各エージェントの実装上のボトルネックに基づいています。

**EgoStatusMLP** (`ml.c7i.xlarge`): データが 8 次元ベクトルの npz で非常に小さく、MLP も軽量です。DataLoader の `num_workers` はデフォルト (0 = メインスレッドのみ) で、DataParallel も使用していません。ボトルネックは純粋な計算量ですが、そもそも計算量が少ないため CPU で十分です。

**Transfuser** (`ml.g6.4xlarge`): CNN + Transformer の forward/backward が GPU バウンドです。DataLoader は `num_workers=2` でデータ読み込みを並列化しています。`ml.g6.4xlarge` は L4 GPU 1 台 + 16 vCPU + 64GB RAM で、十分なメモリと並列データ読み込み性能を備えています。GPU 性能自体を上げたい場合は `ml.g6.12xlarge` (L4 4 台) + DataParallel の実装、または `ml.p4d.24xlarge` (A100) 等が必要です。


## データセット準備

NAVSIM のデータセットは nuPlan / OpenScene に基づいており、特徴量の抽出に navsim devkit (Python 3.9) が必要です。SageMaker AI Notebook (Python 3.11) とは依存関係が競合するため、`prepare_dataset.sh` が conda で Python 3.9 環境を自動作成して特徴量を抽出します。

### 前提条件

以下の条件を満たす環境で実行してください。

- conda がインストール済み (SageMaker AI Notebook にはプリインストール)
- AWS CLI が設定済み
- ディスク容量 500 GB 推奨 (mini split のセンサーデータ ~151 GB + conda 環境 + 展開の一時領域で合計約 210 GB を使用)
- [nuPlan データセットのライセンス](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE)に同意していること

### ディスク容量の目安

mini split を使用する場合の必要容量は以下の通りです。

| 用途 | サイズ |
|------|--------|
| conda 環境 (navsim-py39) | ~5 GB |
| ダウンロード (tgz 圧縮) | ~50 GB |
| 展開後 (ログ + センサーデータ) | ~152 GB |
| 抽出した特徴量 (pt / npz) | 数 MB〜数 GB |
| **合計** | **~210 GB** |

### 実行手順

JupyterLab のターミナルで以下を実行します。

```bash
# EgoStatusMLP 用 (数値データのみ、抽出後 数 MB)
./pipelines/container-navsim-ego-mlp/scripts/prepare_dataset.sh

# Transfuser 用 (カメラ + LiDAR + EgoStatus + 補助タスク、抽出後 数 GB)
./pipelines/container-navsim-transfuser/scripts/prepare_dataset.sh

```

各スクリプトは以下の処理を自動で行います。

1. conda 環境 `navsim-py39` (Python 3.9) を作成 (2 回目以降はスキップ)
2. navsim devkit + nuplan-devkit をインストール
3. mini split データセットをダウンロード (ログ ~1 GB + センサーデータ ~151 GB)
4. `extract_features.py` で特徴量を抽出
5. `balance_dataset.py` でコマンド分布を完全均等化
6. S3 データセットバケットにアップロード

conda 環境は EgoStatusMLP と Transfuser で共有されるため、2 つ目以降のスクリプトは環境作成がスキップされます。

### データバランシング

OpenScene データセットは FORWARD コマンドが大多数を占める不均衡な分布になっており、そのまま学習するとモデルが LEFT / RIGHT コマンドに反応しにくくなります。`prepare_dataset.sh` は抽出した特徴量に対して `balance_dataset.py` を実行し、コマンド分布 (LEFT / FORWARD / RIGHT) を完全均等化します。EgoStatusMLP と Transfuser の両方で実施しています。

- **戦略**: 最小クラスのサンプル数に合わせて他クラスをダウンサンプリング (`--strategy equal`)
- **除外**: `UNKNOWN` コマンド (Index 3) をバランシング対象外 (`--exclude-unknown`)
- **バックアップ**:
  - EgoStatusMLP: 元データを `train_data_original.npz` に保持、バランス後の `train_data.npz` を S3 にアップロード
  - Transfuser: 元データを `train_original/` に保持、バランス後のデータを `train/` として S3 にアップロード
- **スキップ条件**: 元データの不均衡度 (max/min) が 2.0x 未満の場合はスキップ

### データと Pipeline の自動紐づけ

`prepare_dataset.sh` でアップロードしたデータは、Pipeline 実行時に自動的に使用されます。仕組みは以下の通りです。

1. `prepare_dataset.sh` がコンテナ名をプレフィックスとして S3 にアップロード
2. `03-create-and-run-pipeline.py` が同じコンテナ名から S3 パスを自動生成
3. SageMaker がデータをコンテナ内の `/opt/ml/input/data/train/` にマウント
4. `train.py` が環境変数 `SM_CHANNEL_TRAIN` 経由でデータを読み込み

```
s3://{project}-dataset-{account}-{region}/
  ├── container-navsim-ego-mlp/
  │   ├── train/   ← prepare_dataset.sh がアップロード
  │   └── test/
  └── container-navsim-transfuser/
      ├── train/   ← Pipeline 実行時に自動参照
      └── test/
```

データ準備後は `--skip-upload` を指定して Pipeline を実行します。

```bash
./pipelines/scripts/run-pipeline.sh -c container-navsim-transfuser --skip-upload
```

### 出力データ形式

抽出されるデータ形式はモデルによって異なります。

| モデル | 形式 | 内容 | サイズ目安 |
|--------|------|------|-----------|
| EgoStatusMLP | npz | `features` [N, 8] + `targets` [N, 8, 3] | 数 MB |
| Transfuser | pt | `camera` [3, 256, 1024] + `lidar` [1, 256, 256] + `status` [8] + `trajectory` [8, 3] | 数 GB |

### ダミーデータについて

`prepare_dataset.sh` を実行せずに Pipeline を実行した場合、`train.py` のフォールバック機能によりダミーデータ (ランダム生成) で学習が行われます。Pipeline の動作確認には使えますが、モデルの予測精度は期待できません。実データで学習するには `prepare_dataset.sh` の実行が必要です。

## 参考リンク

- [NAVSIM GitHub リポジトリ](https://github.com/autonomousvision/navsim)
- [NAVSIM Paper (NeurIPS 2024)](https://arxiv.org/abs/2406.15349) - v1 の原論文
- [Pseudo-Simulation Paper (CoRL 2025)](https://arxiv.org/abs/2506.04218) - v2 の Pseudo-Simulation 手法
- [Transfuser Paper](https://arxiv.org/abs/2205.15997) - Transfuser アーキテクチャの原論文
- [NAVSIM ベースラインチェックポイント (Hugging Face)](https://huggingface.co/autonomousvision/navsim_baselines)
- [nuPlan データセット](https://www.nuscenes.org/nuplan)
- [ML/AI モデル開発ガイド](model-development-guide.ja.md) - 本プロジェクトのコンテナ構成全般
