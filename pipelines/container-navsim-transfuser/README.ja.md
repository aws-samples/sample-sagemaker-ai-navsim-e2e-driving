# container-navsim-transfuser <!-- omit in toc -->

🌐 **Language**: 🇺🇸 [English](README.md) | 🇯🇵 [日本語](README.ja.md)

[NAVSIM](https://github.com/autonomousvision/navsim) の Transfuser アーキテクチャを SageMaker AI ML Pipeline 上で学習・評価するための BYOC コンテナ構成です。

NAVSIM 自体の概要 (Pseudo-Simulation、PDM Score 等) や本プロジェクトでのコンテナ設計方針については [NAVSIM 自動運転シミュレーション ガイド](../../docs/navsim-guide.ja.md) を参照してください。

- [本コンテナで実装しているもの](#本コンテナで実装しているもの)
- [ディレクトリ構成](#ディレクトリ構成)
- [データセット](#データセット)
- [クイックスタート](#クイックスタート)
- [学習パラメータ](#学習パラメータ)
- [評価メトリクス](#評価メトリクス)
- [カスタマイズのヒント](#カスタマイズのヒント)
- [参考リンク](#参考リンク)

## 本コンテナで実装しているもの

NAVSIM が提供する複数のベースラインエージェントのうち、Transfuser を SageMaker 上で動かせるようにしています。

Transfuser は、`timm` の事前学習済み ResNet-34 をバックボーンに使用し、GPT-style Transformer で画像と LiDAR の特徴量を multi-scale fusion するモデルです。Transformer Decoder で軌跡予測・物体検出・BEV セマンティックセグメンテーションを同時に学習します。

`--latent true` を指定すると Latent Transfuser (LTF) モードになり、LiDAR 入力なしでカメラ画像のみから軌跡を予測します。

学習・評価の流れは以下の通りです。

```
データセット準備 (prepare_dataset.sh)
  ↓ navsim mini split をダウンロード → 特徴量抽出 → S3 アップロード
SageMaker Training Job (train.py)
  ↓ Transfuser を学習 → model.pth を S3 に保存
SageMaker Processing Job (evaluate.py)
  ↓ テストデータで推論 → ADE / FDE / PDM Score を計算
evaluation.json を S3 に出力
```

## ディレクトリ構成

各ファイルの役割は以下の通りです。

| ファイル | 役割 |
|---------|------|
| `Dockerfile` | PyTorch DLC (Python 3.11) ベースの BYOC イメージ。navsim devkit はインストールしない |
| `train.py` | Transfuser の学習スクリプト。SageMaker Training Job の entry_point として実行される |
| `evaluate.py` | 学習済みモデルの評価スクリプト。SageMaker Processing Job として実行される |
| `transfuser_config.py` | TransfuserConfig |
| `transfuser_backbone.py` | TransfuserBackbone: timm ResNet-34 + GPT fusion |
| `transfuser_model.py` | TransfuserModel: Backbone + Decoder + Heads |
| `transfuser_loss.py` | 損失関数: trajectory + detection + BEV semantic |
| `requirements.txt` | コンテナに追加インストールする Python パッケージ |
| `scripts/prepare_dataset.sh` | navsim リポジトリの clone、mini split のダウンロード、特徴量抽出、データバランシング、S3 アップロードを一括実行 |
| `scripts/extract_features.py` | navsim の SceneLoader でカメラ・LiDAR BEV・EgoStatus・GT 軌跡を pt 形式に変換する |
| `scripts/balance_dataset.py` | 学習データのコマンド分布 (LEFT / FORWARD / RIGHT) を完全均等化する |

`train.py` / `evaluate.py` は Dockerfile に COPY せず、SageMaker SDK の `entry_point` + `source_dir` 経由でコンテナに注入されます。そのため、スクリプトを変更してもコンテナの再ビルドは不要です。

## データセット

NAVSIM は [nuPlan](https://www.nuscenes.org/nuplan) / [OpenScene](https://github.com/OpenDriveLab/OpenScene) データセットを使用します。本コンテナではカメラ画像と LiDAR データを含む mini split (約 20 GB) を使用します。

データセットの[ライセンス](https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE)を事前に確認してください。

`prepare_dataset.sh` は conda で Python 3.9 環境 (`navsim-py39`) を自動作成し、navsim devkit をインストールして特徴量を抽出します。SageMaker AI Notebook (Python 3.11) との依存関係の競合を回避するための仕組みです。

1. conda 環境 `navsim-py39` (Python 3.9) を作成 (2 回目以降はスキップ)
2. navsim devkit + nuplan-devkit をインストール
3. mini split のデータをダウンロード (カメラ画像 + LiDAR 含む)
4. navsim の SceneLoader でカメラ画像・LiDAR BEV・EgoStatus・GT 軌跡を抽出
5. `balance_dataset.py` でコマンド分布 (LEFT / FORWARD / RIGHT) を完全均等化
6. pt 形式で train/test に分割し、S3 データセットバケットにアップロード

### データバランシングについて

OpenScene データセットは FORWARD コマンドが大多数を占める不均衡な分布になっており、そのまま学習すると LEFT / RIGHT コマンドへの反応が弱いモデルが出来上がります。`balance_dataset.py` は最小クラスのサンプル数に合わせて他クラスをダウンサンプリングすることで、モデルが全コマンドに均等に反応するように調整します。

- **戦略**: `equal` (最小クラスに合わせた完全均等化)
- **除外**: `UNKNOWN` コマンド (Index 3)
- **バックアップ**: 元のデータは `train_original/` に保持
- **不均衡度が低い場合** (max/min < 2.0x) はスキップ

`prepare_dataset.sh` を実行せずに Pipeline を実行した場合、`train.py` のフォールバック機能によりダミーデータで学習が行われます (動作確認用)。実データで学習するには `prepare_dataset.sh` の実行が必要です。詳細は [NAVSIM ガイド - データセット準備](../../docs/navsim-guide.ja.md#データセット準備) を参照してください。

所要時間の目安は以下の通りです。カメラ画像と LiDAR データを含むため、EgoStatusMLP よりも大きく時間がかかります。

| ステップ | 所要時間 |
|---------|---------|
| conda 環境構築 | 約 5 分 |
| OpenScene mini split ダウンロード | 約 40 分 |
| 地図ダウンロード | 約 1 分 |
| 特徴量抽出 | 約 85 分 |
| データバランシング | 約 5 分 |
| S3 アップロード | 約 5 分 |
| **合計** | **約 140 分** |

### サンプル数と学習時間の目安

デフォルト設定 (`mini` split + バランシング) では約 381 サンプルまで削減されます。用途に応じて以下の選択肢があります。

| 構成 | サンプル数 | 学習時間 (ml.g6.4xlarge, 30 epochs) | 用途 |
|------|-----------|--------------------------------------|------|
| **mini split + バランシング** (デフォルト) | 約 381 | 約 10 分 | 動作確認・プロトタイピング |
| mini split (バランシングなし) | 約 2,892 | 約 90 分 | データ削減を避けたい場合 |
| trainval split + バランシング | 約 38,000 | 約 16 時間 (推定) | 本番向け、より高精度なモデル |

trainval 学習時間は mini split の実測値からの線形外挿で、実測はしていません。GPU 使用率、データローダーのスループット、ストレージ I/O によって変動します。

`trainval` split を使用する場合は、`prepare_dataset.sh` 内の `extract_features.py` 呼び出しで `--split trainval` に変更してください。ダウンロードサイズとディスク消費が大きく増えるため、先に [NAVSIM ガイド - データセット準備](../../docs/navsim-guide.ja.md#データセット準備) のディスク容量見積もりを確認してください。

## クイックスタート

### 1. データセット準備

```bash
./pipelines/container-navsim-transfuser/scripts/prepare_dataset.sh
```

スクリプト実行時にライセンスへの同意を求められます。

### 2. コンテナビルド & Pipeline 実行

```bash
# コンテナビルド & ECR プッシュ
./pipelines/scripts/02-build-and-push-container.sh -c container-navsim-transfuser

# Pipeline 一括実行 (データアップロード → コンテナビルド → Pipeline 実行 → 完了待ち)
./pipelines/scripts/run-pipeline.sh -c container-navsim-transfuser --skip-upload
```

### 3. Notebook から実行

`notebooks/navsim-transfuser-pipeline.ipynb` を JupyterLab で開いて、セルを順に実行してください。データ準備から学習・評価・結果確認まで一通りカバーしています。

## 学習パラメータ

`train.py` で使用可能なハイパーパラメータは以下の通りです。SageMaker Estimator の `hyperparameters` 引数で渡します。

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--epochs` | 30 | 学習エポック数 |
| `--batch-size` | 16 | ミニバッチサイズ |
| `--learning-rate` | 0.0003 | 学習率 |
| `--latent` | `false` | `true` で LTF モード (LiDAR なし、カメラ画像のみ) |

## 評価メトリクス

`evaluate.py` は PDM Score に準じた簡易メトリクスを計算します。完全な PDM Score の計算には nuPlan マップやシーン情報が必要なため、ここでは軌跡予測の精度に焦点を当てています。

| メトリクス | 説明 |
|-----------|------|
| `pdm_score` | ADE ベースの簡易総合スコア (0-1)。低い ADE ほど高スコア |
| `ade` | Average Displacement Error。全タイムステップにおける予測位置と GT 位置の平均 L2 距離 (m) |
| `fde` | Final Displacement Error。最終タイムステップの L2 距離 (m)。長期予測の精度を示す |
| `heading_error` | 予測 heading と GT heading の平均絶対誤差 (rad) |
| `miss_rate` | FDE > 2.0 m のサンプル割合。大きく外れた予測の頻度を示す |

## カスタマイズのヒント

Transfuser はカメラ + LiDAR を使用するセンサーエージェントです。カスタマイズする場合は、以下を参考にしてください。

- `--latent true` で LTF モードに切り替えると、LiDAR なしでカメラ画像のみから軌跡を予測します。LiDAR データの準備が不要になるため、データセット準備が軽量になります
- インスタンスタイプは `ml.g6.4xlarge` (GPU) が自動選択されます。カメラ画像 + LiDAR の CNN 処理に GPU が必須で、データサイズが大きいため 64GB RAM が必要です
- NAVSIM v2 (Pseudo-Simulation 対応) を使用する場合は、Dockerfile の clone ブランチを `main` に変更し、依存関係を調整してください

## 参考リンク

- [NAVSIM GitHub リポジトリ](https://github.com/autonomousvision/navsim)
- [NAVSIM Paper (NeurIPS 2024)](https://arxiv.org/abs/2406.15349) - NAVSIM v1 の原論文
- [Pseudo-Simulation Paper (CoRL 2025)](https://arxiv.org/abs/2506.04218) - NAVSIM v2 の Pseudo-Simulation 手法
- [nuPlan データセット](https://www.nuscenes.org/nuplan)
- [ML/AI モデル開発ガイド](../../docs/model-development-guide.ja.md) - 本プロジェクトのコンテナ構成全般
