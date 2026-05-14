# セットアップと実行ガイド <!-- omit in toc -->

🌐 **Language**: 🇺🇸 [English](setup-guide.md) | 🇯🇵 [日本語](setup-guide.ja.md)

NAVSIM 自動運転モデルの学習パイプラインを Amazon SageMaker AI 上で構築・実行するための手順です。AWS インフラ (S3、ECR、SageMaker AI Notebook、MLflow) のデプロイ、Pipeline の実行、推論エンドポイントのデプロイ、デモアプリの起動、SageMaker Unified Studio との連携までをカバーしています。

- [初期セットアップ (One-time)](#初期セットアップ-one-time)
  - [Step 1: .env ファイルの設定](#step-1-env-ファイルの設定)
      - [GitHub リポジトリ連携](#github-リポジトリ連携)
      - [VPC 構成](#vpc-構成)
  - [Step 2: デプロイの実行](#step-2-デプロイの実行)
- [開発環境の利用](#開発環境の利用)
  - [JupyterLab](#jupyterlab)
  - [MLflow UI](#mlflow-ui)
- [Pipeline 実行 (繰り返し)](#pipeline-実行-繰り返し)
  - [Step 1: ソースコードの準備](#step-1-ソースコードの準備)
  - [Step 2: データセットのアップロード](#step-2-データセットのアップロード)
  - [Step 3: コンテナビルド \& ECR プッシュ](#step-3-コンテナビルド--ecr-プッシュ)
  - [Step 4: Pipeline 作成 \& 実行](#step-4-pipeline-作成--実行)
  - [Step 5: 実行状況の確認](#step-5-実行状況の確認)
  - [Jupyter Notebook での実行](#jupyter-notebook-での実行)
- [CARLA シミュレーションデモ](#carla-シミュレーションデモ)
- [推論エンドポイントとデモアプリ](#推論エンドポイントとデモアプリ)
  - [Step 1: 推論エンドポイントのデプロイ](#step-1-推論エンドポイントのデプロイ)
  - [Step 2: デモアプリの起動](#step-2-デモアプリの起動)
  - [推論エンドポイントの削除](#推論エンドポイントの削除)
- [SageMaker Unified Studio 連携 (オプション)](#sagemaker-unified-studio-連携-オプション)
  - [Step 1: Unified Studio ドメインの作成](#step-1-unified-studio-ドメインの作成)
  - [Step 2: プロジェクトの作成](#step-2-プロジェクトの作成)
  - [Step 3: SageMaker リソースとの連携](#step-3-sagemaker-リソースとの連携)
  - [連携の解除](#連携の解除)
- [テスト](#テスト)
  - [Lint チェック (run-lint.sh)](#lint-チェック-run-lintsh)
  - [統合テスト (run-tests.sh)](#統合テスト-run-testssh)
- [クリーンアップ](#クリーンアップ)
- [トラブルシューティング](#トラブルシューティング)
- [Appendix](#appendix)
  - [作成される AWS リソース](#作成される-aws-リソース)
  - [関連情報](#関連情報)
    - [AWS ドキュメント](#aws-ドキュメント)
    - [関連ワークショップ・サンプル](#関連ワークショップサンプル)

## 初期セットアップ (One-time)

CloudFormation で S3 バケット、ECR リポジトリ、SageMaker AI Notebook、MLflow App などの AWS リソースを一括作成します。サンプルデータセットの S3 アップロードもデプロイ時に自動で行われます。スタックの作成完了まで 10〜15 分程度かかります。

> ⚠️ デプロイ前に AWS CLI の[認証情報を設定](https://docs.aws.amazon.com/ja_jp/cli/latest/userguide/cli-chap-authentication.html)してください。
>
> **IAM Identity Center (SSO) の場合**:
> ```bash
> aws sso login --profile <your-profile>
> export AWS_PROFILE=<your-profile>
> ```
>
> **IAM ユーザーのアクセスキーの場合**:
> ```bash
> aws configure
> ```
>
> **認証の確認**:
> ```bash
> aws sts get-caller-identity
> ```

### Step 1: .env ファイルの設定

`.env` ファイルを作成し、デプロイ先リージョンを確認・設定します。GitHub リポジトリ連携や VPC 構成もここで設定できます。

```bash
cp .env.example.ja .env
# .env を編集してリージョン等の設定を行う
```

`.env` ファイルまたは環境変数でカスタマイズ可能な設定です。

| 環境変数 | デフォルト値 | 説明 |
|---------|------------|------|
| `AWS_DEFAULT_REGION` | `us-east-1` | デプロイ先リージョン |
| `NOTEBOOK_IDLE_TIMEOUT_MIN` | `60` | Notebook のアイドル自動停止までの分数 (最小 5 分) |
| `GITHUB_REPO` | (なし) | Notebook に連携する GitHub リポジトリ URL (オプション)。ユーザー名は URL から自動抽出 |
| `GITHUB_PAT` | (なし) | GitHub Personal Access Token (プライベートリポジトリの場合) |
| `ENABLE_VPC` | `false` | VPC 構成の有効化。`true` で全コンポーネントを VPC 内に配置 |
| `VPC_ID` | (なし) | 既存 VPC の ID (空の場合は新規 VPC を作成) |
| `SUBNET_IDS` | (なし) | 既存サブネット ID (カンマ区切り、`VPC_ID` 指定時に必要) |
| `SECURITY_GROUP_ID` | (なし) | 既存セキュリティグループ ID (`VPC_ID` 指定時に必要) |
| `CREATE_VPC_ENDPOINTS` | `true` | VPC Endpoint の作成。既存 VPC に Endpoint がある場合は `false` |

`.env` ファイルの設定例です。

```bash
# GitHub リポジトリ連携 (オプション)
GITHUB_REPO="https://github.com/your-username/your-repo.git"
GITHUB_PAT="your-github-personal-access-token"

# VPC 構成 (オプション)
ENABLE_VPC=true
```

> ⚠️ `.env` ファイルは `.gitignore` に含まれているため、リポジトリにはコミットされません。PAT を直接コミットしないよう注意してください。

##### GitHub リポジトリ連携


`GITHUB_REPO` が設定されている場合、デプロイスクリプトは CloudFormation デプロイの前に以下を自動で行います。

1. `gh` コマンドでリポジトリの存在を確認し、なければ作成
2. リポジトリの内容をリモートリポジトリに push

CloudFormation デプロイ時に、Secrets Manager へのシークレット作成と SageMaker CodeRepository の連携も自動で行われます。

GitHub Personal Access Token (PAT) の取得方法は [GitHub 公式ドキュメント](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) を参照してください。Classic PAT の場合は `repo` スコープ、Fine-grained PAT の場合は対象リポジトリの `Contents` 権限 (Read-only) が必要です。

##### VPC 構成

VPC を有効にする場合は、`.env` に `ENABLE_VPC=true` を設定します。VPC / サブネット / セキュリティグループ / NAT Gateway / VPC Endpoints が自動作成されます。

```bash
# .env に追加
ENABLE_VPC=true
```

既存の VPC を使用する場合は、以下のリソース ID を指定します。

- `VPC_ID`: 対象の VPC ID。`EnableDnsSupport` と `EnableDnsHostnames` が有効であること
- `SUBNET_IDS`: Notebook / Training Job / Processing Job を配置するプライベートサブネット ID (カンマ区切りで 2 つ以上、異なる AZ)。NAT Gateway へのルートと S3 Gateway Endpoint がルートテーブルに紐付いていること
- `SECURITY_GROUP_ID`: Notebook と Training / Processing Job が共用するセキュリティグループ ID。分散学習を使う場合は同一 SG 内の全トラフィックを許可する自己参照 Ingress ルールが必要

```bash
# .env に追加
ENABLE_VPC=true
VPC_ID="vpc-xxxxxxxxxxxxxxxxx"
SUBNET_IDS="subnet-xxxxxxxxxxxxxxxxx,subnet-yyyyyyyyyyyyyyyyy"
SECURITY_GROUP_ID="sg-xxxxxxxxxxxxxxxxx"
CREATE_VPC_ENDPOINTS=false  # 既存 VPC に Endpoint がある場合
```

VPC 構成の詳細は [VPC 構成の実装](vpc-implementation.ja.md) を参照してください。

### Step 2: デプロイの実行

```bash
./infra/sagemaker-ai-ml-pipeline/scripts/deploy.sh [STACK_NAME] [PROJECT_NAME]
```

パラメータはすべてオプションです。

| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| STACK_NAME | `sagemaker-ai-ml-pipeline-stack` | CloudFormation スタック名 |
| PROJECT_NAME | `sagemaker-ai-ml-pipeline` | リソース名のプレフィックス |

## 開発環境の利用

デプロイ完了後、JupyterLab と MLflow UI にブラウザからアクセスできます。どちらも presigned URL による認証付きアクセスで、ブラウザセッションは 4 時間有効です。

### JupyterLab

Notebook インスタンス上の JupyterLab 環境です。コードの編集、Pipeline の実行、データの確認などに使用します。

```bash
./infra/sagemaker-ai-ml-pipeline/scripts/open-jupyterlab.sh [PROJECT_NAME]
```

Notebook はアイドル状態が一定時間 (デフォルト: 60 分) 続くと自動停止します。停止後にスクリプトを実行すると、自動で再起動を待ってから JupyterLab を開きます。

Notebook には、Lifecycle Config により Kiro CLI が自動インストールされており、JupyterLab のターミナルから AI コーディングツールを利用できます。初回はデバイスフローでログインしてください。

```bash
kiro-cli login --use-device-flow
```

表示されるデバイスコードと URL をローカル PC のブラウザで開き、認証を完了すると `kiro-cli` コマンドが利用可能になります。

### MLflow UI

実験のメトリクス比較やモデルバージョンの管理を行う Web UI です。

```bash
./infra/sagemaker-ai-ml-pipeline/scripts/open-mlflow.sh [PROJECT_NAME]
```

MLflow SDK を使ったメトリクスの記録やモデル登録の方法については [MLflow 実験管理ガイド](mlflow-guide.ja.md) を参照してください。

## Pipeline 実行 (繰り返し)

> ⚠️ **このセクション以降のコマンドはすべて JupyterLab のターミナルで実行してください。**

Step 1〜4 を一括で実行するには `run-pipeline.sh` が便利です。

PyTorch コンテナ (`container-pytorch-dlc` 等) はサンプルデータが同梱されているため、`run-pipeline.sh` だけで実行できます。NAVSIM コンテナ (`container-navsim-ego-mlp` 等) は事前に `prepare_dataset.sh` でデータを準備してから、`--skip-upload` 付きで実行してください。

利用可能なコンテナは以下の通りです。

| コンテナ | 説明 | ビルド | GPU | データ準備 | Pipeline 実行 |
|---------|------|-------|-----|----------|-------------|
| `container-navsim-transfuser` | NAVSIM Transfuser | 必要 | 必要 | 約 140 分 | 約 10 分 |
| `container-navsim-ego-mlp` | NAVSIM EgoStatusMLP ベースライン | 必要 | 不要 | 約 60 分 | 約 15 分 |
| `container-pytorch-dlc` | PyTorch DLC マネージドコンテナ (汎用テンプレート) | 不要 | 対応 | 不要 | 約 8 分 |
| `container-pytorch-dlc-byoc` | PyTorch DLC ベース BYOC | 必要 | 対応 | 不要 | 約 20 分 |

```bash
# PyTorch マネージドコンテナで実行 (デフォルト、ビルド不要)
./pipelines/scripts/run-pipeline.sh

# PyTorch マネージドコンテナで実行 (ビルド不要)
./pipelines/scripts/run-pipeline.sh -c container-pytorch-dlc

# PyTorch DLC BYOC で実行 (Dockerfile ビルドあり)
./pipelines/scripts/run-pipeline.sh -c container-pytorch-dlc-byoc

# NAVSIM EgoStatusMLP で実行 (データ準備 → Pipeline 実行)
./pipelines/container-navsim-ego-mlp/scripts/prepare_dataset.sh
./pipelines/scripts/run-pipeline.sh -c container-navsim-ego-mlp --skip-upload

# NAVSIM Transfuser で実行 (データ準備 → Pipeline 実行)
./pipelines/container-navsim-transfuser/scripts/prepare_dataset.sh
./pipelines/scripts/run-pipeline.sh -c container-navsim-transfuser --skip-upload

# train.py / evaluate.py のみ変更した再実行 (ビルドをスキップ)
./pipelines/scripts/run-pipeline.sh --skip-upload --skip-build
```

各ステップを個別に実行する場合は以下の手順に従ってください。

### Step 1: ソースコードの準備

JupyterLab のターミナルで Pipeline スクリプトを実行するために、ソースコードを Notebook インスタンスに配置します。

- **GitHub リポジトリを連携している場合**: Notebook 起動時にリポジトリが自動で clone されます。ローカル PC で変更を push した後は、Notebook のターミナルで `git pull` を実行して最新の変更を取得してください。
- **GitHub リポジトリを連携していない場合**: リポジトリはデプロイ時に S3 経由で自動配置されます。ローカル PC でファイルを変更した場合、`deploy.sh` を再実行して S3 を更新すると、Notebook を再起動 (停止→起動) した際に最新のファイルが自動的にダウンロードされます。

リポジトリ全体が Notebook インスタンスの `~/SageMaker/{project-name}/` に自動配置されます。GitHub 連携の有無にかかわらず同じパスです。

```bash
cd ~/SageMaker/{project-name}
```

### Step 2: データセットのアップロード

コンテナの種類によってデータの準備方法が異なります。

**PyTorch コンテナの場合**:

`container-pytorch-dlc` / `container-pytorch-dlc-byoc` は、コンテナディレクトリ内のサンプルデータを S3 にアップロードします。

```bash
./pipelines/scripts/01-upload-dataset.sh [PROJECT_NAME]

# コンテナを指定する場合
./pipelines/scripts/01-upload-dataset.sh -c container-pytorch-dlc [PROJECT_NAME]
```

データファイルは各コンテナディレクトリの `data/` にあります (例: `pipelines/container-pytorch-dlc/data/{train,test}.csv`)。自分のデータセットに差し替える場合は、CSV ファイルを更新してから上記コマンドを実行してください。

**NAVSIM コンテナの場合**:

`container-navsim-ego-mlp` / `container-navsim-transfuser` は、専用の `prepare_dataset.sh` でデータを準備して S3 にアップロードします。Pipeline 実行時は `--skip-upload` を指定してください。データセット準備の詳細 (前提条件、ディスク容量、特徴量抽出) については [NAVSIM ガイド - データセット準備](navsim-guide.ja.md#データセット準備) を参照してください。

```bash
# 1. データセットを準備して S3 にアップロード
./pipelines/container-navsim-ego-mlp/scripts/prepare_dataset.sh

# 2. Pipeline 実行時は --skip-upload を指定 (データは準備済みのため)
./pipelines/scripts/run-pipeline.sh -c container-navsim-ego-mlp --skip-upload
```

**データと Pipeline の自動紐づけ**: データはコンテナ名をプレフィックスとして S3 に配置されます。Pipeline 実行時に同じコンテナ名から S3 パスが自動生成されるため、データとコンテナが自動的に紐づきます。

```
s3://{project}-dataset-{account}-{region}/
  ├── container-navsim-ego-mlp/train/        ← container-navsim-ego-mlp の学習データ
  └── container-navsim-transfuser/train/     ← container-navsim-transfuser の学習データ
```

### Step 3: コンテナビルド & ECR プッシュ

BYOC コンテナの Docker イメージをビルドし、ECR にプッシュします。`container-pytorch-dlc` は AWS マネージドコンテナを使用するため、ビルドは不要です (スキップされます)。依存ライブラリ (`pip install`) やベースイメージを変更した場合に実行してください。train.py / evaluate.py のロジック変更のみであれば、SDK が S3 経由でスクリプトを注入するため再ビルドは不要です (詳細は `docs/sagemaker-python-sdk-guide.ja.md` Section 3.3 参照)。

```bash
./pipelines/scripts/02-build-and-push-container.sh -c container-pytorch-dlc-byoc [PROJECT_NAME]
```

| コンテナ | 説明 | ビルド | ECR タグ |
|---------|------|-------|---------|
| `container-pytorch-dlc` | PyTorch DLC ベース、マネージドコンテナ | 不要 | - |
| `container-pytorch-dlc-byoc` | PyTorch DLC ベース BYOC (Train も BYOC) | 必要 (10 GB 以上) | `container-pytorch-dlc-byoc` |
| `container-navsim-ego-mlp` | NAVSIM EgoStatusMLP (CPU) | 必要 | `container-navsim-ego-mlp` |
| `container-navsim-transfuser` | NAVSIM Transfuser (GPU) | 必要 | `container-navsim-transfuser` |

各コンテナは単一の ECR リポジトリ (`{project}-container`) にコンテナディレクトリ名をタグとして push されます。`-c` オプションで指定したディレクトリ名がそのまま ECR タグになるため、複数のコンテナを同時に保持できます。

> ⚠️ `container-pytorch-dlc-byoc` は CUDA、cuDNN、NCCL 等の GPU ライブラリを含むため、初回ビルド時のダウンロードに時間がかかります。十分なディスク空き容量 (30 GB 以上推奨) を確保してください。

### Step 4: Pipeline 作成 & 実行

SageMaker Pipeline を作成して実行します。Pipeline は Train → RegisterModel → Evaluate の 3 ステップで構成されており、モデルの学習・登録・評価を自動的に順番に実行します。

ターミナルから実行する場合は以下のコマンドを使用します。

`container-pytorch-dlc` の場合:

```bash
ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name sagemaker-ai-ml-pipeline-stack \
  --query 'Stacks[0].Outputs[?OutputKey==`SageMakerRoleArn`].OutputValue' \
  --output text)

python pipelines/scripts/03-create-and-run-pipeline.py \
  --project-name sagemaker-ai-ml-pipeline \
  --role-arn "$ROLE_ARN" \
  --container-dir pipelines/container-pytorch-dlc \
  --create --start
```

主要なオプションです。

| オプション | 説明 |
|-----------|------|
| `--project-name` | プロジェクト名 (必須) |
| `--role-arn` | SageMaker 実行ロール ARN (必須) |
| `--region` | AWS リージョン (デフォルト: `us-east-1`) |
| `--container-dir` | コンテナディレクトリのパス (デフォルト: `pipelines/container-pytorch-dlc`) |
| `--create` | Pipeline を作成/更新 |
| `--start` | Pipeline を実行 |
| `--subnet-ids` | VPC サブネット ID (カンマ区切り)。省略時は CFn スタックから自動取得 |
| `--security-group-ids` | セキュリティグループ ID (カンマ区切り)。省略時は CFn スタックから自動取得 |

`--create` も `--start` も指定しない場合、Pipeline 定義の JSON を標準出力に表示します。

> 💡 VPC 構成の場合、`--subnet-ids` / `--security-group-ids` を省略すると CloudFormation スタックの Output から自動取得します。明示的に指定した場合はそちらが優先されます。

### Step 5: 実行状況の確認

Pipeline 実行後、以下のスクリプトで各ステップの状況をターミナルから確認できます。

```bash
./pipelines/scripts/04-check-pipeline-status.sh [PROJECT_NAME]
```

実行例:

```
=== Pipeline 実行状況 ===
Pipeline:  sagemaker-ai-ml-pipeline-container-pytorch-dlc-pipeline
Execution: ooj49xv2k8fc
Status:    🔄 Executing
Started:   1771677200.017

Steps:
🔄 Evaluate: Executing
└─ [Console]  [CW Instance Metrics]
✅ RegisterModel-RegisterModel: Succeeded
✅ Train: Succeeded
└─ [Console]  [CW Instance Metrics]  [CW Algorithm Metrics]
```

`[Console]` / `[CW Instance Metrics]` / `[CW Algorithm Metrics]` はターミナル上でクリックして直接開けるリンクです。

### Jupyter Notebook での実行

`notebooks/` には ML ワークフローの各フェーズに対応した Notebook が用意されています。

| Notebook | 用途 |
|----------|------|
| `pytorch-pipeline.ipynb` | PyTorch DLC: データ確認 → ローカル学習・評価 → SageMaker Job → Pipeline |
| `pytorch-byoc-pipeline.ipynb` | PyTorch BYOC: データ確認 → ローカル学習・評価 → Docker Build → SageMaker Job → Pipeline |
| `navsim-ego-mlp-pipeline.ipynb` | NAVSIM EgoStatusMLP の学習・評価 |
| `navsim-transfuser-pipeline.ipynb` | NAVSIM Transfuser / LTF の学習・評価 (GPU) |

JupyterLab から各 Notebook を開いて実行してください。詳細は各 Notebook を参照してください。

## CARLA シミュレーションデモ

Pipeline で学習した TransFuser モデルを [CARLA](https://carla.org/) 自動運転シミュレーター上で走らせるデモです。NAVSIM のオフラインデータで学習したモデルが、リアルタイムのシミュレーション環境でどう振る舞うかを確認できます。

デモでは以下の流れが一気通貫で実行されます。

- CARLA サーバーの起動
- 学習済み TransFuser モデルを S3 からダウンロード
- 3 台の RGB カメラと LiDAR からセンサーデータをリアルタイム取得
- モデルが予測した将来 4 秒間の軌跡を Pure Pursuit + Lane-Keeping 制御でステアリング・スロットル・ブレーキに変換
- 走行動画を録画

GPU インスタンス (`ml.g4dn.2xlarge` 以上) が必要です。デフォルトの SageMaker AI Notebook インスタンス (`ml.g4dn.2xlarge`) でそのまま実行できます。

詳しい実行方法、アーキテクチャ、カスタマイズ方法は [CARLA シミュレーションデモ README](../demo-carla/transfuser/README.ja.md) を参照してください。

## 推論エンドポイントとデモアプリ

Pipeline で学習したモデルを SageMaker リアルタイム推論エンドポイントとしてデプロイし、デモアプリから推論リクエストを送信できます。

### Step 1: 推論エンドポイントのデプロイ

```bash
# EgoStatusMLP の場合 (CPU)
./infra/sagemaker-ai-inference/scripts/deploy.sh -c navsim-ego-mlp

# Transfuser の場合 (GPU)
./infra/sagemaker-ai-inference/scripts/deploy.sh -c navsim-transfuser
```

デプロイスクリプトは以下を自動で行います。

1. S3 上の最新モデルアーティファクトを検索
2. 推論スクリプト (`inference.py`) を含む `model.tar.gz` を再パッケージ
3. CloudFormation でエンドポイントを作成

### Step 2: デモアプリの起動

```bash
pip install -r demo-app/requirements.txt
streamlit run demo-app/main.py
```

環境変数でエンドポイント名やリージョンを指定できます。

```bash
export AWS_DEFAULT_REGION=us-east-1
export SAGEMAKER_ENDPOINT=my-endpoint-name
streamlit run demo-app/main.py
```

詳細は [demo-app/README.md](../demo-app/README.md) を参照してください。

### 推論エンドポイントの削除

```bash
./infra/sagemaker-ai-inference/scripts/destroy.sh -c navsim-ego-mlp
```

## SageMaker Unified Studio 連携 (オプション)

本リポジトリで作成した SageMaker リソース (Model Registry、Pipeline など) を [Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/what-is-sagemaker-unified-studio.html) から参照・操作できるようにする連携機能です。ML パイプライン環境のデプロイ後、いつでも追加できます。Unified Studio を使用しない場合はスキップしてください。

> ⚠️ 事前に `.env` ファイルに Unified Studio 用の設定を追加してください。
>
> ```bash
> # .env に追加
> UNIFIED_STUDIO_IDC_INSTANCE_ARN="arn:aws:sso:::instance/ssoins-xxxxxxxxxxxx"
> UNIFIED_STUDIO_SSO_USERS="user1@example.com,user2@example.com"
> ```
>
> `UNIFIED_STUDIO_IDC_INSTANCE_ARN` は AWS コンソール → IAM Identity Center → Settings → Instance ARN で確認できます。`UNIFIED_STUDIO_SSO_USERS` はドメイン管理者 / プロジェクトオーナーとして登録する SSO ユーザーのメールアドレスです (カンマ区切りで複数指定可)。

### Step 1: Unified Studio ドメインの作成

DataZone V2 ドメインと IAM ロール (DomainExecutionRole、DomainServiceRole) を作成します。`.env` に `UNIFIED_STUDIO_IDC_INSTANCE_ARN` が設定されている場合、SSO 認証が自動的に有効化されます。

```bash
./infra/unified-studio/scripts/deploy-foundation.sh
```

| オプション | デフォルト値 | 説明 |
|-----------|------------|------|
| `--domain-name` | `sagemaker-ai-ml-pipeline` | ドメイン名 |
| `--project-name` | `sagemaker-ai-ml-pipeline` | リソース命名プレフィックス |
| `--region` | `us-east-1` | AWS リージョン |

デプロイが完了すると、ドメイン ID と次のステップのコマンドが出力されます。

### Step 2: プロジェクトの作成

ブループリントの有効化 (VPC/Subnet 設定含む)、IAM ロール (Provisioning / ManageAccess) の作成、Authorization ポリシーの設定、プロジェクトプロファイル (Tooling + LakehouseCatalog + MLExperiments) の作成、プロジェクトの作成を一括で行います。スクリプトがブループリント ID とデフォルト VPC を自動検出し、必要なロールが存在しない場合は自動作成します。`.env` に `UNIFIED_STUDIO_SSO_USERS` が設定されている場合、SSO ユーザーがプロジェクトオーナーとして自動追加されます。SSO ユーザーのプロファイルは Unified Studio への初回ログイン後に作成されるため、ログイン前の場合はメンバー追加がスキップされます。ログイン後にスクリプトを再実行してください。

```bash
./infra/unified-studio/scripts/deploy-project.sh \
  --domain-id <ドメイン ID>
```

ドメイン ID は Step 1 の出力に表示されます。

| オプション | デフォルト値 | 説明 |
|-----------|------------|------|
| `--domain-id` | (必須) | Step 1 で作成したドメイン ID |
| `--us-project-name` | `ml-pipeline` | Unified Studio プロジェクト名 |
| `--project-name` | `sagemaker-ai-ml-pipeline` | リソース命名プレフィックス |

### Step 3: SageMaker リソースとの連携

ML パイプラインで作成済みの SageMaker リソースを Unified Studio プロジェクトに連携します。

```bash
./infra/unified-studio/scripts/setup-integration.sh \
  --domain-id <ドメイン ID> \
  --project-id <プロジェクト ID>
```

セットアップスクリプトが以下を自動で行います。

- Model Registry の同期 (RAM share + DataZone DataSource)
- MLflow App の接続 (DataZone connection + `AmazonDataZoneProject` タグ)
- Pipeline / Training Job / Processing Job / ECR リポジトリへの `AmazonDataZoneProject` タグ付与

プロジェクト ID は Step 2 の出力に表示されます。CLI で確認する場合は以下のコマンドを使用します。

```bash
# ドメイン一覧
aws datazone list-domains --region us-east-1

# プロジェクト一覧
aws datazone list-projects \
  --domain-identifier <ドメイン ID> \
  --region us-east-1
```

### 連携の解除

連携を解除する場合は、逆順で削除します。

```bash
# Step 3 の解除: SageMaker リソース連携の削除
./infra/unified-studio/scripts/setup-integration.sh \
  --unlink \
  --domain-id <ドメイン ID> \
  --project-id <プロジェクト ID>

# Step 2 の削除: プロジェクト + プロファイル
./infra/unified-studio/scripts/deploy-project.sh --delete --domain-id <ドメイン ID>

# Step 1 の削除: ドメイン + IAM ロール
./infra/unified-studio/scripts/deploy-foundation.sh --delete
```

連携の仕組みや各リソースの詳細については [SageMaker Unified Studio 連携ガイド](unified-studio-integration-guide.ja.md) を参照してください。セットアップ時の制約事項やトラブルシューティングについては [SageMaker Unified Studio セットアップガイド](unified-studio-setup-guide.ja.md) を参照してください。

## テスト

`tests/` ディレクトリには 2 種類のテストスクリプトがあります。インフラのデプロイ後や、スクリプト・Notebook を変更した際の動作確認に利用できます。

- **`run-lint.sh`** — スクリプトの構文や Notebook の構造を静的に検証します。AWS リソースは使わないため、**課金は発生しません**。ローカル PC でも実行可能で、コミット前のチェックに適しています。
- **`run-tests.sh`** — Pipeline と Notebook を実際に動かして、エンドツーエンドで動作することを確認します。**AWS リソースを使用するため課金が発生**し、JupyterLab のターミナル (SageMaker AI Notebook インスタンス上) でのみ実行できます。

### Lint チェック (run-lint.sh)

スクリプトの権限、構文、設定の整合性、Notebook の構造を検証します。AWS リソースは使用せず、課金は発生しません。ローカルでも Notebook インスタンスでも実行できます。

```bash
./tests/run-lint.sh
```

検証内容は以下の通りです。

| チェック | 内容 |
|---------|------|
| スクリプト権限 | `.sh` / `.py` の実行権限 |
| シェル構文 | `bash -n` による構文検証 |
| Python 構文 | `py_compile` による構文検証 |
| `--help` | 全スクリプトが `--help` に応答するか |
| `--show-config` | コンテナごとのインスタンスタイプが正しいか |
| `_common.sh` パス | 全スクリプトが `infra/_common.sh` を正しく参照しているか |
| Notebook | JSON 検証 + `papermill --prepare-only` (カーネル・パラメータ検証) |

### 統合テスト (run-tests.sh)

Pipeline と Notebook を実際に実行し、エンドツーエンドで動作することを検証します。AWS リソースを使用するため課金が発生します。

> ⚠️ このスクリプトは **JupyterLab のターミナル (SageMaker AI Notebook インスタンス上)** で実行してください。ローカル PC では動作しません。

```bash
# 特定のコンテナのみテスト (最速: pytorch-dlc、ビルド不要)
./tests/run-tests.sh -c container-pytorch-dlc

# PyTorch BYOC のみ
./tests/run-tests.sh -c container-pytorch-dlc-byoc

# NAVSIM EgoStatusMLP のみ
./tests/run-tests.sh -c container-navsim-ego-mlp

# NAVSIM Transfuser のみ (GPU、時間がかかる)
./tests/run-tests.sh -c container-navsim-transfuser

# 全コンテナ (最も時間がかかる)
./tests/run-tests.sh

# Pipeline をスキップして Notebook テストのみ
./tests/run-tests.sh --skip-pipeline

# Notebook をスキップして Pipeline テストのみ
./tests/run-tests.sh --skip-notebook -c container-pytorch-dlc
```

検証内容は以下の通りです。

| チェック | 内容 |
|---------|------|
| Pipeline (`run-pipeline.sh`) | データアップロード → コンテナビルド → Pipeline 実行までの一連のフローが正常完了するか |
| Notebook (papermill) | 対応する Pipeline 系 Notebook を最後までセル実行できるか (学習・評価・モデル登録の全工程) |

主要なオプションは以下の通りです。

| オプション | 説明 |
|-----------|------|
| `-c, --container DIR` | テスト対象コンテナ (複数指定可) |
| `--skip-pipeline` | Pipeline 実行をスキップ |
| `--skip-notebook` | Notebook 実行をスキップ |
| `--auto-approve` | 実行前の確認プロンプトをスキップ |

## クリーンアップ

S3 バケットの中身を空にし、ECR イメージと SageMaker モデルパッケージグループを削除してからスタックを削除します。`destroy.sh` がすべて自動で行います。

```bash
./infra/sagemaker-ai-ml-pipeline/scripts/destroy.sh [STACK_NAME] [PROJECT_NAME]
```

| パラメータ | デフォルト値 | 説明 |
|-----------|------------|------|
| STACK_NAME | `sagemaker-ai-ml-pipeline-stack` | CloudFormation スタック名 |
| PROJECT_NAME | `sagemaker-ai-ml-pipeline` | リソース名のプレフィックス |

MLflow App の削除に数分かかる場合があります。

## トラブルシューティング

デプロイ時のエラー (Service Quotas など) や Pipeline 実行時のエラーの調査・解決方法は [トラブルシューティングガイド](troubleshooting-guide.ja.md) を参照してください。

## Appendix

### 作成される AWS リソース

CloudFormation スタックにより以下のリソースが作成されます。

| リソース | 名前パターン | 用途 |
|---------|------------|------|
| Amazon S3 Bucket | `{project}-dataset-{account}` | 学習・テストデータセット (`train/`、`test/`) |
| Amazon S3 Bucket | `{project}-model-artifact-{account}` | モデルアーティファクト |
| Amazon S3 Bucket | `{project}-eval-artifact-{account}` | 評価結果 |
| Amazon S3 Bucket | `{project}-mlflow-artifacts-{account}` | MLflow アーティファクト |
| Amazon ECR Repository | `{project}-container` | 学習・評価コンテナ |
| SageMaker AI Notebook | `{project}-notebook` | 開発用ノートブック |
| SageMaker Lifecycle Config | `{project}-notebook-lcc` | Notebook 起動時の初期設定 |
| SageMaker Code Repository | `{project}` | GitHub リポジトリ連携 (GitHub URL 設定時のみ) |
| SageMaker Model Package Group | `{project}-model-group` | モデルレジストリ |
| SageMaker MLflow App | `{project}-mlflow` | 実験管理 |
| AWS Secrets Manager Secret | `{project}-sagemaker-github-credentials` | GitHub 認証情報 (PAT 設定時のみ) |
| IAM Role | `{project}-sagemaker-role` | SageMaker 実行ロール |
| VPC | `{project}-vpc` | VPC (`ENABLE_VPC=true` 時のみ) |
| プライベートサブネット ×2 | `{project}-private-1`, `{project}-private-2` | Notebook / Training / Processing Job 配置 |
| パブリックサブネット ×2 | `{project}-public-1`, `{project}-public-2` | NAT Gateway 配置 |
| NAT Gateway | `{project}-nat` | プライベートサブネットからのインターネットアクセス |
| VPC Endpoints | - | S3 Gateway Endpoint + SageMaker API / MLflow / Notebook / ECR / CW Logs / STS (Interface) |

### 関連情報

#### AWS ドキュメント

本プロジェクトで使用している AWS サービスの公式ドキュメントです。

**SageMaker Pipelines**:

- [Amazon SageMaker Pipelines 概要](https://aws.amazon.com/sagemaker-ai/pipelines/)
- [Pipelines Overview - Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-overview.html)
- [Define a Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/define-pipeline.html)
- [Create a Pipeline with @step Decorator](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-step-decorator-create-pipeline.html)

**SageMaker Model Registry**:

- [Model Registry Overview](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)
- [Model Registry - Models, Versions, and Groups](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry-models.html)

**MLflow on SageMaker**:

- [Create an MLflow App](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-app-create-app-cli.html)
- [MLflow UI を起動する](https://docs.aws.amazon.com/ja_jp/sagemaker/latest/dg/mlflow-launch-ui.html)
- [MLflow Tutorials - Example Notebooks](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-tutorials.html)
- [Auto-register Models with Model Registry via MLflow](https://docs.aws.amazon.com/sagemaker/latest/dg/mlflow-track-experiments-model-registration.html)
- [MLflow Integration with SageMaker Pipelines](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps-integration.html)
- [Managed MLflow 3.0 on SageMaker (Blog)](https://aws.amazon.com/blogs/machine-learning/accelerating-generative-ai-development-with-fully-managed-mlflow-3-0-on-amazon-sagemaker-ai/)

**SageMaker MLOps**:

- [Amazon SageMaker MLOps](https://aws.amazon.com/sagemaker/ai/mlops/)

**CloudFormation リファレンス**:

- [AWS::SageMaker::Pipeline](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-sagemaker-pipeline.html)
- [AWS::SageMaker::NotebookInstance](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-sagemaker-notebookinstance.html)
- [AWS::SageMaker::ModelPackageGroup](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-resource-sagemaker-modelpackagegroup.html)

**MLflow App API リファレンス**:

- [CreateMlflowApp](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateMlflowApp.html)
- [DescribeMlflowApp](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DescribeMlflowApp.html)
- [DeleteMlflowApp](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_DeleteMlflowApp.html)
- [CreatePresignedMlflowAppUrl](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreatePresignedMlflowAppUrl.html)
- [Boto3 create_presigned_mlflow_app_url](https://docs.aws.amazon.com/boto3/latest/reference/services/sagemaker/client/create_presigned_mlflow_app_url.html)

#### 関連ワークショップ・サンプル

SageMaker AI を使った ML ワークフローをさらに学ぶためのワークショップとサンプルリポジトリです。

- [Amazon SageMaker AI Immersion Day](https://catalog.us-east-1.prod.workshops.aws/workshops/63069e26-921c-4ce1-9cc7-dd882ff62575/ja-JP) - SageMaker AI の主要機能をハンズオンで体験するワークショップ。データ準備、モデル学習、デプロイ、MLOps まで幅広くカバー
- [amazon-sagemaker-from-idea-to-production](https://github.com/aws-samples/amazon-sagemaker-from-idea-to-production) - アイデアから本番環境までの ML ワークフローを SageMaker で構築するエンドツーエンドのサンプル。Studio、Pipelines、Model Registry、Feature Store を活用
- [sagemaker-end-to-end-workshop](https://github.com/aws-samples/sagemaker-end-to-end-workshop) - SageMaker のエンドツーエンドワークショップ。データ探索からモデルのデプロイ・モニタリングまでの一連の ML ライフサイクルをハンズオンで学習

