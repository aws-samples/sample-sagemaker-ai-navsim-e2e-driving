# SageMaker Unified Studio 連携ガイド <!-- omit in toc -->

🌐 **Language**: 🇺🇸 [English](unified-studio-integration-guide.md) | 🇯🇵 [日本語](unified-studio-integration-guide.ja.md)

本プロジェクトと Amazon SageMaker Unified Studio を連携する方法をまとめたガイドです。

Unified Studio に表示されるリソースはプロジェクトに紐付いたものに限られます。本プロジェクトで作成した SageMaker リソース (Model Package Group、MLflow App、Pipeline など) を Unified Studio から利用するには、それぞれ連携設定が必要です。

## 目次 <!-- omit in toc -->

- [1. 連携方法の全体像](#1-連携方法の全体像)
- [2. クイックスタート: セットアップスクリプトによる自動化](#2-クイックスタート-セットアップスクリプトによる自動化)
  - [前提条件](#前提条件)
  - [使い方](#使い方)
  - [スクリプトの処理内容](#スクリプトの処理内容)
  - [スクリプトがカバーする範囲](#スクリプトがカバーする範囲)
  - [セットアップ後の確認](#セットアップ後の確認)
  - [タグ付与スクリプトの単独実行](#タグ付与スクリプトの単独実行)
- [3. Model Registry 連携 (DataSource 方式)](#3-model-registry-連携-datasource-方式)
  - [概要](#概要)
  - [仕組み](#仕組み)
  - [連携手順](#連携手順)
    - [Step 1: RAM share の作成](#step-1-ram-share-の作成)
    - [Step 2: DataZone data source の作成](#step-2-datazone-data-source-の作成)
    - [Step 3: 自動登録の確認](#step-3-自動登録の確認)
- [4. MLflow App の連携](#4-mlflow-app-の連携)
  - [概要](#概要-1)
  - [方法 A: UI から接続する](#方法-a-ui-から接続する)
  - [方法 B: API から接続する (自動化向け)](#方法-b-api-から接続する-自動化向け)
  - [注意点](#注意点)
- [5. SageMaker Pipeline の連携](#5-sagemaker-pipeline-の連携)
  - [概要](#概要-2)
  - [手動でのタグ付与](#手動でのタグ付与)
  - [確認方法](#確認方法)
  - [注意点](#注意点-1)
- [6. Training Job / Processing Job の連携](#6-training-job--processing-job-の連携)
  - [概要](#概要-3)
  - [手動でのタグ付与](#手動でのタグ付与-1)
  - [注意点](#注意点-2)
- [7. ECR リポジトリの連携](#7-ecr-リポジトリの連携)
  - [概要](#概要-4)
  - [手動でのタグ付与](#手動でのタグ付与-2)
  - [注意点](#注意点-3)
- [8. カスタムタグの注意点](#8-カスタムタグの注意点)
  - [AmazonDataZoneProject タグについて](#amazondatazoneproject-タグについて)
  - [その他の DataZone 関連タグ](#その他の-datazone-関連タグ)


## 1. 連携方法の全体像

リソースの種類によって連携方法が異なる。以下の表にまとめる。

| リソース | 連携方法 | 自動化 |
|---------|---------|--------|
| Model Package Group | DataZone DataSource (API/CLI) | ✅ `setup-integration.sh` |
| MLflow App | DataZone `create-connection` API + `AmazonDataZoneProject` タグ (両方必要) | ✅ `setup-integration.sh` |
| SageMaker Pipeline | `AmazonDataZoneProject` タグ (値=プロジェクト ID) | ✅ `setup-integration.sh` |
| Training Job | `AmazonDataZoneProject` タグ (値=プロジェクト ID) | ✅ `setup-integration.sh` |
| Processing Job | `AmazonDataZoneProject` タグ (値=プロジェクト ID) | ✅ `setup-integration.sh` |
| ECR リポジトリ | `AmazonDataZoneProject` タグ (値=プロジェクト ID) | ✅ `setup-integration.sh` |


連携の仕組みは大きく 2 つのパターンに分かれます。

1. **DataSource 方式** (Model Package Group): RAM share + DataZone DataSource を作成し、DataZone がリソースをスキャン・同期する
2. **タグ方式** (Pipeline、Training Job、Processing Job、ECR): リソースに `AmazonDataZoneProject` タグを付与し、値にプロジェクト ID を指定すると、そのプロジェクトに表示される

参考:

- [Machine learning - Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/sagemaker.html)
- [Bringing existing resources into Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/bring-resources-scripts.html)


## 2. クイックスタート: セットアップスクリプトによる自動化

本リポジトリでは、Unified Studio 連携を自動化するスクリプトを提供しています。スクリプト 1 つで Model Registry 連携 (RAM share + DataSource)、MLflow App の接続、既存 SageMaker リソースへの `AmazonDataZoneProject` タグ付与をすべて行います。各リソースの連携の仕組みについてはセクション 3 以降を参照してください。

### 前提条件

セットアップスクリプトを実行する前に、以下が必要です。

- メインスタック (`sagemaker-ai-ml-pipeline-stack`) がデプロイ済みであること
- Unified Studio のドメインとプロジェクトが作成済みであること (`deploy-foundation.sh` + `deploy-project.sh` で作成)

ドメイン ID とプロジェクト ID は `deploy-foundation.sh` / `deploy-project.sh` の出力に表示されます。CLI で確認する場合は以下のコマンドを使用してください。

```bash
# ドメイン一覧
aws datazone list-domains --region <region>

# プロジェクト一覧
aws datazone list-projects \
  --domain-identifier <ドメイン ID> \
  --region <region>
```

### 使い方

```bash
# セットアップ
./infra/unified-studio/scripts/setup-integration.sh \
  --domain-id <ドメイン ID> \
  --project-id <プロジェクト ID>

# 連携解除
./infra/unified-studio/scripts/setup-integration.sh \
  --unlink \
  --domain-id <ドメイン ID> \
  --project-id <プロジェクト ID>
```

### スクリプトの処理内容

セットアップ時に以下の 5 ステップを実行します。

1. DataZone connection ID の自動取得
2. CloudFormation スタックのデプロイ (RAM share + DataZone DataSource)
3. DataSource の初回実行 (Unified Studio への初回同期)
4. MLflow App の接続 (DataZone `create-connection` API)
5. 既存 SageMaker リソースへの `AmazonDataZoneProject` タグ付与

Step 4 では、`datazone create-connection` API で MLflow App をプロジェクトに接続します。Tooling 環境に紐づけて接続を作成し、同名の接続がまだない場合にのみ作成します。MLflow App の UI 表示には `AmazonDataZoneProject` タグと接続の両方が必要です。

Step 5 では、`tag-resources.py` がプロジェクト名プレフィックス (`sagemaker-ai-ml-pipeline-`) に一致するリソースを検索し、`AmazonDataZoneProject` タグ (値=プロジェクト ID) を付与します。対象リソースは Pipeline、Training Job、Processing Job、Model Package Group、MLflow App、Model、Endpoint、ECR リポジトリです。

連携解除時は以下の 3 ステップを実行します。

1. 既存 SageMaker リソースから `AmazonDataZoneProject` タグを削除
2. MLflow App の接続を削除
3. CloudFormation スタックの削除 (RAM share + DataZone DataSource)

### スクリプトがカバーする範囲

| リソース | カバー |
|---------|--------|
| Model Package Group (DataSource) | ✅ 自動化済み |
| MLflow App | ✅ 自動化済み (`create-connection` API) |
| Pipeline / Training Job / Processing Job | ✅ タグ付与を自動化済み |
| Model / Endpoint | ✅ タグ付与を自動化済み |
| ECR リポジトリ | ✅ タグ付与を自動化済み |

### セットアップ後の確認

セットアップ完了後、Unified Studio の **Build** メニューから以下を確認できます。

- **Model Registry → Registered Models**: DataSource 経由で同期されたモデル
- **ML Pipelines**: タグ経由で表示される Pipeline
- **MLflow**: 接続された MLflow App の実験・run

初回同期が完了するまで数分かかる場合がある。

### タグ付与スクリプトの単独実行

`tag-resources.py` はセットアップスクリプトから呼ばれますが、単独でも実行できます。

```bash
# タグ付与
python3 infra/unified-studio/scripts/tag-resources.py \
  --project-id <プロジェクト ID> \
  --region <region>

# タグ削除
python3 infra/unified-studio/scripts/tag-resources.py \
  --project-id <プロジェクト ID> \
  --region <region> \
  --unlink
```


参考:

- [CloudFormation テンプレート](../infra/unified-studio/cfn/integration.yaml)
- [セットアップスクリプト](../infra/unified-studio/scripts/setup-integration.sh)
- [タグ付与スクリプト](../infra/unified-studio/scripts/tag-resources.py)


## 3. Model Registry 連携 (DataSource 方式)

### 概要

Unified Studio の Model Registry に表示されるモデルは、プロジェクトに紐付いたものに限られます。Model Package Group を Unified Studio から見えるようにするには、DataSource を作成します (UI からは不可)。

参考:

- [Create a data source for SageMaker AI](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/create-sagemaker-data-source.html)

### 仕組み

Unified Studio で作成した Model Group には以下のタグが自動付与されます。

```
AmazonDataZoneProject: <プロジェクト ID>
AmazonDataZoneDomain: <ドメイン ID>
AmazonDataZoneScopeName: dev
AmazonDataZoneUser: <ユーザー ID>
```

本リポジトリの Model Package Group にはこれらのタグがないため、Unified Studio の Model Registry に表示されません。DataSource を作成することで、DataZone が Model Package Group をスキャンし、Unified Studio に同期します。

### 連携手順

`setup-integration.sh` で自動化済み。手動で行う場合は以下の手順を参照。

#### Step 1: RAM share の作成

SageMaker と DataZone の信頼関係を確立します。

1. [RAM コンソール](https://console.aws.amazon.com/ram/home) を開く
2. **Create resource share** を選択
3. 名前: `DataZone-<ドメイン ID>-SageMaker`
4. Resources: **DataZone Domains** から対象ドメインを選択
5. Managed Permissions: `AWSRAMSageMakerServicePrincipalPermissionAmazonDataZoneDomain`
6. Principals: Service principal = `sagemaker.amazonaws.com`
7. Sources: 自アカウント ID を指定 (service principal への共有時に必須)
8. **Create resource share**

#### Step 2: DataZone data source の作成

以下の JSON を `create-sagemaker-datasource.json` として保存します。

```json
{
  "name": "sagemaker-ai-ml-pipeline-datasource",
  "projectIdentifier": "<プロジェクト ID>",
  "type": "SAGEMAKER",
  "description": "sagemaker-ai-ml-pipeline の Model Package Group を Unified Studio に連携",
  "connectionIdentifier": "<connection ID>",
  "configuration": {
    "sageMakerRunConfiguration": {
      "trackingAssets": {
        "SageMakerModelPackageGroupAssetType": [
          "arn:aws:sagemaker:<region>:<account-id>:model-package-group/sagemaker-ai-ml-pipeline-pytorch"
        ]
      }
    }
  },
  "enableSetting": "ENABLED",
  "publishOnImport": "True"
}
```

```bash
aws datazone create-data-source \
  --domain-identifier <ドメイン ID> \
  --cli-input-json file://create-sagemaker-datasource.json \
  --region <region>
```

#### Step 3: 自動登録の確認

連携後、Pipeline 実行のたびに以下のフローで自動登録されます。

1. Pipeline 実行 → Train → RegisterModel ステップ
2. `sagemaker-ai-ml-pipeline-pytorch` に新バージョンが登録される
3. DataZone data source のスケジュール (または手動 run) で Unified Studio に同期
4. Unified Studio の **Build → Model Registry → Registered Models** に表示される

参考:

- [Create a data source for SageMaker AI](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/create-sagemaker-data-source.html)
- [Model registry - Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/sagemaker-register-models.xml.html)


## 4. MLflow App の連携

### 概要

本リポジトリの MLflow App (`sagemaker-ai-ml-pipeline-mlflow`) を Unified Studio プロジェクトに接続する方法です。Unified Studio では新規の MLflow App を作成することはできず、SageMaker AI で作成済みのサーバーを「接続」する形になります。

参考:

- [Track experiments using MLflow - Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/use-mlflow-experiments.html)

### 方法 A: UI から接続する

Unified Studio の UI から既存の MLflow App を接続できます。

1. Unified Studio プロジェクトにサインイン
2. 左メニューから **MLflow** を選択
3. **Connect MLflow App** を選択
4. MLflow App Name を入力
5. Connection name を入力
6. MLflow App ARN を入力: `arn:aws:sagemaker:<region>:<account-id>:mlflow-app/app-XXXXXXXXXXXX` (末尾はサービス側で自動採番されるリソース ID)
7. **Connect to server** を選択

接続後、**Open MLflow** から MLflow UI を開き、実験・モデル・トレースを確認できます。

### 方法 B: API から接続する (自動化向け)

DataZone の `create-connection` API を使って、プログラムから MLflow 接続を作成できます。`setup-integration.sh` ではこの方法を使用しています。

参考:

- [CreateConnection - Amazon DataZone API Reference](https://docs.aws.amazon.com/datazone/latest/APIReference/API_CreateConnection.html)
- [MlflowPropertiesInput](https://docs.aws.amazon.com/datazone/latest/APIReference/API_MlflowPropertiesInput.html)

```bash
aws datazone create-connection \
  --domain-identifier <ドメイン ID> \
  --environment-identifier <環境 ID> \
  --name "sagemaker-ai-ml-pipeline-mlflow" \
  --props '{
    "mlflowProperties": {
      "mlflowAppArn": "arn:aws:sagemaker:<region>:<account-id>:mlflow-app/app-XXXXXXXXXXXX"
    }
  }' \
  --region <region>
```

`ConnectionPropertiesInput` は union 型で、`mlflowProperties` メンバーに `mlflowAppArn` を指定します。`environment-identifier` には Tooling 環境の ID を使用します (`setup-integration.sh` が自動検出)。

### 注意点

- Unified Studio の Build → MLflow に MLflow App を表示するには、`AmazonDataZoneProject` タグと `create-connection` API (または UI からの接続) の両方が必要。タグだけでは表示されず、接続だけでも表示されない
- `create-connection` の `environment-identifier` には Tooling 環境の ID を使用する。MLExperiments 環境は不要
- Unified Studio では新規の MLflow App を作成できない。SageMaker AI 側で作成済みのサーバーを接続する形になる
- MLflow の実験・run は MLflow App に紐付くため、MLflow App が異なると実験履歴が分断される
- 本リポジトリの既存 MLflow App を接続すれば、これまでの実験履歴をそのまま Unified Studio から参照できる

参考:

- [Track experiments using MLflow - Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/use-mlflow-experiments.html)
- [CreateConnection - Amazon DataZone API Reference](https://docs.aws.amazon.com/datazone/latest/APIReference/API_CreateConnection.html)
- [MLflow 実験管理ガイド](mlflow-guide.ja.md)


## 5. SageMaker Pipeline の連携

### 概要

SageMaker Pipeline を Unified Studio に表示するには、Pipeline リソースに `AmazonDataZoneProject` タグを付与します。タグの値にはプロジェクト ID を指定します。

参考:

- [Machine learning - Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/sagemaker.html)

### 手動でのタグ付与

```bash
aws sagemaker add-tags \
  --resource-arn arn:aws:sagemaker:<region>:<account-id>:pipeline/sagemaker-ai-ml-pipeline-container-pytorch-dlc-pipeline \
  --tags Key=AmazonDataZoneProject,Value=<プロジェクト ID> \
  --region <region>
```

### 確認方法

タグ付与後、Unified Studio の **Build → ML Pipelines** に Pipeline が表示されます。Pipeline の実行履歴やステップの詳細も確認できます。

### 注意点

- タグ付与後に作成された Pipeline 実行 (Execution) は自動的に Unified Studio に表示される
- タグ付与前に実行された Pipeline 実行も表示される (タグは Pipeline リソース自体に付与するため)
- `setup-integration.sh` を使えば自動的にタグが付与される (セクション 2 参照)


## 6. Training Job / Processing Job の連携

### 概要

Training Job と Processing Job を Unified Studio に表示するには、各ジョブに `AmazonDataZoneProject` タグを付与します。Pipeline 経由で作成されたジョブも、個別に作成されたジョブも同様です。

参考:

- [Machine learning - Amazon SageMaker Unified Studio](https://docs.aws.amazon.com/sagemaker-unified-studio/latest/userguide/sagemaker.html)

### 手動でのタグ付与

```bash
# Training Job
aws sagemaker add-tags \
  --resource-arn arn:aws:sagemaker:<region>:<account-id>:training-job/<ジョブ名> \
  --tags Key=AmazonDataZoneProject,Value=<プロジェクト ID> \
  --region <region>

# Processing Job
aws sagemaker add-tags \
  --resource-arn arn:aws:sagemaker:<region>:<account-id>:processing-job/<ジョブ名> \
  --tags Key=AmazonDataZoneProject,Value=<プロジェクト ID> \
  --region <region>
```

### 注意点

- Training Job / Processing Job は数が多くなりがちなので、`tag-resources.py` での一括タグ付与が便利 (セクション 2 参照)
- タグ付与スクリプトはプロジェクト名プレフィックス (`sagemaker-ai-ml-pipeline-`) に一致するジョブを自動検索する
- Pipeline 経由で新しいジョブが作成された場合、再度タグ付与スクリプトを実行する必要がある


## 7. ECR リポジトリの連携

### 概要

BYOC (Bring Your Own Container) で使用する ECR リポジトリを Unified Studio に表示するには、リポジトリに `AmazonDataZoneProject` タグを付与します。

### 手動でのタグ付与

```bash
aws ecr tag-resource \
  --resource-arn arn:aws:ecr:<region>:<account-id>:repository/sagemaker-ai-ml-pipeline-byoc \
  --tags Key=AmazonDataZoneProject,Value=<プロジェクト ID> \
  --region <region>
```

### 注意点

- ECR リポジトリのタグ付与には `ecr:tag-resource` API を使用する (SageMaker の `add-tags` ではない)
- `setup-integration.sh` では `tag-resources.py` 経由で自動的にタグが付与される (セクション 2 参照)


## 8. カスタムタグの注意点

### AmazonDataZoneProject タグについて

`AmazonDataZoneProject` タグは Unified Studio (DataZone) が使用する予約済みタグです。以下の点に注意してください。

- タグの値はプロジェクト ID (例: `abc1defgh2ijkl`) を指定する
- 1 つのリソースに対して 1 つのプロジェクト ID のみ指定可能 (複数プロジェクトへの同時表示は不可)
- Unified Studio で作成したリソースには自動的にこのタグが付与される
- 手動で付与したタグを削除すると、Unified Studio からリソースが非表示になる

### その他の DataZone 関連タグ

Unified Studio で作成したリソースには、以下のタグも自動付与されます。

```
AmazonDataZoneDomain: <ドメイン ID>
AmazonDataZoneScopeName: dev
AmazonDataZoneUser: <ユーザー ID>
```

これらのタグは手動で付与する必要はありません。`AmazonDataZoneProject` タグのみで Unified Studio への表示は機能します。


