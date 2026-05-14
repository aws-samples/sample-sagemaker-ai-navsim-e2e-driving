# トラブルシューティングガイド

🌐 **Language**: 🇺🇸 [English](troubleshooting-guide.md) | 🇯🇵 [日本語](troubleshooting-guide.ja.md)

SageMaker AI ML Pipeline のデプロイや実行で発生しうるエラーと、その調査・解決方法をまとめています。

- [デプロイ時のエラー](#デプロイ時のエラー)
  - [ResourceLimitExceeded (Service Quotas)](#resourcelimitexceeded-service-quotas)
- [Pipeline 実行時のエラー](#pipeline-実行時のエラー)
  - [エラーの調査方法](#エラーの調査方法)
  - [AlgorithmError: ExecuteUserScriptError (exit code 1)](#algorithmerror-executeuserscripterror-exit-code-1)
  - [FileNotFoundError: No CSV files found](#filenotfounderror-no-csv-files-found)
- [Notebook 環境のエラー](#notebook-環境のエラー)
  - [Python パッケージのバージョン互換性](#python-パッケージのバージョン互換性)
  - [git pull で認証エラーが発生する](#git-pull-で認証エラーが発生する)

## デプロイ時のエラー

CloudFormation スタックのデプロイ (`deploy.sh`) 時に発生するエラーと対処法です。

### ResourceLimitExceeded (Service Quotas)

SageMaker AI の GPU インスタンスはデフォルトのクォータが 0 に設定されていることがあります。`ResourceLimitExceeded` エラーが発生した場合は、AWS Service Quotas でクォータの引き上げをリクエストしてください。`deploy.sh` は一般的なインスタンスタイプのクォータ引き上げを自動リクエストしますが、承認には数分〜数時間かかる場合があります。

以下のクォータが必要になることがあります。

| クォータ名 | クォータコード | 用途 | 推奨値 |
|-----------|-------------|------|--------|
| `ml.g6.xlarge for training job usage` | `L-1B43CB89` | Training Job (予備) | 5 |
| `ml.g6.xlarge for processing job usage` | `L-B0D09498` | Processing Job (予備) | 5 |
| `ml.g6.4xlarge for training job usage` | `L-07B51BA0` | Training Job (Transfuser) | 5 |
| `ml.g6.4xlarge for processing job usage` | `L-8BF5F502` | Processing Job (Transfuser) | 5 |

AWS CLI でクォータの引き上げをリクエストできます。

```bash
aws service-quotas request-service-quota-increase \
    --service-code sagemaker \
    --quota-code L-07B51BA0 \
    --desired-value 5 \
    --region us-east-1
```

ステータスは [Service Quotas コンソール](https://console.aws.amazon.com/servicequotas/home/services/sagemaker/quotas) で確認できます。

EgoStatusMLP で使用する CPU インスタンス (`ml.c7i.xlarge`) はデフォルトで十分なクォータが設定されているため、通常は引き上げ不要です。

## Pipeline 実行時のエラー

SageMaker Pipeline の Training Job や Processing Job が失敗したときの調査方法とよくあるエラーの対処法です。

### エラーの調査方法

SageMaker コンソールに表示されるエラーメッセージだけでは原因がわからないことがあります。以下の手順で CloudWatch Logs から詳細なエラーログを取得できます。

#### Step 1: Pipeline の実行ステータスを確認する

SageMaker コンソール、または AWS CLI で Pipeline の実行状況を確認します。

```bash
aws sagemaker list-pipeline-executions \
  --pipeline-name sagemaker-ai-ml-pipeline-container-pytorch-dlc-pipeline \
  --sort-by CreationTime \
  --sort-order Descending \
  --max-results 5 \
  --region us-east-1
```

#### Step 2: 失敗した Job を特定する

Pipeline の実行 ARN から、失敗したステップの Job 名を確認します。

```bash
aws sagemaker list-pipeline-execution-steps \
  --pipeline-execution-arn <execution-arn> \
  --region us-east-1
```

Training Job の場合は `FailureReason` フィールドにエラーの概要が表示されますが、`ErrorMessage ""` のように空の場合があります。その場合は次のステップで CloudWatch Logs を確認します。

#### Step 3: CloudWatch Logs で詳細なエラーログを確認する

Training Job と Processing Job のログは CloudWatch Logs に出力されます。

Training Job のログを確認する手順です。

1. ログストリームの一覧を取得します。

```bash
aws logs describe-log-streams \
  --log-group-name "/aws/sagemaker/TrainingJobs" \
  --region us-east-1 \
  --order-by LastEventTime \
  --descending \
  --max-items 5 \
  --query 'logStreams[].logStreamName' \
  --output json
```

2. 該当する Job のログストリーム名を指定して、ログイベントを取得します。ログストリーム名は `{job-name}/algo-1-{timestamp}` の形式です。

```bash
aws logs get-log-events \
  --log-group-name "/aws/sagemaker/TrainingJobs" \
  --log-stream-name "<log-stream-name>" \
  --region us-east-1 \
  --query 'events[].message' \
  --output json
```

Processing Job の場合はロググループ名が異なります。

```bash
aws logs describe-log-streams \
  --log-group-name "/aws/sagemaker/ProcessingJobs" \
  --region us-east-1 \
  --order-by LastEventTime \
  --descending \
  --max-items 5 \
  --query 'logStreams[].logStreamName' \
  --output json
```

> 💡 SageMaker コンソールの Training Job 詳細画面からも「View logs」リンクで CloudWatch Logs に直接アクセスできます。

### AlgorithmError: ExecuteUserScriptError (exit code 1)

Training Job や Processing Job で以下のようなエラーが表示される場合があります。

```
AlgorithmError: ExecuteUserScriptError:
ExitCode 1
ErrorMessage ""
Command "/usr/local/bin/python3.10 train.py"
```

`ErrorMessage` が空のため、このメッセージだけでは原因がわかりません。[Step 3: CloudWatch Logs で詳細なエラーログを確認する](#step-3-cloudwatch-logs-で詳細なエラーログを確認する) の手順で CloudWatch Logs を確認してください。

よくある原因は以下の通りです。

- Python の依存パッケージが不足している (Dockerfile に追加が必要)
- 入力データのパスが間違っている (S3 にデータがアップロードされていない)
- IAM 権限が不足している (MLflow、S3 など外部サービスへのアクセス)
- コードのバグ (ローカルでは動くがコンテナ内で失敗するケース)

### FileNotFoundError: No CSV files found

```
FileNotFoundError: No CSV files found in /opt/ml/input/data/train
```

S3 にデータがアップロードされていない可能性があります。以下のコマンドでデータをアップロードしてください。

```bash
./pipelines/scripts/01-upload-dataset.sh
```

アップロード後、S3 バケットにデータが存在するか確認します。

```bash
aws s3 ls s3://sagemaker-ai-ml-pipeline-dataset-<account-id>/train/
aws s3 ls s3://sagemaker-ai-ml-pipeline-dataset-<account-id>/test/
```

## Notebook 環境のエラー

SageMaker AI Notebook インスタンスで作業するときに発生する環境設定関連のエラーです。

### Python パッケージのバージョン互換性

SageMaker AI Notebook インスタンスの pytorch 環境では、プリインストールされたパッケージ間のバージョン互換性に注意が必要です。`pip install -U` で一部のパッケージを更新すると、依存関係の変化により他のパッケージとの互換性が壊れることがあります。

#### NumPy 2.x と PyTorch の非互換

`pip install -U scikit-learn` などを実行すると、依存関係により NumPy が 2.x に更新される場合があります。しかし、プリインストールの PyTorch (2.0.1) は NumPy 1.x でコンパイルされているため、以下のエラーが発生します。

```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6 as it may crash.
RuntimeError: Could not infer dtype of numpy.float32
```

pip install 時に `"numpy<2"` を指定して NumPy 1.x に固定してください。

```bash
pip install -U scikit-learn pandas "numpy<2"
```

#### MLflow 3.x と PyTorch 2.0 の非互換

MLflow 3.9 以降の `mlflow.pytorch.save_model()` は `torch.export` モジュール (PyTorch 2.1 で追加) を内部で import します。PyTorch 2.0.x 環境では以下のエラーが発生します。

```
ModuleNotFoundError: No module named 'torch.export'
```

MLflow App は MLflow 3.x で動作しています。`sagemaker-mlflow` パッケージと合わせてインストールしてください。

```bash
pip install -U mlflow sagemaker-mlflow
```

#### 推奨される pip install コマンド

各ノートブックの pip install セルでは、以下のバージョン制約を使用しています。

PyTorch ノートブックの場合です。

```bash
pip install -U scikit-learn pandas mlflow sagemaker-mlflow matplotlib seaborn "numpy<2"
```

> ⚠️ PyTorch 自体は upgrade 対象に含めないでください。プリインストール版を使用します。`pip install -U torch` を実行すると NumPy 2.x が引き込まれ、上記の互換性問題が発生します。

### git pull で認証エラーが発生する

Notebook インスタンスのターミナルで `git pull` を実行した際に、以下のようなエラーが発生する場合があります。

```
Username for 'https://github.com/...':
Password for 'https://...@github.com/...':
remote: Invalid username or token. Password authentication is not supported for Git operations.
fatal: Authentication failed for 'https://github.com/...'
```

**原因**: SageMaker CodeRepository は Notebook 起動時に AWS Secrets Manager に保存された GitHub Personal Access Token (PAT) を使ってリポジトリを clone しますが、その後のターミナルからの `git pull` / `git push` では同じ認証情報が自動的に使われない場合があります。

**対処法**: Secrets Manager から GitHub PAT を取得して、リポジトリの remote URL に設定します。PAT はインフラデプロイ時に `.env` の `GITHUB_PAT` から Secrets Manager (`{プロジェクト名}-sagemaker-github-credentials`) に保存されています。

```bash
# Secrets Manager から PAT を取得
PAT=$(aws secretsmanager get-secret-value \
  --secret-id sagemaker-ai-ml-pipeline-sagemaker-github-credentials \
  --query 'SecretString' --output text \
  | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['password'])")

# remote URL に PAT を設定
REPO_URL=$(git remote get-url origin | sed 's|https://[^@]*@|https://|' | sed 's|https://|https://'"${PAT}"'@|')
git remote set-url origin "${REPO_URL}"

# 動作確認
git pull --all
```

> **Note**: この設定は `.git/config` に保存されるため、Notebook インスタンスを停止・再起動しても維持されます。ただし、PAT を再発行した場合は再度設定が必要です。
