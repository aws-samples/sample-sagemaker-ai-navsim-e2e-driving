#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail

# ============================================================
# Pipeline 一括実行スクリプト
# ============================================================
#
# 使い方:
#   ./run-pipeline.sh [オプション] [プロジェクト名]
#
# オプション:
#   -c, --container DIR   コンテナディレクトリ名 (デフォルト: container-navsim-transfuser)
#   -w, --watch SEC       Pipeline 完了まで指定秒間隔でポーリング (デフォルト: 30)
#   --skip-upload         データセットアップロードをスキップ
#   --skip-build          コンテナビルド & ECR プッシュをスキップ
#   -h, --help            このヘルプを表示
#
# 処理内容:
#   Step 1: データセットを S3 にアップロード  (01-upload-dataset.sh)
#   Step 2: コンテナをビルドして ECR にプッシュ (02-build-and-push-container.sh)
#   Step 3: Pipeline を作成して実行           (03-create-and-run-pipeline.py)
#   Step 4: 実行状況をポーリングして表示       (04-check-pipeline-status.sh)
# ============================================================

source "$(dirname "$0")/../../infra/_common.sh"

SCRIPTS_DIR="$(dirname "$0")"

usage() {
  cat << EOF
使い方: $(basename "$0") [オプション] [プロジェクト名]

Pipeline の一括実行 (データアップロード → コンテナビルド → Pipeline 実行 → 状況確認) を行います。

オプション:
  -c, --container DIR   コンテナディレクトリ名 (デフォルト: container-navsim-transfuser)
                        選択肢:
                          container-navsim-transfuser       NAVSIM Transfuser (BYOC, GPU)
                          container-navsim-ego-mlp     NAVSIM EgoStatusMLP (BYOC, GPU)
                          container-pytorch-dlc        PyTorch DLC (マネージドコンテナ, GPU)
                          container-pytorch-dlc-byoc   PyTorch DLC ベース BYOC
  -w, --watch SEC       Pipeline 完了まで指定秒間隔でポーリング (デフォルト: 30)
  --skip-upload         データセットアップロードをスキップ
  --skip-build          コンテナビルド & ECR プッシュをスキップ
  --auto-approve        確認プロンプトをスキップ
  -h, --help            このヘルプを表示

引数:
  プロジェクト名    プロジェクト名 (デフォルト: sagemaker-ai-ml-pipeline)

例:
  $(basename "$0")                                    # デフォルト設定で全ステップ実行
  $(basename "$0") -c container-pytorch-dlc           # PyTorch コンテナで実行
  $(basename "$0") --skip-upload --skip-build         # Pipeline 実行のみ (再実行時)
  $(basename "$0") -w 60                              # 60 秒間隔でポーリング

NAVSIM コンテナの場合:
  NAVSIM コンテナ (container-navsim-*) は事前にデータセットの準備が必要です。

  # 1. データセット準備 (初回のみ、OpenScene データセットをダウンロード・前処理・S3 アップロード)
  ./pipelines/container-navsim-transfuser/scripts/prepare_dataset.sh

  # 2. パイプライン実行 (--skip-upload でデータアップロードをスキップ)
  $(basename "$0") -c container-navsim-transfuser --skip-upload

  prepare_dataset.sh を実行せずにパイプラインを実行すると、ダミーデータで学習されます。

⏱️  所要時間の目安:
  - prepare_dataset.sh (ego-mlp):    約 60 分  (初回のみ)
  - prepare_dataset.sh (transfuser): 約 140 分 (初回のみ)
  - Pipeline 実行 (ego-mlp):         約 15 分
  - Pipeline 実行 (transfuser):      約 90 分
EOF
}

# --- オプション解析 ---
CONTAINER_DIR="container-navsim-transfuser"
WATCH_INTERVAL=30
SKIP_UPLOAD=false
SKIP_BUILD=false
AUTO_APPROVE=false
PROJECT_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)       usage; exit 0 ;;
    -c|--container)  [[ $# -lt 2 ]] && { echo "ERROR: -c にはディレクトリ名が必要です。"; exit 1; }; CONTAINER_DIR="$2"; shift 2 ;;
    -w|--watch)      [[ $# -lt 2 ]] && { echo "ERROR: -w には秒数が必要です。"; exit 1; }; WATCH_INTERVAL="$2"; shift 2 ;;
    --skip-upload)   SKIP_UPLOAD=true; shift ;;
    --skip-build)    SKIP_BUILD=true; shift ;;
    --auto-approve)  AUTO_APPROVE=true; shift ;;
    *)               PROJECT_NAME="$1"; shift ;;
  esac
done

PROJECT_NAME="${PROJECT_NAME:-${DEFAULT_PROJECT_NAME}}"
REGION="${AWS_DEFAULT_REGION:-${DEFAULT_REGION}}"

# --- Role ARN を CloudFormation から取得 ---
echo -e "${BOLD}=== Pipeline 一括実行 ===${RESET}"
echo ""
echo -e "${BOLD}Project:${RESET}   ${PROJECT_NAME}"
echo -e "${BOLD}Container:${RESET} ${CONTAINER_DIR}"
echo -e "${BOLD}Region:${RESET}    ${REGION}"

# インスタンスタイプを 03-create-and-run-pipeline.py から実際の設定値を取得して表示
INSTANCE_INFO=$(python "${SCRIPTS_DIR}/03-create-and-run-pipeline.py" \
  --project-name "${PROJECT_NAME}" \
  --role-arn "dummy" \
  --region "${REGION}" \
  --container-dir "pipelines/${CONTAINER_DIR}" \
  --show-config 2>/dev/null) || INSTANCE_INFO=""
if [[ -n "${INSTANCE_INFO}" ]]; then
  echo "${INSTANCE_INFO}"
fi

# S3 データパスの表示
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
DATASET_BUCKET="${PROJECT_NAME}-dataset-${ACCOUNT_ID}-${REGION}"
echo -e "${BOLD}Data:${RESET}      s3://${DATASET_BUCKET}/${CONTAINER_DIR}/train/"
echo ""

ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name "${PROJECT_NAME}-stack" \
  --region "${REGION}" \
  --query 'Stacks[0].Outputs[?OutputKey==`SageMakerRoleArn`].OutputValue' \
  --output text 2>/dev/null) || {
  echo -e "${RED}ERROR: CloudFormation スタック '${PROJECT_NAME}-stack' が見つかりません。${RESET}"
  echo "先に deploy.sh を実行してください。"
  exit 1
}

if [[ -z "${ROLE_ARN}" ]]; then
  echo -e "${RED}ERROR: SageMakerRoleArn が取得できませんでした。${RESET}"
  exit 1
fi

echo -e "${BOLD}Role ARN:${RESET}  ${ROLE_ARN}"
echo ""

if [[ "${AUTO_APPROVE}" == "false" ]]; then
  read -rp "上記の設定で Pipeline を実行しますか？ [y/N]: " confirm
  if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
    echo "中止しました。"
    exit 0
  fi
  echo ""
fi
if [[ "${SKIP_UPLOAD}" == "false" ]]; then
  echo -e "${BOLD}${CYAN}[Step 1/3] データセットアップロード${RESET}"
  "${SCRIPTS_DIR}/01-upload-dataset.sh" -c "${CONTAINER_DIR}" --auto-approve "${PROJECT_NAME}"
  echo ""
else
  echo -e "${YELLOW}[Step 1/3] データセットアップロード: スキップ${RESET}"
  echo ""
fi

# --- Step 2: コンテナビルド & ECR プッシュ ---
if [[ "${SKIP_BUILD}" == "true" ]]; then
  echo -e "${YELLOW}[Step 2/3] コンテナビルド & ECR プッシュ: スキップ (--skip-build)${RESET}"
  echo ""
elif [[ ! -f "pipelines/${CONTAINER_DIR}/Dockerfile" ]]; then
  echo -e "${YELLOW}[Step 2/3] コンテナビルド & ECR プッシュ: スキップ (${CONTAINER_DIR} はマネージドコンテナ)${RESET}"
  echo ""
else
  echo -e "${BOLD}${CYAN}[Step 2/3] コンテナビルド & ECR プッシュ${RESET}"
  "${SCRIPTS_DIR}/02-build-and-push-container.sh" -c "${CONTAINER_DIR}" --auto-approve "${PROJECT_NAME}"
  echo ""
fi

# --- Step 3: Pipeline 作成 & 実行 ---
echo -e "${BOLD}${CYAN}[Step 3/3] Pipeline 作成 & 実行${RESET}"

PIPELINE_ARGS=(
  --project-name "${PROJECT_NAME}"
  --role-arn "${ROLE_ARN}"
  --region "${REGION}"
  --container-dir "pipelines/${CONTAINER_DIR}"
  --create --start
  --auto-approve
)

python "${SCRIPTS_DIR}/03-create-and-run-pipeline.py" "${PIPELINE_ARGS[@]}"
echo ""

# --- Step 4: 実行状況のポーリング ---
echo -e "${BOLD}${CYAN}[完了待ち] Pipeline 実行状況をポーリング中 (${WATCH_INTERVAL} 秒間隔)${RESET}"
echo -e "${YELLOW}Ctrl+C で停止できます。${RESET}"
echo ""

trap 'echo ""; echo "ポーリングを停止しました。"; exit 0' INT

while true; do
  "${SCRIPTS_DIR}/04-check-pipeline-status.sh" -c "${CONTAINER_DIR}" "${PROJECT_NAME}"

  # 最新実行のステータスを取得
  EXEC_STATUS=$(aws sagemaker list-pipeline-executions \
    --pipeline-name "${PROJECT_NAME}-${CONTAINER_DIR}-pipeline" \
    --sort-by CreationTime \
    --sort-order Descending \
    --max-results 1 \
    --region "${REGION}" \
    --query 'PipelineExecutionSummaries[0].PipelineExecutionStatus' \
    --output text 2>/dev/null) || true

  if [[ "${EXEC_STATUS}" == "Succeeded" ]]; then
    echo ""
    echo -e "${GREEN}${BOLD}Pipeline が正常に完了しました。${RESET}"
    break
  elif [[ "${EXEC_STATUS}" == "Failed" || "${EXEC_STATUS}" == "Stopped" ]]; then
    echo ""
    echo -e "${RED}${BOLD}Pipeline が ${EXEC_STATUS} で終了しました。${RESET}"
    exit 1
  fi

  echo ""
  echo -e "${YELLOW}次の更新: ${WATCH_INTERVAL} 秒後 ($(date '+%H:%M:%S %Z'))${RESET}"
  sleep "${WATCH_INTERVAL}"
done
