#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail

# ============================================================
# SageMaker Unified Studio 連携セットアップ / 解除スクリプト
# ============================================================
#
# 使い方:
#   セットアップ:
#     ./setup-unified-studio.sh --domain-id <ID> --project-id <ID> [オプション]
#   連携解除:
#     ./setup-unified-studio.sh --unlink --domain-id <ID> --project-id <ID> [オプション]
#
# 必須引数 (セットアップ時):
#   --domain-id       Unified Studio (DataZone) ドメイン ID
#   --project-id      Unified Studio (DataZone) プロジェクト ID
#
# オプション:
#   --unlink          連携を解除する (CloudFormation スタックの削除)
#   --region          AWS リージョン (デフォルト: us-east-1)
#   --project-name    リソース命名プレフィックス (デフォルト: sagemaker-ai-ml-pipeline)
#   --auto-approve    確認プロンプトをスキップして実行
#   -h, --help        ヘルプを表示
#
# セットアップ時の処理内容:
#   1. DataZone connection ID の自動取得
#   2. CloudFormation スタックのデプロイ (RAM share + DataZone data source)
#   3. Data source の初回実行 (Unified Studio への初回同期)
#   4. MLflow App の接続 (DataZone create-connection)
#   5. 既存 SageMaker リソースへの AmazonDataZoneProject タグ付与
#
# 連携解除時の処理内容:
#   1. 既存 SageMaker リソースから AmazonDataZoneProject タグを削除
#   2. MLflow App の接続を削除
#   3. CloudFormation スタックの削除 (RAM share + DataZone data source)
#
# 前提条件:
#   - メインスタック (sagemaker-ai-ml-pipeline-stack) がデプロイ済みであること
#   - Unified Studio のドメインとプロジェクトが作成済みであること
#
# 参考:
#   docs/unified-studio-integration-guide.md
# ============================================================

source "$(dirname "$0")/../../_common.sh"

# --- ヘルプ ---
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  awk '/^# ====/{n++; next} n>=1 && n<=2{sub(/^# ?/,""); print}' "$0"
  exit 0
fi

# --- 引数解析 ---
DOMAIN_ID=""
PROJECT_ID=""
REGION="${AWS_DEFAULT_REGION:-${DEFAULT_REGION}}"
PROJECT_NAME="${DEFAULT_PROJECT_NAME}"
AUTO_APPROVE=false
UNLINK=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain-id)
      DOMAIN_ID="$2"
      shift 2
      ;;
    --project-id)
      PROJECT_ID="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --project-name)
      PROJECT_NAME="$2"
      shift 2
      ;;
    --auto-approve)
      AUTO_APPROVE=true
      shift
      ;;
    --unlink)
      UNLINK=true
      shift
      ;;
    *)
      printf "${RED}不明なオプション: $1${RESET}\n"
      exit 1
      ;;
  esac
done

STACK_NAME="${PROJECT_NAME}-unified-studio-integration-stack"

# ============================================================
# 連携解除モード
# ============================================================
if [[ "${UNLINK}" == "true" ]]; then
  # --domain-id と --project-id は unlink 時も必要
  if [[ -z "${DOMAIN_ID}" ]]; then
    printf "${RED}エラー: --domain-id は必須です${RESET}\n"
    exit 1
  fi
  if [[ -z "${PROJECT_ID}" ]]; then
    printf "${RED}エラー: --project-id は必須です${RESET}\n"
    exit 1
  fi

  echo ""
  printf "${BOLD}${CYAN}=== SageMaker Unified Studio 連携の解除 ===${RESET}\n"
  printf "${BLUE}ドメイン ID     :${RESET} ${DOMAIN_ID}\n"
  printf "${BLUE}プロジェクト ID :${RESET} ${PROJECT_ID}\n"
  printf "${BLUE}スタック名      :${RESET} ${STACK_NAME}\n"
  printf "${BLUE}リージョン      :${RESET} ${REGION}\n"
  echo ""

  confirm_or_abort "上記のスタックを削除して連携を解除しますか?"

  # --- Step 1/3: AmazonDataZoneProject タグの削除 ---
  printf "\n${BOLD}${CYAN}=== Step 1/3: AmazonDataZoneProject タグの削除 ===${RESET}\n"

  SCRIPT_DIR="$(dirname "$0")"
  python3 "${SCRIPT_DIR}/tag-resources.py" \
    --project-id "${PROJECT_ID}" \
    --region "${REGION}" \
    --project-name "${PROJECT_NAME}" \
    --unlink

  # --- Step 2/3: MLflow App 接続の削除 ---
  printf "\n${BOLD}${CYAN}=== Step 2/3: MLflow App 接続の削除 ===${RESET}\n"

  MLFLOW_CONNECTION_ID=$(aws datazone list-connections \
    --domain-identifier "${DOMAIN_ID}" \
    --project-identifier "${PROJECT_ID}" \
    --type MLFLOW \
    --region "${REGION}" \
    --query "items[?name=='${PROJECT_NAME}-mlflow'].connectionId | [0]" \
    --output text \
    2>/dev/null || echo "")

  if [[ -n "${MLFLOW_CONNECTION_ID}" && "${MLFLOW_CONNECTION_ID}" != "None" ]]; then
    aws datazone delete-connection \
      --domain-identifier "${DOMAIN_ID}" \
      --identifier "${MLFLOW_CONNECTION_ID}" \
      --region "${REGION}" \
      > /dev/null
    printf "${GREEN}✔ MLflow 接続を削除しました (${MLFLOW_CONNECTION_ID})${RESET}\n"
  else
    printf "${YELLOW}MLflow 接続が見つかりませんでした (スキップ)${RESET}\n"
  fi

  # --- Step 3/3: CloudFormation スタックの削除 ---
  printf "\n${BOLD}${CYAN}=== Step 3/3: CloudFormation スタックの削除 ===${RESET}\n"
  aws cloudformation delete-stack \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
   

  printf "${BLUE}スタック削除を待機中...${RESET}\n"
  aws cloudformation wait stack-delete-complete \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
   

  echo ""
  printf "${BOLD}${GREEN}✔ 連携を解除しました${RESET}\n"
  echo ""
  exit 0
fi

# ============================================================
# セットアップモード
# ============================================================

# --- 必須引数チェック ---
if [[ -z "${DOMAIN_ID}" ]]; then
  printf "${RED}エラー: --domain-id は必須です${RESET}\n"
  exit 1
fi
if [[ -z "${PROJECT_ID}" ]]; then
  printf "${RED}エラー: --project-id は必須です${RESET}\n"
  exit 1
fi

# --- セットアップ情報の表示 ---
echo ""
printf "${BOLD}${CYAN}=== SageMaker Unified Studio 連携セットアップ ===${RESET}\n"
printf "${BLUE}ドメイン ID     :${RESET} ${DOMAIN_ID}\n"
printf "${BLUE}プロジェクト ID :${RESET} ${PROJECT_ID}\n"
printf "${BLUE}リージョン      :${RESET} ${REGION}\n"
printf "${BLUE}プロジェクト名  :${RESET} ${PROJECT_NAME}\n"
printf "${BLUE}スタック名      :${RESET} ${STACK_NAME}\n"
echo ""

confirm_or_abort "上記の内容で Unified Studio 連携をセットアップしますか?"

# --- Step 1/5: Connection ID の自動取得 ---
printf "\n${BOLD}${CYAN}=== Step 1/5: Connection ID の取得 ===${RESET}\n"

CONNECTION_ID=$(aws datazone list-connections \
  --domain-identifier "${DOMAIN_ID}" \
  --project-identifier "${PROJECT_ID}" \
  --region "${REGION}" \
  --query 'items[0].connectionId' \
  --output text \
 )

if [[ -z "${CONNECTION_ID}" || "${CONNECTION_ID}" == "None" ]]; then
  printf "${RED}エラー: Connection ID を取得できませんでした${RESET}\n"
  printf "${RED}ドメイン ID とプロジェクト ID が正しいか確認してください${RESET}\n"
  exit 1
fi

printf "${GREEN}✔ Connection ID: ${CONNECTION_ID}${RESET}\n"

# --- Step 2/5: CloudFormation スタックのデプロイ ---
printf "\n${BOLD}${CYAN}=== Step 2/5: CloudFormation スタックのデプロイ ===${RESET}\n"

TEMPLATE_FILE="$(dirname "$0")/../cfn/integration.yaml"

aws cloudformation deploy \
  --template-file "${TEMPLATE_FILE}" \
  --stack-name "${STACK_NAME}" \
  --parameter-overrides \
    "DomainId=${DOMAIN_ID}" \
    "ProjectId=${PROJECT_ID}" \
    "ConnectionId=${CONNECTION_ID}" \
    "ProjectName=${PROJECT_NAME}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "${REGION}" \
 

printf "${GREEN}✔ スタック '${STACK_NAME}' のデプロイが完了しました${RESET}\n"

# --- Step 3/5: Data source の初回実行 ---
printf "\n${BOLD}${CYAN}=== Step 3/5: Data source の初回実行 ===${RESET}\n"

# CloudFormation の出力から Data Source ID を取得
DATA_SOURCE_ID=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='DataSourceId'].OutputValue" \
  --output text \
 )

# Data Source ID は "DomainId|DataSourceId" 形式で返される場合があるため、
# パイプ区切りの2番目の値を取得する
if [[ "${DATA_SOURCE_ID}" == *"|"* ]]; then
  DATA_SOURCE_ID="${DATA_SOURCE_ID##*|}"
fi

aws datazone start-data-source-run \
  --domain-identifier "${DOMAIN_ID}" \
  --data-source-identifier "${DATA_SOURCE_ID}" \
  --region "${REGION}" \
  > /dev/null

printf "${GREEN}✔ Data source の初回実行を開始しました${RESET}\n"

# --- Step 4/5: MLflow App の接続 ---
printf "\n${BOLD}${CYAN}=== Step 4/5: MLflow App の接続 ===${RESET}\n"

MLFLOW_SERVER_NAME="${PROJECT_NAME}-mlflow"
MLFLOW_ARN=$(aws cloudformation describe-stacks \
  --stack-name "${PROJECT_NAME}-stack" \
  --region "${REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='MlflowAppArn'].OutputValue | [0]" \
  --output text \
  2>/dev/null || echo "")

if [[ -z "${MLFLOW_ARN}" || "${MLFLOW_ARN}" == "None" ]]; then
  printf "${YELLOW}MLflow App '${MLFLOW_SERVER_NAME}' が見つかりませんでした (スキップ)${RESET}\n"
else
  # 既存の MLflow 接続を確認
  EXISTING_MLFLOW=$(aws datazone list-connections \
    --domain-identifier "${DOMAIN_ID}" \
    --project-identifier "${PROJECT_ID}" \
    --type MLFLOW \
    --region "${REGION}" \
    --query "items[?name=='${MLFLOW_SERVER_NAME}'].connectionId | [0]" \
    --output text \
    2>/dev/null || echo "")

  if [[ -n "${EXISTING_MLFLOW}" && "${EXISTING_MLFLOW}" != "None" ]]; then
    printf "${GREEN}✔ MLflow 接続は既に存在します (${EXISTING_MLFLOW})${RESET}\n"
  else
    # Tooling 環境 ID を取得 (MLflow 接続のコンテナとして使用)
    TOOLING_ENV_ID=$(aws datazone list-environments \
      --domain-identifier "${DOMAIN_ID}" \
      --project-identifier "${PROJECT_ID}" \
      --region "${REGION}" \
      --query "items[?name=='Tooling'].id | [0]" \
      --output text \
      2>/dev/null || echo "")

    if [[ -z "${TOOLING_ENV_ID}" || "${TOOLING_ENV_ID}" == "None" ]]; then
      printf "${YELLOW}Tooling 環境が見つかりませんでした (スキップ)${RESET}\n"
    else
      aws datazone create-connection \
        --domain-identifier "${DOMAIN_ID}" \
        --environment-identifier "${TOOLING_ENV_ID}" \
        --name "${MLFLOW_SERVER_NAME}" \
        --props '{"mlflowProperties":{"trackingServerArn":"'"${MLFLOW_ARN}"'"}}' \
        --region "${REGION}" \
        > /dev/null

      printf "${GREEN}✔ MLflow App を接続しました${RESET}\n"
      printf "  ARN: ${MLFLOW_ARN}\n"
    fi
  fi
fi

# --- Step 5/5: 既存リソースへの AmazonDataZoneProject タグ付与 ---
printf "\n${BOLD}${CYAN}=== Step 5/5: 既存リソースへの AmazonDataZoneProject タグ付与 ===${RESET}\n"

SCRIPT_DIR="$(dirname "$0")"
python3 "${SCRIPT_DIR}/tag-resources.py" \
  --project-id "${PROJECT_ID}" \
  --region "${REGION}" \
  --project-name "${PROJECT_NAME}"

# --- 完了 ---
echo ""
printf "${BOLD}${GREEN}=== セットアップ完了 ===${RESET}\n"
printf "Unified Studio の ${BOLD}Build${RESET} メニューから以下を確認してください。\n"
printf "  - ${BOLD}Model Registry → Registered Models${RESET} (DataSource 経由)\n"
printf "  - ${BOLD}ML Pipelines${RESET} (タグ経由)\n"
printf "  - ${BOLD}MLflow${RESET} (接続経由)\n"
printf "${YELLOW}※ 初回同期が完了するまで数分かかる場合があります${RESET}\n"
echo ""
