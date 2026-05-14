#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail

# ============================================================
# SageMaker AI ML Pipeline - CloudFormation 削除スクリプト
# ============================================================
#
# 使い方:
#   ./destroy.sh [オプション] [スタック名] [プロジェクト名]
#
# オプション:
#   -h, --help          ヘルプを表示
#   --auto-approve      確認プロンプトをスキップして実行
#
# 引数:
#   $1 - CloudFormation スタック名 (デフォルト: sagemaker-ai-ml-pipeline-stack)
#   $2 - プロジェクト名。リソースの命名プレフィックスに使用 (デフォルト: sagemaker-ai-ml-pipeline)
#
# 環境変数:
#   AWS_DEFAULT_REGION - 対象リージョン (デフォルト: us-east-1)
#
# 処理内容:
#   1. Amazon S3 バケットの中身を削除 (スタック削除前に必要)
#   2. Amazon ECR リポジトリのイメージを削除
#   3. CloudFormation スタックの削除・完了待機
# ============================================================

source "$(dirname "$0")/../../_common.sh"

# --- ヘルプ ---
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  awk '/^# ====/{n++; next} n>=1 && n<=2{sub(/^# ?/,""); print}' "$0"
  exit 0
fi

# --- オプション解析 ---
AUTO_APPROVE=false
POSITIONAL_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --auto-approve) AUTO_APPROVE=true ;;
    *) POSITIONAL_ARGS+=("${arg}") ;;
  esac
done

# --- パラメータ ---
STACK_NAME="${POSITIONAL_ARGS[0]:-${DEFAULT_STACK_NAME}}"
PROJECT_NAME="${POSITIONAL_ARGS[1]:-${DEFAULT_PROJECT_NAME}}"
REGION="${AWS_DEFAULT_REGION:-${DEFAULT_REGION}}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# --- 削除情報の表示 ---
echo ""
printf "${BOLD}${CYAN}=== SageMaker AI ML Pipeline スタックの削除 ===${RESET}\n"
printf "${BLUE}スタック名     :${RESET} ${STACK_NAME}\n"
printf "${BLUE}プロジェクト名 :${RESET} ${PROJECT_NAME}\n"
printf "${BLUE}リージョン     :${RESET} ${REGION}\n"
printf "${BLUE}アカウント     :${RESET} ${ACCOUNT_ID}\n"
echo ""
printf "${YELLOW}以下のリソースが削除されます:${RESET}\n"
printf "  - S3: ${PROJECT_NAME}-dataset-${ACCOUNT_ID}-${REGION}\n"
printf "  - S3: ${PROJECT_NAME}-model-${ACCOUNT_ID}-${REGION}\n"
printf "  - S3: ${PROJECT_NAME}-eval-${ACCOUNT_ID}-${REGION}\n"
printf "  - S3: ${PROJECT_NAME}-mlflow-${ACCOUNT_ID}-${REGION}\n"
printf "  - ECR: ${PROJECT_NAME}-container\n"
printf "  - CloudFormation: ${STACK_NAME} (Notebook, MLflow, IAM Role 等)\n"
echo ""

confirm_or_abort "上記の内容でスタックを削除しますか?"

# --- Amazon S3 バケットの中身を削除 ---
# CloudFormation はバケットが空でないと削除できないため、先に中身を削除する。
# バケットが存在しない場合はスキップ。
BUCKETS=(
  "${PROJECT_NAME}-dataset-${ACCOUNT_ID}-${REGION}"
  "${PROJECT_NAME}-model-${ACCOUNT_ID}-${REGION}"
  "${PROJECT_NAME}-eval-${ACCOUNT_ID}-${REGION}"
  "${PROJECT_NAME}-mlflow-${ACCOUNT_ID}-${REGION}"
)

printf "${BOLD}${CYAN}=== Amazon S3 バケットの削除 ===${RESET}\n"
for BUCKET in "${BUCKETS[@]}"; do
  if aws s3api head-bucket --bucket "${BUCKET}" --region "${REGION}" 2>/dev/null; then
    printf "${BLUE}削除中:${RESET} s3://${BUCKET} ...\n"
    aws s3 rm "s3://${BUCKET}" --recursive --region "${REGION}"
  else
    printf "${YELLOW}バケット ${BUCKET} は存在しません。スキップします。${RESET}\n"
  fi
done

# --- Amazon ECR リポジトリのイメージを削除 ---
# CloudFormation はイメージが残っている ECR リポジトリを削除できないため、
# 先にイメージを削除する。リポジトリが存在しない場合はスキップ。
ECR_REPO="${PROJECT_NAME}-container"
echo ""
printf "${BOLD}${CYAN}=== Amazon ECR イメージの削除 ===${RESET}\n"
if aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${REGION}" >/dev/null 2>&1; then
  IMAGE_IDS=$(aws ecr list-images --repository-name "${ECR_REPO}" --region "${REGION}" --query 'imageIds[*]' --output json)
  if [ "${IMAGE_IDS}" != "[]" ]; then
    printf "${BLUE}イメージを削除中:${RESET} ${ECR_REPO} ...\n"
    aws ecr batch-delete-image \
      --repository-name "${ECR_REPO}" \
      --image-ids "${IMAGE_IDS}" \
      --region "${REGION}" \
     
  else
    printf "${YELLOW}${ECR_REPO} にイメージはありません。${RESET}\n"
  fi
else
  printf "${YELLOW}ECR リポジトリ ${ECR_REPO} は存在しません。スキップします。${RESET}\n"
fi

# --- SageMaker モデルパッケージグループの削除 ---
# モデルパッケージが残っているとグループを削除できないため、先にパッケージを削除する。
# グループが存在しない場合はスキップ。
echo ""
printf "${BOLD}${CYAN}=== SageMaker モデルパッケージグループの削除 ===${RESET}\n"

# 対象グループ名のパターン: {PROJECT_NAME}-* (例: {PROJECT_NAME}-pytorch, {PROJECT_NAME}-navsim-transfuser)
MODEL_GROUPS=$(aws sagemaker list-model-package-groups \
  --region "${REGION}" \
  --query "ModelPackageGroupSummaryList[?starts_with(ModelPackageGroupName, '${PROJECT_NAME}')].ModelPackageGroupName" \
  --output text \
  2>/dev/null || true)

if [ -z "${MODEL_GROUPS}" ]; then
  printf "${YELLOW}モデルパッケージグループは存在しません。スキップします。${RESET}\n"
else
  for GROUP in ${MODEL_GROUPS}; do
    printf "${BLUE}グループ処理中:${RESET} ${GROUP}\n"
    # グループ内のモデルパッケージを全件削除 (ページネーション対応)
    while true; do
      PACKAGE_ARNS=$(aws sagemaker list-model-packages \
        --model-package-group-name "${GROUP}" \
        --max-results 100 \
        --region "${REGION}" \
        --query 'ModelPackageSummaryList[].ModelPackageArn' \
        --output text \
        2>/dev/null || true)
      if [ -z "${PACKAGE_ARNS}" ]; then
        break
      fi
      for ARN in ${PACKAGE_ARNS}; do
        printf "  ${BLUE}パッケージ削除:${RESET} ${ARN}\n"
        aws sagemaker delete-model-package \
          --model-package-name "${ARN}" \
          --region "${REGION}" \
          2>/dev/null || true
      done
      # API の結果整合性を考慮して少し待つ
      sleep 1
    done
    # パッケージ削除後にグループを削除
    aws sagemaker delete-model-package-group \
      --model-package-group-name "${GROUP}" \
      --region "${REGION}" \
     
    printf "  ${GREEN}✔ グループ削除完了:${RESET} ${GROUP}\n"
  done
fi

# --- CloudFormation スタックの削除 ---
# delete-stack は非同期なので、wait で完了を待機する。
echo ""
printf "${BOLD}${CYAN}=== Lambda Layer の削除 ===${RESET}\n"
LAYER_NAME="${PROJECT_NAME}-boto3"
LAYER_VERSIONS=$(aws lambda list-layer-versions \
  --layer-name "${LAYER_NAME}" \
  --region "${REGION}" \
  --query 'LayerVersions[].Version' \
  --output text 2>/dev/null || true)
if [ -n "${LAYER_VERSIONS}" ]; then
  for VER in ${LAYER_VERSIONS}; do
    aws lambda delete-layer-version --layer-name "${LAYER_NAME}" --version-number "${VER}" --region "${REGION}"
    printf "  ${GREEN}✔${RESET} ${LAYER_NAME}:${VER} を削除しました\n"
  done
else
  printf "${YELLOW}Lambda Layer ${LAYER_NAME} は存在しません。スキップします。${RESET}\n"
fi

echo ""
printf "${BOLD}${CYAN}=== CloudFormation スタックの削除 ===${RESET}\n"
aws cloudformation delete-stack \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
 

printf "${BLUE}スタック削除を待機中...${RESET}\n"
aws cloudformation wait stack-delete-complete \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
 

echo ""
printf "${BOLD}${GREEN}✔ スタック '${STACK_NAME}' の削除が完了しました${RESET}\n"

# --- Secrets Manager シークレットの削除 ---
# deploy.sh で CFn 外に作成した GitHub 認証情報を削除する。
# CodeRepository がシークレットを参照しているため、スタック削除後に削除する。
echo ""
printf "${BOLD}${CYAN}=== Secrets Manager シークレットの削除 ===${RESET}\n"
SECRET_NAME="${PROJECT_NAME}-sagemaker-github-credentials"
if aws secretsmanager describe-secret --secret-id "${SECRET_NAME}" --region "${REGION}" >/dev/null 2>&1; then
  aws secretsmanager delete-secret \
    --secret-id "${SECRET_NAME}" \
    --region "${REGION}" >/dev/null
  printf "  ${GREEN}✔${RESET} ${SECRET_NAME} を削除しました\n"
else
  printf "${YELLOW}シークレット ${SECRET_NAME} は存在しません。スキップします。${RESET}\n"
fi
echo ""
