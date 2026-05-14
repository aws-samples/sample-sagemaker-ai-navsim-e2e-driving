#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail

# ============================================================
# Build and push training/evaluation container to ECR
# ============================================================
#
# 使い方:
#   ./02-build-and-push-container.sh [オプション] [プロジェクト名]
#
# 引数:
#   $1 - プロジェクト名。ECR リポジトリ名 ({プロジェクト名}-container) に使用 (デフォルト: sagemaker-ai-ml-pipeline)
#
# オプション:
#   --container, -c  コンテナディレクトリ名 (デフォルト: container-navsim-transfuser)
#                    例: container-navsim-transfuser, container-pytorch-dlc
#
# 環境変数:
#   AWS_DEFAULT_REGION - デプロイ先リージョン (デフォルト: us-east-1)
#
# 処理内容:
#   1. ECR にログイン
#   2. 指定されたコンテナディレクトリの Dockerfile をビルド
#   3. ECR リポジトリにプッシュ
# ============================================================

usage() {
  cat << EOF
使い方: $(basename "$0") [オプション] [プロジェクト名]

Build and push training/evaluation container to ECR.

オプション:
  -c, --container DIR   コンテナディレクトリ名 (デフォルト: container-navsim-transfuser)
  --auto-approve        確認プロンプトをスキップ
  -h, --help            このヘルプを表示

引数:
  プロジェクト名    ECR リポジトリ名 ({プロジェクト名}-container) に使用
                    (デフォルト: sagemaker-ai-ml-pipeline)

利用可能なコンテナ:
  container-navsim-transfuser       NAVSIM Transfuser (GPU)
  container-navsim-ego-mlp     NAVSIM EgoStatusMLP (CPU)
  container-pytorch-dlc        PyTorch DLC ベース (マネージドコンテナ、ビルド不要)
  container-pytorch-dlc-byoc   PyTorch DLC ベース BYOC (Train も BYOC イメージ)

環境変数:
  AWS_DEFAULT_REGION    デプロイ先リージョン (デフォルト: us-east-1)

例:
  $(basename "$0")
  $(basename "$0") -c container-pytorch-dlc
  $(basename "$0") -c container-pytorch-dlc my-project
  AWS_DEFAULT_REGION=us-east-1 $(basename "$0") my-project
EOF
}

# ヘルプオプションの処理
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# オプション解析
CONTAINER_DIR="container-navsim-transfuser"
AUTO_APPROVE=false
PROJECT_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--container)
      CONTAINER_DIR="$2"
      shift 2
      ;;
    --auto-approve)
      AUTO_APPROVE=true
      shift
      ;;
    *)
      PROJECT_NAME="$1"
      shift
      ;;
  esac
done

PROJECT_NAME="${PROJECT_NAME:-sagemaker-ai-ml-pipeline}"
source "$(dirname "$0")/../../infra/_common.sh"
REGION="${AWS_DEFAULT_REGION:-${DEFAULT_REGION}}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${PROJECT_NAME}-container"
CONTAINER_PATH="$(dirname "$0")/../${CONTAINER_DIR}/"

# コンテナディレクトリの存在確認
if [[ ! -d "${CONTAINER_PATH}" ]]; then
  echo "ERROR: Container directory not found: ${CONTAINER_PATH}"
  echo "Available containers:"
  find "$(dirname "$0")"/../ -maxdepth 1 -name "container-*" -type d -exec basename {} \;
  exit 1
fi

# Dockerfile の存在確認 (マネージドコンテナのみの構成ではビルド不要)
if [[ ! -f "${CONTAINER_PATH}/Dockerfile" ]]; then
  echo "INFO: No Dockerfile found in ${CONTAINER_DIR}."
  echo "This container uses AWS managed containers for both Train and Evaluate."
  echo "No build/push is required. Skipping."
  exit 0
fi

echo "=== Building and pushing container ==="
echo "Container: ${CONTAINER_DIR}"
echo "ECR Repo:  ${ECR_REPO}"
echo ""

if [[ "${AUTO_APPROVE}" == "false" ]]; then
  read -rp "コンテナをビルドして ECR にプッシュしますか？ [y/N]: " confirm
  if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
    echo "中止しました。"
    exit 0
  fi
  echo ""
fi

# Login to ECR (自アカウント)
aws ecr get-login-password --region "${REGION}" | \
  docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# Login to DLC ECR (AWS 提供の Deep Learning Containers) - DLC ベースの場合のみ
if grep -q "763104351884" "${CONTAINER_PATH}/Dockerfile" 2>/dev/null; then
  echo "DLC base image detected - logging in to DLC ECR..."
  aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "763104351884.dkr.ecr.${REGION}.amazonaws.com"
fi

# Tag name = container directory name (e.g. container-navsim-transfuser, container-pytorch-dlc)
IMAGE_TAG="${CONTAINER_DIR}"

echo "Image Tag: ${IMAGE_TAG}"

# Build
docker build --build-arg AWS_REGION="${REGION}" -t "${PROJECT_NAME}-container:${IMAGE_TAG}" "${CONTAINER_PATH}"

# Tag and push
docker tag "${PROJECT_NAME}-container:${IMAGE_TAG}" "${ECR_REPO}:${IMAGE_TAG}"
docker push "${ECR_REPO}:${IMAGE_TAG}"

echo "Container pushed to ${ECR_REPO}:${IMAGE_TAG}"
