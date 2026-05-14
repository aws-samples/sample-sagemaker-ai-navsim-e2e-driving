#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail

# ============================================================
# サンプルデータセットを S3 にアップロード
# ============================================================
#
# 使い方:
#   ./01-upload-dataset.sh [オプション] [プロジェクト名]
#
# オプション:
#   -c, --container DIR   コンテナディレクトリ名 (デフォルト: container-navsim-transfuser)
#   -h, --help            このヘルプを表示
#
# 処理内容:
#   指定コンテナの data/ ディレクトリにある train.csv と test.csv を
#   S3 データセットバケットの train/ と test/ プレフィックスにアップロード
# ============================================================

usage() {
  cat << EOF
使い方: $(basename "$0") [オプション] [プロジェクト名]

サンプルデータセットを S3 にアップロードします。

オプション:
  -c, --container DIR   コンテナディレクトリ名 (デフォルト: container-navsim-transfuser)
  --auto-approve        確認プロンプトをスキップ
  -h, --help            このヘルプを表示

引数:
  プロジェクト名    S3 バケット名 ({プロジェクト名}-dataset-{アカウントID}) に使用
                    (デフォルト: sagemaker-ai-ml-pipeline)

例:
  $(basename "$0")
  $(basename "$0") -c container-pytorch-dlc
  $(basename "$0") -c container-pytorch-dlc my-project
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
BUCKET="${PROJECT_NAME}-dataset-${ACCOUNT_ID}-${REGION}"
DATA_DIR="$(dirname "$0")/../${CONTAINER_DIR}/data"

# データディレクトリの存在確認
# NAVSIM コンテナは prepare_dataset.sh で S3 に直接アップロードするため data/ を持たない
if [[ ! -d "${DATA_DIR}" ]]; then
  if [[ "${CONTAINER_DIR}" == container-navsim-* ]]; then
    echo "NAVSIM コンテナ: ダミーデータを S3 に配置します (train.py がダミーデータを自動生成)"
    CONTAINER_TAG="${CONTAINER_DIR}"
    aws s3api put-object --bucket "${BUCKET}" --key "${CONTAINER_TAG}/train/.dummy" --region "${REGION}" > /dev/null
    aws s3api put-object --bucket "${BUCKET}" --key "${CONTAINER_TAG}/test/.dummy" --region "${REGION}" > /dev/null
    echo "Done."
    exit 0
  fi
  echo "ERROR: Data directory not found: ${DATA_DIR}"
  echo "Available containers with data:"
  for d in "$(dirname "$0")"/../container-*/data; do
    [[ -d "$d" ]] && basename "$(dirname "$d")"
  done
  exit 1
fi

echo "=== Uploading dataset to S3 ==="
echo "Container: ${CONTAINER_DIR}"
echo "Bucket:    s3://${BUCKET}"
echo ""

if [[ "${AUTO_APPROVE}" == "false" ]]; then
  read -rp "データセットを S3 にアップロードしますか？ [y/N]: " confirm
  if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
    echo "中止しました。"
    exit 0
  fi
  echo ""
fi

# コンテナ名をそのまま S3 プレフィックスに使用する
# 例: s3://{bucket}/container-navsim-transfuser/train/
#     s3://{bucket}/container-navsim-transfuser/train/
CONTAINER_TAG="${CONTAINER_DIR}"

# train データのアップロード
if [[ -f "${DATA_DIR}/train.csv" ]]; then
  aws s3 cp "${DATA_DIR}/train.csv" "s3://${BUCKET}/${CONTAINER_TAG}/train/train.csv" --region "${REGION}"
  echo "Uploaded train.csv -> s3://${BUCKET}/${CONTAINER_TAG}/train/train.csv"
else
  echo "WARNING: train.csv not found in ${DATA_DIR}"
fi

# test データのアップロード
if [[ -f "${DATA_DIR}/test.csv" ]]; then
  aws s3 cp "${DATA_DIR}/test.csv" "s3://${BUCKET}/${CONTAINER_TAG}/test/test.csv" --region "${REGION}"
  echo "Uploaded test.csv -> s3://${BUCKET}/${CONTAINER_TAG}/test/test.csv"
else
  echo "WARNING: test.csv not found in ${DATA_DIR}"
fi

echo ""
echo "Done."
