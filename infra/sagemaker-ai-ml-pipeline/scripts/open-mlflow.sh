#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail

# ============================================================
# MLflow UI をブラウザで開く
# ============================================================
#
# 使い方:
#   ./open-mlflow.sh [プロジェクト名]
#
# 引数:
#   $1 - プロジェクト名 (デフォルト: sagemaker-ai-ml-pipeline)
#
# 環境変数:
#   AWS_DEFAULT_REGION - 対象リージョン (デフォルト: us-east-1)
# ============================================================

source "$(dirname "$0")/../../_common.sh"

PROJECT_NAME="${1:-${DEFAULT_PROJECT_NAME}}"
REGION="${AWS_DEFAULT_REGION:-${DEFAULT_REGION}}"
MLFLOW_NAME="${PROJECT_NAME}-mlflow"
SESSION_DURATION=14400

printf "${BOLD}${CYAN}=== MLflow UI ===${RESET}\n"
STACK_NAME="${PROJECT_NAME}-stack"
MLFLOW_ARN=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='MlflowAppArn'].OutputValue" \
  --output text)
URL=$(aws sagemaker create-presigned-mlflow-app-url \
  --arn "${MLFLOW_ARN}" \
  --session-expiration-duration-in-seconds "${SESSION_DURATION}" \
  --region "${REGION}" \
  --query AuthorizedUrl \
  --output text \
 )
print_hyperlink "${URL}" "MLflow UI を開く"
open "${URL}"
printf "${YELLOW}※ ブラウザセッションは 4 時間有効です。${RESET}\n"
