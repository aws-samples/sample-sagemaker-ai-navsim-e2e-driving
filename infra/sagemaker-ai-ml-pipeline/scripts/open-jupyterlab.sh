#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail

# ============================================================
# JupyterLab をブラウザで開く
# ============================================================
#
# 使い方:
#   ./open-jupyterlab.sh [プロジェクト名]
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
NOTEBOOK_NAME="${PROJECT_NAME}-notebook"
SESSION_DURATION=14400

printf "${BOLD}${CYAN}=== JupyterLab ===${RESET}\n"

# ノートブックインスタンスの状態を確認
STATUS=$(aws sagemaker describe-notebook-instance \
  --notebook-instance-name "${NOTEBOOK_NAME}" \
  --region "${REGION}" \
  --query 'NotebookInstanceStatus' \
  --output text \
  2>&1) || {
  printf "${BOLD}${YELLOW}エラー: ノートブックインスタンス '${NOTEBOOK_NAME}' が見つかりません。${RESET}\n"
  exit 1
}

CONSOLE_URL="https://${REGION}.console.aws.amazon.com/sagemaker/home?region=${REGION}#/notebooks-and-git-repos"

printf "  ステータス: ${BOLD}${STATUS}${RESET}\n"

case "${STATUS}" in
  InService)
    printf "  ${GREEN}起動済みです。${RESET}\n"
    ;;
  Stopped)
    printf "  ${YELLOW}停止中です。起動します...${RESET}\n"
    aws sagemaker start-notebook-instance \
      --notebook-instance-name "${NOTEBOOK_NAME}" \
      --region "${REGION}" \
     
    printf "  ${YELLOW}InService になるまで待機中... (2-3 分かかる場合があります)${RESET}\n"
    printf "  ${BLUE}コンソール: ${CONSOLE_URL}${RESET}\n"
    aws sagemaker wait notebook-instance-in-service \
      --notebook-instance-name "${NOTEBOOK_NAME}" \
      --region "${REGION}" \
     
    printf "  ${GREEN}起動完了しました。${RESET}\n"
    ;;
  Pending)
    printf "  ${YELLOW}起動中です。InService になるまで待機中... (2-3 分かかる場合があります)${RESET}\n"
    printf "  ${BLUE}コンソール: ${CONSOLE_URL}${RESET}\n"
    aws sagemaker wait notebook-instance-in-service \
      --notebook-instance-name "${NOTEBOOK_NAME}" \
      --region "${REGION}" \
     
    printf "  ${GREEN}起動完了しました。${RESET}\n"
    ;;
  Stopping)
    printf "  ${YELLOW}停止処理中です。完了を待ってから起動します... (2-3 分かかる場合があります)${RESET}\n"
    printf "  ${BLUE}コンソール: ${CONSOLE_URL}${RESET}\n"
    aws sagemaker wait notebook-instance-stopped \
      --notebook-instance-name "${NOTEBOOK_NAME}" \
      --region "${REGION}" \
     
    printf "  ${YELLOW}停止完了。起動します...${RESET}\n"
    aws sagemaker start-notebook-instance \
      --notebook-instance-name "${NOTEBOOK_NAME}" \
      --region "${REGION}" \
     
    printf "  ${YELLOW}InService になるまで待機中... (2-3 分かかる場合があります)${RESET}\n"
    aws sagemaker wait notebook-instance-in-service \
      --notebook-instance-name "${NOTEBOOK_NAME}" \
      --region "${REGION}" \
     
    printf "  ${GREEN}起動完了しました。${RESET}\n"
    ;;
  Updating)
    printf "  ${YELLOW}更新中です。完了を待機中... (2-3 分かかる場合があります)${RESET}\n"
    printf "  ${BLUE}コンソール: ${CONSOLE_URL}${RESET}\n"
    aws sagemaker wait notebook-instance-in-service \
      --notebook-instance-name "${NOTEBOOK_NAME}" \
      --region "${REGION}" \
     
    printf "  ${GREEN}更新完了しました。${RESET}\n"
    ;;
  *)
    printf "  ${YELLOW}予期しないステータスです: ${STATUS}${RESET}\n"
    printf "  ${YELLOW}コンソールで確認してください: ${CONSOLE_URL}${RESET}\n"
    exit 1
    ;;
esac

URL=$(aws sagemaker create-presigned-notebook-instance-url \
  --notebook-instance-name "${NOTEBOOK_NAME}" \
  --session-expiration-duration-in-seconds "${SESSION_DURATION}" \
  --region "${REGION}" \
  --query AuthorizedUrl \
  --output text \
 )
# presigned URL はデフォルトで Jupyter クラシックが開くため、
# ?authToken= の前に /lab を挿入して JupyterLab を直接開けるようにする
# URL 形式: https://<name>.notebook.<region>.sagemaker.aws?authToken=...
#        → https://<name>.notebook.<region>.sagemaker.aws/lab?authToken=...
JUPYTERLAB_URL="${URL/\?//lab?}"
print_hyperlink "${JUPYTERLAB_URL}" "JupyterLab を開く"
open "${JUPYTERLAB_URL}"
printf "${YELLOW}※ ブラウザセッションは 4 時間有効です。${RESET}\n"
