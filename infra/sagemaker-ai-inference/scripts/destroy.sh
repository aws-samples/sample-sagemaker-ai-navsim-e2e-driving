#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail

# ============================================================
# SageMaker Inference Endpoint - 削除スクリプト
# ============================================================
#
# 使い方:
#   ./destroy.sh [オプション] [スタック名]
#
# オプション:
#   -h, --help              ヘルプを表示
#   -c, --container MODEL   モデル種別 (navsim-ego-mlp | navsim-transfuser)
#   --auto-approve          確認プロンプトをスキップして実行
#
# 処理内容:
#   1. CloudFormation スタックの削除 (Endpoint → EndpointConfig → Model)
# ============================================================

source "$(dirname "$0")/../../_common.sh"

# --- ヘルプ ---
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  awk '/^# ====/{n++; next} n>=1 && n<=2{sub(/^# ?/,""); print}' "$0"
  exit 0
fi

# --- オプション解析 ---
AUTO_APPROVE=false
MODEL_TYPE=""
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto-approve) AUTO_APPROVE=true; shift ;;
    -c|--container) MODEL_TYPE="$2"; shift 2 ;;
    *) POSITIONAL_ARGS+=("$1"); shift ;;
  esac
done

# --- パラメータ ---
if [[ -n "${MODEL_TYPE}" ]]; then
  case "${MODEL_TYPE}" in
    navsim-ego-mlp|navsim-transfuser) ;;
    *)
      echo -e "${RED}❌ 未対応のモデル: ${MODEL_TYPE}${RESET}"
      echo "   対応モデル: navsim-ego-mlp, navsim-transfuser"
      exit 1 ;;
  esac
  DEFAULT_STACK_NAME="sagemaker-ai-inference-${MODEL_TYPE}-stack"
else
  DEFAULT_STACK_NAME="sagemaker-ai-inference-stack"
fi
STACK_NAME="${POSITIONAL_ARGS[0]:-${DEFAULT_STACK_NAME}}"
REGION="${AWS_DEFAULT_REGION:-${DEFAULT_REGION}}"

echo -e "${BOLD}${CYAN}============================================${RESET}"
echo -e "${BOLD}${CYAN} SageMaker Inference Endpoint - Destroy${RESET}"
echo -e "${BOLD}${CYAN}============================================${RESET}"
echo -e "Stack:  ${BOLD}${STACK_NAME}${RESET}"
echo -e "Region: ${BOLD}${REGION}${RESET}"

# エンドポイント名を取得して表示
ENDPOINT_NAME=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" --region "${REGION}" \
  --query 'Stacks[0].Outputs[?OutputKey==`EndpointName`].OutputValue' \
  --output text 2>/dev/null) || true
if [[ -n "${ENDPOINT_NAME}" && "${ENDPOINT_NAME}" != "None" ]]; then
  echo -e "Endpoint: ${BOLD}${ENDPOINT_NAME}${RESET}"
fi
echo ""

# スタック存在確認
if ! aws cloudformation describe-stacks --stack-name "${STACK_NAME}" --region "${REGION}" &>/dev/null; then
  echo -e "${YELLOW}スタック '${STACK_NAME}' は存在しません。${RESET}"
  exit 0
fi

confirm_or_abort "エンドポイントを削除しますか?"

echo "CloudFormation スタックを削除中..."
aws cloudformation delete-stack \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}"

echo "削除完了を待機中..."
aws cloudformation wait stack-delete-complete \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}"

echo -e "${BOLD}${GREEN}✅ 削除完了${RESET}"
