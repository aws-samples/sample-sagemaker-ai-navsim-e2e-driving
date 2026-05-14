#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail

# ============================================================
# SageMaker Inference Endpoint - デプロイスクリプト
# ============================================================
#
# 使い方:
#   ./deploy.sh [オプション] [スタック名] [プロジェクト名]
#
# オプション:
#   -h, --help              ヘルプを表示
#   -c, --container MODEL   モデル種別 (navsim-ego-mlp | navsim-transfuser)
#   --auto-approve          確認プロンプトをスキップして実行
#   --model-s3-uri URI      model.tar.gz の S3 URI を直接指定
#   --instance-type TYPE    インスタンスタイプ (デフォルト: モデルに応じて自動選択)
#
# 処理内容:
#   1. 最新のモデルアーティファクトを検索
#   2. inference.py を含む model.tar.gz を再パッケージして S3 にアップロード
#   3. CloudFormation でエンドポイントをデプロイ
# ============================================================

source "$(dirname "$0")/../../_common.sh"

# --- ヘルプ ---
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  awk '/^# ====/{n++; next} n>=1 && n<=2{sub(/^# ?/,""); print}' "$0"
  exit 0
fi

# --- オプション解析 ---
AUTO_APPROVE=false
MODEL_S3_URI=""
MODEL_TYPE="navsim-ego-mlp"
INSTANCE_TYPE=""
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --auto-approve) AUTO_APPROVE=true; shift ;;
    -c|--container) MODEL_TYPE="$2"; shift 2 ;;
    --model-s3-uri) MODEL_S3_URI="$2"; shift 2 ;;
    --instance-type) INSTANCE_TYPE="$2"; shift 2 ;;
    *) POSITIONAL_ARGS+=("$1"); shift ;;
  esac
done

# --- モデル別設定 ---
case "${MODEL_TYPE}" in
  navsim-ego-mlp)
    INFERENCE_SCRIPT="inference_navsim_ego_mlp.py"
    DEFAULT_INSTANCE="ml.m5.large"
    CONTAINER_IMAGE_SUFFIX_GPU="pytorch-inference:2.5.1-cpu-py311-ubuntu22.04-sagemaker"
    CONTAINER_IMAGE_SUFFIX_CPU="pytorch-inference:2.5.1-cpu-py311-ubuntu22.04-sagemaker"
    ;;
  navsim-transfuser)
    INFERENCE_SCRIPT="inference_navsim_transfuser.py"
    DEFAULT_INSTANCE="ml.g6.xlarge"
    CONTAINER_IMAGE_SUFFIX_GPU="pytorch-inference:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker"
    CONTAINER_IMAGE_SUFFIX_CPU="pytorch-inference:2.5.1-cpu-py311-ubuntu22.04-sagemaker"
    ;;
  *)
    echo -e "${RED}❌ 未対応のモデル: ${MODEL_TYPE}${RESET}"
    echo "   対応モデル: navsim-ego-mlp, navsim-transfuser"
    exit 1
    ;;
esac

INSTANCE_TYPE="${INSTANCE_TYPE:-${DEFAULT_INSTANCE}}"

# インスタンスタイプに応じて CPU/GPU コンテナイメージを選択
if [[ "${INSTANCE_TYPE}" =~ ^ml\.(g[0-9]|p[0-9]|trn|inf) ]]; then
  CONTAINER_IMAGE_SUFFIX="${CONTAINER_IMAGE_SUFFIX_GPU}"
else
  CONTAINER_IMAGE_SUFFIX="${CONTAINER_IMAGE_SUFFIX_CPU}"
fi

# --- パラメータ ---
STACK_NAME="${POSITIONAL_ARGS[0]:-sagemaker-ai-inference-${MODEL_TYPE}-stack}"
PROJECT_NAME="${POSITIONAL_ARGS[1]:-${DEFAULT_PROJECT_NAME}}"
REGION="${AWS_DEFAULT_REGION:-${DEFAULT_REGION}}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
MODEL_BUCKET="${PROJECT_NAME}-model-${ACCOUNT_ID}-${REGION}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEMPLATE_FILE="${SCRIPT_DIR}/../cfn/sagemaker-ai-inference.yaml"
CONTAINER_IMAGE="763104351884.dkr.ecr.${REGION}.amazonaws.com/${CONTAINER_IMAGE_SUFFIX}"

echo -e "${BOLD}${CYAN}============================================${RESET}"
echo -e "${BOLD}${CYAN} SageMaker Inference Endpoint - Deploy${RESET}"
echo -e "${BOLD}${CYAN}============================================${RESET}"
echo -e "Model:    ${BOLD}${MODEL_TYPE}${RESET}"
echo -e "Stack:    ${BOLD}${STACK_NAME}${RESET}"
echo -e "Project:  ${BOLD}${PROJECT_NAME}${RESET}"
echo -e "Instance: ${BOLD}${INSTANCE_TYPE}${RESET}"
echo -e "Region:   ${BOLD}${REGION}${RESET}"
echo ""

# -------------------------------------------------------------------------
# Step 1: モデルアーティファクトの検索
# -------------------------------------------------------------------------
if [[ -z "${MODEL_S3_URI}" ]]; then
  echo -e "${CYAN}[Step 1/4]${RESET} 最新の ${MODEL_TYPE} モデルを検索..."
  # endpoint/ (前回リパッケージ済み) または output/{MODEL_TYPE}/ (Pipeline 出力) から最新を探す
  LATEST_MODEL=$(aws s3 ls "s3://${MODEL_BUCKET}/" --recursive --region "${REGION}" \
    | grep -E "(endpoint/${MODEL_TYPE}/|output/${MODEL_TYPE}/)" \
    | grep "model.tar.gz$" \
    | sort -k1,2 | tail -1 | awk '{print $4}') || true

  if [[ -z "${LATEST_MODEL}" ]]; then
    echo -e "${RED}❌ モデルアーティファクトが見つかりません。${RESET}"
    echo "   先に Pipeline を実行してモデルを学習してください。"
    echo "   ./pipelines/scripts/run-pipeline.sh -c container-${MODEL_TYPE}"
    exit 1
  fi
  SOURCE_MODEL_URI="s3://${MODEL_BUCKET}/${LATEST_MODEL}"
  echo "  Found: ${SOURCE_MODEL_URI}"
else
  SOURCE_MODEL_URI="${MODEL_S3_URI}"
  echo -e "${CYAN}[Step 1/4]${RESET} 指定されたモデルを使用: ${SOURCE_MODEL_URI}"
fi

# -------------------------------------------------------------------------
# Step 2: inference.py を含む model.tar.gz を再パッケージ
# -------------------------------------------------------------------------
echo -e "${CYAN}[Step 2/4]${RESET} inference.py を含む model.tar.gz を再パッケージ..."
WORK_DIR=$(mktemp -d)
trap 'rm -rf "${WORK_DIR}"' EXIT

aws s3 cp "${SOURCE_MODEL_URI}" "${WORK_DIR}/original.tar.gz" --region "${REGION}" --quiet
cd "${WORK_DIR}"
tar xzf original.tar.gz

mkdir -p code
cp "${SCRIPT_DIR}/${INFERENCE_SCRIPT}" code/inference.py

# 公式版 Transfuser はモデル定義が複数ファイルに分かれているため追加コピー
TRANSFUSER_DIR="${SCRIPT_DIR}/../../../pipelines/container-navsim-transfuser"
if [[ "${MODEL_TYPE}" == "navsim-transfuser" && -d "${TRANSFUSER_DIR}" ]]; then
  cp "${TRANSFUSER_DIR}/transfuser_config.py" code/
  cp "${TRANSFUSER_DIR}/transfuser_backbone.py" code/
  cp "${TRANSFUSER_DIR}/transfuser_model.py" code/
  # Lite 版チェックポイントの後方互換用
  cp "${SCRIPT_DIR}/transfuser_model_lite.py" code/
fi

tar czf model.tar.gz model.pth code/

ENDPOINT_MODEL_URI="s3://${MODEL_BUCKET}/endpoint/${MODEL_TYPE}/model.tar.gz"
aws s3 cp model.tar.gz "${ENDPOINT_MODEL_URI}" --region "${REGION}" --quiet
echo "  Uploaded: ${ENDPOINT_MODEL_URI}"

# -------------------------------------------------------------------------
# Step 3: Service Quotas チェック & 引き上げリクエスト
# -------------------------------------------------------------------------
echo -e "${CYAN}[Step 3/4]${RESET} Service Quotas を確認..."

# インスタンスタイプに対応する endpoint 用 Quota Code
case "${INSTANCE_TYPE}" in
  ml.m5.large)     QUOTA_CODE="L-614B09FD" ;;
  ml.g4dn.xlarge)  QUOTA_CODE="L-B8A65202" ;;
  ml.g4dn.2xlarge) QUOTA_CODE="L-B19B6BAE" ;;
  ml.g6.xlarge)    QUOTA_CODE="L-D470D954" ;;
  ml.g6.4xlarge)   QUOTA_CODE="L-E8498C83" ;;
  *)               QUOTA_CODE="" ;;
esac

if [[ -n "${QUOTA_CODE}" && "${QUOTA_CODE}" != "None" ]]; then
  current_value=$(aws service-quotas get-service-quota \
    --service-code sagemaker \
    --quota-code "${QUOTA_CODE}" \
    --region "${REGION}" \
    --query 'Quota.Value' \
    --output text 2>/dev/null) || current_value="0"

  current_int=${current_value%.*}
  desired_value=2

  if [[ "${current_int}" -ge "${desired_value}" ]]; then
    echo -e "  ✅ ${INSTANCE_TYPE} for endpoint usage: ${current_int} (十分)"
  else
    echo -e "  ⚠️  ${INSTANCE_TYPE} for endpoint usage: ${current_int} → ${desired_value} にリクエスト中..."
    request_output=$(aws service-quotas request-service-quota-increase \
      --service-code sagemaker \
      --quota-code "${QUOTA_CODE}" \
      --desired-value "${desired_value}" \
      --region "${REGION}" 2>&1) \
      && echo "  リクエスト完了 (承認まで数分〜数時間かかる場合があります)" \
      || echo "  ${request_output}"
    echo "  確認: aws service-quotas get-service-quota --service-code sagemaker --quota-code ${QUOTA_CODE} --region ${REGION} --query Quota.Value"
  fi
else
  echo -e "  ⚠️  ${INSTANCE_TYPE} の endpoint 用 Quota Code が見つかりません。手動で確認してください。"
fi

# -------------------------------------------------------------------------
# Step 4: CloudFormation デプロイ
# -------------------------------------------------------------------------
echo -e "${CYAN}[Step 4/4]${RESET} CloudFormation スタックをデプロイ..."

# ML Pipeline スタックから Role ARN を取得
PIPELINE_STACK="${PROJECT_NAME}-stack"
ROLE_ARN=$(aws cloudformation describe-stacks \
  --stack-name "${PIPELINE_STACK}" \
  --query 'Stacks[0].Outputs[?OutputKey==`SageMakerRoleArn`].OutputValue' \
  --output text --region "${REGION}" 2>/dev/null) || true

if [[ -z "${ROLE_ARN}" || "${ROLE_ARN}" == "None" ]]; then
  echo -e "${RED}❌ SageMaker Role ARN が取得できません。${RESET}"
  echo "   ML Pipeline スタック '${PIPELINE_STACK}' がデプロイ済みか確認してください。"
  exit 1
fi
echo "  Role: ${ROLE_ARN}"

confirm_or_abort "エンドポイントをデプロイしますか?"

aws cloudformation deploy \
  --template-file "${TEMPLATE_FILE}" \
  --stack-name "${STACK_NAME}" \
  --parameter-overrides \
    ProjectName="${PROJECT_NAME}" \
    ModelName="${MODEL_TYPE}" \
    ModelDataUrl="${ENDPOINT_MODEL_URI}" \
    InstanceType="${INSTANCE_TYPE}" \
    ContainerImage="${CONTAINER_IMAGE}" \
    RoleArn="${ROLE_ARN}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "${REGION}" \
  --no-fail-on-empty-changeset

# --- 結果表示 ---
ENDPOINT_NAME=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --query 'Stacks[0].Outputs[?OutputKey==`EndpointName`].OutputValue' \
  --output text --region "${REGION}")

echo ""
echo -e "${BOLD}${GREEN}============================================${RESET}"
echo -e "${BOLD}${GREEN} ✅ デプロイ完了${RESET}"
echo -e "${BOLD}${GREEN}============================================${RESET}"
echo -e "Endpoint: ${BOLD}${ENDPOINT_NAME}${RESET}"
echo ""
echo "テスト:"
echo "  SAGEMAKER_ENDPOINT=${ENDPOINT_NAME} streamlit run demo-app/main.py"
echo ""
echo "削除:"
echo "  ./infra/sagemaker-ai-inference/scripts/destroy.sh ${STACK_NAME}"
