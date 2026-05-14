#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail

# ============================================================
# Unified Studio Foundation デプロイスクリプト
# ============================================================
#
# 使い方:
#   ./deploy-foundation.sh [オプション]
#
# オプション:
#   --region            AWS リージョン (デフォルト: us-east-1)
#   --project-name      リソース命名プレフィックス (デフォルト: sagemaker-ai-ml-pipeline)
#   --domain-name       Unified Studio ドメイン名 (デフォルト: sagemaker-unified-studio)
#   --auto-approve      確認プロンプトをスキップ
#   --delete            スタックを削除
#   -h, --help          ヘルプを表示
#
# デプロイ順序:
#   1. deploy-foundation.sh (このスクリプト)
#   2. deploy-project.sh (ProjectProfile + Project)
#   3. sagemaker-ai-ml-pipeline deploy.sh
#   4. setup-integration.sh (連携)
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

# --- .env ファイルの読み込み ---
ENV_FILE="${REPO_ROOT}/.env"
if [ -f "${ENV_FILE}" ]; then
  set -a
  eval "$(grep -v '^\s*#' "${ENV_FILE}" | grep -v '^\s*$')"
  set +a
fi

# --- 引数解析 ---
REGION="${AWS_DEFAULT_REGION:-${DEFAULT_REGION}}"
PROJECT_NAME="${DEFAULT_PROJECT_NAME}"
DOMAIN_NAME="sagemaker-ai-ml-pipeline"
AUTO_APPROVE=false
DELETE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --region)
      REGION="$2"
      shift 2
      ;;
    --project-name)
      PROJECT_NAME="$2"
      shift 2
      ;;
    --domain-name)
      DOMAIN_NAME="$2"
      shift 2
      ;;
    --auto-approve)
      AUTO_APPROVE=true
      shift
      ;;
    --delete)
      DELETE=true
      shift
      ;;
    *)
      printf "${RED}不明なオプション: $1${RESET}\n"
      exit 1
      ;;
  esac
done

STACK_NAME="${PROJECT_NAME}-unified-studio-foundation-stack"
TEMPLATE_FILE="$(dirname "$0")/../cfn/foundation.yaml"

# ============================================================
# 削除モード
# ============================================================
if [[ "${DELETE}" == "true" ]]; then
  echo ""
  printf "${BOLD}${CYAN}=== Unified Studio Foundation スタックの削除 ===${RESET}\n"
  printf "${BLUE}スタック名 :${RESET} ${STACK_NAME}\n"
  printf "${BLUE}リージョン :${RESET} ${REGION}\n"
  echo ""

  confirm_or_abort "上記のスタックを削除しますか?"

  # ドメイン ID を取得
  DOMAIN_ID=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
    --query "Stacks[0].Outputs[?OutputKey=='DomainId'].OutputValue" \
    --output text \
    2>/dev/null || echo "")

  # ドメイン配下の全プロジェクトを削除 (デフォルトプロジェクト含む)
  if [[ -n "${DOMAIN_ID}" && "${DOMAIN_ID}" != "None" ]]; then
    PROJECT_IDS=$(aws datazone list-projects \
      --domain-identifier "${DOMAIN_ID}" \
      --region "${REGION}" \
      --query 'items[].id' --output text \
      2>/dev/null || echo "")

    if [[ -n "${PROJECT_IDS}" ]]; then
      printf "${BLUE}ドメイン配下のプロジェクトを削除中...${RESET}\n"
      for PID in ${PROJECT_IDS}; do
        aws datazone delete-project \
          --domain-identifier "${DOMAIN_ID}" \
          --identifier "${PID}" \
          --skip-deletion-check \
          --region "${REGION}" \
          2>/dev/null || true
        printf "  ${GREEN}✔ ${PID}${RESET}\n"
      done

      # プロジェクト削除完了を待機
      printf "${BLUE}プロジェクト削除の完了を待機中...${RESET}\n"
      while true; do
        REMAINING=$(aws datazone list-projects \
          --domain-identifier "${DOMAIN_ID}" \
          --region "${REGION}" \
          --query 'items[?projectStatus!=`DELETE_FAILED`].id' --output text \
          2>/dev/null || echo "")
        [[ -z "${REMAINING}" ]] && break
        sleep 5
      done
      printf "${GREEN}✔ 全プロジェクトの削除が完了しました${RESET}\n"
    fi
  fi

  aws cloudformation delete-stack \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
   

  printf "${BLUE}スタック削除を待機中...${RESET}\n"
  aws cloudformation wait stack-delete-complete \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
   

  printf "${BOLD}${GREEN}✔ スタックを削除しました${RESET}\n"

  # ManageAccess ロールとカスタムポリシーの削除
  # (deploy-project.sh で作成されたドメイン固有のロール)
  # Provisioning ロールはアカウント共有のため削除しない
  if [[ -n "${DOMAIN_ID}" && "${DOMAIN_ID}" != "None" ]]; then
    MANAGE_ACCESS_ROLE_NAME="AmazonSageMakerManageAccess-${REGION}-${DOMAIN_ID}"
    DOMAIN_ID_SUFFIX="${DOMAIN_ID#dzd-}"
    CUSTOM_POLICY_ARN="arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/service-role/AmazonSageMakerManageAccessPolicy-${DOMAIN_ID_SUFFIX}"

    if aws iam get-role --role-name "${MANAGE_ACCESS_ROLE_NAME}" > /dev/null 2>&1; then
      printf "${BLUE}ManageAccess ロールを削除中...${RESET}\n"
      # アタッチされたポリシーをすべてデタッチ
      for P in $(aws iam list-attached-role-policies --role-name "${MANAGE_ACCESS_ROLE_NAME}" --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null); do
        aws iam detach-role-policy --role-name "${MANAGE_ACCESS_ROLE_NAME}" --policy-arn "${P}" 2>/dev/null
      done
      aws iam delete-role --role-name "${MANAGE_ACCESS_ROLE_NAME}" 2>/dev/null
      printf "${GREEN}✔ ${MANAGE_ACCESS_ROLE_NAME} を削除しました${RESET}\n"
    fi

    # カスタムポリシーの削除
    if aws iam get-policy --policy-arn "${CUSTOM_POLICY_ARN}" > /dev/null 2>&1; then
      aws iam delete-policy --policy-arn "${CUSTOM_POLICY_ARN}" 2>/dev/null
      printf "${GREEN}✔ AmazonSageMakerManageAccessPolicy-${DOMAIN_ID_SUFFIX} を削除しました${RESET}\n"
    fi

    # Lake Formation Data Lake Admin から削除済みロールのクリーンアップ
    # (削除済みの ManageAccess ロールが残っていると、新しいドメインのプロビジョニングが失敗する)
    MANAGE_ACCESS_ROLE_ARN="arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/service-role/${MANAGE_ACCESS_ROLE_NAME}"
    CURRENT_ADMINS=$(aws lakeformation get-data-lake-settings \
      --region "${REGION}" \
      --query 'DataLakeSettings.DataLakeAdmins[].DataLakePrincipalIdentifier' \
      --output json 2>/dev/null || echo "[]")

    if echo "${CURRENT_ADMINS}" | grep -q "${MANAGE_ACCESS_ROLE_ARN}"; then
      printf "${BLUE}Lake Formation Data Lake Admin から削除中...${RESET}\n"
      REMAINING_ADMINS=$(echo "${CURRENT_ADMINS}" | python3 -c "
import json,sys
admins=[a for a in json.load(sys.stdin) if a != '${MANAGE_ACCESS_ROLE_ARN}']
print(json.dumps({'DataLakeAdmins':[{'DataLakePrincipalIdentifier':a} for a in admins]}))")
      aws lakeformation put-data-lake-settings \
        --region "${REGION}" \
        --data-lake-settings "${REMAINING_ADMINS}" \
        > /dev/null
      printf "${GREEN}✔ Lake Formation から ${MANAGE_ACCESS_ROLE_NAME} を削除しました${RESET}\n"
    fi
  fi

  exit 0
fi

# ============================================================
# デプロイモード
# ============================================================
echo ""
printf "${BOLD}${CYAN}=== Unified Studio Foundation デプロイ ===${RESET}\n"
printf "${BLUE}スタック名           :${RESET} ${STACK_NAME}\n"
printf "${BLUE}リージョン           :${RESET} ${REGION}\n"
printf "${BLUE}プロジェクト名       :${RESET} ${PROJECT_NAME}\n"
printf "${BLUE}ドメイン名           :${RESET} ${DOMAIN_NAME}\n"
echo ""

confirm_or_abort "上記の内容でデプロイしますか?"

printf "\n${BOLD}${CYAN}=== CloudFormation スタックのデプロイ ===${RESET}\n"

IDC_INSTANCE_ARN="${UNIFIED_STUDIO_IDC_INSTANCE_ARN:-}"

aws cloudformation deploy \
  --template-file "${TEMPLATE_FILE}" \
  --stack-name "${STACK_NAME}" \
  --parameter-overrides \
    "ProjectName=${PROJECT_NAME}" \
    "DomainName=${DOMAIN_NAME}" \
    "IdcInstanceArn=${IDC_INSTANCE_ARN}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "${REGION}" \
 

printf "${GREEN}✔ スタック '${STACK_NAME}' のデプロイが完了しました${RESET}\n"

# --- 出力の表示 ---
printf "\n${BOLD}${CYAN}=== スタック出力 ===${RESET}\n"

OUTPUTS=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --query 'Stacks[0].Outputs' \
  --output table \
 )

echo "${OUTPUTS}"

# Domain ID を取得して表示
DOMAIN_ID=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='DomainId'].OutputValue" \
  --output text \
 )

# --- SSO ユーザーをドメイン管理者として登録 ---
SSO_USERS="${UNIFIED_STUDIO_SSO_USERS:-}"
if [[ -n "${SSO_USERS}" && -n "${IDC_INSTANCE_ARN}" ]]; then
  printf "\n${BOLD}${CYAN}=== SSO ユーザーのドメイン管理者登録 ===${RESET}\n"

  IFS=',' read -ra SSO_USER_ARRAY <<< "${SSO_USERS}"
  for SSO_USER in "${SSO_USER_ARRAY[@]}"; do
    SSO_USER=$(echo "${SSO_USER}" | xargs)  # trim spaces
    [[ -z "${SSO_USER}" ]] && continue

    SSO_PROFILE_ID=$(aws datazone search-user-profiles \
      --domain-identifier "${DOMAIN_ID}" \
      --user-type DATAZONE_SSO_USER \
      --region "${REGION}" \
      --no-paginate \
      --query "items[?details.sso.username=='${SSO_USER}'].id | [0]" \
      --output text 2>/dev/null || echo "")

    if [[ -n "${SSO_PROFILE_ID}" && "${SSO_PROFILE_ID}" != "None" ]]; then
      printf "${GREEN}✔ SSO ユーザー '${SSO_USER}' をドメイン管理者として確認しました${RESET}\n"
    else
      printf "${YELLOW}⚠ SSO ユーザー '${SSO_USER}' の DataZone プロファイルが見つかりませんでした${RESET}\n"
      printf "${YELLOW}  Unified Studio ポータルに一度 SSO でログインしてプロファイルを作成してください${RESET}\n"
    fi
  done

  printf "${YELLOW}  ポータル URL: https://${DOMAIN_ID}.sagemaker.${REGION}.on.aws${RESET}\n"
fi

echo ""
printf "${BOLD}${GREEN}=== デプロイ完了 ===${RESET}\n"
printf "次のステップとして、以下のコマンドでプロジェクトを作成できます。\n"
echo ""
printf "  ${BOLD}./infra/unified-studio/scripts/deploy-project.sh --domain-id ${DOMAIN_ID}${RESET}\n"
echo ""
