#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail

# ============================================================
# Unified Studio Project デプロイスクリプト
# ============================================================
#
# 使い方:
#   ./deploy-project.sh --domain-id <ID> [オプション]
#
# 必須引数:
#   --domain-id         Unified Studio (DataZone) ドメイン ID
#
# オプション:
#   --region            AWS リージョン (デフォルト: us-east-1)
#   --project-name      リソース命名プレフィックス (デフォルト: sagemaker-ai-ml-pipeline)
#   --us-project-name   Unified Studio プロジェクト名 (デフォルト: ml-pipeline)
#   --auto-approve      確認プロンプトをスキップ
#   --delete            スタックを削除
#   -h, --help          ヘルプを表示
#
# 処理内容:
#   1. ドメインから managed ブループリント ID を取得 (Tooling, MLflowApp)
#   2. Provisioning / ManageAccess ロール ARN を取得
#   3. ブループリントの Authorization ポリシーを設定
#   4. CloudFormation スタックのデプロイ
#      (EnvironmentBlueprintConfiguration + ProjectProfile + Project)
#
# 前提条件:
#   - foundation スタック (Domain) がデプロイ済みであること
#
# デプロイ順序:
#   1. deploy-foundation.sh (Domain + IAM Role)
#   2. deploy-project.sh (このスクリプト)
#   3. sagemaker-ai-ml-pipeline deploy.sh (ML パイプライン)
#   4. setup-integration.sh (連携)
# ============================================================

source "$(dirname "$0")/../../_common.sh"

# --- .env ファイルの読み込み ---
ENV_FILE="${REPO_ROOT}/.env"
if [ -f "${ENV_FILE}" ]; then
  set -a
  eval "$(grep -v '^\s*#' "${ENV_FILE}" | grep -v '^\s*$')"
  set +a
fi

# --- ヘルプ ---
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  awk '/^# ====/{n++; next} n>=1 && n<=2{sub(/^# ?/,""); print}' "$0"
  exit 0
fi

# --- 引数解析 ---
DOMAIN_ID=""
REGION="${AWS_DEFAULT_REGION:-${DEFAULT_REGION}}"
PROJECT_NAME="${DEFAULT_PROJECT_NAME}"
US_PROJECT_NAME="ml-pipeline"
AUTO_APPROVE=false
DELETE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --domain-id)
      DOMAIN_ID="$2"
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
    --us-project-name)
      US_PROJECT_NAME="$2"
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

STACK_NAME="${PROJECT_NAME}-unified-studio-project-stack"
TEMPLATE_FILE="$(dirname "$0")/../cfn/project.yaml"

# --- 必須引数チェック ---
if [[ -z "${DOMAIN_ID}" ]]; then
  printf "${RED}エラー: --domain-id は必須です${RESET}\n"
  exit 1
fi

# ============================================================
# 削除モード
# ============================================================
if [[ "${DELETE}" == "true" ]]; then
  echo ""
  printf "${BOLD}${CYAN}=== Unified Studio Project スタックの削除 ===${RESET}\n"
  printf "${BLUE}スタック名 :${RESET} ${STACK_NAME}\n"
  printf "${BLUE}リージョン :${RESET} ${REGION}\n"
  echo ""

  confirm_or_abort "上記のスタックを削除しますか?"

  aws cloudformation delete-stack \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
   

  printf "${BLUE}スタック削除を待機中...${RESET}\n"
  aws cloudformation wait stack-delete-complete \
    --stack-name "${STACK_NAME}" \
    --region "${REGION}" \
   

  printf "${BOLD}${GREEN}✔ スタックを削除しました${RESET}\n"
  exit 0
fi

# ============================================================
# デプロイモード
# ============================================================

# --- Step 1/5: ブループリント ID の取得 ---
printf "\n${BOLD}${CYAN}=== Step 1/5: ブループリント ID の取得 ===${RESET}\n"

TOOLING_BP_ID=$(aws datazone list-environment-blueprints \
  --domain-identifier "${DOMAIN_ID}" \
  --managed \
  --region "${REGION}" \
  --output json | jq -r '[.items[] | select(.name=="Tooling")] | .[0].id // empty')

MLFLOW_BP_ID=$(aws datazone list-environment-blueprints \
  --domain-identifier "${DOMAIN_ID}" \
  --managed \
  --region "${REGION}" \
  --output json | jq -r '[.items[] | select(.name=="MLflowApp")] | .[0].id // empty')

LAKEHOUSE_BP_ID=$(aws datazone list-environment-blueprints \
  --domain-identifier "${DOMAIN_ID}" \
  --managed \
  --region "${REGION}" \
  --output json | jq -r '[.items[] | select(.name=="LakehouseCatalog")] | .[0].id // empty')

if [[ -z "${TOOLING_BP_ID}" || "${TOOLING_BP_ID}" == "None" ]]; then
  printf "${RED}エラー: Tooling ブループリントが見つかりません${RESET}\n"
  exit 1
fi
if [[ -z "${MLFLOW_BP_ID}" || "${MLFLOW_BP_ID}" == "None" ]]; then
  printf "${RED}エラー: MLflowApp ブループリントが見つかりません${RESET}\n"
  exit 1
fi
if [[ -z "${LAKEHOUSE_BP_ID}" || "${LAKEHOUSE_BP_ID}" == "None" ]]; then
  printf "${RED}エラー: LakehouseCatalog ブループリントが見つかりません${RESET}\n"
  exit 1
fi

printf "${GREEN}✔ Tooling          : ${TOOLING_BP_ID}${RESET}\n"
printf "${GREEN}✔ MLflowApp        : ${MLFLOW_BP_ID}${RESET}\n"
printf "${GREEN}✔ LakehouseCatalog : ${LAKEHOUSE_BP_ID}${RESET}\n"

# --- Step 2/5: ブループリント設定の有効化 (API) ---
printf "\n${BOLD}${CYAN}=== Step 2/5: ブループリント設定の有効化 ===${RESET}\n"

# Provisioning / ManageAccess ロール ARN を構成
# これらのロールは管理コンソールの Configure で作成されるが、
# 存在しない場合はスクリプトで自動作成する
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
PROVISIONING_ROLE_NAME="AmazonSageMakerProvisioning-${ACCOUNT_ID}"
PROVISIONING_ROLE="arn:aws:iam::${ACCOUNT_ID}:role/service-role/${PROVISIONING_ROLE_NAME}"
MANAGE_ACCESS_ROLE_NAME="AmazonSageMakerManageAccess-${REGION}-${DOMAIN_ID}"
MANAGE_ACCESS_ROLE="arn:aws:iam::${ACCOUNT_ID}:role/service-role/${MANAGE_ACCESS_ROLE_NAME}"

# ドメイン ARN (ManageAccess ロールの信頼ポリシーで使用)
DOMAIN_ARN="arn:aws:datazone:${REGION}:${ACCOUNT_ID}:domain/${DOMAIN_ID}"

# Provisioning ロール: なければ作成
if aws iam get-role --role-name "${PROVISIONING_ROLE_NAME}" > /dev/null 2>&1; then
  printf "${GREEN}✔ Provisioning   : ${PROVISIONING_ROLE} (既存)${RESET}\n"
else
  printf "${BLUE}Provisioning ロールを作成中...${RESET}\n"
  aws iam create-role \
    --role-name "${PROVISIONING_ROLE_NAME}" \
    --path "/service-role/" \
    --assume-role-policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"datazone.amazonaws.com\"},\"Action\":\"sts:AssumeRole\",\"Condition\":{\"StringEquals\":{\"aws:SourceAccount\":\"${ACCOUNT_ID}\"}}}]}" \
    > /dev/null
  aws iam attach-role-policy \
    --role-name "${PROVISIONING_ROLE_NAME}" \
    --policy-arn "arn:aws:iam::aws:policy/service-role/SageMakerStudioProjectProvisioningRolePolicy" \
   
  printf "${GREEN}✔ Provisioning   : ${PROVISIONING_ROLE} (作成)${RESET}\n"
fi

# ManageAccess ロール: なければ作成
if aws iam get-role --role-name "${MANAGE_ACCESS_ROLE_NAME}" > /dev/null 2>&1; then
  printf "${GREEN}✔ ManageAccess   : ${MANAGE_ACCESS_ROLE} (既存)${RESET}\n"
else
  printf "${BLUE}ManageAccess ロールを作成中...${RESET}\n"
  aws iam create-role \
    --role-name "${MANAGE_ACCESS_ROLE_NAME}" \
    --path "/service-role/" \
    --assume-role-policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"datazone.amazonaws.com\"},\"Action\":\"sts:AssumeRole\",\"Condition\":{\"StringEquals\":{\"aws:SourceAccount\":\"${ACCOUNT_ID}\"},\"ArnEquals\":{\"aws:SourceArn\":\"${DOMAIN_ARN}\"}}}]}" \
    > /dev/null

  # AWS マネージドポリシーをアタッチ
  for POLICY_ARN in \
    "arn:aws:iam::aws:policy/service-role/AmazonDataZoneRedshiftManageAccessRolePolicy" \
    "arn:aws:iam::aws:policy/service-role/AmazonDataZoneGlueManageAccessRolePolicy" \
    "arn:aws:iam::aws:policy/AmazonDataZoneSageMakerManageAccessRolePolicy"; do
    aws iam attach-role-policy \
      --role-name "${MANAGE_ACCESS_ROLE_NAME}" \
      --policy-arn "${POLICY_ARN}" \
     
  done

  # ドメイン固有のカスタムポリシー (Redshift Secret アクセス)
  DOMAIN_ID_SUFFIX="${DOMAIN_ID#dzd-}"
  CUSTOM_POLICY_NAME="AmazonSageMakerManageAccessPolicy-${DOMAIN_ID_SUFFIX}"
  aws iam create-policy \
    --policy-name "${CUSTOM_POLICY_NAME}" \
    --path "/service-role/" \
    --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Sid\":\"RedshiftSecretStatement\",\"Effect\":\"Allow\",\"Action\":\"secretsmanager:GetSecretValue\",\"Resource\":\"*\",\"Condition\":{\"StringEquals\":{\"secretsmanager:ResourceTag/AmazonDataZoneDomain\":\"${DOMAIN_ID}\"}}}]}" \
    > /dev/null
  aws iam attach-role-policy \
    --role-name "${MANAGE_ACCESS_ROLE_NAME}" \
    --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/service-role/${CUSTOM_POLICY_NAME}" \
   

  printf "${GREEN}✔ ManageAccess   : ${MANAGE_ACCESS_ROLE} (作成)${RESET}\n"
fi

# ブループリント ID → 名前のマッピング (表示用)
bp_name() {
  case "$1" in
    "${TOOLING_BP_ID}") echo "Tooling" ;;
    "${MLFLOW_BP_ID}") echo "MLflowApp" ;;
    "${LAKEHOUSE_BP_ID}") echo "LakehouseCatalog" ;;
    *) echo "$1" ;;
  esac
}

# CFn の EnvironmentBlueprintConfiguration は V2 ドメインの managed ブループリントを
# サポートしていないため、API で直接設定する。
# ただし、管理コンソールの Configure で既に設定済みの場合は上書きしない
# (PUT API は regionalParameters 等を消してしまうため)

# デフォルト VPC の情報を取得 (regionalParameters に必要)
DEFAULT_VPC_ID=$(aws ec2 describe-vpcs --region "${REGION}" \
  --filters "Name=isDefault,Values=true" \
  --query 'Vpcs[0].VpcId' --output text)

if [[ -z "${DEFAULT_VPC_ID}" || "${DEFAULT_VPC_ID}" == "None" ]]; then
  printf "${RED}エラー: デフォルト VPC が見つかりません${RESET}\n"
  exit 1
fi

DEFAULT_SUBNETS=$(aws ec2 describe-subnets --region "${REGION}" \
  --filters "Name=vpc-id,Values=${DEFAULT_VPC_ID}" \
  --query 'Subnets[].SubnetId' --output text | tr '\t' ',')
DEFAULT_AZS=$(aws ec2 describe-subnets --region "${REGION}" \
  --filters "Name=vpc-id,Values=${DEFAULT_VPC_ID}" \
  --query 'Subnets[].AvailabilityZone' --output text | tr '\t' '\n' | sort -u | head -2 | tr '\n' ',' | sed 's/,$//')
# Unified Studio の Tooling 環境が使用する S3 バケットを検出。
# ドメイン作成時に AWS が自動生成する amazon-sagemaker-{account}-{region}-{hash} を探す。
# 見つからなければ新規作成する。
S3_BUCKET=$(aws s3api list-buckets \
  --query "Buckets[?starts_with(Name,'amazon-sagemaker-${ACCOUNT_ID}-${REGION}-')].Name | [0]" \
  --output text)
if [[ -z "${S3_BUCKET}" || "${S3_BUCKET}" == "None" ]]; then
  S3_HASH=$(openssl rand -hex 6)
  S3_BUCKET="amazon-sagemaker-${ACCOUNT_ID}-${REGION}-${S3_HASH}"
  printf "${BLUE}S3 バケットを作成中: ${S3_BUCKET}${RESET}\n"
  # us-east-1 は LocationConstraint を指定するとエラーになる (AWS S3 API の仕様)
  if [[ "${REGION}" == "us-east-1" ]]; then
    aws s3api create-bucket --bucket "${S3_BUCKET}" --region "${REGION}"
  else
    aws s3api create-bucket --bucket "${S3_BUCKET}" --region "${REGION}" \
      --create-bucket-configuration LocationConstraint="${REGION}"
  fi
fi
S3_LOCATION="s3://${S3_BUCKET}"
REGIONAL_PARAMS="{\"${REGION}\":{\"VpcId\":\"${DEFAULT_VPC_ID}\",\"Subnets\":\"${DEFAULT_SUBNETS}\",\"AZs\":\"${DEFAULT_AZS}\",\"S3Location\":\"${S3_LOCATION}\"}}"

printf "${GREEN}✔ VPC: ${DEFAULT_VPC_ID} (Subnets: ${DEFAULT_AZS})${RESET}\n"

for BP_ID in "${TOOLING_BP_ID}" "${MLFLOW_BP_ID}" "${LAKEHOUSE_BP_ID}"; do
  BP_LABEL="$(bp_name "${BP_ID}") (${BP_ID})"
  aws datazone put-environment-blueprint-configuration \
    --domain-identifier "${DOMAIN_ID}" \
    --environment-blueprint-identifier "${BP_ID}" \
    --enabled-regions "${REGION}" \
    --provisioning-role-arn "${PROVISIONING_ROLE}" \
    --manage-access-role-arn "${MANAGE_ACCESS_ROLE}" \
    --regional-parameters "${REGIONAL_PARAMS}" \
    --region "${REGION}" \
    > /dev/null
  printf "${GREEN}✔ ${BP_LABEL}: 設定しました${RESET}\n"
done

# --- Step 3/5: ブループリントの Authorization ポリシー設定 ---
printf "\n${BOLD}${CYAN}=== Step 3/5: Authorization ポリシーの設定 ===${RESET}\n"

# ルートドメインユニット ID を取得
ROOT_DOMAIN_UNIT_ID=$(aws datazone get-domain \
  --identifier "${DOMAIN_ID}" \
  --region "${REGION}" \
  --query 'rootDomainUnitId' \
  --output text \
 )

if [[ -z "${ROOT_DOMAIN_UNIT_ID}" || "${ROOT_DOMAIN_UNIT_ID}" == "None" ]]; then
  printf "${RED}エラー: ルートドメインユニット ID を取得できませんでした${RESET}\n"
  exit 1
fi

printf "${GREEN}✔ ルートドメインユニット: ${ROOT_DOMAIN_UNIT_ID}${RESET}\n"

# entityIdentifier のフォーマット: {AWSアカウントID}:{environmentBlueprintId}
# (API ドキュメントに未記載、CloudTrail の AddPolicyGrant イベントから確認)
add_blueprint_policy() {
  local bp_id="$1"
  local policy_type="$2"
  local detail="$3"
  local include_children="$4"
  local entity_id="${ACCOUNT_ID}:${bp_id}"

  # 既存ポリシーを確認
  local existing
  existing=$(aws datazone list-policy-grants \
    --domain-identifier "${DOMAIN_ID}" \
    --entity-type ENVIRONMENT_BLUEPRINT_CONFIGURATION \
    --entity-identifier "${entity_id}" \
    --policy-type "${policy_type}" \
    --region "${REGION}" \
    --query 'grantList[0].grantId' \
    --output text \
    2>/dev/null || echo "")

  if [[ -n "${existing}" && "${existing}" != "None" ]]; then
    printf "  ${GREEN}✔ ${policy_type}: 設定済み (スキップ)${RESET}\n"
    return
  fi

  aws datazone add-policy-grant \
    --domain-identifier "${DOMAIN_ID}" \
    --entity-type ENVIRONMENT_BLUEPRINT_CONFIGURATION \
    --entity-identifier "${entity_id}" \
    --policy-type "${policy_type}" \
    --principal "{\"project\":{\"projectDesignation\":\"CONTRIBUTOR\",\"projectGrantFilter\":{\"domainUnitFilter\":{\"domainUnit\":\"${ROOT_DOMAIN_UNIT_ID}\",\"includeChildDomainUnits\":${include_children}}}}}" \
    --detail "${detail}" \
    --region "${REGION}" \
    > /dev/null

  printf "  ${GREEN}✔ ${policy_type}: 設定しました${RESET}\n"
}

for BP_ID in "${TOOLING_BP_ID}" "${MLFLOW_BP_ID}" "${LAKEHOUSE_BP_ID}"; do
  printf "${BLUE}$(bp_name "${BP_ID}") (${BP_ID}):${RESET}\n"
  add_blueprint_policy "${BP_ID}" "CREATE_ENVIRONMENT_FROM_BLUEPRINT" \
    '{"createEnvironmentFromBlueprint":{}}' "true"
  add_blueprint_policy "${BP_ID}" "CREATE_ENVIRONMENT_PROFILE" \
    "{\"createEnvironmentProfile\":{\"domainUnitId\":\"${ROOT_DOMAIN_UNIT_ID}\"}}" "false"
done

# --- デプロイ情報の表示 ---
echo ""
printf "${BOLD}${CYAN}=== Unified Studio Project デプロイ ===${RESET}\n"
printf "${BLUE}スタック名           :${RESET} ${STACK_NAME}\n"
printf "${BLUE}リージョン           :${RESET} ${REGION}\n"
printf "${BLUE}ドメイン ID          :${RESET} ${DOMAIN_ID}\n"
printf "${BLUE}プロジェクト名       :${RESET} ${PROJECT_NAME}\n"
printf "${BLUE}Unified Studio プロジェクト名 :${RESET} ${US_PROJECT_NAME}\n"
printf "${BLUE}Tooling Blueprint         :${RESET} ${TOOLING_BP_ID}\n"
printf "${BLUE}MLflowApp Blueprint       :${RESET} ${MLFLOW_BP_ID}\n"
printf "${BLUE}LakehouseCatalog Blueprint:${RESET} ${LAKEHOUSE_BP_ID}\n"
echo ""

confirm_or_abort "上記の内容でデプロイしますか?"

# --- Step 4/5: CloudFormation スタックのデプロイ ---
printf "\n${BOLD}${CYAN}=== Step 4/5: CloudFormation スタックのデプロイ ===${RESET}\n"

aws cloudformation deploy \
  --template-file "${TEMPLATE_FILE}" \
  --stack-name "${STACK_NAME}" \
  --parameter-overrides \
    "ProjectName=${PROJECT_NAME}" \
    "DomainId=${DOMAIN_ID}" \
    "ToolingBlueprintId=${TOOLING_BP_ID}" \
    "MLflowAppBlueprintId=${MLFLOW_BP_ID}" \
    "LakehouseCatalogBlueprintId=${LAKEHOUSE_BP_ID}" \
    "UnifiedStudioProjectName=${US_PROJECT_NAME}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "${REGION}" \
 

printf "${GREEN}✔ スタック '${STACK_NAME}' のデプロイが完了しました${RESET}\n"

# --- 出力の表示 ---
printf "\n${BOLD}${CYAN}=== スタック出力 ===${RESET}\n"

aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --query 'Stacks[0].Outputs' \
  --output table \
 

PROJECT_ID=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --query "Stacks[0].Outputs[?OutputKey=='ProjectId'].OutputValue" \
  --output text \
 )

# --- Step 5/5: プロジェクトメンバーの追加 ---
printf "\n${BOLD}${CYAN}=== Step 5/5: プロジェクトメンバーの追加 ===${RESET}\n"

# メンバー追加の共通関数
add_project_member() {
  local user_id="$1"
  local label="$2"

  # 既にメンバーかどうか確認
  local existing
  existing=$(aws datazone list-project-memberships \
    --domain-identifier "${DOMAIN_ID}" \
    --project-identifier "${PROJECT_ID}" \
    --region "${REGION}" \
    --no-paginate \
    --query "members[?memberDetails.user.userId=='${user_id}'].designation | [0]" \
    --output text 2>/dev/null || echo "")

  if [[ -n "${existing}" && "${existing}" != "None" ]]; then
    printf "${GREEN}✔ ${label}: 既にメンバーです (${existing})${RESET}\n"
  else
    aws datazone create-project-membership \
      --domain-identifier "${DOMAIN_ID}" \
      --project-identifier "${PROJECT_ID}" \
      --member "{\"userIdentifier\":\"${user_id}\"}" \
      --designation PROJECT_OWNER \
      --region "${REGION}" \
      > /dev/null
    printf "${GREEN}✔ ${label}: プロジェクトオーナーとして追加しました${RESET}\n"
  fi
}

# 現在の IAM プリンシパルを追加
MEMBER_ADDED=false
CALLER_ARN=$(aws sts get-caller-identity --query Arn --output text)
IAM_PROFILE_ID=$(aws datazone search-user-profiles \
  --domain-identifier "${DOMAIN_ID}" \
  --user-type DATAZONE_IAM_USER \
  --region "${REGION}" \
  --no-paginate \
  --query "items[?details.iam.arn=='${CALLER_ARN}'].id | [0]" \
  --output text 2>/dev/null || echo "")

if [[ -n "${IAM_PROFILE_ID}" && "${IAM_PROFILE_ID}" != "None" ]]; then
  add_project_member "${IAM_PROFILE_ID}" "IAM (${CALLER_ARN})"
  MEMBER_ADDED=true
fi

# .env の UNIFIED_STUDIO_SSO_USERS で指定された SSO ユーザーを追加 (カンマ区切り)
SSO_USERS="${UNIFIED_STUDIO_SSO_USERS:-}"
if [[ -n "${SSO_USERS}" ]]; then
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
      add_project_member "${SSO_PROFILE_ID}" "SSO (${SSO_USER})"
      MEMBER_ADDED=true
    else
      printf "${YELLOW}⚠ SSO ユーザー '${SSO_USER}' の DataZone プロファイルが見つかりませんでした${RESET}\n"
      printf "${YELLOW}  Unified Studio に一度 SSO でログインした後、以下のコマンドで追加してください。${RESET}\n"
      echo ""
      printf "  ${BOLD}SSO_PROFILE_ID=\$(aws datazone search-user-profiles \\\\${RESET}\n"
      printf "    --domain-identifier ${DOMAIN_ID} \\\\\\n"
      printf "    --user-type DATAZONE_SSO_USER --region ${REGION} \\\\\\n"
      printf "    --query \"items[?details.sso.username=='${SSO_USER}'].id | [0]\" --output text)\n"
      echo ""
      printf "  ${BOLD}aws datazone create-project-membership \\\\${RESET}\n"
      printf "    --domain-identifier ${DOMAIN_ID} \\\\\\n"
      printf "    --project-identifier ${PROJECT_ID} \\\\\\n"
      printf "    --member \"{\\\\\"userIdentifier\\\\\":\\\\\"\${SSO_PROFILE_ID}\\\\\"}\" \\\\\\n"
      printf "    --designation PROJECT_OWNER --region ${REGION}\n"
    fi
  done
fi

if [[ "${MEMBER_ADDED}" == "false" ]]; then
  printf "${YELLOW}⚠ プロジェクトメンバーを追加できませんでした${RESET}\n"
  printf "${YELLOW}  Unified Studio にログイン後、このスクリプトを再実行するとメンバーが追加されます。${RESET}\n"
  echo ""
  printf "  ${BOLD}./infra/unified-studio/scripts/deploy-project.sh --domain-id ${DOMAIN_ID}${RESET}\n"
fi

echo ""
printf "${BOLD}${GREEN}=== デプロイ完了 ===${RESET}\n"
printf "次のステップとして、以下のコマンドで Unified Studio 連携をセットアップできます。\n"
echo ""
printf "  ${BOLD}./infra/unified-studio/scripts/setup-integration.sh \\\\${RESET}\n"
printf "    --domain-id ${DOMAIN_ID} \\\\\n"
printf "    --project-id ${PROJECT_ID}\n"
echo ""
