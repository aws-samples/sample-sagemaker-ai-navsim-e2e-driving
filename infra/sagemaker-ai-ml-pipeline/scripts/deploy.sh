#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail

# ============================================================
# SageMaker AI ML Pipeline - CloudFormation デプロイスクリプト
# ============================================================
#
# 使い方:
#   ./deploy.sh [オプション] [スタック名] [プロジェクト名]
#
# オプション:
#   -h, --help          ヘルプを表示
#   --auto-approve      確認プロンプトをスキップして実行
#
# 引数:
#   $1 - CloudFormation スタック名 (デフォルト: sagemaker-ai-ml-pipeline-stack)
#   $2 - プロジェクト名。リソースの命名プレフィックスに使用 (デフォルト: sagemaker-ai-ml-pipeline)
#
# 環境変数 (.env ファイルまたはシェル環境変数で設定):
#   AWS_DEFAULT_REGION          - デプロイ先リージョン (デフォルト: us-east-1)
#   NOTEBOOK_IDLE_TIMEOUT_MIN   - アイドル自動停止までの分数 (デフォルト: 60)
#   GITHUB_REPO                 - Notebook に連携する GitHub リポジトリ URL (省略可)
#   GITHUB_PAT                  - GitHub Personal Access Token (プライベートリポジトリの場合)
#
# 処理内容:
#   1. .env ファイルの読み込み
#   2. GitHub リポジトリの確認・作成 (GITHUB_REPO_URL が設定されている場合)
#   3. リポジトリ全体の push
#   4. CloudFormation スタックのデプロイ (S3, ECR, IAM, Notebook, MLflow 等)
#   5. スタック出力の表示
#   6. JupyterLab / MLflow UI の presigned URL 生成・表示
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

# --- .env ファイルの読み込み ---
ENV_FILE="${REPO_ROOT}/.env"
if [ -f "${ENV_FILE}" ]; then
  echo "Loading .env file..."
  # コメント行と空行を除外し、export して読み込む
  set -a
  # shellcheck disable=SC1090
  eval "$(grep -v '^\s*#' "${ENV_FILE}" | grep -v '^\s*$')"
  set +a
fi

# --- パラメータ ---
STACK_NAME="${POSITIONAL_ARGS[0]:-${DEFAULT_STACK_NAME}}"
PROJECT_NAME="${POSITIONAL_ARGS[1]:-${DEFAULT_PROJECT_NAME}}"
REGION="${AWS_DEFAULT_REGION:-${DEFAULT_REGION}}"
GITHUB_REPO="${GITHUB_REPO:-}"
GITHUB_PAT="${GITHUB_PAT:-}"
IDLE_TIMEOUT_MIN="${NOTEBOOK_IDLE_TIMEOUT_MIN:-60}"

# --- VPC 設定 ---
ENABLE_VPC="${ENABLE_VPC:-false}"
VPC_ID="${VPC_ID:-}"
SUBNET_IDS="${SUBNET_IDS:-}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-}"
CREATE_VPC_ENDPOINTS="${CREATE_VPC_ENDPOINTS:-true}"

# --- デプロイ情報の表示 ---
echo ""
printf "${BOLD}${CYAN}=== SageMaker AI ML Pipeline のデプロイ ===${RESET}\n"
printf "${BLUE}スタック名     :${RESET} ${STACK_NAME}\n"
printf "${BLUE}プロジェクト名 :${RESET} ${PROJECT_NAME}\n"
printf "${BLUE}リージョン     :${RESET} ${REGION}\n"
if [ -n "${GITHUB_REPO}" ]; then
  # URL からオーナー/リポジトリ名を抽出
  # https://github.com/owner/repo.git → owner/repo
  # https://github.example.com/owner/repo.git → owner/repo
  REPO_FULL=$(echo "${GITHUB_REPO}" | sed -E 's|https?://[^/]+/||; s|\.git$||; s|/$||')
  REPO_OWNER=$(echo "${REPO_FULL}" | cut -d'/' -f1)
  REPO_NAME=$(echo "${REPO_FULL}" | cut -d'/' -f2)

  # .git 拡張子を除去してブラウザ用 URL を生成
  GITHUB_BROWSER_URL="${GITHUB_REPO%.git}"
  printf "${BLUE}GitHub リポジトリ :${RESET} "
  printf '\e]8;;%s\e\\' "${GITHUB_BROWSER_URL}"
  printf '%s' "${GITHUB_REPO}"
  printf '\e]8;;\e\\\n'
  printf "${BLUE}GitHub ユーザー   :${RESET} ${REPO_OWNER}\n"
  if [ -n "${GITHUB_PAT}" ]; then
    printf "${BLUE}GitHub PAT        :${RESET} *****(設定済み)\n"
  fi
fi
printf "${BLUE}Notebook アイドル停止 :${RESET} ${IDLE_TIMEOUT_MIN} 分\n"
if [[ "${ENABLE_VPC}" == "true" ]]; then
  printf "${BLUE}VPC 構成             :${RESET} 有効\n"
  if [ -n "${VPC_ID}" ]; then
    printf "${BLUE}  VPC ID             :${RESET} ${VPC_ID}\n"
    printf "${BLUE}  Subnet IDs         :${RESET} ${SUBNET_IDS}\n"
    printf "${BLUE}  Security Group ID  :${RESET} ${SECURITY_GROUP_ID}\n"
  else
    # 既存スタックに VPC がある場合は表示
    EXISTING_VPC_SUBNETS=$(aws cloudformation describe-stacks \
      --stack-name "${STACK_NAME}" --region "${REGION}" \
      --query 'Stacks[0].Outputs[?OutputKey==`VpcSubnetIds`].OutputValue' \
      --output text 2>/dev/null) || true
    if [[ -n "${EXISTING_VPC_SUBNETS}" && "${EXISTING_VPC_SUBNETS}" != "None" ]]; then
      printf "${BLUE}  VPC                :${RESET} CFn 管理 (作成済み)\n"
    else
      printf "${BLUE}  VPC                :${RESET} 新規作成\n"
    fi
  fi
  printf "${BLUE}  VPC Endpoints      :${RESET} ${CREATE_VPC_ENDPOINTS}\n"
fi
echo ""

confirm_or_abort "上記の内容でデプロイしますか?"

# --- GitHub リポジトリの確認・作成 ---
if [ -n "${GITHUB_REPO}" ]; then
  if ! command -v gh &>/dev/null; then
    printf "${YELLOW}⚠ gh (GitHub CLI) がインストールされていません。リポジトリの自動作成をスキップします。${RESET}\n"
    printf "${YELLOW}  インストール: https://cli.github.com/${RESET}\n"
    echo ""
  else
    printf "${BOLD}${CYAN}=== GitHub リポジトリの確認 ===${RESET}\n"

    if gh repo view "${REPO_FULL}" &>/dev/null; then
      printf "${GREEN}✔ リポジトリ '${REPO_FULL}' は既に存在します${RESET}\n"
    else
      printf "${YELLOW}リポジトリ '${REPO_FULL}' が見つかりません。作成します...${RESET}\n"
      gh repo create "${REPO_FULL}" --private --description "SageMaker AI ML Pipeline"
      printf "${GREEN}✔ リポジトリ '${REPO_FULL}' を作成しました${RESET}\n"
    fi
    echo ""

    # リポジトリ全体の push
    printf "${BOLD}${CYAN}=== リポジトリの push ===${RESET}\n"

    TEMP_DIR=$(mktemp -d)
    trap 'rm -rf "${TEMP_DIR}"' EXIT

    git clone "https://${REPO_OWNER}:${GITHUB_PAT}@github.com/${REPO_FULL}.git" "${TEMP_DIR}/repo" 2>/dev/null || \
    git clone "${GITHUB_REPO}" "${TEMP_DIR}/repo" 2>/dev/null

    # .git, .env, .kiro を除外してリポジトリ全体をコピー
    rsync -a --delete --exclude='.git' --exclude='.env' --exclude='.kiro' "${REPO_ROOT}/" "${TEMP_DIR}/repo/"

    cd "${TEMP_DIR}/repo"
    git add -A
    if git diff --cached --quiet; then
      printf "${GREEN}✔ リポジトリに変更はありません${RESET}\n"
    else
      git commit -m "Sync repository contents"
      git push origin "$(git branch --show-current)"
      printf "${GREEN}✔ リポジトリを push しました${RESET}\n"
    fi
    cd - > /dev/null

    # trap で TEMP_DIR は自動削除される
    echo ""
  fi
fi

# --- CloudFormation デプロイ ---
TEMPLATE_FILE="$(dirname "$0")/../cfn/sagemaker-ai-ml-pipeline.yaml"

CFN_PARAMS=(
  "ProjectName=${PROJECT_NAME}"
  "NotebookIdleTimeoutMinutes=${IDLE_TIMEOUT_MIN}"
  "EnableVPC=${ENABLE_VPC}"
  "CreateVpcEndpoints=${CREATE_VPC_ENDPOINTS}"
)
if [ -n "${VPC_ID}" ]; then
  CFN_PARAMS+=("VpcId=${VPC_ID}")
  CFN_PARAMS+=("SubnetIds=${SUBNET_IDS}")
  CFN_PARAMS+=("SecurityGroupId=${SECURITY_GROUP_ID}")
fi
if [ -n "${GITHUB_REPO}" ]; then
  CFN_PARAMS+=("GitHubRepoUrl=${GITHUB_REPO}")
fi
if [ -n "${REPO_OWNER:-}" ] && [ -n "${GITHUB_PAT}" ]; then
  CFN_PARAMS+=("GitHubUsername=${REPO_OWNER}")
  # GitHubPAT is NOT passed to CloudFormation to avoid exposure in event history.
  # Secret is created/updated here BEFORE stack deployment because SageMaker
  # CodeRepository validates GitHub credentials at Notebook creation time.
fi

# --- GitHub 認証情報を Secrets Manager に事前保存 ---
if [ -n "${REPO_OWNER:-}" ] && [ -n "${GITHUB_PAT:-}" ]; then
  SECRET_NAME="${PROJECT_NAME}-sagemaker-github-credentials"
  printf "${BOLD}${CYAN}=== GitHub 認証情報を Secrets Manager に保存 ===${RESET}\n"
  if aws secretsmanager describe-secret --secret-id "${SECRET_NAME}" --region "${REGION}" >/dev/null 2>&1; then
    aws secretsmanager put-secret-value \
      --secret-id "${SECRET_NAME}" \
      --secret-string "{\"username\":\"${REPO_OWNER}\",\"password\":\"${GITHUB_PAT}\"}" \
      --region "${REGION}" >/dev/null
  else
    aws secretsmanager create-secret \
      --name "${SECRET_NAME}" \
      --description "GitHub credentials for ${PROJECT_NAME} SageMaker AI Notebook" \
      --secret-string "{\"username\":\"${REPO_OWNER}\",\"password\":\"${GITHUB_PAT}\"}" \
      --region "${REGION}" >/dev/null
  fi
  # Get the secret ARN and pass it to CloudFormation
  SECRET_ARN=$(aws secretsmanager describe-secret --secret-id "${SECRET_NAME}" --region "${REGION}" --query "ARN" --output text)
  CFN_PARAMS+=("GitHubSecretArn=${SECRET_ARN}")
  printf "${GREEN}✔ 認証情報を Secrets Manager に保存しました${RESET}\n"
  echo ""
fi

printf "${BOLD}${CYAN}=== Service Quotas の確認・上限緩和リクエスト ===${RESET}\n"

# 必要な SageMaker Quotas: インスタンスタイプ、用途、Quota コード、必要数
# Pipeline のインスタンスタイプはコンテナごとに異なる:
#   container-navsim-transfuser:      ml.g6.4xlarge (GPU)
#   container-navsim-ego-mlp:    ml.c7i.xlarge (CPU)
#   container-pytorch-dlc:       ml.c7i.xlarge (CPU)
#   container-pytorch-dlc-byoc:  ml.c7i.xlarge (CPU)
# Notebook は ml.g4dn.2xlarge を 1 台使用 (CARLA シミュレーション + TransFuser 推論に 32 GB RAM が必要)
QUOTA_REQUESTS=(
  # QuotaCode:DesiredValue:Description
  # Notebook: ml.g4dn.2xlarge (GPU)
  "L-D8B97089:2:ml.g4dn.2xlarge for notebook instance usage"
  # CPU: ml.c7i.xlarge (navsim-ego-mlp, pytorch-dlc, pytorch-dlc-byoc)
  "L-194981EA:5:ml.c7i.xlarge for training job usage"
  "L-719822E7:5:ml.c7i.xlarge for processing job usage"
  # GPU: ml.g6.4xlarge (navsim-transfuser)
  "L-07B51BA0:5:ml.g6.4xlarge for training job usage"
  "L-8BF5F502:5:ml.g6.4xlarge for processing job usage"
  # GPU: ml.g6.xlarge (予備: GPU 推論テスト用)
  "L-56AE9D73:5:ml.g6.xlarge for training job usage"
  "L-49E4D2AB:5:ml.g6.xlarge for processing job usage"
)

for entry in "${QUOTA_REQUESTS[@]}"; do
  IFS=':' read -r quota_code desired_value description <<< "${entry}"

  current_value=$(aws service-quotas get-service-quota \
    --service-code sagemaker \
    --quota-code "${quota_code}" \
    --region "${REGION}" \
    --query 'Quota.Value' \
    --output text 2>/dev/null) || current_value="0"

  # 小数点を除去して比較 (API が 4.0 のように返す場合がある)
  current_int=${current_value%.*}
  if [[ "${current_int}" -ge "${desired_value}" ]]; then
    printf "  ✅ ${description}: ${current_int} (十分)\n"
  else
    printf "  ⚠️  ${description}: ${current_int} → ${desired_value} にリクエスト中..."
    quota_result=$(aws service-quotas request-service-quota-increase \
      --service-code sagemaker \
      --quota-code "${quota_code}" \
      --desired-value "${desired_value}" \
      --region "${REGION}" 2>&1) && printf " 完了\n" || printf " (既にリクエスト済み or エラー: ${quota_result})\n"
    printf "    確認: aws service-quotas get-service-quota --service-code sagemaker --quota-code ${quota_code} --region ${REGION} --query Quota.Value\n"
  fi
done
echo ""

printf "${BOLD}${CYAN}=== boto3 Lambda Layer の作成 ===${RESET}\n"
BOTO3_LAYER_ARN=$(aws lambda list-layer-versions \
  --layer-name "${PROJECT_NAME}-boto3" \
  --region "${REGION}" \
  --query 'LayerVersions[0].LayerVersionArn' \
  --output text 2>/dev/null || true)
if [ -n "${BOTO3_LAYER_ARN}" ] && [ "${BOTO3_LAYER_ARN}" != "None" ]; then
  printf "${GREEN}✔ 既存の Lambda Layer を使用: ${BOTO3_LAYER_ARN}${RESET}\n"
else
  LAYER_DIR=$(mktemp -d)
  LAYER_ZIP="/tmp/boto3-layer.zip"
  pip install boto3 -t "${LAYER_DIR}/python" -q
  (cd "${LAYER_DIR}" && zip -r "${LAYER_ZIP}" python -q)
  rm -rf "${LAYER_DIR}"
  BOTO3_LAYER_ARN=$(aws lambda publish-layer-version \
    --layer-name "${PROJECT_NAME}-boto3" \
    --compatible-runtimes python3.12 \
    --zip-file "fileb://${LAYER_ZIP}" \
    --region "${REGION}" \
    --query LayerVersionArn \
    --output text)
  rm -f "${LAYER_ZIP}"
  printf "${GREEN}✔ Lambda Layer を作成しました: ${BOTO3_LAYER_ARN}${RESET}\n"
fi
CFN_PARAMS+=("Boto3LayerArn=${BOTO3_LAYER_ARN}")
echo ""

printf "${BOLD}${CYAN}=== CloudFormation スタックのデプロイ ===${RESET}\n"
aws cloudformation deploy \
  --template-file "${TEMPLATE_FILE}" \
  --stack-name "${STACK_NAME}" \
  --parameter-overrides "${CFN_PARAMS[@]}" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "${REGION}" \
 

printf "${BOLD}${GREEN}✔ スタック '${STACK_NAME}' のデプロイが完了しました${RESET}\n"
echo ""

# --- サンプルデータセットを S3 にアップロード ---
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
DATASET_BUCKET="${PROJECT_NAME}-dataset-${ACCOUNT_ID}-${REGION}"

printf "${BOLD}${CYAN}=== サンプルデータセットを S3 にアップロード ===${RESET}\n"
UPLOADED_COUNT=0
for DATA_DIR in "${REPO_ROOT}"/pipelines/container-*/data; do
  [ -d "${DATA_DIR}" ] || continue
  CONTAINER_DIR=$(basename "$(dirname "${DATA_DIR}")")
  # train.csv または test.csv が存在するコンテナのみアップロード
  if [ -f "${DATA_DIR}/train.csv" ] || [ -f "${DATA_DIR}/test.csv" ]; then
    for CSV_FILE in "${DATA_DIR}"/{train,test}.csv; do
      [ -f "${CSV_FILE}" ] || continue
      SPLIT=$(basename "$(dirname "$(echo "${CSV_FILE}")")" | sed 's/.*//'; basename "${CSV_FILE}" .csv)
      SPLIT=$(basename "${CSV_FILE}" .csv)
      aws s3 cp "${CSV_FILE}" "s3://${DATASET_BUCKET}/${CONTAINER_DIR}/${SPLIT}/${SPLIT}.csv" \
        --region "${REGION}" --quiet
    done
    printf "  ${GREEN}✔${RESET} ${CONTAINER_DIR}\n"
    UPLOADED_COUNT=$((UPLOADED_COUNT + 1))
  fi
done
if [ "${UPLOADED_COUNT}" -eq 0 ]; then
  printf "  ${YELLOW}⚠ アップロード対象のデータが見つかりませんでした${RESET}\n"
else
  printf "${GREEN}✔ ${UPLOADED_COUNT} コンテナのサンプルデータを s3://${DATASET_BUCKET}/ にアップロードしました${RESET}\n"
fi
echo ""

# --- リポジトリ全体を S3 にアップロード (GitHub 連携なしの場合のみ) ---
# GitHub 連携がない場合、Lifecycle Config が S3 からリポジトリをダウンロードするため、
# ここで S3 にアップロードしておく。
if [ -z "${GITHUB_REPO}" ]; then
  ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
  DATASET_BUCKET="${PROJECT_NAME}-dataset-${ACCOUNT_ID}-${REGION}"

  printf "${BOLD}${CYAN}=== リポジトリを S3 にアップロード ===${RESET}\n"
  aws s3 sync "${REPO_ROOT}" "s3://${DATASET_BUCKET}/_repo-source/" \
    --exclude '.git/*' --exclude '.env' --exclude '.kiro/*' \
    --delete \
    --region "${REGION}" \
   
  printf "${GREEN}✔ リポジトリを s3://${DATASET_BUCKET}/_repo-source/ にアップロードしました${RESET}\n"
  echo ""

  # Notebook インスタンスを再起動して OnStart スクリプトを再実行する。
  # 初回デプロイ時は CloudFormation が Notebook を起動した時点で S3 にリポジトリが
  # まだアップロードされていないため、OnStart の s3 sync では何もダウンロードされない。
  # 再起動することで、アップロード済みのリポジトリが Notebook にダウンロードされる。
  if aws sagemaker describe-notebook-instance --notebook-instance-name "${PROJECT_NAME}-notebook" --region "${REGION}" &>/dev/null; then
    printf "${BOLD}${CYAN}=== Notebook インスタンスの再起動 ===${RESET}\n"
    printf "S3 にアップロードしたリポジトリを反映するため、Notebook を再起動します...\n"
    aws sagemaker stop-notebook-instance \
      --notebook-instance-name "${PROJECT_NAME}-notebook" \
      --region "${REGION}"
    aws sagemaker wait notebook-instance-stopped \
      --notebook-instance-name "${PROJECT_NAME}-notebook" \
      --region "${REGION}"
    aws sagemaker start-notebook-instance \
      --notebook-instance-name "${PROJECT_NAME}-notebook" \
      --region "${REGION}"
    aws sagemaker wait notebook-instance-in-service \
      --notebook-instance-name "${PROJECT_NAME}-notebook" \
      --region "${REGION}"
    printf "${GREEN}✔ Notebook インスタンスを再起動しました${RESET}\n"
    echo ""
  fi
fi

# --- スタック出力の表示 ---
printf "${BOLD}${CYAN}=== スタック出力 ===${RESET}\n"
OUTPUTS=$(aws cloudformation describe-stacks \
  --stack-name "${STACK_NAME}" \
  --region "${REGION}" \
  --query 'Stacks[0].Outputs' \
  --output json \
 )

echo "${OUTPUTS}" | python3 -c "
import json, sys
outputs = json.load(sys.stdin)
for o in sorted(outputs, key=lambda x: x['OutputKey']):
    print(f\"  {o['OutputKey']:40s} {o['OutputValue']}\")
"
echo ""

# --- Presigned URL の生成 ---
# ブラウザセッションの有効期限を 4 時間 (14400 秒) に設定。
# Notebook の presigned URL 自体の有効期限は 5 分間だが、
# セッション開始後はここで指定した時間だけブラウザセッションが維持される。
MLFLOW_NAME="${PROJECT_NAME}-mlflow"
SESSION_DURATION=14400

printf "${BOLD}${CYAN}=== 開発環境 URL ===${RESET}\n"
echo ""

# JupyterLab (Notebook がデプロイされている場合のみ)
NOTEBOOK_NAME=$(echo "${OUTPUTS}" | python3 -c "import json,sys;[print(o['OutputValue']) for o in json.load(sys.stdin) if o['OutputKey']=='NotebookInstanceName']" 2>/dev/null || true)

if [ -n "${NOTEBOOK_NAME}" ]; then
  printf "${BLUE}JupyterLab:${RESET}\n"
  NOTEBOOK_URL=$(aws sagemaker create-presigned-notebook-instance-url \
    --notebook-instance-name "${NOTEBOOK_NAME}" \
    --session-expiration-duration-in-seconds "${SESSION_DURATION}" \
    --region "${REGION}" \
    --query AuthorizedUrl \
    --output text \
   )
  JUPYTERLAB_URL="${NOTEBOOK_URL/\?//lab?}"
  print_hyperlink "${JUPYTERLAB_URL}" "JupyterLab を開く"
  echo ""
fi

# MLflow UI
printf "${BLUE}MLflow UI:${RESET}\n"
MLFLOW_ARN=$(echo "${OUTPUTS}" | python3 -c "import json,sys;[print(o['OutputValue']) for o in json.load(sys.stdin) if o['OutputKey']=='MlflowAppArn']" 2>/dev/null || true)
if [ -n "${MLFLOW_ARN}" ]; then
  MLFLOW_URL=$(aws sagemaker create-presigned-mlflow-app-url \
    --arn "${MLFLOW_ARN}" \
    --session-expiration-duration-in-seconds "${SESSION_DURATION}" \
    --region "${REGION}" \
    --query AuthorizedUrl \
    --output text \
   )
  print_hyperlink "${MLFLOW_URL}" "MLflow UI を開く"
  echo ""
fi

printf "${YELLOW}※ presigned URL の有効期限は 5 分間です。期限切れの場合は open-jupyterlab.sh / open-mlflow.sh で再取得できます。${RESET}\n"
printf "${YELLOW}※ ブラウザセッションは 4 時間有効です。${RESET}\n"
echo ""

# --- GitHub リポジトリ情報の表示 ---
if [ -n "${GITHUB_REPO}" ]; then
  printf "${BOLD}${CYAN}=== GitHub リポジトリ情報 ===${RESET}\n"
  printf "${BLUE}リポジトリ URL :${RESET} "
  GITHUB_BROWSER_URL="${GITHUB_REPO%.git}"
  printf '\e]8;;%s\e\\' "${GITHUB_BROWSER_URL}"
  printf '%s' "${GITHUB_REPO}"
  printf '\e]8;;\e\\\n'
  printf "${BLUE}ユーザー名     :${RESET} ${REPO_OWNER}\n"
  printf "${BLUE}認証方式       :${RESET} "
  if [ -n "${GITHUB_PAT}" ]; then
    printf "Secrets Manager (自動作成)\n"
  else
    printf "なし (パブリックリポジトリ)\n"
  fi
  echo ""
fi
