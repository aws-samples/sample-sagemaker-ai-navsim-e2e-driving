#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail

# ============================================================
# Pipeline 実行状況の確認
# ============================================================
#
# 使い方:
#   ./04-check-pipeline-status.sh [オプション] [プロジェクト名]
#
# オプション:
#   -c, --container DIR   コンテナディレクトリ名 (パイプライン名の特定に使用)
#   -n, --num NUM         表示する実行数 (デフォルト: 1)
#   -L, --log-lines NUM   失敗時のログ表示行数 (デフォルト: 30)
#   -w, --watch SEC       指定秒間隔でポーリング (Ctrl+C で停止)
#   -h, --help            このヘルプを表示
#
# 表示内容:
#   - Pipeline 実行ステータス
#   - 各ステップ (Train / RegisterModel / Evaluate) の状況
#   - 失敗ステップの FailureReason
#   - 失敗ステップの CloudWatch Logs (末尾)
# ============================================================

source "$(dirname "$0")/../../infra/_common.sh"

RED='\033[0;31m'

usage() {
  cat << EOF
使い方: $(basename "$0") [オプション] [プロジェクト名]

Pipeline の実行状況を確認します。失敗時は CloudWatch Logs も自動表示します。

オプション:
  -c, --container DIR   コンテナディレクトリ名 (パイプライン名の特定に使用)
  -n, --num NUM         表示する実行数 (デフォルト: 1)
  -L, --log-lines NUM   失敗時のログ表示行数 (デフォルト: 30)
  -w, --watch SEC       指定秒間隔でポーリング (Ctrl+C で停止)
  -h, --help            このヘルプを表示

引数:
  プロジェクト名    Pipeline 名 ({プロジェクト名}-{コンテナ名}-pipeline) に使用
                    (デフォルト: sagemaker-ai-ml-pipeline)

例:
  $(basename "$0") -c container-pytorch-dlc          # PyTorch DLC パイプライン
  $(basename "$0") -c container-navsim-transfuser    # NAVSIM Transfuser パイプライン
  $(basename "$0") -c container-pytorch-dlc -n 3     # 最新 3 件を表示
  $(basename "$0") -c container-pytorch-dlc -w 30    # 30 秒間隔でポーリング
EOF
}

# --- オプション解析 ---
NUM_EXECUTIONS=1
LOG_LINES=30
WATCH_INTERVAL=0
PROJECT_NAME=""
CONTAINER_TAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)       usage; exit 0 ;;
    -n|--num)        NUM_EXECUTIONS="$2"; shift 2 ;;
    -L|--log-lines)  LOG_LINES="$2"; shift 2 ;;
    -w|--watch)      WATCH_INTERVAL="$2"; shift 2 ;;
    -c|--container)  CONTAINER_TAG="$2"; shift 2 ;;
    *)               PROJECT_NAME="$1"; shift ;;
  esac
done

PROJECT_NAME="${PROJECT_NAME:-${DEFAULT_PROJECT_NAME}}"
REGION="${AWS_DEFAULT_REGION:-${DEFAULT_REGION}}"
if [[ -n "${CONTAINER_TAG}" ]]; then
  PIPELINE_NAME="${PROJECT_NAME}-${CONTAINER_TAG}-pipeline"
else
  PIPELINE_NAME="${PROJECT_NAME}-pipeline"
fi

# --- ステータスアイコン ---
status_icon() {
  case "$1" in
    Succeeded|Completed) echo "✅" ;;
    Failed)              echo "❌" ;;
    Executing|InProgress) echo "🔄" ;;
    Starting)            echo "⏳" ;;
    Stopped)             echo "⏹️" ;;
    *)                   echo "❓" ;;
  esac
}

status_color() {
  case "$1" in
    Succeeded|Completed) echo -e "${GREEN}" ;;
    Failed|Stopped)      echo -e "${RED}" ;;
    Executing|Starting|InProgress) echo -e "${YELLOW}" ;;
    *)                   echo -e "${RESET}" ;;
  esac
}

# --- CloudWatch Logs を表示 ---
show_cloudwatch_logs() {
  local job_name="$1"
  local log_group="$2"
  local label="$3"

  echo ""
  echo -e "  ${CYAN}📋 ${label} CloudWatch Logs (/.../${job_name}):${RESET}"

  local stream
  stream=$(aws logs describe-log-streams \
    --log-group-name "${log_group}" \
    --log-stream-name-prefix "${job_name}" \
    --region "${REGION}" \
    --order-by LastEventTime \
    --descending \
    --max-items 1 \
    --query 'logStreams[0].logStreamName' \
    --output text 2>/dev/null) || true

  if [[ -z "${stream}" || "${stream}" == "None" ]]; then
    echo "    (ログストリームが見つかりません)"
    return
  fi

  aws logs get-log-events \
    --log-group-name "${log_group}" \
    --log-stream-name "${stream}" \
    --region "${REGION}" \
    --limit "${LOG_LINES}" \
    --query 'events[].message' \
    --output text 2>/dev/null | while IFS= read -r line; do
      # エラー行をハイライト
      if echo "${line}" | grep -qiE "error|exception|traceback|failed"; then
        echo -e "    ${RED}${line}${RESET}"
      else
        echo "    ${line}"
      fi
    done
}

# --- 失敗ステップのログを自動取得 ---
show_failed_step_logs() {
  local exec_arn="$1"
  local step_name="$2"

  # Training Job のログ
  local train_arn
  train_arn=$(aws sagemaker list-pipeline-execution-steps \
    --pipeline-execution-arn "${exec_arn}" \
    --region "${REGION}" \
    --query "PipelineExecutionSteps[?StepName=='${step_name}'].Metadata.TrainingJob.Arn | [0]" \
    --output text 2>/dev/null) || true

  if [[ -n "${train_arn}" && "${train_arn}" != "None" ]]; then
    local job_name="${train_arn##*/}"
    show_cloudwatch_logs "${job_name}" "/aws/sagemaker/TrainingJobs" "Training Job"
    return
  fi

  # Processing Job のログ
  local proc_arn
  proc_arn=$(aws sagemaker list-pipeline-execution-steps \
    --pipeline-execution-arn "${exec_arn}" \
    --region "${REGION}" \
    --query "PipelineExecutionSteps[?StepName=='${step_name}'].Metadata.ProcessingJob.Arn | [0]" \
    --output text 2>/dev/null) || true

  if [[ -n "${proc_arn}" && "${proc_arn}" != "None" ]]; then
    local job_name="${proc_arn##*/}"
    show_cloudwatch_logs "${job_name}" "/aws/sagemaker/ProcessingJobs" "Processing Job"
    return
  fi

  echo "    (ログ取得先の Job が特定できませんでした)"
}

# --- OSC 8 ハイパーリンク ---
osc8() {
  local url="$1" text="$2"
  local ESC=$'\x1b'
  printf '%s' "${ESC}]8;;${url}${ESC}\\${text}${ESC}]8;;${ESC}\\"
}

# --- CloudWatch instance metrics URL (Training/Processing 共通) ---
cw_instance_url() {
  local ns_enc="$1" job_name="$2"
  local host="${job_name}*2falgo-1"
  local Q="'"
  local graph="~(view~${Q}timeSeries~stacked~false~metrics~(~(~${Q}${ns_enc}~${Q}MemoryUtilization~${Q}Host~${Q}${host}~(id~${Q}m1))~(~${Q}.~${Q}DiskUtilization~${Q}.~${Q}.~(id~${Q}m2))~(~${Q}.~${Q}CPUUtilization~${Q}.~${Q}.~(id~${Q}m3)))~region~${Q}${REGION})"
  local query="~${Q}*7b${ns_enc}*2cHost*7d"
  echo "https://${REGION}.console.aws.amazon.com/cloudwatch/home?region=${REGION}#metricsV2?graph=${graph}&query=${query}"
}

# --- CloudWatch algorithm metrics URL (Training のみ) ---
cw_algo_url() {
  local ns_enc="$1" job_name="$2"
  local Q="'"
  local graph="~(view~${Q}timeSeries~stacked~false~metrics~(~(~${Q}${ns_enc}~${Q}train*3aprecision~${Q}TrainingJobName~${Q}${job_name}~(id~${Q}m1))~(~${Q}.~${Q}train*3aaccuracy~${Q}.~${Q}.~(id~${Q}m2))~(~${Q}.~${Q}train*3arecall~${Q}.~${Q}.~(id~${Q}m3))~(~${Q}.~${Q}train*3af1~${Q}.~${Q}.~(id~${Q}m4)))~region~${Q}${REGION})"
  local query="~${Q}*7b${ns_enc}*2cTrainingJobName*7d"
  echo "https://${REGION}.console.aws.amazon.com/cloudwatch/home?region=${REGION}#metricsV2?graph=${graph}&query=${query}"
}

# --- 1 回分の実行状況を表示 ---
show_execution() {
  local exec_arn="$1"
  local exec_status="$2"
  local start_time="$3"
  local exec_id="${exec_arn##*/}"

  local color
  color=$(status_color "${exec_status}")
  local icon
  icon=$(status_icon "${exec_status}")

  echo -e "${BOLD}Pipeline:${RESET}  ${PIPELINE_NAME}"
  echo -e "${BOLD}Execution:${RESET} ${exec_id}"
  echo -e "${BOLD}Status:${RESET}    ${icon} ${color}${exec_status}${RESET}"
  echo -e "${BOLD}Started:${RESET}   ${start_time}"
  echo ""

  # ステップ一覧を取得
  local steps_json
  steps_json=$(aws sagemaker list-pipeline-execution-steps \
    --pipeline-execution-arn "${exec_arn}" \
    --region "${REGION}" \
    --output json 2>/dev/null) || return

  local step_count
  step_count=$(echo "${steps_json}" | jq '.PipelineExecutionSteps | length')

  if [[ "${step_count}" -eq 0 ]]; then
    echo "  (ステップ情報なし - Pipeline 開始直後の可能性があります)"
    return
  fi

  echo -e "${BOLD}Steps:${RESET}"

  local failed_names=""

  while IFS=$'\t' read -r step_name step_status step_reason train_arn proc_arn; do
    local icon
    icon=$(status_icon "${step_status}")
    echo "  ${icon} ${step_name}: ${step_status}"

    if [[ -n "${step_reason}" && "${step_reason}" != "-" ]]; then
      echo "     └─ Reason: ${step_reason}"
    fi

    if [[ -n "${train_arn}" && "${train_arn}" != "-" ]]; then
      local job_name="${train_arn##*/}"
      local ns="*2faws*2fsagemaker*2fTrainingJobs"
      local sm_url="https://${REGION}.console.aws.amazon.com/sagemaker/home?region=${REGION}#/jobs/${job_name}"
      printf "     └─ ${CYAN}%s${RESET}  ${CYAN}%s${RESET}  ${CYAN}%s${RESET}\n" \
        "$(osc8 "${sm_url}" "[Console]")" \
        "$(osc8 "$(cw_instance_url "${ns}" "${job_name}")" "[CW Instance Metrics]")" \
        "$(osc8 "$(cw_algo_url "${ns}" "${job_name}")" "[CW Algorithm Metrics]")"
    elif [[ -n "${proc_arn}" && "${proc_arn}" != "-" ]]; then
      local job_name="${proc_arn##*/}"
      local ns="*2faws*2fsagemaker*2fProcessingJobs"
      local sm_url="https://${REGION}.console.aws.amazon.com/sagemaker/home?region=${REGION}#/processing-jobs/${job_name}"
      printf "     └─ ${CYAN}%s${RESET}  ${CYAN}%s${RESET}\n" \
        "$(osc8 "${sm_url}" "[Console]")" \
        "$(osc8 "$(cw_instance_url "${ns}" "${job_name}")" "[CW Instance Metrics]")"
    fi

    if [[ "${step_status}" == "Failed" ]]; then
      failed_names="${failed_names:+${failed_names},}${step_name}"
    fi
  done < <(echo "${steps_json}" | jq -r '
    .PipelineExecutionSteps[] |
    [
      .StepName,
      .StepStatus,
      (if has("FailureReason") then .FailureReason else "-" end),
      (.Metadata.TrainingJob.Arn // "-"),
      (.Metadata.ProcessingJob.Arn // "-")
    ] | @tsv
  ')

  if [[ -n "${failed_names}" ]]; then
    echo ""
    echo -e "${BOLD}${RED}=== 失敗ステップのログ ===${RESET}"
    IFS=',' read -ra names <<< "${failed_names}"
    for step_name in "${names[@]}"; do
      show_failed_step_logs "${exec_arn}" "${step_name}"
    done
  fi
}

# --- メイン処理 ---
run_check() {
  echo -e "${BOLD}=== Pipeline 実行状況 ===${RESET}"
  echo ""

  local executions_json
  executions_json=$(aws sagemaker list-pipeline-executions \
    --pipeline-name "${PIPELINE_NAME}" \
    --sort-by CreationTime \
    --sort-order Descending \
    --max-results "${NUM_EXECUTIONS}" \
    --region "${REGION}" \
    --output json 2>/dev/null)

  local exec_list
  exec_list=$(echo "${executions_json}" | jq -r '
    .PipelineExecutionSummaries[] |
    [.PipelineExecutionArn, .PipelineExecutionStatus, (.StartTime // "N/A")] | @tsv
  ') || true

  if [[ -z "${exec_list}" ]]; then
    echo "Pipeline '${PIPELINE_NAME}' の実行履歴がありません。"
    return
  fi

  local i=0
  while IFS=$'\t' read -r arn exec_status start_time; do
    if [[ $i -gt 0 ]]; then
      echo ""
      echo "─────────────────────────────────────────"
      echo ""
    fi
    show_execution "${arn}" "${exec_status}" "${start_time}"
    ((i++)) || true
  done <<< "${exec_list}"
}

# --- 実行 ---
if [[ "${WATCH_INTERVAL}" -gt 0 ]]; then
  trap 'echo ""; echo "ポーリングを停止しました。"; exit 0' INT
  while true; do
    clear
    run_check
    echo ""
    echo -e "${YELLOW}次の更新: ${WATCH_INTERVAL} 秒後 ($(date '+%H:%M:%S'))  Ctrl+C で停止${RESET}"
    sleep "${WATCH_INTERVAL}"
  done
else
  run_check
fi
