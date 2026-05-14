#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -uo pipefail

# ============================================================
# 統合テストスクリプト (実際の AWS リソースを使用)
# ============================================================
#
# 使い方:
#   ./tests/run-tests.sh [オプション]
#
# オプション:
#   -c, --container DIR   テスト対象コンテナ (デフォルト: 全コンテナ)
#   --skip-pipeline       Pipeline 実行をスキップ
#   --skip-notebook       Notebook テストをスキップ
#   --auto-approve        実行前の確認プロンプトをスキップ
#   -h, --help            ヘルプを表示
#
# ⚠️  注意:
#   - このスクリプトは JupyterLab のターミナル (SageMaker AI Notebook インスタンス上) で
#     実行してください。ローカル PC 上では動作しません。
#   - 実際の AWS リソースを作成・実行するため課金が発生します。
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1
source infra/_common.sh

usage() {
  cat << EOF
使い方: $(basename "$0") [オプション]

統合テスト (実際の AWS リソースを使用)。

⚠️  このスクリプトは JupyterLab のターミナル (SageMaker AI Notebook インスタンス上) で実行してください。
    実際の AWS リソースを作成・実行するため課金が発生します。

オプション:
  -c, --container DIR   テスト対象コンテナ (複数指定可、デフォルト: 全コンテナ)
                        例: -c container-pytorch-dlc -c container-navsim-ego-mlp
  --skip-pipeline       Pipeline 実行をスキップ
  --skip-notebook       Notebook テストをスキップ
  --auto-approve        実行前の確認プロンプトをスキップ
  -h, --help            ヘルプを表示

利用可能なコンテナ:
  container-pytorch-dlc         最速 (マネージドコンテナ、ビルド不要)
  container-pytorch-dlc-byoc    PyTorch DLC ベース BYOC (GPU 対応)
  container-navsim-ego-mlp      NAVSIM EgoStatusMLP (BYOC、CPU)
  container-navsim-transfuser   NAVSIM Transfuser (BYOC、GPU)

例:
  $(basename "$0")                                    # 全テスト
  $(basename "$0") -c container-pytorch-dlc           # PyTorch のみ
  $(basename "$0") --skip-notebook                    # Notebook テストをスキップ
EOF
}

# --- オプション解析 ---
CONTAINERS=()
SKIP_PIPELINE=false
SKIP_NOTEBOOK=false
AUTO_APPROVE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)        usage; exit 0 ;;
    -c|--container)   CONTAINERS+=("$2"); shift 2 ;;
    --skip-pipeline)  SKIP_PIPELINE=true; shift ;;
    --skip-notebook)  SKIP_NOTEBOOK=true; shift ;;
    --auto-approve)   AUTO_APPROVE=true; shift ;;
    *)                echo "ERROR: 不明なオプション: $1"; usage; exit 1 ;;
  esac
done

# デフォルト: Dockerfile がないコンテナ (ビルド不要) を優先
if [[ ${#CONTAINERS[@]} -eq 0 ]]; then
  for d in pipelines/container-*/; do
    CONTAINERS+=("$(basename "$d")")
  done
fi

PASS=0
FAIL=0
SKIP=0
RESULTS=()

record() {
  local status="$1" name="$2"
  if [[ "${status}" == "pass" ]]; then
    echo -e "  ${GREEN}✅ ${name}${RESET}"
    ((PASS++))
  elif [[ "${status}" == "fail" ]]; then
    echo -e "  ${RED}❌ ${name}${RESET}"
    ((FAIL++))
  else
    echo -e "  ${YELLOW}⏭️  ${name}${RESET}"
    ((SKIP++))
  fi
  RESULTS+=("${status}:${name}")
}

REGION="${AWS_DEFAULT_REGION:-${DEFAULT_REGION}}"
PROJECT_NAME="${DEFAULT_PROJECT_NAME}"

echo ""
echo -e "${BOLD}${CYAN}============================================${RESET}"
echo -e "${BOLD}${CYAN} 統合テスト${RESET}"
echo -e "${BOLD}${CYAN}============================================${RESET}"
echo -e "${YELLOW}⚠️  JupyterLab のターミナル (SageMaker AI Notebook インスタンス上) で実行してください。${RESET}"
echo -e "${YELLOW}⚠️  実際の AWS リソースを使用するため課金が発生します。${RESET}"
echo ""
echo -e "Project:    ${PROJECT_NAME}"
echo -e "Region:     ${REGION}"
echo -e "Containers: ${CONTAINERS[*]}"
echo -e "Pipeline:   $(if ${SKIP_PIPELINE}; then echo 'スキップ'; else echo '実行'; fi)"
echo -e "Notebook:   $(if ${SKIP_NOTEBOOK}; then echo 'スキップ'; else echo '実行'; fi)"
echo ""

# --- 実行確認 ---
if [[ "${AUTO_APPROVE}" == "false" ]]; then
  read -r -p "上記の設定でテストを実行しますか? [y/N]: " reply
  if [[ ! "${reply}" =~ ^[Yy]$ ]]; then
    echo "中止しました。"
    exit 0
  fi
  echo ""
fi

# ==================================================================
# 1. Pipeline テスト
# ==================================================================
if [[ "${SKIP_PIPELINE}" == "false" ]]; then
  echo -e "${BOLD}${CYAN}[1] Pipeline テスト${RESET}"
  echo ""

  for container in "${CONTAINERS[@]}"; do
    echo -e "${BOLD}--- ${container} ---${RESET}"

    # --show-config テスト
    config_output=$(python pipelines/scripts/03-create-and-run-pipeline.py \
      --project-name "${PROJECT_NAME}" --role-arn dummy --region "${REGION}" \
      --container-dir "pipelines/${container}" --show-config 2>/dev/null) || config_output=""
    if [[ "${config_output}" == *"Train:"* ]]; then
      record pass "${container}: --show-config"
      echo "    ${config_output}" | head -2
    else
      record fail "${container}: --show-config"
    fi

    # Pipeline 実行
    echo "    Pipeline を実行中..."
    start_time=$(date +%s)
    if ./pipelines/scripts/run-pipeline.sh -c "${container}" --auto-approve 2>&1 | tee "/tmp/test-${container}.log" | tail -5; then
      # ログから最終ステータスを確認
      if grep -q "正常に完了しました" "/tmp/test-${container}.log"; then
        elapsed=$(( $(date +%s) - start_time ))
        record pass "${container}: Pipeline 実行 (${elapsed}s)"
      else
        record fail "${container}: Pipeline 実行 (ログに完了メッセージなし)"
      fi
    else
      record fail "${container}: Pipeline 実行"
    fi
    echo ""
  done
else
  echo -e "${YELLOW}[1] Pipeline テスト: スキップ${RESET}"
  echo ""
fi

# ==================================================================
# 2. Notebook テスト
# ==================================================================
if [[ "${SKIP_NOTEBOOK}" == "false" ]]; then
  echo -e "${BOLD}${CYAN}[2] Notebook テスト${RESET}"
  echo ""

  if command -v papermill >/dev/null 2>&1; then
    :
  else
    echo "    papermill をインストール中..."
    pip install -q papermill
  fi

  if command -v papermill >/dev/null 2>&1; then
    for nb in notebooks/*.ipynb; do
      [[ -f "$nb" ]] || continue
      nb_name=$(basename "$nb")

      # -c 指定がある場合、対応する Notebook のみ実行
      if [[ ${#CONTAINERS[@]} -gt 0 ]]; then
        match=false
        for container in "${CONTAINERS[@]}"; do
          # コンテナ名 → Notebook 名のマッピング
          case "${container}" in
            container-pytorch-dlc)       nb_pattern="pytorch-pipeline.ipynb" ;;
            container-pytorch-dlc-byoc)  nb_pattern="pytorch-byoc-pipeline.ipynb" ;;
            container-navsim-ego-mlp)    nb_pattern="navsim-ego-mlp-pipeline.ipynb" ;;
            container-navsim-transfuser) nb_pattern="navsim-transfuser-pipeline.ipynb" ;;
            *)                           nb_pattern="" ;;
          esac
          if [[ "${nb_name}" == "${nb_pattern}" ]]; then
            match=true
            break
          fi
        done
        if [[ "${match}" == "false" ]]; then
          continue
        fi
      fi
      echo "    ${nb_name} を実行中..."

      output_nb="/tmp/test-${nb_name}"
      if (cd notebooks && papermill "../${nb}" "$output_nb" \
          -p PROJECT_NAME "${PROJECT_NAME}" \
          --log-output \
          2>&1); then
        record pass "notebook: ${nb_name}"
      else
        record fail "notebook: ${nb_name}"
        echo "    詳細: ${output_nb}"
      fi
    done
  fi
  echo ""
else
  echo -e "${YELLOW}[3] Notebook テスト: スキップ${RESET}"
  echo ""
fi

# ==================================================================
# 結果サマリー
# ==================================================================
echo -e "${BOLD}${CYAN}============================================${RESET}"
echo -e "${BOLD}${CYAN} テスト結果${RESET}"
echo -e "${BOLD}${CYAN}============================================${RESET}"
echo -e "  ${GREEN}✅ Pass: ${PASS}${RESET}"
echo -e "  ${RED}❌ Fail: ${FAIL}${RESET}"
echo -e "  ${YELLOW}⏭️  Skip: ${SKIP}${RESET}"
echo ""

if [[ ${FAIL} -gt 0 ]]; then
  echo -e "${RED}❌ ${FAIL} 件のテストが失敗しました。${RESET}"
  echo ""
  echo "失敗したテスト:"
  for r in "${RESULTS[@]}"; do
    if [[ "${r}" == fail:* ]]; then
      echo "  - ${r#fail:}"
    fi
  done
  exit 1
else
  echo -e "${GREEN}✅ すべてのテストに合格しました。${RESET}"
fi
