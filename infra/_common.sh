#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# ============================================================
# 共通定義 - カラー、デフォルトパラメータ
# ============================================================
#
# 使い方:
#   source "$(dirname "$0")/_common.sh"
# ============================================================

# --- カラー定義 (ANSI エスケープコード) ---
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

# --- AWS CLI pager 無効化 ---
export AWS_PAGER=""

# --- デフォルトパラメータ ---
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# .env ファイルがあれば読み込む (既存の環境変数は上書きしない)
if [[ -f "${REPO_ROOT}/.env" ]]; then
  while IFS='=' read -r key value; do
    [[ -z "${key}" || "${key}" =~ ^# ]] && continue
    value="${value%\"}" && value="${value#\"}"
    [[ -z "${!key:-}" ]] && export "${key}=${value}"
  done < "${REPO_ROOT}/.env"
fi

DEFAULT_STACK_NAME="sagemaker-ai-ml-pipeline-stack"
DEFAULT_PROJECT_NAME="sagemaker-ai-ml-pipeline"
# リージョン: AWS_DEFAULT_REGION (.env 含む) → AWS CLI 設定 → us-east-1
DEFAULT_REGION="${AWS_DEFAULT_REGION:-$(aws configure get region 2>/dev/null || echo "us-east-1")}"

# --- OSC 8 ハイパーリンク表示関数 ---
# ターミナル上でクリック可能なリンクを表示する。
# 引数: $1 = URL, $2 = ラベルテキスト
print_hyperlink() {
  local url="$1"
  local label="$2"
  printf "  ${BOLD}${GREEN}>>>${RESET} "
  printf "${BOLD}${YELLOW}"
  printf '\e]8;;%s\e\\' "${url}"
  printf '%s' "${label}"
  printf '\e]8;;\e\\'
  printf "${RESET}"
  printf " ${BOLD}${GREEN}<<<${RESET}\n"
}

# --- クロスプラットフォーム URL オープン関数 ---
# macOS では自動でブラウザを開く。それ以外は URL を表示する。
# 引数: $1 = URL
open_url() {
  local url="$1"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    open "$url"
  else
    printf "${YELLOW}ブラウザで開いてください: %s${RESET}\n" "$url"
  fi
}

# --- 確認プロンプト関数 ---
# AUTO_APPROVE=true の場合はスキップ。それ以外は y/N で確認する。
# 引数: $1 = 確認メッセージ
confirm_or_abort() {
  local message="${1:-続行しますか?}"
  if [[ "${AUTO_APPROVE:-false}" == "true" ]]; then
    return 0
  fi
  printf "${BOLD}${YELLOW}%s [y/N]: ${RESET}" "${message}"
  read -r answer
  if [[ "${answer}" != "y" && "${answer}" != "Y" ]]; then
    echo "中断しました。"
    exit 0
  fi
}
