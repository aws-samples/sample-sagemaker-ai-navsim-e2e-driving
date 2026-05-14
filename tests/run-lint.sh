#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -uo pipefail

# ============================================================
# Lint / 静的検証スクリプト
# ============================================================
# スクリプトの整合性と基本機能を検証する。AWS リソースの作成・課金は発生しない。
# ローカル PC でも SageMaker AI Notebook インスタンスのターミナルでも実行できる。
#
# 使い方:
#   ./tests/run-lint.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

PASS=0
FAIL=0
SKIP=0

# timeout コマンドの互換性 (macOS は gtimeout)
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT="gtimeout"
else
  TIMEOUT=""
fi
run_with_timeout() {
  if [[ -n "${TIMEOUT}" ]]; then
    ${TIMEOUT} 5 "$@"
  else
    "$@"
  fi
}

pass() { echo "  ✅ $1"; ((PASS++)); }
fail() { echo "  ❌ $1"; [[ -n "${2:-}" ]] && echo "     → $2"; ((FAIL++)); }
skip() { echo "  ⏭️  $1"; ((SKIP++)); }

echo "============================================"
echo " 機能テスト"
echo "============================================"
echo ""

# ------------------------------------------------------------------
# 1. ファイル権限
# ------------------------------------------------------------------
echo "📋 [1/7] スクリプトの実行権限"
for f in \
    pipelines/scripts/run-pipeline.sh \
    pipelines/scripts/01-upload-dataset.sh \
    pipelines/scripts/02-build-and-push-container.sh \
    pipelines/scripts/03-create-and-run-pipeline.py \
    pipelines/scripts/04-check-pipeline-status.sh \
    infra/sagemaker-ai-ml-pipeline/scripts/deploy.sh \
    infra/sagemaker-ai-ml-pipeline/scripts/destroy.sh \
    infra/sagemaker-ai-inference/scripts/deploy.sh \
    infra/sagemaker-ai-inference/scripts/destroy.sh; do
  if [[ -f "$f" ]]; then
    if [[ -x "$f" ]]; then
      pass "$f"
    else
      fail "$f" "実行権限なし (chmod +x $f)"
    fi
  else
    skip "$f (ファイルなし)"
  fi
done
echo ""

# ------------------------------------------------------------------
# 2. シェルスクリプト構文チェック
# ------------------------------------------------------------------
echo "📋 [2/7] シェルスクリプト構文チェック"
while IFS= read -r -d '' f; do
  err=$(bash -n "$f" 2>&1)
  if [[ $? -eq 0 ]]; then
    pass "$f"
  else
    fail "$f" "${err}"
  fi
done < <(find pipelines/scripts infra -name "*.sh" -print0 2>/dev/null)
echo ""

# ------------------------------------------------------------------
# 3. Python 構文チェック
# ------------------------------------------------------------------
echo "📋 [3/7] Python 構文チェック"
while IFS= read -r -d '' f; do
  err=$(python -m py_compile "$f" 2>&1)
  if [[ $? -eq 0 ]]; then
    pass "$f"
  else
    fail "$f" "${err}"
  fi
done < <(find pipelines demo-app -name "*.py" -print0 2>/dev/null)
echo ""

# ------------------------------------------------------------------
# 4. --help オプション
# ------------------------------------------------------------------
echo "📋 [4/7] --help オプション"
for f in \
    "pipelines/scripts/run-pipeline.sh" \
    "pipelines/scripts/01-upload-dataset.sh" \
    "pipelines/scripts/02-build-and-push-container.sh" \
    "pipelines/scripts/04-check-pipeline-status.sh"; do
  if [[ -x "$f" ]]; then
    err=$(run_with_timeout "$f" --help 2>&1)
    if [[ $? -eq 0 ]]; then
      pass "$f --help"
    else
      fail "$f --help" "${err}"
    fi
  else
    skip "$f (実行権限なし)"
  fi
done
if run_with_timeout python pipelines/scripts/03-create-and-run-pipeline.py --help >/dev/null 2>&1; then
  pass "03-create-and-run-pipeline.py --help"
else
  fail "03-create-and-run-pipeline.py --help" "$(run_with_timeout python pipelines/scripts/03-create-and-run-pipeline.py --help 2>&1 | tail -3)"
fi
echo ""

# ------------------------------------------------------------------
# 5. --show-config オプション
# ------------------------------------------------------------------
echo "📋 [5/7] --show-config (インスタンスタイプ確認)"
for container in container-navsim-transfuser container-navsim-ego-mlp container-pytorch-dlc container-pytorch-dlc-byoc; do
  output=$(run_with_timeout python pipelines/scripts/03-create-and-run-pipeline.py \
    --project-name test --role-arn dummy --region us-east-1 \
    --container-dir "pipelines/${container}" \
    --show-config 2>/dev/null) || output=""
  if [[ "${output}" == *"Train:"* && "${output}" == *"Evaluate:"* ]]; then
    train=$(echo "${output}" | grep "Train:" | awk '{print $2}')
    eval=$(echo "${output}" | grep "Evaluate:" | awk '{print $2}')
    pass "${container}: Train=${train}, Eval=${eval}"
  else
    fail "${container}: --show-config" "出力: ${output:-empty}"
  fi
done
echo ""

# ------------------------------------------------------------------
# 6. _common.sh の読み込み
# ------------------------------------------------------------------
echo "📋 [6/7] _common.sh の読み込み"
if [[ -f "infra/_common.sh" ]]; then
  if bash -c "source infra/_common.sh && echo \${DEFAULT_PROJECT_NAME}" >/dev/null 2>&1; then
    pass "infra/_common.sh"
  else
    fail "infra/_common.sh" "$(bash -c 'source infra/_common.sh' 2>&1 | tail -3)"
  fi
else
  fail "infra/_common.sh" "ファイルが存在しません"
fi
# 各スクリプトからの参照パスが正しいか
for f in \
    pipelines/scripts/run-pipeline.sh \
    pipelines/scripts/01-upload-dataset.sh \
    pipelines/scripts/02-build-and-push-container.sh \
    pipelines/scripts/04-check-pipeline-status.sh; do
  if [[ -f "$f" ]]; then
    ref=$(grep '_common.sh' "$f" | head -1)
    if echo "${ref}" | grep -q 'infra/_common.sh'; then
      pass "$f → infra/_common.sh"
    elif echo "${ref}" | grep -q 'infra/scripts/_common.sh'; then
      fail "$f" "infra/scripts/_common.sh を参照 (正: infra/_common.sh)"
    else
      skip "$f (_common.sh 参照なし)"
    fi
  fi
done
echo ""

# ------------------------------------------------------------------
# 7. Notebook チェック
# ------------------------------------------------------------------
echo "📋 [7/7] Notebook チェック"

# papermill がなければインストール
if ! command -v papermill >/dev/null 2>&1; then
  echo "  papermill をインストール中..."
  pip install -q papermill
fi

if command -v jupyter >/dev/null 2>&1; then
  for nb in notebooks/*.ipynb; do
    if [[ -f "$nb" ]]; then
      nb_name=$(basename "$nb")

      # JSON チェック
      if python -c "import json; json.load(open('${nb}'))" 2>/dev/null; then
        pass "${nb_name} (valid JSON)"
      else
        fail "${nb_name} (invalid JSON)"
        continue
      fi

      # papermill --prepare-only (パラメータ注入 + カーネル検証)
      if command -v papermill >/dev/null 2>&1; then
        prepare_output=$(papermill "$nb" /dev/null --prepare-only 2>&1)
        if [[ $? -eq 0 ]]; then
          pass "${nb_name} (papermill prepare OK)"
        else
          fail "${nb_name} (papermill prepare)" "$(echo "${prepare_output}" | tail -3)"
        fi
      fi
    fi
  done
else
  skip "jupyter 未インストール"
fi
echo ""

# ------------------------------------------------------------------
# 結果サマリー
# ------------------------------------------------------------------
echo "============================================"
echo " テスト結果"
echo "============================================"
echo "  ✅ Pass: ${PASS}"
echo "  ❌ Fail: ${FAIL}"
echo "  ⏭️  Skip: ${SKIP}"
echo ""

if [[ ${FAIL} -gt 0 ]]; then
  echo "❌ ${FAIL} 件のテストが失敗しました。"
  exit 1
else
  echo "✅ すべてのテストに合格しました。"
fi
