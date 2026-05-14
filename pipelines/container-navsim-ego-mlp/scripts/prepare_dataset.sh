#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
set -euo pipefail

# ============================================================
# NAVSIM データセット準備スクリプト (EgoStatusMLP)
# ============================================================
#
# conda で Python 3.9 環境を自動作成し、navsim devkit をインストールして
# mini split から EgoStatusMLP 用の特徴量を抽出する。
#
# 使い方:
#   ./pipelines/container-navsim-ego-mlp/scripts/prepare_dataset.sh [プロジェクト名]
#
# 前提条件:
#   - conda がインストール済み (SageMaker AI Notebook にはプリインストール)
#   - AWS CLI が設定済み
#   - ディスク容量 10 GB 以上
#
# 所要時間の目安:
#   - 全体: 約 60 分
#     - Step 1 (conda 環境構築): 約 5 分
#     - Step 2 (OpenScene mini split ダウンロード): 約 40 分
#     - Step 3 (特徴量抽出): 約 10 分
#     - Step 4 (データバランシング): 約 1 分
#     - Step 5 (S3 アップロード): 約 4 分
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORK_DIR="/home/ec2-user/SageMaker/.navsim_workspace"
PROJECT_NAME="${1:-sagemaker-ai-ml-pipeline}"
source "${PROJECT_ROOT}/infra/_common.sh"
REGION="${AWS_DEFAULT_REGION:-${DEFAULT_REGION}}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
DATASET_BUCKET="${PROJECT_NAME}-dataset-${ACCOUNT_ID}-${REGION}"
CONTAINER_TAG="container-navsim-ego-mlp"
CONDA_ENV_DIR="/home/ec2-user/SageMaker/.conda_envs"
CONDA_ENV="navsim-py39"

echo "============================================"
echo "NAVSIM データセット準備 (EgoStatusMLP)"
echo "============================================"
echo "Project:    ${PROJECT_NAME}"
echo "Region:     ${REGION}"
echo "S3 Bucket:  s3://${DATASET_BUCKET}"
echo "Conda Env:  ${CONDA_ENV}"
echo ""

# ライセンス確認
echo "⚠️  NAVSIM データセットは nuPlan / OpenScene データセットに基づいています。"
echo "    https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/LICENSE"
echo ""
echo "⏱️  所要時間の目安: 約 60 分"
echo "    - conda 環境構築:                約 5 分"
echo "    - OpenScene mini split ダウンロード: 約 40 分"
echo "    - 特徴量抽出:                    約 10 分"
echo "    - データバランシング:            約 1 分"
echo "    - S3 アップロード:               約 4 分"
echo ""
read -p "ライセンスに同意しますか? [y/N]: " answer
if [[ "${answer}" != "y" && "${answer}" != "Y" ]]; then
    echo "中断しました。"
    exit 0
fi

# -------------------------------------------------------------------------
# Step 1: conda 環境の作成
# -------------------------------------------------------------------------
echo ""
echo "[Step 1/5] conda 環境を作成..."

# conda パッケージキャッシュをルートディスクではなく SageMaker ボリュームに配置
export CONDA_PKGS_DIRS="${CONDA_ENV_DIR}/.pkgs"
mkdir -p "${CONDA_PKGS_DIRS}"

if [[ -x "${CONDA_ENV_DIR}/${CONDA_ENV}/bin/python" ]]; then
    echo "既存の conda 環境を使用: ${CONDA_ENV_DIR}/${CONDA_ENV}"
else
    echo "Python 3.9 環境を作成中 (${CONDA_ENV_DIR}/${CONDA_ENV})..."
    rm -rf "${CONDA_ENV_DIR:?}/${CONDA_ENV}"
    conda create -p "${CONDA_ENV_DIR}/${CONDA_ENV}" python=3.9 -y
fi

# conda env のパスを解決
if [[ -d "${CONDA_ENV_DIR}/${CONDA_ENV}" ]]; then
    CONDA_PREFIX="${CONDA_ENV_DIR}/${CONDA_ENV}"
else
    CONDA_PREFIX="$(conda info --base)/envs/${CONDA_ENV}"
fi
PYTHON="${CONDA_PREFIX}/bin/python"
PIP="${CONDA_PREFIX}/bin/pip"

# navsim 本体を clone (pip install より先に行う)
mkdir -p "${WORK_DIR}"
if [[ ! -d "${WORK_DIR}/navsim" ]]; then
    echo "navsim リポジトリを clone..."
    git clone --depth 1 --branch v1.1 \
        https://github.com/autonomousvision/navsim.git \
        "${WORK_DIR}/navsim"
fi

echo "navsim devkit をインストール中 (requirements.txt から)..."
"${PIP}" install --quiet -r "${WORK_DIR}/navsim/requirements.txt" 2>&1 | tail -5

# -------------------------------------------------------------------------
# Step 2: mini split データセットをダウンロード
# -------------------------------------------------------------------------
echo ""
echo "[Step 2/5] mini split データセットをダウンロード..."
DATASET_DIR="${WORK_DIR}/dataset"
mkdir -p "${DATASET_DIR}"

DOWNLOAD_DIR="${WORK_DIR}/navsim/download"

# ダウンロード済みデータのリンクを作成する関数
link_downloaded_data() {
    local linked=false
    for d in navsim_logs sensor_blobs maps; do
        for src in "${DOWNLOAD_DIR}/${d}" "${DOWNLOAD_DIR}/mini_${d}"; do
            if [[ -d "${src}" ]]; then
                ln -sfn "${src}" "${DATASET_DIR}/${d}"
                echo "  Linked $(basename "${src}") → ${DATASET_DIR}/${d}"
                linked=true
                break
            fi
        done
    done
    ${linked}
}

# 既にダウンロード済みかチェック (リンク先が有効ならスキップ)
if [[ -d "${DATASET_DIR}/navsim_logs/mini" ]]; then
    echo "  ダウンロード済みデータを検出。スキップします。"
else
    # リンクが未作成の場合、既存のダウンロードデータからリンクを試行
    if link_downloaded_data && [[ -d "${DATASET_DIR}/navsim_logs/mini" ]]; then
        echo "  既存のダウンロードデータからリンクを作成しました。"
    elif [[ -d "${DOWNLOAD_DIR}" ]]; then
        # ダウンロードを実行
        pushd "${DOWNLOAD_DIR}" > /dev/null
        if [[ -f "./download_mini.sh" ]]; then
            chmod +x ./download_mini.sh
            yes | bash ./download_mini.sh || echo "WARNING: download script exited with non-zero (some files may still have been downloaded)"
        fi
        popd > /dev/null
        link_downloaded_data
    fi
fi

# データの存在確認
if [[ ! -d "${DATASET_DIR}/navsim_logs/mini" ]]; then
    echo ""
    echo "❌ navsim_logs/mini が見つかりません。"
    echo "   ダウンロードディレクトリの内容:"
    find "${DOWNLOAD_DIR}" -maxdepth 3 -type d 2>/dev/null | head -20
    echo ""
    echo "   dataset ディレクトリの内容:"
    find "${DATASET_DIR}" -maxdepth 3 -type d 2>/dev/null | head -20
    exit 1
fi

# -------------------------------------------------------------------------
# Step 3: 特徴量抽出 (Python 3.9 環境で実行)
# -------------------------------------------------------------------------
echo ""
echo "[Step 3/5] 特徴量を抽出..."
CACHE_DIR="${WORK_DIR}/cache/ego-mlp"
mkdir -p "${CACHE_DIR}"

# NUPLAN_MAPS_ROOT を設定 (navsim が地図ファイルにアクセスするために必要)
export NUPLAN_MAPS_ROOT="${DATASET_DIR}/maps"

"${PYTHON}" -u "${CONTAINER_DIR}/scripts/extract_features.py" \
    --data-root "${DATASET_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    --navsim-root "${WORK_DIR}/navsim" \
    --split mini \
    2>&1 || {
    echo ""
    echo "❌ 特徴量抽出に失敗しました。"
    echo "   navsim devkit のインストールまたはデータのダウンロードに問題がある可能性があります。"
    echo "   ログを確認してください。"
    exit 1
}

# -------------------------------------------------------------------------
# Step 4: データバランシング
# -------------------------------------------------------------------------
echo ""
echo "[Step 4/6] データをバランシング (コマンド分布を完全均等化)..."

"${PYTHON}" -u "${CONTAINER_DIR}/scripts/balance_dataset.py" \
    --cache-dir "${CACHE_DIR}" \
    --strategy equal \
    --exclude-unknown \
    2>&1 || {
    echo ""
    echo "⚠️  バランシングに失敗しました（不均衡度が低い場合はスキップされます）"
    echo "   学習は元のデータで続行されます"
}

# -------------------------------------------------------------------------
# Step 5: S3 にアップロード
# -------------------------------------------------------------------------
echo ""
echo "[Step 5/6] データを S3 にアップロード..."

for prefix in train test; do
    if ls "${CACHE_DIR}"/${prefix}*.npz 1>/dev/null 2>&1; then
        aws s3 cp "${CACHE_DIR}/" "s3://${DATASET_BUCKET}/${CONTAINER_TAG}/${prefix}/" \
            --recursive --exclude "*" --include "${prefix}*.npz" \
            --region "${REGION}"
        echo "  Uploaded ${prefix} data"
    fi
done

# -------------------------------------------------------------------------
# Step 6: 完了
# -------------------------------------------------------------------------
echo ""
echo "[Step 6/6] クリーンアップ..."
echo "一時ファイル: ${WORK_DIR} (不要なら rm -rf ${WORK_DIR})"
echo "元のデータバックアップ: ${CACHE_DIR}/train_data_original.npz (バランシング前)"

echo ""
echo "============================================"
echo "✅ データセット準備完了 (EgoStatusMLP)"
echo "============================================"
echo "Train: s3://${DATASET_BUCKET}/${CONTAINER_TAG}/train/"
echo "Test:  s3://${DATASET_BUCKET}/${CONTAINER_TAG}/test/"
echo ""
echo "次のステップ:"
echo "  ./pipelines/scripts/run-pipeline.sh -c container-navsim-ego-mlp --skip-upload"
