# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# NAVSIM EgoStatusMLP: 学習・評価コンテナ
#
# PyTorch DLC をベースに、EgoStatusMLP の学習・評価に必要な最小限のパッケージを
# 追加した BYOC イメージ。
#
# navsim devkit 本体はインストールしない。理由:
#   - navsim v1.1 は Python 3.9 + numpy==1.23.4 等の古いバージョンをピン留めしており、
#     DLC (Python 3.11 + numpy 2.x) と依存関係が競合する
#   - train.py / evaluate.py は npz 形式の前処理済みデータを読み込むため、
#     navsim の SceneLoader への直接依存がない
#   - 特徴量抽出 (extract_features.py) は notebook / ローカル環境で事前に実行する
#
# 使い方:
#   ./pipelines/scripts/02-build-and-push-container.sh -c container-navsim-ego-mlp
#   ./pipelines/scripts/run-pipeline.sh -c container-navsim-ego-mlp --skip-upload

ARG AWS_REGION
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# train.py / evaluate.py は COPY しない。
# SDK が entry_point + source_dir 経由で S3 からコンテナに注入する。
ENV SAGEMAKER_PROGRAM=train.py
