# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# NAVSIM Transfuser: 学習・評価コンテナ
#
# NAVSIM 公式の Transfuser アーキテクチャに準拠した実装。
# timm の事前学習済み ResNet-34 バックボーン + GPT-style Transformer による
# multi-scale fusion を使用する。
#
# Transfuser と Latent Transfuser (LTF) はハイパーパラメータ --latent で切り替え:
#   --latent false  → Transfuser (カメラ + LiDAR)
#   --latent true   → LTF (カメラのみ、LiDAR を位置エンコーディングで代替)
#
# 使い方:
#   ./pipelines/scripts/02-build-and-push-container.sh -c container-navsim-transfuser
#   ./pipelines/scripts/run-pipeline.sh -c container-navsim-transfuser
#
# 推奨インスタンス:
#   ml.g6.4xlarge (L4 1 台, 16 vCPU, 64 GiB)

ARG AWS_REGION
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

ENV SAGEMAKER_PROGRAM=train.py
