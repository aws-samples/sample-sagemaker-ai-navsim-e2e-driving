# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
# PyTorch DLC ベース BYOC - Train / Evaluate 両方でこのイメージを使用
#
# container-pytorch-dlc との違い:
#   container-pytorch-dlc       → Train は AWS マネージドコンテナ (PyTorch Estimator)
#   container-pytorch-dlc-byoc  → Train も Evaluate も この BYOC イメージを使用 (汎用 Estimator)
#
# Train ステップでもカスタムイメージを使いたい場合 (DLC にないライブラリが
# 学習時にも必要な場合など) にこの構成を選択してください。
#
# 使い方:
#   ./pipelines/scripts/02-build-and-push-container.sh -c container-pytorch-dlc-byoc
#   ./pipelines/scripts/run-pipeline.sh -c container-pytorch-dlc-byoc

ARG AWS_REGION
FROM 763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker

RUN pip install --no-cache-dir \
    mlflow>=3.0 \
    sagemaker-mlflow \
    scikit-learn \
    pandas \
    numpy \
    "pyarrow>=4.0.0,<16" \
    boto3

# requirements.txt は Notebook から直接 Processing Job を実行するケースに対応するため配置。
# Pipeline 実行時は source_dir を使わないため、この requirements.txt は参照されない。

# train.py / evaluate.py は COPY しない。
# SDK が entry_point + source_dir 経由で S3 からコンテナに注入するため、
# スクリプト変更時にコンテナの再ビルドが不要になる。
# 同様に ENTRYPOINT も指定しない。sagemaker-training ライブラリが
# SAGEMAKER_PROGRAM で指定されたスクリプトを自動的に実行する。
ENV SAGEMAKER_PROGRAM=train.py
