# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Evaluation script - runs as a SageMaker Processing job.
Loads a PyTorch model, evaluates on test dataset, and writes evaluation artifacts.

Environment variables (set by 03-create-and-run-pipeline.py):
    MLFLOW_APP_ARN : MLflow App の ARN (未設定時は MLflow 記録をスキップ)

Processing job のマウントパス:
    /opt/ml/processing/model : Training Job が出力した model.tar.gz
    /opt/ml/processing/test  : テストデータ (CSV)
    /opt/ml/processing/evaluation : 評価結果の出力先 (evaluation.json)
"""

import os
import json
import tarfile
import pandas as pd
import numpy as np
import mlflow
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class SimpleClassifier(nn.Module):
    """Simple feedforward classifier for tabular data.

    train.py と同じアーキテクチャを定義する必要がある。
    モデルの復元に使用するため、構造を変更した場合は両ファイルを同期すること。
    """

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def main():
    # --- パス設定 ---
    # Processing job のコンテナ内マウントパス (03-create-and-run-pipeline.py で定義)
    model_dir = "/opt/ml/processing/model"
    test_dir = "/opt/ml/processing/test"
    eval_dir = "/opt/ml/processing/evaluation"

    # --- モデル読み込み ---
    print("Loading model...")
    # Training Job の出力は model.tar.gz にアーカイブされるため、まず展開する
    tar_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=model_dir, filter="data")

    # チェックポイントから input_dim / num_classes を取得してモデルを再構築する
    # weights_only=True で安全にロードする (PyTorch 2.0 以降推奨)
    checkpoint = torch.load(
        os.path.join(model_dir, "model.pth"),
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    model = SimpleClassifier(
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    # 推論モードに切り替え (Dropout が無効化される)
    model.eval()

    # --- テストデータ読み込み ---
    print("Loading test data...")
    test_files = [f for f in os.listdir(test_dir) if f.endswith(".csv")]
    if not test_files:
        raise FileNotFoundError(f"No CSV files found in {test_dir}")

    # 複数 CSV ファイルを結合して 1 つの DataFrame にする
    df = pd.concat(
        [pd.read_csv(os.path.join(test_dir, f)) for f in test_files]
    )
    # 最終列をターゲット (ラベル)、それ以外を特徴量として扱う (train.py と同じ規則)
    X_test = torch.tensor(df.iloc[:, :-1].values.astype(np.float32))
    y_test = df.iloc[:, -1].values.astype(np.int64)

    # --- 推論 ---
    # torch.no_grad() で勾配計算を無効化してメモリ使用量を削減する
    with torch.no_grad():
        outputs = model(X_test)
        y_pred = outputs.argmax(dim=1).numpy()

    # --- メトリクス計算 ---
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
    }
    print(f"Evaluation metrics: {json.dumps(metrics, indent=2)}")

    # --- MLflow 記録 ---
    # MLFLOW_APP_ARN が設定されている場合のみ実行する
    tracking_arn = os.environ.get("MLFLOW_APP_ARN", "")
    tracking_url = os.environ.get("MLFLOW_APP_URL", "")
    if tracking_arn:
        mlflow.set_tracking_uri(tracking_arn)
        # training experiment とは別に evaluation experiment に記録する
        mlflow.set_experiment("evaluation")

        with mlflow.start_run() as run:
            # テストセットのメトリクスをまとめて記録する
            mlflow.log_metrics(metrics)
            # どのデータセットで評価したかをタグで記録する
            mlflow.set_tag("dataset", "test")

            # 正しい MLflow UI リンクを出力する
            if tracking_url:
                exp_id = run.info.experiment_id
                run_id = run.info.run_id
                base = f"{tracking_url}/#/experiments"
                print(f"MLflow UI (run): {base}/{exp_id}"
                      f"/runs/{run_id}")

    # --- 評価レポート出力 ---
    # SageMaker Pipeline の後続ステップや Model Registry の条件判定に使用できる
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation report saved to {eval_dir}/evaluation.json")


if __name__ == "__main__":
    main()
