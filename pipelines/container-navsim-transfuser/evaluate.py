# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
NAVSIM Transfuser 評価スクリプト

学習済みの Transfuser モデルをロードし、テストデータで軌跡予測の精度を評価する。

【Processing Job のマウントパス】
    /opt/ml/processing/model      : model.tar.gz
    /opt/ml/processing/test       : テストデータ (pt or npz)
    /opt/ml/processing/evaluation : 評価結果の出力先 (evaluation.json)
"""

import glob
import json
import os
import tarfile

import mlflow
import numpy as np
import torch
import torch.nn.functional as F

from transfuser_config import TransfuserConfig
from transfuser_model import TransfuserModel


def load_test_data(test_dir, num_poses=8):
    """テストデータを読み込む。pt → npz → ダミーの順で試行。"""
    pt_files = sorted(glob.glob(os.path.join(test_dir, "**/*.pt"), recursive=True))
    if pt_files:
        return [torch.load(f, map_location="cpu", weights_only=True) for f in pt_files]

    npz_files = sorted(glob.glob(os.path.join(test_dir, "*.npz")))
    if npz_files:
        samples = []
        for f in npz_files:
            data = np.load(f)
            feats, tgts = data["features"], data["targets"]
            for i in range(feats.shape[0]):
                status = torch.tensor(feats[i], dtype=torch.float32)
                traj = torch.tensor(tgts[i], dtype=torch.float32)
                if status.shape[0] < 8:
                    status = F.pad(status, (0, 8 - status.shape[0]))
                samples.append({
                    "camera": torch.randn(3, 256, 1024),
                    "lidar": torch.randn(1, 256, 256),
                    "status": status[:8], "trajectory": traj,
                })
        return samples

    print("WARNING: No test data found. Generating dummy data.")
    return [
        {
            "camera": torch.randn(3, 256, 1024),
            "lidar": torch.randn(1, 256, 256),
            "status": torch.randn(8),
            "trajectory": torch.randn(num_poses, 3),
        }
        for _ in range(50)
    ]


def main():
    model_dir = "/opt/ml/processing/model"
    test_dir = "/opt/ml/processing/test"
    eval_dir = "/opt/ml/processing/evaluation"

    # --- モデル読み込み ---
    tar_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=model_dir, filter="data")

    checkpoint = torch.load(
        os.path.join(model_dir, "model.pth"), map_location="cpu", weights_only=True
    )
    cfg = checkpoint["config"]
    config = TransfuserConfig(
        latent=cfg["latent"],
        image_architecture=cfg["image_architecture"],
        lidar_architecture=cfg["lidar_architecture"],
        num_poses=cfg["num_poses"],
        num_bounding_boxes=cfg["num_bounding_boxes"],
    )

    model = TransfuserModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- テストデータ読み込み ---
    samples = load_test_data(test_dir, config.num_poses)
    print(f"Test samples: {len(samples)}")

    # --- 推論 ---
    all_preds, all_targets = [], []
    with torch.no_grad():
        for s in samples:
            features = {
                "camera_feature": s["camera"].unsqueeze(0),
                "status_feature": s["status"].unsqueeze(0),
            }
            if not config.latent:
                features["lidar_feature"] = s["lidar"].unsqueeze(0)

            out = model(features)
            all_preds.append(out["trajectory"].squeeze(0).numpy())
            all_targets.append(s["trajectory"].numpy())

    preds = np.stack(all_preds)
    targets = np.stack(all_targets)

    # --- メトリクス計算 ---
    disp = np.sqrt(np.sum((preds[:, :, :2] - targets[:, :, :2]) ** 2, axis=-1))
    metrics = {
        "pdm_score": round(float(max(0, 1.0 - np.mean(disp) / 10.0)), 4),
        "ade": round(float(np.mean(disp)), 4),
        "fde": round(float(np.mean(disp[:, -1])), 4),
        "heading_error": round(float(np.mean(np.abs(preds[:, :, 2] - targets[:, :, 2]))), 4),
        "miss_rate": round(float(np.mean(disp[:, -1] > 2.0)), 4),
    }
    mode = "LTF" if config.latent else "Transfuser"
    print(f"Mode: {mode}")
    print(f"Metrics: {json.dumps(metrics, indent=2)}")

    # --- MLflow 記録 ---
    tracking_arn = os.environ.get("MLFLOW_APP_ARN", "")
    if tracking_arn:
        mlflow.set_tracking_uri(tracking_arn)
        mlflow.set_experiment("navsim-transfuser-evaluation")
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.set_tag("mode", mode)

    # --- 評価レポート出力 ---
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {eval_dir}/evaluation.json")


if __name__ == "__main__":
    main()
