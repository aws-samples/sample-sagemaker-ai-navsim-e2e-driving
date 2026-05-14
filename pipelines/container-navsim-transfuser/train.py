# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
NAVSIM Transfuser 学習スクリプト

NAVSIM 公式の Transfuser アーキテクチャ (timm ResNet-34 + GPT-style multi-scale
fusion + Transformer Decoder) を SageMaker Training Job で学習する。

【環境変数】 (SageMaker が自動設定)
    SM_CHANNEL_TRAIN    : データセットのローカルパス
    SM_MODEL_DIR        : 学習済みモデルの保存先パス
    MLFLOW_APP_ARN : MLflow App の ARN (未設定時はスキップ)
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transfuser_config import TransfuserConfig
from transfuser_model import TransfuserModel
from transfuser_loss import transfuser_loss


# =========================================================================
# データセット
# =========================================================================

class TransfuserDataset(Dataset):
    """前処理済みの Transfuser 特徴量を読み込む Dataset。"""

    def __init__(self, data_dir: str, latent: bool = False):
        self.latent = latent
        pt_files = sorted(glob.glob(os.path.join(data_dir, "**/*.pt"), recursive=True))
        if pt_files:
            self.samples = pt_files
            return
        npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.samples = npz_files if npz_files else []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        if path.endswith(".pt"):
            return torch.load(path, map_location="cpu", weights_only=True)

        # npz フォールバック: ダミーの画像・LiDAR を生成
        data = np.load(path)
        status = torch.tensor(
            data["features"][0] if data["features"].ndim > 1 else data["features"],
            dtype=torch.float32,
        )
        trajectory = torch.tensor(
            data["targets"][0] if data["targets"].ndim > 2 else data["targets"],
            dtype=torch.float32,
        )
        camera = torch.randn(3, 256, 1024)
        lidar = torch.randn(1, 256, 256)
        if status.shape[0] < 8:
            status = F.pad(status, (0, 8 - status.shape[0]))
        return {
            "camera": camera, "lidar": lidar,
            "status": status[:8], "trajectory": trajectory,
        }


def collate_fn(batch):
    keys = set(batch[0].keys())
    for b in batch:
        keys &= b.keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


def generate_dummy_data(data_dir: str, n_samples: int = 200, config: TransfuserConfig = None):
    """パイプラインテスト用のダミーデータを生成する。"""
    os.makedirs(data_dir, exist_ok=True)
    num_poses = config.num_poses if config else 8
    num_bboxes = config.num_bounding_boxes if config else 30
    bev_h = config.lidar_resolution_height // 2 if config else 128
    bev_w = config.lidar_resolution_width if config else 256

    for i in range(n_samples):
        camera = torch.randn(3, 256, 1024)
        lidar = torch.randn(1, 256, 256)
        status = torch.randn(8)
        trajectory = torch.zeros(num_poses, 3)
        for t in range(num_poses):
            trajectory[t, 0] = status[0] * (t + 1) * 0.5
            trajectory[t, 1] = status[1] * (t + 1) * 0.5
        trajectory += torch.randn_like(trajectory) * 0.1

        # 補助タスク用のダミーターゲット
        agent_states = torch.randn(num_bboxes, 5)
        agent_labels = torch.zeros(num_bboxes)
        agent_labels[:5] = 1.0  # 最初の 5 個を有効なエージェントとする
        bev_semantic_map = torch.zeros(bev_h, bev_w, dtype=torch.long)

        torch.save({
            "camera": camera, "lidar": lidar,
            "status": status, "trajectory": trajectory,
            "agent_states": agent_states, "agent_labels": agent_labels,
            "bev_semantic_map": bev_semantic_map,
        }, os.path.join(data_dir, f"sample_{i:04d}.pt"))


# =========================================================================
# 学習ループ
# =========================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--latent", type=str, default="false")
    parser.add_argument("--train", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    return parser.parse_args()


def main():
    args = parse_args()
    latent = args.latent.lower() == "true"

    config = TransfuserConfig(latent=latent)
    params = {
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "latent": latent,
    }
    mode = "Latent Transfuser (LTF)" if latent else "Transfuser"
    print(f"Mode: {mode}")
    print(f"Hyperparameters: {json.dumps(params, indent=2, default=str)}")

    # --- データ読み込み ---
    print(f"Looking for data in: {args.train}")
    dataset = TransfuserDataset(args.train, latent=latent)
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("No data found. Generating dummy data...")
        dummy_dir = "/tmp/dummy_train_data"
        generate_dummy_data(dummy_dir, n_samples=200, config=config)
        dataset = TransfuserDataset(dummy_dir, latent=latent)
        print(f"Dataset size after dummy generation: {len(dataset)}")

    if len(dataset) == 0:
        raise RuntimeError("Failed to create dataset.")

    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"],
                              shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"],
                            shuffle=False, collate_fn=collate_fn, num_workers=2)
    print(f"Train: {n_train}, Val: {n_val}")

    # --- モデル初期化 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TransfuserModel(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"])

    # --- 学習ループ ---
    best_val_loss = float("inf")
    epoch_metrics = []

    for epoch in range(params["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            features = {
                "camera_feature": batch["camera"].to(device),
                "status_feature": batch["status"].to(device),
            }
            if not latent:
                features["lidar_feature"] = batch["lidar"].to(device)

            targets = {"trajectory": batch["trajectory"].to(device)}
            if "agent_states" in batch:
                targets["agent_states"] = batch["agent_states"].to(device)
                targets["agent_labels"] = batch["agent_labels"].to(device)
                targets["bev_semantic_map"] = batch["bev_semantic_map"].to(device)

            optimizer.zero_grad()
            predictions = model(features)

            # 補助タスクのターゲットがある場合は公式の損失関数を使用
            if "agent_states" in targets:
                loss = transfuser_loss(targets, predictions, config)
            else:
                loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch["camera"].size(0)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = {
                    "camera_feature": batch["camera"].to(device),
                    "status_feature": batch["status"].to(device),
                }
                if not latent:
                    features["lidar_feature"] = batch["lidar"].to(device)

                predictions = model(features)
                val_loss += F.l1_loss(
                    predictions["trajectory"], batch["trajectory"].to(device)
                ).item() * batch["camera"].size(0)

        avg_train = train_loss / max(n_train, 1)
        avg_val = val_loss / max(n_val, 1)
        print(f"Epoch {epoch+1}/{params['epochs']} - train_loss: {avg_train:.4f} - val_loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
        epoch_metrics.append({"train_loss": avg_train, "val_loss": avg_val})

    # --- 最終メトリクス (ADE / FDE) ---
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            features = {
                "camera_feature": batch["camera"].to(device),
                "status_feature": batch["status"].to(device),
            }
            if not latent:
                features["lidar_feature"] = batch["lidar"].to(device)
            out = model(features)
            all_preds.append(out["trajectory"].cpu().numpy())
            all_targets.append(batch["trajectory"].numpy())

    preds = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    ade = float(np.mean(np.sqrt(np.sum((preds[:, :, :2] - targets_np[:, :, :2]) ** 2, axis=-1))))
    fde = float(np.mean(np.sqrt(np.sum((preds[:, -1, :2] - targets_np[:, -1, :2]) ** 2, axis=-1))))

    print(f"Training best_val_loss: {best_val_loss:.4f}")
    print(f"Training ADE: {ade:.4f}")
    print(f"Training FDE: {fde:.4f}")

    # --- MLflow 記録 ---
    tracking_arn = os.environ.get("MLFLOW_APP_ARN", "")
    if tracking_arn:
        import mlflow
        mlflow.set_tracking_uri(tracking_arn)
        mlflow.set_experiment("navsim-transfuser-training")
        with mlflow.start_run():
            mlflow.log_params(params)
            for step, m in enumerate(epoch_metrics):
                mlflow.log_metric("train_loss", m["train_loss"], step=step)
                mlflow.log_metric("val_loss", m["val_loss"], step=step)
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("ade", ade)
            mlflow.log_metric("fde", fde)
            # モデルファイルを Run の Artifacts にも保存 (MLflow UI で確認可能にする)
            tmp_model_path = "/tmp/model.pth"
            torch.save(model.state_dict(), tmp_model_path)
            mlflow.log_artifact(tmp_model_path, artifact_path="model")

    # --- モデル保存 ---
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "latent": config.latent,
            "image_architecture": config.image_architecture,
            "lidar_architecture": config.lidar_architecture,
            "num_poses": config.num_poses,
            "num_bounding_boxes": config.num_bounding_boxes,
        },
    }, os.path.join(args.model_dir, "model.pth"))
    print(f"Model saved to {args.model_dir}/model.pth")


if __name__ == "__main__":
    main()
