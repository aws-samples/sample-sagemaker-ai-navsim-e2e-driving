# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
NAVSIM EgoStatusMLP 学習スクリプト。

NAVSIM の EgoStatusMLP ベースラインエージェントを SageMaker Training Job で学習する。
EgoStatusMLP はセンサー入力を使わず、自車の速度・加速度・走行コマンドのみから
将来の軌跡を予測する軽量なベースラインモデル。

【実行の流れ】
  1. SageMaker が S3 の NAVSIM データセットをコンテナの SM_CHANNEL_TRAIN にダウンロード
  2. navsim の SceneLoader でシーンデータを読み込み
  3. EgoStatusMLP エージェントを PyTorch Lightning で学習
  4. 学習済みモデルを SM_MODEL_DIR に保存

【環境変数】 (SageMaker が自動設定)
    SM_CHANNEL_TRAIN    : NAVSIM データセットのローカルパス
    SM_MODEL_DIR        : 学習済みモデルの保存先パス
    MLFLOW_APP_ARN : MLflow App の ARN (未設定時はスキップ)

【ハイパーパラメータ】
    --epochs        : 学習エポック数 (default: 50)
    --batch-size    : ミニバッチサイズ (default: 64)
    --learning-rate : 学習率 (default: 0.001)
    --hidden-dim    : MLP 隠れ層の次元数 (default: 128)
    --num-poses     : 予測する将来の軌跡ポーズ数 (default: 8)
    --time-horizon  : 予測時間ホライズン秒 (default: 4.0)
"""

import argparse
import json
import os
import sys
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class EgoStatusMLPModel(nn.Module):
    """EgoStatusMLP: 自車状態のみから将来軌跡を予測する MLP モデル。

    NAVSIM の EgoStatusMLPAgent と同等のアーキテクチャ。
    カメラや LiDAR などのセンサー入力を一切使わず、自車の運動状態だけで
    将来 4 秒間 (0.5 秒間隔 × 8 ポーズ) の軌跡を回帰する。
    「センサーなしでどこまでいけるか」を示すベースラインとして位置づけられる。

    SageMaker Training Job で独立して実行できるよう、navsim devkit への
    依存を排除している (navsim は Python 3.9 必須のため DLC と競合する)。

    入力 (8 次元):
        [velocity_x, velocity_y, accel_x, accel_y,
         cmd_left, cmd_straight, cmd_right, cmd_unknown]

    出力:
        [num_poses, 3] 各ポーズは (x, y, heading) のローカル座標

    アーキテクチャ:
        Linear(8 → hidden_dim) → ReLU
        → Linear(hidden_dim → hidden_dim) → ReLU
        → Linear(hidden_dim → hidden_dim) → ReLU
        → Linear(hidden_dim → num_poses * 3)
        → reshape to [num_poses, 3]
    """

    # 入力次元: velocity(2) + acceleration(2) + driving_command(4, one-hot)
    INPUT_DIM = 8

    def __init__(self, hidden_dim: int = 128, num_poses: int = 8):
        """
        Args:
            hidden_dim: MLP 隠れ層の次元数 (NAVSIM 公式デフォルト: 128)
            num_poses: 予測する将来の軌跡ポーズ数 (0.5 秒間隔 × 8 = 4 秒)
        """
        super().__init__()
        self.num_poses = num_poses
        self.mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # 出力: num_poses * 3 (各ポーズの x, y, heading をフラットに出力)
            nn.Linear(hidden_dim, num_poses * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播: EgoStatus 特徴量から将来軌跡を予測する。

        Args:
            x: [B, 8] EgoStatus 特徴量のバッチ

        Returns:
            [B, num_poses, 3] 予測軌跡 (x, y, heading)
        """
        out = self.mlp(x)
        # フラットな出力を [B, num_poses, 3] にリシェイプ
        return out.reshape(-1, self.num_poses, 3)


def extract_features_and_targets(data_dir: str, num_poses: int = 8):
    """NAVSIM データセットから EgoStatus 特徴量と GT 軌跡を読み込む。

    3 つのデータ形式に対応しており、以下の優先順位で読み込みを試みる:
      1. 前処理済み npz ファイル (notebook の extract_features.py で生成)
      2. navsim キャッシュ形式 (gz 圧縮 pickle、navsim devkit が生成)
      3. フォールバック: ダミーデータ生成 (パイプラインの動作確認用)

    通常は notebook 上で extract_features.py を実行して npz を生成し、
    S3 にアップロードしてから Training Job を実行するため、1 のパスを通る。

    Args:
        data_dir: SageMaker がマウントしたデータディレクトリのパス
                  (SM_CHANNEL_TRAIN 環境変数で指定される)
        num_poses: 予測する将来の軌跡ポーズ数

    Returns:
        features: [N, 8] numpy array - EgoStatus 特徴量
        targets: [N, num_poses, 3] numpy array - GT 軌跡 (x, y, heading)
    """
    # 前処理済み npz ファイルがある場合はそのまま読み込む
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if npz_files:
        print(f"Loading preprocessed data from {len(npz_files)} npz files...")
        all_features = []
        all_targets = []
        for npz_file in sorted(npz_files):
            data = np.load(npz_file)
            all_features.append(data["features"])
            all_targets.append(data["targets"])
        features = np.concatenate(all_features, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        print(f"Loaded {features.shape[0]} samples from npz files")
        return features.astype(np.float32), targets.astype(np.float32)

    # navsim キャッシュ形式の場合: gz ファイルから読み込む
    print("Attempting to load from navsim cache format...")
    try:
        import gzip
        import pickle

        all_features = []
        all_targets = []

        # キャッシュディレクトリ構造: data_dir/log_name/token/feature_name.gz
        for log_dir in sorted(os.listdir(data_dir)):
            log_path = os.path.join(data_dir, log_dir)
            if not os.path.isdir(log_path):
                continue
            for token_dir in sorted(os.listdir(log_path)):
                token_path = os.path.join(log_path, token_dir)
                if not os.path.isdir(token_path):
                    continue

                feature_path = os.path.join(token_path, "ego_status_feature.gz")
                target_path = os.path.join(token_path, "trajectory_target.gz")

                if os.path.exists(feature_path) and os.path.exists(target_path):
                    with gzip.open(feature_path, "rb") as f:
                        feat_dict = pickle.load(f)
                    with gzip.open(target_path, "rb") as f:
                        tgt_dict = pickle.load(f)

                    ego_status = feat_dict["ego_status"].numpy()
                    trajectory = tgt_dict["trajectory"].numpy()
                    all_features.append(ego_status)
                    all_targets.append(trajectory)

        if all_features:
            features = np.stack(all_features, axis=0)
            targets = np.stack(all_targets, axis=0)
            print(f"Loaded {features.shape[0]} samples from navsim cache")
            return features.astype(np.float32), targets.astype(np.float32)
    except Exception as e:
        print(f"Warning: Failed to load navsim cache: {e}")

    # フォールバック: ダミーデータ生成 (テスト用)
    print("WARNING: No valid data found. Generating dummy data for testing.")
    n_samples = 1000
    features = np.random.randn(n_samples, 8).astype(np.float32)
    # ダミー軌跡: 等速直線運動 + ノイズ
    targets = np.zeros((n_samples, num_poses, 3), dtype=np.float32)
    for i in range(num_poses):
        targets[:, i, 0] = features[:, 0] * (i + 1) * 0.5  # x = vx * t
        targets[:, i, 1] = features[:, 1] * (i + 1) * 0.5  # y = vy * t
    targets += np.random.randn(*targets.shape).astype(np.float32) * 0.1
    return features, targets


def parse_args():
    """コマンドライン引数をパースする。

    SageMaker SDK の Estimator.hyperparameters で渡した値は、
    --key value 形式の CLI 引数としてこのスクリプトに渡される。
    例: hyperparameters={"epochs": 50} → --epochs 50

    SageMaker 環境変数 (SM_CHANNEL_TRAIN, SM_MODEL_DIR) はデフォルト値として
    使用される。ローカルでテスト実行する場合は --train / --model-dir で上書き可能。
    """
    parser = argparse.ArgumentParser()

    # ハイパーパラメータ (notebook の Estimator.hyperparameters から渡される)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-poses", type=int, default=8)
    parser.add_argument("--time-horizon", type=float, default=4.0)

    # SageMaker が自動設定するパス (環境変数から取得)
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "num_poses": args.num_poses,
        "time_horizon": args.time_horizon,
    }

    print(f"Hyperparameters: {json.dumps(params, indent=2)}")

    # -------------------------------------------------------------------------
    # データ読み込み
    # -------------------------------------------------------------------------
    # SageMaker が S3 のデータを SM_CHANNEL_TRAIN にダウンロード済み。
    # extract_features_and_targets() で npz / navsim キャッシュ / ダミーの
    # いずれかの形式からデータを読み込む。
    print(f"Loading training data from {args.train}...")
    features, targets = extract_features_and_targets(args.train, args.num_poses)
    print(f"Features shape: {features.shape}, Targets shape: {targets.shape}")

    # Train/Val split (80/20)
    # ランダムシャッフルしてから分割。Validation は学習中の過学習チェックに使用。
    n_total = features.shape[0]
    n_train = int(n_total * 0.8)
    indices = np.random.permutation(n_total)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_dataset = TensorDataset(
        torch.tensor(features[train_idx]),
        torch.tensor(targets[train_idx]),
    )
    val_dataset = TensorDataset(
        torch.tensor(features[val_idx]),
        torch.tensor(targets[val_idx]),
    )
    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"], shuffle=False)

    # -------------------------------------------------------------------------
    # モデル初期化
    # -------------------------------------------------------------------------
    # GPU が利用可能なら GPU を使用 (EgoStatusMLP は軽量なので CPU でも十分)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = EgoStatusMLPModel(
        hidden_dim=params["hidden_dim"],
        num_poses=params["num_poses"],
    ).to(device)

    # NAVSIM 公式の EgoStatusMLP は L1 Loss (MAE) を使用。
    # L2 Loss (MSE) より外れ値に対してロバスト。
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # -------------------------------------------------------------------------
    # 学習ループ
    # -------------------------------------------------------------------------
    # エポックごとに Train → Validation を繰り返す。
    # best_val_loss を追跡して、最良のモデルを最終的に保存する。
    best_val_loss = float("inf")
    epoch_metrics = []
    for epoch in range(params["epochs"]):
        # --- Train phase ---
        model.train()
        train_loss = 0.0
        train_count = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            train_count += X_batch.size(0)

        # --- Validation phase ---
        # 勾配計算を無効化して推論のみ実行
        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_count += X_batch.size(0)

        avg_train_loss = train_loss / max(train_count, 1)
        avg_val_loss = val_loss / max(val_count, 1)

        # SageMaker の metric_definitions で正規表現キャプチャされ、
        # CloudWatch Metrics に自動送信される
        print(
            f"Epoch {epoch + 1}/{params['epochs']}"
            f" - train_loss: {avg_train_loss:.4f}"
            f" - val_loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        epoch_metrics.append({
            "train_loss": avg_train_loss, "val_loss": avg_val_loss,
        })

    # -------------------------------------------------------------------------
    # 最終メトリクス (ADE / FDE)
    # -------------------------------------------------------------------------
    # Validation データに対する軌跡予測精度を計算。
    # ADE (Average Displacement Error): 全タイムステップの平均 L2 距離 (m)
    # FDE (Final Displacement Error): 最終タイムステップの L2 距離 (m)
    # これらは自動運転の軌跡予測で標準的に使われるメトリクス。
    model.eval()
    all_preds = []
    all_targets_list = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets_list.append(y_batch.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets_arr = np.concatenate(all_targets_list, axis=0)

    # xy 座標のみで L2 距離を計算 (heading は含めない)
    ade = np.mean(np.sqrt(np.sum((all_preds[:, :, :2] - all_targets_arr[:, :, :2]) ** 2, axis=-1)))
    fde = np.mean(np.sqrt(np.sum((all_preds[:, -1, :2] - all_targets_arr[:, -1, :2]) ** 2, axis=-1)))

    # SageMaker の metric_definitions でキャプチャされる
    print(f"Training best_val_loss: {best_val_loss:.4f}")
    print(f"Training ADE: {ade:.4f}")
    print(f"Training FDE: {fde:.4f}")

    # -------------------------------------------------------------------------
    # MLflow 記録 (オプション)
    # -------------------------------------------------------------------------
    # notebook 側の Estimator.environment で渡された環境変数を参照する。
    # MLFLOW_APP_ARN が設定されていれば MLflow App に
    # ハイパーパラメータ・メトリクス・モデルを記録する。
    # 未設定の場合はこのセクション全体がスキップされ、学習には影響しない。
    tracking_arn = os.environ.get("MLFLOW_APP_ARN", "")
    tracking_url = os.environ.get("MLFLOW_APP_URL", "")
    model_group_name = os.environ.get(
        "MODEL_GROUP_NAME", "sagemaker-ai-ml-pipeline-navsim-ego-mlp"
    )
    if tracking_arn:
        import mlflow

        mlflow.set_tracking_uri(tracking_arn)
        mlflow.set_experiment("navsim-training")

        with mlflow.start_run() as run:
            mlflow.log_params(params)

            # epoch ごとのメトリクスを記録 (学習曲線の可視化用)
            for step, m in enumerate(epoch_metrics):
                mlflow.log_metric("train_loss", m["train_loss"], step=step)
                mlflow.log_metric("val_loss", m["val_loss"], step=step)

            # 最終メトリクスを記録
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("ade", ade)
            mlflow.log_metric("fde", fde)

            # MLflow Model Registry にモデルを登録
            # registered_model_name を指定すると新バージョンとして自動登録される
            mlflow.pytorch.log_model(
                pytorch_model=model,
                name="navsim-model",
                registered_model_name=model_group_name,
            )
            # モデルファイルを Run の Artifacts にも保存 (MLflow UI で確認可能にする)
            tmp_model_path = "/tmp/model.pth"
            torch.save(model.state_dict(), tmp_model_path)
            mlflow.log_artifact(tmp_model_path, artifact_path="model")
            print(f"Model registered to MLflow: {model_group_name}")

            # MLflow UI への直リンクを出力 (CloudWatch Logs から確認可能)
            if tracking_url:
                exp_id = run.info.experiment_id
                run_id = run.info.run_id
                base = f"{tracking_url}/#/experiments"
                print(f"MLflow UI (run): {base}/{exp_id}/runs/{run_id}")

    # -------------------------------------------------------------------------
    # モデル保存
    # -------------------------------------------------------------------------
    # SM_MODEL_DIR に保存したファイルは、Training Job 完了後に SageMaker が
    # 自動的に model.tar.gz に圧縮して output_path の S3 パスにアップロードする。
    # evaluate.py はこの model.tar.gz を展開してモデルを読み込む。
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hidden_dim": params["hidden_dim"],
            "num_poses": params["num_poses"],
            "time_horizon": params["time_horizon"],
            "params": params,  # evaluate.py でモデル再構築時に参照
        },
        model_path,
    )
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
