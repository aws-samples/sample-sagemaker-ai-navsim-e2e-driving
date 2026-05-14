# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
NAVSIM EgoStatusMLP 評価スクリプト。

学習済みの EgoStatusMLP モデルをロードし、テストデータで軌跡予測の精度を評価する。
NAVSIM の PDM Score に準じた簡易メトリクスを計算して evaluation.json に出力する。

【実行の流れ】
  1. SageMaker Processing Job がモデルとテストデータをコンテナにマウント
  2. model.tar.gz を展開して model.pth を読み込み、モデルを再構築
  3. テストデータに対して推論を実行
  4. PDM Score ベースのメトリクス (ADE, FDE, heading_error, miss_rate) を計算
  5. evaluation.json に出力
  6. (オプション) MLflow App にメトリクスを記録

【Processing Job のマウントパス】
    /opt/ml/processing/model      : Training Job が出力した model.tar.gz
    /opt/ml/processing/test       : テストデータ (npz or navsim キャッシュ)
    /opt/ml/processing/evaluation : 評価結果の出力先 (evaluation.json)

【環境変数】 (notebook の ScriptProcessor.env から渡される)
    MLFLOW_APP_ARN : MLflow App の ARN (未設定時はスキップ)
    MLFLOW_APP_URL : MLflow UI の URL (ログ出力用、未設定時はスキップ)
"""

import json
import os
import sys
import glob
import tarfile

import mlflow
import numpy as np
import torch
import torch.nn as nn


class EgoStatusMLPModel(nn.Module):
    """train.py と同じモデル定義 (推論時にモデルを再構築するために必要)。

    train.py の EgoStatusMLPModel と完全に同一のアーキテクチャでなければ、
    state_dict のロードに失敗する。モデル構造を変更した場合は
    train.py と evaluate.py の両方を更新すること。
    """

    INPUT_DIM = 8

    def __init__(self, hidden_dim: int = 128, num_poses: int = 8):
        super().__init__()
        self.num_poses = num_poses
        self.mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_poses * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        return out.reshape(-1, self.num_poses, 3)


def extract_features_and_targets(data_dir: str, num_poses: int = 8):
    """テストデータを読み込む (train.py と同じデータ読み込みロジック)。

    npz → navsim キャッシュ → ダミーデータの優先順位で読み込みを試みる。
    詳細は train.py の同名関数を参照。
    """
    # 前処理済み npz ファイル
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
        return features.astype(np.float32), targets.astype(np.float32)

    # navsim キャッシュ形式
    try:
        import gzip
        import pickle

        all_features = []
        all_targets = []
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
                    all_features.append(feat_dict["ego_status"].numpy())
                    all_targets.append(tgt_dict["trajectory"].numpy())

        if all_features:
            features = np.stack(all_features, axis=0)
            targets = np.stack(all_targets, axis=0)
            return features.astype(np.float32), targets.astype(np.float32)
    except Exception as e:
        print(f"Warning: Failed to load navsim cache: {e}")

    # フォールバック: ダミーデータ
    print("WARNING: No valid data found. Generating dummy data for testing.")
    n_samples = 200
    features = np.random.randn(n_samples, 8).astype(np.float32)
    targets = np.zeros((n_samples, num_poses, 3), dtype=np.float32)
    for i in range(num_poses):
        targets[:, i, 0] = features[:, 0] * (i + 1) * 0.5
        targets[:, i, 1] = features[:, 1] * (i + 1) * 0.5
    targets += np.random.randn(*targets.shape).astype(np.float32) * 0.1
    return features, targets


def compute_simplified_pdm_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
) -> dict:
    """PDM Score に準じた簡易メトリクスを計算する。

    NAVSIM の完全な PDM Score は nuPlan マップやシーン情報が必要だが、
    ここでは軌跡予測の精度に焦点を当てた簡易版を計算する。

    Args:
        predictions: [N, T, 3] 予測軌跡 (x, y, heading)
        ground_truth: [N, T, 3] GT 軌跡

    Returns:
        メトリクス辞書
    """
    # ADE (Average Displacement Error): 全タイムステップの平均 L2 距離
    displacement = np.sqrt(
        np.sum((predictions[:, :, :2] - ground_truth[:, :, :2]) ** 2, axis=-1)
    )
    ade = float(np.mean(displacement))

    # FDE (Final Displacement Error): 最終タイムステップの L2 距離
    fde = float(np.mean(displacement[:, -1]))

    # Heading Error: heading の平均絶対誤差 (rad)
    heading_error = float(
        np.mean(np.abs(predictions[:, :, 2] - ground_truth[:, :, 2]))
    )

    # Miss Rate: FDE > 2.0m のサンプル割合
    miss_rate = float(np.mean(displacement[:, -1] > 2.0))

    # 簡易 PDM Score: ADE ベースのスコア (0-1, 低い ADE ほど高スコア)
    # PDM Score の完全な計算には衝突判定・走行可能領域判定が必要
    pdm_score = float(max(0.0, 1.0 - ade / 10.0))

    return {
        "pdm_score": round(pdm_score, 4),
        "ade": round(ade, 4),
        "fde": round(fde, 4),
        "heading_error": round(heading_error, 4),
        "miss_rate": round(miss_rate, 4),
    }


def main():
    # Processing Job のマウントパス (SageMaker が自動設定)
    model_dir = "/opt/ml/processing/model"
    test_dir = "/opt/ml/processing/test"
    eval_dir = "/opt/ml/processing/evaluation"

    # -------------------------------------------------------------------------
    # モデル読み込み
    # -------------------------------------------------------------------------
    # Training Job の出力は model.tar.gz として S3 に保存される。
    # Processing Job の inputs で指定すると、SageMaker がこのファイルを
    # model_dir にダウンロードする。tar.gz を展開して model.pth を取得する。
    print("Loading model...")
    tar_path = os.path.join(model_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=model_dir, filter="data")

    # チェックポイントには model_state_dict に加えて hidden_dim, num_poses が
    # 含まれているため、モデルのアーキテクチャを自動的に再構築できる。
    # weights_only=True は PyTorch 2.0 以降で推奨される安全なロード方法。
    checkpoint = torch.load(
        os.path.join(model_dir, "model.pth"),
        map_location=torch.device("cpu"),
        weights_only=True,
    )
    model = EgoStatusMLPModel(
        hidden_dim=checkpoint["hidden_dim"],
        num_poses=checkpoint["num_poses"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Dropout 等を無効化して推論モードに切り替え

    # -------------------------------------------------------------------------
    # テストデータ読み込み
    # -------------------------------------------------------------------------
    print("Loading test data...")
    num_poses = checkpoint["num_poses"]
    features, targets = extract_features_and_targets(test_dir, num_poses)
    print(f"Test features: {features.shape}, targets: {targets.shape}")

    # -------------------------------------------------------------------------
    # 推論
    # -------------------------------------------------------------------------
    # 勾配計算を無効化してメモリ効率を上げる
    X_test = torch.tensor(features)
    with torch.no_grad():
        predictions = model(X_test).numpy()

    # -------------------------------------------------------------------------
    # メトリクス計算
    # -------------------------------------------------------------------------
    # PDM Score ベースの簡易メトリクスを計算 (ADE, FDE, heading_error, miss_rate)
    metrics = compute_simplified_pdm_metrics(predictions, targets)
    print(f"Evaluation metrics: {json.dumps(metrics, indent=2)}")

    # -------------------------------------------------------------------------
    # MLflow 記録 (オプション)
    # -------------------------------------------------------------------------
    # notebook 側の ScriptProcessor.env で渡された環境変数を参照する。
    # MLFLOW_APP_ARN が設定されていれば評価メトリクスを記録する。
    # 学習時のメトリクスは train.py 側で別の Run として記録されるため、
    # MLflow UI 上で学習 Run と評価 Run を比較できる。
    tracking_arn = os.environ.get("MLFLOW_APP_ARN", "")
    tracking_url = os.environ.get("MLFLOW_APP_URL", "")
    if tracking_arn:
        mlflow.set_tracking_uri(tracking_arn)
        mlflow.set_experiment("navsim-evaluation")

        with mlflow.start_run() as run:
            mlflow.log_metrics(metrics)
            mlflow.set_tag("dataset", "test")
            mlflow.set_tag("model", "EgoStatusMLP")

            if tracking_url:
                exp_id = run.info.experiment_id
                run_id = run.info.run_id
                base = f"{tracking_url}/#/experiments"
                print(f"MLflow UI (run): {base}/{exp_id}/runs/{run_id}")

    # -------------------------------------------------------------------------
    # 評価レポート出力
    # -------------------------------------------------------------------------
    # evaluation.json を出力ディレクトリに保存。Processing Job 完了後、
    # SageMaker が outputs で指定した S3 パスに自動アップロードする。
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation report saved to {eval_dir}/evaluation.json")


if __name__ == "__main__":
    main()
