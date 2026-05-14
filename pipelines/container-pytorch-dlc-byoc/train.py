# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
Training script - PyTorch SimpleClassifier (3-layer MLP).

このスクリプトは SageMaker Training Job のコンテナ内で実行される。
PyTorch Estimator の entry_point として指定することで、SageMaker が
マネージドの PyTorch DLC (Deep Learning Container) 上でこのスクリプトを呼び出す。

【実行の流れ】
  1. SageMaker が S3 の学習データをコンテナの SM_CHANNEL_TRAIN にダウンロード
  2. このスクリプトが実行される
  3. 学習済みモデルを SM_MODEL_DIR に保存
  4. SageMaker が SM_MODEL_DIR の内容を model.tar.gz に圧縮して S3 にアップロード

【環境変数】 (SageMaker が自動設定する)
    SM_CHANNEL_TRAIN    : S3 から同期されたトレーニングデータのローカルパス
    SM_MODEL_DIR        : 学習済みモデルの保存先パス (S3 にアップロードされる)
    MLFLOW_APP_ARN : MLflow App の ARN (未設定時は MLflow 記録をスキップ)
    MODEL_GROUP_NAME    : MLflow / SageMaker Model Registry のモデルグループ名

【ハイパーパラメータ】 (PyTorch Estimator の hyperparameters 引数で渡す)
    --epochs        : 学習エポック数 (default: 20)
    --batch-size    : ミニバッチサイズ (default: 32)
    --learning-rate : Adam オプティマイザの学習率 (default: 0.001)

    SageMaker SDK は hyperparameters dict のキーをそのまま CLI 引数に変換する。
    例: {"batch-size": 32} → スクリプトに "--batch-size 32" として渡される。
    argparse 側では dest が自動的に "batch_size" (ハイフン→アンダースコア) になる。
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score


class SimpleClassifier(nn.Module):
    """表形式データ向けの 3 層フィードフォワード分類器。

    アーキテクチャ: 入力次元 → Linear(64) → ReLU → Dropout(0.2)
                              → Linear(32) → ReLU → Dropout(0.2)
                              → Linear(num_classes)

    Dropout(0.2) を各隠れ層に挿入することで過学習を抑制する。
    evaluate.py でモデルを復元する際も同じクラス定義が必要なため、
    アーキテクチャを変更した場合は evaluate.py も同様に更新すること。
    """

    def __init__(self, input_dim: int, num_classes: int):
        """
        Args:
            input_dim:   入力特徴量の次元数 (CSV の列数 - 1)
            num_classes: 分類クラス数 (ターゲット列のユニーク値の数)
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            # 出力層は Softmax なし。CrossEntropyLoss が内部で Softmax を計算する。
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args():
    """コマンドライン引数とデフォルト値を定義する。

    SageMaker SDK の hyperparameters 引数で渡した値は
    "--key value" 形式の CLI 引数としてこのスクリプトに渡される。

    SM_CHANNEL_TRAIN / SM_MODEL_DIR は SageMaker が自動設定する環境変数。
    ローカル実行時はデフォルト値 (/opt/ml/...) にフォールバックする。
    """
    parser = argparse.ArgumentParser()

    # --- ハイパーパラメータ ---
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    # バッチサイズは GPU メモリ量に応じて調整する
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size for DataLoader.",
    )
    # Adam の学習率。大きすぎると発散、小さすぎると収束が遅くなる。
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer.",
    )

    # --- SageMaker が自動設定するパス ---
    # fit() の inputs={"train": ...} で渡した S3 URI が SM_CHANNEL_TRAIN に展開される
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
        help="Path to training data directory (set automatically by SageMaker).",
    )
    # SageMaker はこのディレクトリの内容を model.tar.gz に圧縮して S3 に保存する
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Path to save the trained model (set automatically by SageMaker).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # argparse は "--batch-size" を args.batch_size (アンダースコア) に変換する
    params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }

    # -------------------------------------------------------------------------
    # データ読み込み
    # -------------------------------------------------------------------------
    # SM_CHANNEL_TRAIN 配下に S3 のデータが展開されている。
    # 複数ファイルが存在する場合はすべて結合して 1 つの DataFrame にする。
    print("Loading training data...")
    train_files = [f for f in os.listdir(args.train) if f.endswith(".csv")]
    if not train_files:
        raise FileNotFoundError(f"No CSV files found in {args.train}")

    df = pd.concat(
        [pd.read_csv(os.path.join(args.train, f)) for f in train_files]
    )
    print(f"Training data shape: {df.shape}")

    # データ規約: 最終列をターゲット (ラベル)、それ以外を特徴量として扱う。
    # evaluate.py も同じ規約に従っているため、変更する場合は両ファイルを同期すること。
    # PyTorch の要件: 特徴量は float32、ラベルは int64 (CrossEntropyLoss の入力型)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    # モデル構築に必要な次元情報をデータから動的に取得する
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))

    # -------------------------------------------------------------------------
    # DataLoader 作成 (train / validation 分割)
    # -------------------------------------------------------------------------
    # 学習データの 80% を学習、20% を validation に使用する。
    # validation は epoch ごとの過学習検出に使用する。
    val_ratio = 0.2
    n_val = int(len(X) * val_ratio)
    indices = np.random.RandomState(42).permutation(len(X))
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    train_dataset = TensorDataset(
        torch.tensor(X[train_idx]), torch.tensor(y[train_idx])
    )
    val_dataset = TensorDataset(
        torch.tensor(X[val_idx]), torch.tensor(y[val_idx])
    )
    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=params["batch_size"], shuffle=False
    )
    print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

    # -------------------------------------------------------------------------
    # モデル・損失関数・オプティマイザ初期化
    # -------------------------------------------------------------------------
    # GPU が利用可能な場合は自動的に GPU を使用する (ml.p3 系インスタンスなど)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleClassifier(input_dim, num_classes).to(device)

    # CrossEntropyLoss: 多クラス分類の標準的な損失関数。
    # 内部で LogSoftmax + NLLLoss を計算するため、モデル出力に Softmax は不要。
    criterion = nn.CrossEntropyLoss()

    # Adam: 適応的学習率を持つ最適化アルゴリズム。SGD より収束が速い傾向がある。
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # -------------------------------------------------------------------------
    # 学習ループ
    # -------------------------------------------------------------------------
    # MLflow の epoch メトリクス記録用 (後でまとめて送信)
    epoch_metrics = []

    for epoch in range(params["epochs"]):
        # --- Train ---
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            correct += (outputs.argmax(dim=1) == y_batch).sum().item()
            total += X_batch.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss_sum += loss.item() * X_batch.size(0)
                val_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
                val_total += X_batch.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        epoch_metrics.append({
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        print(
            f"Epoch {epoch + 1}/{params['epochs']}"
            f" - loss: {train_loss:.4f}, acc: {train_acc:.4f}"
            f" - val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

    # -------------------------------------------------------------------------
    # 最終メトリクス計算
    # -------------------------------------------------------------------------
    # SageMaker は標準出力を CloudWatch Logs に転送する。
    # metric_definitions の Regex パターンと一致する行が CloudWatch Metrics に記録される。
    # 03-create-and-run-pipeline.py の metric_definitions を参照。
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    train_accuracy = (all_preds == all_labels).mean()
    train_precision = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    train_recall = recall_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    train_f1 = f1_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    avg_loss = epoch_metrics[-1]["train_loss"]

    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Training precision: {train_precision:.4f}")
    print(f"Training recall: {train_recall:.4f}")
    print(f"Training f1: {train_f1:.4f}")

    # -------------------------------------------------------------------------
    # MLflow 記録 (オプション)
    # -------------------------------------------------------------------------
    # MLFLOW_APP_ARN が設定されている場合のみ実行する。
    # 未設定の場合 (ローカル実行・MLflow なし環境) は静かにスキップする。
    tracking_arn = os.environ.get("MLFLOW_APP_ARN", "")
    tracking_url = os.environ.get("MLFLOW_APP_URL", "")
    model_group_name = os.environ.get(
        "MODEL_GROUP_NAME", "sagemaker-ai-ml-pipeline-pytorch-byoc"
    )
    if tracking_arn:
        import mlflow
        from mlflow.models import infer_signature

        # SageMaker Managed MLflow の場合、ARN を tracking URI として直接指定できる
        mlflow.set_tracking_uri(tracking_arn)
        mlflow.set_experiment("training")

        with mlflow.start_run() as run:
            # ハイパーパラメータを記録 (MLflow UI で実験間の比較に使用)
            mlflow.log_params(params)

            # epoch ごとのメトリクスを記録 (MLflow UI で学習曲線を可視化)
            for step, m in enumerate(epoch_metrics):
                mlflow.log_metric("train_loss", m["train_loss"], step=step)
                mlflow.log_metric("train_acc", m["train_acc"], step=step)
                mlflow.log_metric("val_loss", m["val_loss"], step=step)
                mlflow.log_metric("val_acc", m["val_acc"], step=step)

            # 最終メトリクスを記録
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("train_precision", train_precision)
            mlflow.log_metric("train_recall", train_recall)
            mlflow.log_metric("train_f1", train_f1)

            # signature: モデルの入出力スキーマ (型・shape) を記録する。
            # infer_signature は numpy array を受け取るため、
            # GPU 上のテンソルを CPU → numpy に変換してから渡す。
            sample_input = torch.tensor(X[:5]).to(device)
            with torch.no_grad():
                sample_output = model(sample_input).cpu().numpy()
            signature = infer_signature(X[:5], sample_output)

            # log_model + registered_model_name で記録と Model Registry 登録を同時に行う。
            # SageMaker の AutomaticModelRegistration が有効な場合、
            # MLflow Model Registry への登録が SageMaker Model Registry にも自動反映される。
            mlflow.pytorch.log_model(
                pytorch_model=model,
                name="pytorch-model",
                signature=signature,
                registered_model_name=model_group_name,
            )
            # モデルファイルを Run の Artifacts にも保存 (MLflow UI で確認可能にする)
            tmp_model_path = "/tmp/model.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                },
                tmp_model_path,
            )
            mlflow.log_artifact(tmp_model_path, artifact_path="model")
            print(
                f"Model registered to MLflow Model Registry: {model_group_name}"
            )

            # sagemaker-mlflow プラグインが出力するリンクには MLflow App の
            # URL プレフィックスが含まれないため、正しいリンクを別途出力する。
            if tracking_url:
                experiment_id = run.info.experiment_id
                run_id = run.info.run_id
                base = f"{tracking_url}/#/experiments"
                print(
                    f"MLflow UI (experiment): {base}/{experiment_id}"
                )
                print(
                    f"MLflow UI (run): {base}/{experiment_id}"
                    f"/runs/{run_id}"
                )

    # -------------------------------------------------------------------------
    # モデル保存
    # -------------------------------------------------------------------------
    # SageMaker は SM_MODEL_DIR の内容を model.tar.gz に圧縮して S3 にアップロードする。
    # Pipeline の RegisterModel ステップと Evaluate ステップがこのアーティファクトを参照する。
    #
    # state_dict だけでなく input_dim / num_classes も一緒に保存する理由:
    # evaluate.py でモデルを復元する際、SimpleClassifier のコンストラクタに
    # input_dim と num_classes を渡す必要があるため。
    # weights_only=True でロードする場合も dict 形式なら安全に復元できる。
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes,
        },
        model_path,
    )
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
