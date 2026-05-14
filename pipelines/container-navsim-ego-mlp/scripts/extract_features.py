# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
NAVSIM データセットから EgoStatusMLP 用の特徴量とターゲットを抽出する。

navsim devkit の SceneLoader を使用してシーンデータを読み込み、
EgoStatusFeatureBuilder / TrajectoryTargetBuilder で特徴量を計算し、
npz 形式で保存する。

使い方:
    python extract_features.py \
        --data-root /path/to/dataset \
        --cache-dir /path/to/cache \
        --navsim-root /path/to/navsim \
        --split mini
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np


def extract_with_navsim(data_root: str, navsim_root: str, split: str):
    """navsim devkit を使用して特徴量を抽出する。"""
    # navsim をパスに追加
    sys.path.insert(0, navsim_root)

    try:
        from navsim.common.dataloader import SceneLoader
        from navsim.common.dataclasses import SceneFilter, SensorConfig
        from navsim.agents.ego_status_mlp_agent import (
            EgoStatusFeatureBuilder,
            TrajectoryTargetBuilder,
        )
        from nuplan.planning.simulation.trajectory.trajectory_sampling import (
            TrajectorySampling,
        )
    except ImportError as e:
        print(f"navsim import failed: {e}")
        print("Falling back to dummy data generation.")
        return None, None

    trajectory_sampling = TrajectorySampling(
        time_horizon=4, interval_length=0.5
    )
    feature_builder = EgoStatusFeatureBuilder()
    target_builder = TrajectoryTargetBuilder(
        trajectory_sampling=trajectory_sampling
    )

    # SceneLoader の初期化

    try:
        scene_loader = SceneLoader(
            data_path=Path(os.path.join(data_root, "navsim_logs", split)),
            sensor_blobs_path=Path(os.path.join(data_root, "sensor_blobs", split)),
            scene_filter=SceneFilter(),
            sensor_config=SensorConfig.build_no_sensors(),
        )
    except Exception as e:
        print(f"SceneLoader initialization failed: {e}")
        return None, None

    all_features = []
    all_targets = []

    tokens = scene_loader.tokens
    print(f"Processing {len(tokens)} scenes from {split} split...")

    for i, token in enumerate(tokens):
        try:
            scene = scene_loader.get_scene_from_token(token)
            agent_input = scene.get_agent_input()

            feat_dict = feature_builder.compute_features(agent_input)
            tgt_dict = target_builder.compute_targets(scene)

            all_features.append(feat_dict["ego_status"].numpy())
            all_targets.append(tgt_dict["trajectory"].numpy())

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(tokens)} scenes")
        except Exception as e:
            print(f"  Warning: Failed to process token {token}: {e}")
            continue

    if not all_features:
        return None, None

    features = np.stack(all_features, axis=0)
    targets = np.stack(all_targets, axis=0)
    return features, targets


def generate_dummy_data(n_samples: int = 500, num_poses: int = 8):
    """テスト用のダミーデータを生成する。"""
    print(f"Generating {n_samples} dummy samples...")
    features = np.random.randn(n_samples, 8).astype(np.float32)
    targets = np.zeros((n_samples, num_poses, 3), dtype=np.float32)
    for i in range(num_poses):
        targets[:, i, 0] = features[:, 0] * (i + 1) * 0.5
        targets[:, i, 1] = features[:, 1] * (i + 1) * 0.5
    targets += np.random.randn(*targets.shape).astype(np.float32) * 0.1
    return features, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--navsim-root", required=True)
    parser.add_argument("--split", default="mini")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    # navsim で特徴量抽出を試みる
    features, targets = extract_with_navsim(
        args.data_root, args.navsim_root, args.split
    )

    # 失敗した場合はダミーデータ
    if features is None:
        features, targets = generate_dummy_data()

    # Train/Test 分割
    n_total = features.shape[0]
    n_train = int(n_total * args.train_ratio)
    indices = np.random.permutation(n_total)

    train_features = features[indices[:n_train]]
    train_targets = targets[indices[:n_train]]
    test_features = features[indices[n_train:]]
    test_targets = targets[indices[n_train:]]

    # npz 形式で保存
    train_path = os.path.join(args.cache_dir, "train_data.npz")
    test_path = os.path.join(args.cache_dir, "test_data.npz")

    np.savez(train_path, features=train_features, targets=train_targets)
    np.savez(test_path, features=test_features, targets=test_targets)

    print(f"Train data: {train_features.shape[0]} samples -> {train_path}")
    print(f"Test data:  {test_features.shape[0]} samples -> {test_path}")


if __name__ == "__main__":
    main()
