# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
EgoMLP データセットのコマンドバランシングスクリプト

EgoMLP学習データのコマンド分布を均等化し、モデルが全コマンド（LEFT/FORWARD/RIGHT）に
反応できるようにします。

使い方:
    python balance_dataset.py \
        --cache-dir /path/to/cache/ego-mlp \
        --strategy equal \
        --exclude-unknown

処理内容:
    1. train_data.npz のコマンド分布を調査
    2. 最小クラスのサンプル数に基づいて他のクラスをダウンサンプリング
    3. バランス済みデータを train_data_balanced.npz に保存
    4. 元の train_data.npz は train_data_original.npz にバックアップ
"""

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


def analyze_command_distribution(npz_path: str):
    """npzファイル内の全サンプルのコマンド分布を分析する"""
    data = np.load(npz_path)
    features = data['features']  # [N, 8]: [vx, vy, ax, ay, cmd(4)]

    commands = []
    for i in range(len(features)):
        cmd = features[i, 4:8]  # Last 4 dimensions are command one-hot
        cmd_idx = np.argmax(cmd)
        commands.append(cmd_idx)

    return features, data['targets'], Counter(commands)


def balance_dataset(features, targets, counter, strategy: str = "equal",
                   exclude_unknown: bool = False, min_samples: int = None):
    """コマンド分布をバランシングする

    Args:
        features: [N, 8] numpy array
        targets: [N, num_poses, 3] numpy array
        counter: コマンド分布カウンター
        strategy: バランシング戦略 ("equal" = 完全均等化)
        exclude_unknown: UNKNOWN（Index 3）を除外するか
        min_samples: 各クラスの最小サンプル数
    """
    cmd_names = {0: "LEFT", 1: "FORWARD", 2: "RIGHT", 3: "UNKNOWN"}

    # 各コマンドのサンプルをグループ化
    cmd_samples = {i: [] for i in range(4)}
    for i in range(len(features)):
        cmd = features[i, 4:8]
        cmd_idx = np.argmax(cmd)
        cmd_samples[cmd_idx].append(i)

    # UNKNOWNを除外する場合
    active_classes = [0, 1, 2] if exclude_unknown else [0, 1, 2, 3]
    active_counter = {k: v for k, v in counter.items() if k in active_classes and v > 0}

    if not active_counter:
        print("Error: No samples found in active classes")
        return None, None, {}

    # 最小クラスのサンプル数を決定
    if min_samples is not None:
        target_count = min_samples
        print(f"\nBalancing strategy: {strategy} (forced min_samples={min_samples})")
    else:
        target_count = min(active_counter.values())
        print(f"\nBalancing strategy: {strategy}")

    print(f"  Target samples per class: {target_count}")

    # 各クラスの目標数を設定
    target_counts = {}
    for cmd_idx in active_classes:
        if cmd_idx in active_counter:
            if strategy == "equal":
                target_counts[cmd_idx] = min(target_count, active_counter[cmd_idx])
            else:
                target_counts[cmd_idx] = active_counter[cmd_idx]

    print(f"\nTarget distribution:")
    for cmd_idx in sorted(active_classes):
        if cmd_idx in active_counter:
            original = active_counter[cmd_idx]
            target = target_counts.get(cmd_idx, 0)
            print(f"  {cmd_names[cmd_idx]}: {original} → {target} samples")

    # ダウンサンプリングを実行
    selected_indices = []
    balanced_counter = Counter()

    for cmd_idx, target_count in target_counts.items():
        samples = cmd_samples[cmd_idx]

        if len(samples) <= target_count:
            # 全サンプルを使用
            selected_samples = samples
        else:
            # ランダムダウンサンプリング
            np.random.shuffle(samples)
            selected_samples = samples[:target_count]

        selected_indices.extend(selected_samples)
        balanced_counter[cmd_idx] = len(selected_samples)

    # インデックスでソート（時系列順を保持）
    selected_indices.sort()

    # バランス済みデータを作成
    balanced_features = features[selected_indices]
    balanced_targets = targets[selected_indices]

    return balanced_features, balanced_targets, balanced_counter


def main():
    parser = argparse.ArgumentParser(description="Balance EgoMLP dataset by command distribution")
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Cache directory containing train_data.npz")
    parser.add_argument("--strategy", type=str, default="equal",
                        choices=["equal", "proportional"],
                        help="Balancing strategy: equal (完全均等化) or proportional (比例削減)")
    parser.add_argument("--exclude-unknown", action="store_true",
                        help="Exclude UNKNOWN command (Index 3) from balancing")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="Force minimum samples per class (overrides auto-detection)")
    args = parser.parse_args()

    train_path = os.path.join(args.cache_dir, "train_data.npz")
    train_original_path = os.path.join(args.cache_dir, "train_data_original.npz")
    train_balanced_path = os.path.join(args.cache_dir, "train_data_balanced.npz")

    if not os.path.exists(train_path):
        print(f"Error: {train_path} does not exist")
        return

    print("=" * 60)
    print("EgoMLP Dataset Balancing")
    print("=" * 60)
    print(f"Source:   {train_path}")
    print(f"Strategy: {args.strategy}")
    if args.exclude_unknown:
        print(f"Excluding: UNKNOWN command")
    if args.min_samples:
        print(f"Min samples: {args.min_samples}")
    print()

    # Step 1: 元のデータ分布を分析
    print("[Step 1/4] Analyzing original distribution...")
    features, targets, original_counter = analyze_command_distribution(train_path)

    cmd_names = {0: "LEFT", 1: "FORWARD", 2: "RIGHT", 3: "UNKNOWN"}
    total_original = sum(original_counter.values())

    print(f"\nOriginal distribution ({total_original} samples):")
    for cmd_idx, count in sorted(original_counter.items()):
        pct = count / total_original * 100 if total_original > 0 else 0
        print(f"  {cmd_names[cmd_idx]}: {count} ({pct:.1f}%)")

    # 不均衡度を計算
    if original_counter:
        max_count = max(original_counter.values())
        min_count = min(original_counter.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\nImbalance ratio: {imbalance_ratio:.1f}x (max/min)")

        if imbalance_ratio < 2.0:
            print(f"  → Distribution is already reasonably balanced (< 2.0x)")
            print(f"  → Skipping balancing")
            return

    # Step 2: 元のデータをバックアップ
    print(f"\n[Step 2/4] Backing up original data...")
    if os.path.exists(train_original_path):
        print(f"  Warning: {train_original_path} already exists, skipping backup")
    else:
        shutil.copy2(train_path, train_original_path)
        print(f"  Copied: {train_path} → {train_original_path}")

    # Step 3: バランシングを実行
    print(f"\n[Step 3/4] Balancing dataset...")

    balanced_features, balanced_targets, balanced_counter = balance_dataset(
        features,
        targets,
        original_counter,
        strategy=args.strategy,
        exclude_unknown=args.exclude_unknown,
        min_samples=args.min_samples
    )

    if balanced_features is None:
        print("  Balancing failed")
        return

    total_balanced = sum(balanced_counter.values())

    print(f"\nBalanced distribution ({total_balanced} samples):")
    for cmd_idx, count in sorted(balanced_counter.items()):
        pct = count / total_balanced * 100 if total_balanced > 0 else 0
        print(f"  {cmd_names[cmd_idx]}: {count} ({pct:.1f}%)")

    # バランス済みデータを保存
    np.savez(train_balanced_path, features=balanced_features, targets=balanced_targets)
    print(f"\nSaved balanced data to: {train_balanced_path}")

    # Step 4: バランス済みデータを train_data.npz に配置
    print(f"\n[Step 4/4] Replacing train_data.npz with balanced data...")
    shutil.move(train_balanced_path, train_path)
    print(f"  Moved: {train_balanced_path} → {train_path}")

    print()
    print("=" * 60)
    print("✅ Dataset balancing completed")
    print("=" * 60)
    print(f"Original data backup: {train_original_path}")
    print(f"Balanced data:        {train_path}")
    print(f"Samples: {total_original} → {total_balanced}")
    print()


if __name__ == "__main__":
    main()
