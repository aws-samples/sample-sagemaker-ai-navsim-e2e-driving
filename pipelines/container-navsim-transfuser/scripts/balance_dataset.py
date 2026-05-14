# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
データセットのコマンドバランシングスクリプト

TransFuser学習データのコマンド分布を均等化し、モデルが全コマンド（LEFT/FORWARD/RIGHT）に
反応できるようにします。

使い方:
    python balance_dataset.py \
        --cache-dir /path/to/cache/transfuser \
        --target-ratio 0.5

処理内容:
    1. train/ ディレクトリ内の全サンプルのコマンド分布を調査
    2. 最小クラスのサンプル数に基づいて他のクラスをダウンサンプリング
    3. バランス済みデータを train_balanced/ に保存
    4. 元の train/ は train_original/ にバックアップ
"""

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def analyze_command_distribution(data_dir: str):
    """データディレクトリ内の全サンプルのコマンド分布を分析する"""
    sample_files = sorted(Path(data_dir).glob("*.pt"))

    commands = []
    sample_info = []  # [(filepath, cmd_idx), ...]

    print(f"Analyzing {len(sample_files)} samples...")

    for sample_file in sample_files:
        try:
            data = torch.load(sample_file, map_location="cpu", weights_only=True)
            status = data["status"]
            cmd_idx = torch.argmax(status[:4]).item()
            commands.append(cmd_idx)
            sample_info.append((sample_file, cmd_idx))
        except Exception as e:
            print(f"Warning: Failed to load {sample_file}: {e}")
            continue

    return sample_info, Counter(commands)


def balance_dataset(sample_info, counter, output_dir: str, strategy: str = "equal",
                   exclude_unknown: bool = False, min_samples: int = None):
    """コマンド分布をバランシングする

    Args:
        sample_info: [(filepath, cmd_idx), ...] のリスト
        counter: コマンド分布カウンター
        output_dir: 出力先ディレクトリ
        strategy: バランシング戦略
            - "equal": 完全均等化（全クラスを最小クラスに合わせる）
            - "proportional": 比例的削減（target_ratioベース、廃止予定）
        exclude_unknown: UNKNOWN（Index 3）を除外するか
        min_samples: 各クラスの最小サンプル数（Noneなら最小クラスのサンプル数を使用）
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd_names = {0: "LEFT", 1: "FORWARD", 2: "RIGHT", 3: "UNKNOWN"}

    # 各コマンドのサンプルをグループ化
    cmd_samples = {i: [] for i in range(4)}
    for filepath, cmd_idx in sample_info:
        cmd_samples[cmd_idx].append(filepath)

    # UNKNOWNを除外する場合
    active_classes = [0, 1, 2] if exclude_unknown else [0, 1, 2, 3]
    active_counter = {k: v for k, v in counter.items() if k in active_classes and v > 0}

    if not active_counter:
        print("Error: No samples found in active classes")
        return 0, {}

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
                # 完全均等化：全クラスを同じサンプル数に
                target_counts[cmd_idx] = min(target_count, active_counter[cmd_idx])
            else:
                # proportional戦略（後方互換性のため残す）
                target_counts[cmd_idx] = active_counter[cmd_idx]

    print(f"\nTarget distribution:")
    for cmd_idx in sorted(active_classes):
        if cmd_idx in active_counter:
            original = active_counter[cmd_idx]
            target = target_counts.get(cmd_idx, 0)
            print(f"  {cmd_names[cmd_idx]}: {original} → {target} samples")

    # ダウンサンプリングを実行
    balanced_counter = Counter()
    sample_count = 0

    for cmd_idx, target_count in target_counts.items():
        samples = cmd_samples[cmd_idx]

        if len(samples) <= target_count:
            # 全サンプルを使用
            selected_samples = samples
        else:
            # ランダムダウンサンプリング
            np.random.shuffle(samples)
            selected_samples = samples[:target_count]

        # ファイルをコピー
        for src_path in selected_samples:
            dst_path = Path(output_dir) / f"sample_{sample_count:05d}.pt"
            shutil.copy2(src_path, dst_path)
            balanced_counter[cmd_idx] += 1
            sample_count += 1

    return sample_count, balanced_counter


def main():
    parser = argparse.ArgumentParser(description="Balance TransFuser dataset by command distribution")
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Cache directory containing train/ subdirectory")
    parser.add_argument("--strategy", type=str, default="equal",
                        choices=["equal", "proportional"],
                        help="Balancing strategy: equal (完全均等化) or proportional (比例削減)")
    parser.add_argument("--exclude-unknown", action="store_true",
                        help="Exclude UNKNOWN command (Index 3) from balancing")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="Force minimum samples per class (overrides auto-detection)")
    args = parser.parse_args()

    train_dir = os.path.join(args.cache_dir, "train")
    train_original_dir = os.path.join(args.cache_dir, "train_original")
    train_balanced_dir = os.path.join(args.cache_dir, "train_balanced")

    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} does not exist")
        return

    print("=" * 60)
    print("TransFuser Dataset Balancing")
    print("=" * 60)
    print(f"Source:   {train_dir}")
    print(f"Strategy: {args.strategy}")
    if args.exclude_unknown:
        print(f"Excluding: UNKNOWN command")
    if args.min_samples:
        print(f"Min samples: {args.min_samples}")
    print()

    # Step 1: 元のデータ分布を分析
    print("[Step 1/4] Analyzing original distribution...")
    sample_info, original_counter = analyze_command_distribution(train_dir)

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
    if os.path.exists(train_original_dir):
        print(f"  Warning: {train_original_dir} already exists, skipping backup")
    else:
        shutil.move(train_dir, train_original_dir)
        print(f"  Moved: {train_dir} → {train_original_dir}")

    # Step 3: バランシングを実行
    print(f"\n[Step 3/4] Balancing dataset...")

    # train_original_dir から読み込み、train_balanced_dir に出力
    sample_info_from_original = []
    for filepath, cmd_idx in sample_info:
        # パスを train_original に変更
        new_path = Path(train_original_dir) / filepath.name
        sample_info_from_original.append((new_path, cmd_idx))

    balanced_count, balanced_counter = balance_dataset(
        sample_info_from_original,
        original_counter,
        train_balanced_dir,
        strategy=args.strategy,
        exclude_unknown=args.exclude_unknown,
        min_samples=args.min_samples
    )

    total_balanced = sum(balanced_counter.values())

    print(f"\nBalanced distribution ({total_balanced} samples):")
    for cmd_idx, count in sorted(balanced_counter.items()):
        pct = count / total_balanced * 100 if total_balanced > 0 else 0
        print(f"  {cmd_names[cmd_idx]}: {count} ({pct:.1f}%)")

    # Step 4: バランス済みデータを train/ に配置
    print(f"\n[Step 4/4] Moving balanced data to {train_dir}...")
    shutil.move(train_balanced_dir, train_dir)
    print(f"  Moved: {train_balanced_dir} → {train_dir}")

    print()
    print("=" * 60)
    print("✅ Dataset balancing completed")
    print("=" * 60)
    print(f"Original data backup: {train_original_dir}")
    print(f"Balanced data:        {train_dir}")
    print(f"Samples: {total_original} → {total_balanced}")
    print()


if __name__ == "__main__":
    main()
