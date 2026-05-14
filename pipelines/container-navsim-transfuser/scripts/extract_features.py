# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
NAVSIM データセットから Transfuser 用の特徴量を抽出する。

navsim devkit の SceneLoader を使用してカメラ画像、LiDAR BEV、
EgoStatus、GT 軌跡を抽出し、pt (PyTorch tensor) 形式で保存する。

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
import traceback
from pathlib import Path
import numpy as np
import torch


def extract_with_navsim(data_root: str, navsim_root: str, split: str, cache_dir: str, train_ratio: float = 0.8):
    """navsim devkit を使用して Transfuser 用特徴量を抽出し、逐次的に保存する。"""
    sys.path.insert(0, navsim_root)

    try:
        from navsim.common.dataloader import SceneLoader
        from navsim.common.dataclasses import SceneFilter, SensorConfig
    except ImportError as e:
        print(f"navsim import failed: {e}")
        return 0

    sensor_config = SensorConfig.build_all_sensors()

    try:
        scene_loader = SceneLoader(
            data_path=Path(os.path.join(data_root, "navsim_logs", split)),
            sensor_blobs_path=Path(os.path.join(data_root, "sensor_blobs", split)),
            scene_filter=SceneFilter(),
            sensor_config=sensor_config,
        )
    except Exception as e:
        print(f"SceneLoader initialization failed: {e}")
        return 0

    tokens = scene_loader.tokens
    print(f"Processing {len(tokens)} scenes from {split} split...")

    # Train/Test ディレクトリを作成
    train_dir = os.path.join(cache_dir, "train")
    test_dir = os.path.join(cache_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # ランダムシャッフル用のインデックス
    n_total = len(tokens)
    n_train = int(n_total * train_ratio)
    indices = np.random.permutation(n_total)
    train_indices = set(indices[:n_train].tolist())

    train_count = 0
    test_count = 0

    for i, token in enumerate(tokens):
        try:
            scene = scene_loader.get_scene_from_token(token)
            agent_input = scene.get_agent_input()

            # 最新フレーム (リストの末尾) を使用
            cameras = agent_input.cameras[-1]
            lidar_frame = agent_input.lidars[-1]
            ego = agent_input.ego_statuses[-1]

            # カメラ画像: 前方 3 台をスティッチしてリサイズ (3, 256, 1024)
            CAM_H, CAM_W = 256, 1024
            cam_images = []
            for cam_key in ["cam_l0", "cam_f0", "cam_r0"]:
                cam_obj = getattr(cameras, cam_key, None)
                if cam_obj is not None and hasattr(cam_obj, 'image'):
                    img = cam_obj.image
                    # numpy配列のチェック: Noneでなく、サイズが0より大きいか
                    if img is not None and (not isinstance(img, np.ndarray) or img.size > 0):
                        cam_images.append(img)
            if cam_images:
                stitched = np.concatenate(cam_images, axis=1)  # 横方向に結合
                # [H, W, 3] -> [3, H, W], 正規化
                camera = torch.tensor(stitched, dtype=torch.float32).permute(2, 0, 1) / 255.0
                # 統一サイズにリサイズ
                camera = torch.nn.functional.interpolate(
                    camera.unsqueeze(0), size=(CAM_H, CAM_W), mode="area"
                ).squeeze(0)
            else:
                camera = torch.zeros(3, CAM_H, CAM_W)

            # LiDAR BEV ヒストグラム
            lidar_pc = lidar_frame.lidar_pc
            # numpy配列のチェック: Noneでなく、サイズが0より大きいか
            if lidar_pc is not None and (not isinstance(lidar_pc, np.ndarray) or lidar_pc.size > 0):
                # 点群を BEV ヒストグラムに変換 (256x256)
                bev = _pointcloud_to_bev(lidar_pc, resolution=256)
                lidar = torch.tensor(bev, dtype=torch.float32).unsqueeze(0)
            else:
                lidar = torch.zeros(1, 256, 256)

            # EgoStatus
            velocity = np.array([ego.ego_velocity[0], ego.ego_velocity[1]])
            accel = np.array([ego.ego_acceleration[0], ego.ego_acceleration[1]])
            cmd = ego.driving_command

            # OpenSceneでは既にone-hot encodingされている場合がある
            if isinstance(cmd, np.ndarray) and cmd.shape == (4,):
                # 既にone-hot encoding → そのまま使用
                cmd_onehot = cmd.astype(np.float32)
            else:
                # スカラー値 → one-hot encodingに変換
                if isinstance(cmd, (np.ndarray, list, tuple)):
                    cmd_value = int(cmd[0]) if len(cmd) > 0 else 0
                else:
                    cmd_value = int(cmd)
                cmd_onehot = np.zeros(4, dtype=np.float32)
                cmd_onehot[min(cmd_value, 3)] = 1.0

            status = torch.tensor(
                np.concatenate([cmd_onehot, velocity, accel]),
                dtype=torch.float32,
            )

            # GT 軌跡
            traj_obj = scene.get_future_trajectory()
            # Trajectoryオブジェクトからnumpy配列を取得
            if hasattr(traj_obj, 'poses'):
                traj_data = traj_obj.poses
            elif hasattr(traj_obj, 'data'):
                traj_data = traj_obj.data
            elif isinstance(traj_obj, np.ndarray):
                traj_data = traj_obj
            else:
                # オブジェクトを配列に変換
                traj_data = np.array(traj_obj)
            trajectory = torch.tensor(
                traj_data[:8],  # 8 poses
                dtype=torch.float32,
            )

            sample = {
                "camera": camera,
                "lidar": lidar,
                "status": status,
                "trajectory": trajectory,
            }

            # 補助タスク用ターゲットの抽出
            # agent_states: 周囲エージェントの BBox (x, y, heading, length, width)
            # agent_labels: 有効フラグ (1=有効, 0=パディング)
            # bev_semantic_map: BEV セマンティックマップ (クラス ID)
            NUM_BBOXES = 30
            BEV_H, BEV_W = 128, 256

            try:
                tracked = scene.get_tracked_objects()
                agent_states = torch.zeros(NUM_BBOXES, 5)
                agent_labels = torch.zeros(NUM_BBOXES)
                if tracked is not None:
                    boxes = []
                    for obj in tracked:
                        if hasattr(obj, 'box'):
                            b = obj.box
                            boxes.append([b.center_x, b.center_y, b.heading, b.length, b.width])
                        elif hasattr(obj, 'center'):
                            boxes.append([
                                obj.center[0], obj.center[1],
                                getattr(obj, 'heading', 0.0),
                                getattr(obj, 'length', 4.0),
                                getattr(obj, 'width', 2.0),
                            ])
                    n = min(len(boxes), NUM_BBOXES)
                    if n > 0:
                        agent_states[:n] = torch.tensor(boxes[:n], dtype=torch.float32)
                        agent_labels[:n] = 1.0
            except Exception:
                agent_states = torch.zeros(NUM_BBOXES, 5)
                agent_labels = torch.zeros(NUM_BBOXES)

            bev_semantic_map = _generate_bev_semantic_map(
                scene, tracked if 'tracked' in dir() else None, BEV_H, BEV_W
            )

            sample["agent_states"] = agent_states
            sample["agent_labels"] = agent_labels
            sample["bev_semantic_map"] = bev_semantic_map

            # 直接ファイルに保存（メモリ節約のため）
            if i in train_indices:
                torch.save(sample, os.path.join(train_dir, f"sample_{train_count:05d}.pt"))
                train_count += 1
            else:
                torch.save(sample, os.path.join(test_dir, f"sample_{test_count:05d}.pt"))
                test_count += 1

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(tokens)} scenes (train: {train_count}, test: {test_count})")

        except Exception as e:
            if i < 5:  # 最初の5件のみ詳細を表示
                print(f"  Warning: Failed to process token {token}:")
                print(f"    Error: {e}")
                traceback.print_exc()
            else:
                print(f"  Warning: Failed to process token {token}: {e}")
            continue

    print(f"\nTrain: {train_count} samples -> {train_dir}")
    print(f"Test:  {test_count} samples -> {test_dir}")
    return train_count + test_count


def _generate_bev_semantic_map(scene, tracked_objects, bev_h: int = 128, bev_w: int = 256,
                               x_range: float = 32.0, y_range: float = 32.0) -> torch.Tensor:
    """BEV セマンティックマップを生成する。

    NAVSIM 公式の TransfuserConfig.bev_semantic_classes に準拠:
        0: background
        1: road (lane, intersection)
        2: walkways
        3: centerline
        4: static objects (barriers, cones)
        5: vehicles
        6: pedestrians

    map 情報が取得できない場合は tracked objects のみからマップを生成する。
    """
    bev = np.zeros((bev_h, bev_w), dtype=np.int64)

    def world_to_bev(x, y):
        """ワールド座標 (自車中心) を BEV ピクセル座標に変換する。"""
        bx = int((x + x_range) / (2 * x_range) * bev_w)
        by = int((y + y_range) / (2 * y_range) * bev_h)
        return np.clip(bx, 0, bev_w - 1), np.clip(by, 0, bev_h - 1)

    def fill_box(bev, cx, cy, heading, length, width, class_id):
        """回転した BBox を BEV マップに描画する。"""
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        corners = []
        for dx, dy in [(-length/2, -width/2), (length/2, -width/2),
                        (length/2, width/2), (-length/2, width/2)]:
            rx = cx + dx * cos_h - dy * sin_h
            ry = cy + dx * sin_h + dy * cos_h
            corners.append(world_to_bev(rx, ry))
        # 簡易的に BBox の AABB で塗りつぶし
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        x_min, x_max = max(0, min(xs)), min(bev_w - 1, max(xs))
        y_min, y_max = max(0, min(ys)), min(bev_h - 1, max(ys))
        if x_min <= x_max and y_min <= y_max:
            bev[y_min:y_max + 1, x_min:x_max + 1] = class_id

    # tracked objects から vehicles (5) と pedestrians (6) を描画
    if tracked_objects is not None:
        try:
            for obj in tracked_objects:
                obj_type = getattr(obj, 'tracked_object_type', None)
                if obj_type is None:
                    obj_type = getattr(obj, 'type', None)
                type_name = str(obj_type).lower() if obj_type is not None else ""

                if hasattr(obj, 'box'):
                    b = obj.box
                    cx, cy, h = b.center_x, b.center_y, b.heading
                    l, w = b.length, b.width
                elif hasattr(obj, 'center'):
                    cx, cy = obj.center[0], obj.center[1]
                    h = getattr(obj, 'heading', 0.0)
                    l = getattr(obj, 'length', 4.0)
                    w = getattr(obj, 'width', 2.0)
                else:
                    continue

                if "pedestrian" in type_name:
                    fill_box(bev, cx, cy, h, l, w, 6)
                elif "vehicle" in type_name:
                    fill_box(bev, cx, cy, h, l, w, 5)
                elif any(k in type_name for k in ["barrier", "cone", "sign", "generic"]):
                    fill_box(bev, cx, cy, h, l, w, 4)
                else:
                    fill_box(bev, cx, cy, h, l, w, 5)  # デフォルトは vehicle
        except Exception:
            pass

    # map 情報から road (1) を描画
    try:
        map_api = scene.map_api if hasattr(scene, 'map_api') else None
        if map_api is not None and hasattr(map_api, 'get_proximal_map_objects'):
            ego_pose = scene.get_ego_state() if hasattr(scene, 'get_ego_state') else None
            if ego_pose is not None:
                from nuplan.common.maps.abstract_map import SemanticMapLayer
                layers = [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]
                map_objects = map_api.get_proximal_map_objects(
                    ego_pose.center, x_range, layers
                )
                for layer, objects in map_objects.items():
                    for obj in objects:
                        if hasattr(obj, 'polygon'):
                            coords = np.array(obj.polygon.exterior.coords)
                            for coord in coords:
                                bx, by = world_to_bev(
                                    coord[0] - ego_pose.center.x,
                                    coord[1] - ego_pose.center.y,
                                )
                                if bev[by, bx] == 0:
                                    bev[by, bx] = 1
    except Exception:
        pass  # map API が利用できない場合はスキップ

    return torch.tensor(bev, dtype=torch.long)


def _pointcloud_to_bev(pc: np.ndarray, resolution: int = 256,
                        x_range: float = 50.0, y_range: float = 50.0) -> np.ndarray:
    """点群を BEV ヒストグラムに変換する。"""
    x, y = pc[:, 0], pc[:, 1]
    mask = (np.abs(x) < x_range) & (np.abs(y) < y_range)
    x, y = x[mask], y[mask]
    xi = ((x + x_range) / (2 * x_range) * resolution).astype(int).clip(0, resolution - 1)
    yi = ((y + y_range) / (2 * y_range) * resolution).astype(int).clip(0, resolution - 1)
    bev = np.zeros((resolution, resolution), dtype=np.float32)
    np.add.at(bev, (xi, yi), 1)
    bev = np.clip(bev / max(bev.max(), 1), 0, 1)
    return bev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--navsim-root", required=True)
    parser.add_argument("--split", default="mini")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    n_samples = extract_with_navsim(
        args.data_root,
        args.navsim_root,
        args.split,
        args.cache_dir,
        args.train_ratio
    )

    if n_samples == 0:
        print("❌ Feature extraction failed. No samples extracted.")
        sys.exit(1)

    print(f"\n✅ Successfully extracted {n_samples} samples")


if __name__ == "__main__":
    main()
