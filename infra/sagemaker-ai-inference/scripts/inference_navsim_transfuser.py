# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
SageMaker PyTorch Inference Script for NAVSIM Transfuser.

EgoStatus (速度・加速度・コマンド) のみを入力として受け取り、
カメラ・LiDAR はダミー入力で推論する簡易版。

公式版チェックポイント (config キー) と Lite 版チェックポイント (params キー) の
両方に対応する。
"""

import json
import os
import sys

import torch

# code/ ディレクトリにモデル定義ファイルが同梱されている
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

COMMAND_MAP = {"LEFT": 0, "FORWARD": 1, "STRAIGHT": 1, "RIGHT": 2, "UNKNOWN": 3}


def model_fn(model_dir):
    ckpt = torch.load(f"{model_dir}/model.pth", map_location="cpu", weights_only=True)

    if "config" in ckpt:
        # 公式版チェックポイント
        from transfuser_config import TransfuserConfig
        from transfuser_model import TransfuserModel

        cfg = ckpt["config"]
        config = TransfuserConfig(
            latent=cfg["latent"],
            image_architecture=cfg["image_architecture"],
            lidar_architecture=cfg["lidar_architecture"],
            num_poses=cfg["num_poses"],
            num_bounding_boxes=cfg["num_bounding_boxes"],
        )
        model = TransfuserModel(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return {"model": model, "config": config, "version": "official"}
    else:
        # Lite 版チェックポイント (後方互換)
        from transfuser_model_lite import TransfuserModelLite

        params = ckpt["params"]
        model = TransfuserModelLite(
            hidden_dim=params["hidden_dim"],
            num_poses=params["num_poses"],
            latent=params.get("latent", False),
            num_bboxes=params.get("num_bboxes", 30),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return {"model": model, "params": params, "version": "lite"}


def input_fn(request_body, request_content_type):
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    data = json.loads(request_body)
    vx, vy = data.get("velocity", [0.0, 0.0])
    ax, ay = data.get("acceleration", [0.0, 0.0])
    cmd = data.get("command", "FORWARD")

    cmd_onehot = [0.0, 0.0, 0.0, 0.0]
    cmd_onehot[COMMAND_MAP.get(cmd.upper(), 3)] = 1.0

    status = torch.tensor([[vx, vy, ax, ay] + cmd_onehot], dtype=torch.float32)
    camera = torch.zeros(1, 3, 256, 1024)
    lidar = torch.zeros(1, 1, 256, 256)
    return {"camera": camera, "lidar": lidar, "status": status}


def predict_fn(input_data, model_dict):
    model = model_dict["model"]
    with torch.no_grad():
        if model_dict["version"] == "official":
            features = {
                "camera_feature": input_data["camera"],
                "status_feature": input_data["status"],
            }
            config = model_dict["config"]
            if not config.latent:
                features["lidar_feature"] = input_data["lidar"]
            output = model(features)
        else:
            output = model(
                input_data["camera"], input_data["lidar"], input_data["status"]
            )
    return output["trajectory"]


def output_fn(prediction, response_content_type):
    traj = prediction[0].numpy()
    return json.dumps({
        "trajectory": [[float(p[0]), float(p[1])] for p in traj],
        "trajectory_with_heading": [[float(p[0]), float(p[1]), float(p[2])] for p in traj],
    })
