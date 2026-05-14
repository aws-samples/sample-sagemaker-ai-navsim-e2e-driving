# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
SageMaker PyTorch Inference Script for NAVSIM EgoStatusMLP.

SageMaker PyTorch Serving Container が呼び出す 4 つの関数を定義する。
"""

import json
import torch
import torch.nn as nn


class EgoStatusMLPModel(nn.Module):
    INPUT_DIM = 8

    def __init__(self, hidden_dim=128, num_poses=8):
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

    def forward(self, x):
        return self.mlp(x).reshape(-1, self.num_poses, 3)


COMMAND_MAP = {"LEFT": 0, "FORWARD": 1, "STRAIGHT": 1, "RIGHT": 2, "UNKNOWN": 3}


def model_fn(model_dir):
    ckpt = torch.load(f"{model_dir}/model.pth", map_location="cpu", weights_only=True)
    model = EgoStatusMLPModel(
        hidden_dim=ckpt["hidden_dim"], num_poses=ckpt["num_poses"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {"model": model, "num_poses": ckpt["num_poses"]}


def input_fn(request_body, request_content_type):
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    data = json.loads(request_body)
    vx, vy = data.get("velocity", [0.0, 0.0])
    ax, ay = data.get("acceleration", [0.0, 0.0])
    cmd = data.get("command", "FORWARD")

    cmd_onehot = [0.0, 0.0, 0.0, 0.0]
    cmd_idx = COMMAND_MAP.get(cmd.upper(), 3)
    cmd_onehot[cmd_idx] = 1.0

    features = [vx, vy, ax, ay] + cmd_onehot
    return torch.tensor([features], dtype=torch.float32)


def predict_fn(input_data, model_dict):
    model = model_dict["model"]
    with torch.no_grad():
        output = model(input_data)
    return output


def output_fn(prediction, response_content_type):
    traj = prediction[0].numpy()  # [num_poses, 3]
    result = {
        "trajectory": [[float(p[0]), float(p[1])] for p in traj],
        "trajectory_with_heading": [[float(p[0]), float(p[1]), float(p[2])] for p in traj],
    }
    return json.dumps(result)
