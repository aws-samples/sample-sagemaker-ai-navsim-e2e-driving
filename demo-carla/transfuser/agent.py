# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
TransFuser Agent: loads NAVSIM TransFuser model and runs inference
with camera, LiDAR, and ego status inputs from CARLA.

The model architecture is imported from pipelines/container-navsim-transfuser/.
"""

import os
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms

# Add pipeline container to path for model imports
_PIPELINE_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "pipelines", "container-navsim-transfuser")
)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from transfuser_config import TransfuserConfig  # noqa: E402
from transfuser_model import TransfuserModel  # noqa: E402

COMMAND_MAP = {"LEFT": 0, "FORWARD": 1, "STRAIGHT": 1, "RIGHT": 2, "UNKNOWN": 3}


class TransfuserAgent:
    """Agent that predicts trajectory using TransFuser with camera + LiDAR + ego status.

    Supports local mode only (model.pth loaded directly on GPU/CPU).
    """

    def __init__(self, model_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        cfg_dict = ckpt["config"]
        config = TransfuserConfig(
            pretrained=False,
            latent=cfg_dict.get("latent", False),
            image_architecture=cfg_dict.get("image_architecture", "resnet34"),
            lidar_architecture=cfg_dict.get("lidar_architecture", "resnet34"),
            num_poses=cfg_dict.get("num_poses", 8),
            num_bounding_boxes=cfg_dict.get("num_bounding_boxes", 30),
        )
        self.config = config
        self.model = TransfuserModel(config)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval().to(self.device)
        self._to_tensor = transforms.ToTensor()
        print(f"TransFuser loaded (latent={config.latent}, device={self.device})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, camera_images, lidar_points, velocity, acceleration,
                command="FORWARD"):
        """Predict future trajectory from sensor data and ego state.

        Args:
            camera_images: dict {"l0": HxWx3, "f0": HxWx3, "r0": HxWx3} BGR uint8
            lidar_points: Nx3 numpy array (x, y, z) in CARLA coords
            velocity: [vx, vy] in m/s (ego-local, nuPlan convention: x=fwd, y=left)
            acceleration: [ax, ay] in m/s^2 (ego-local, nuPlan convention)
            command: "LEFT", "FORWARD", or "RIGHT"

        Returns:
            list of [x, y] waypoints in ego-local frame
        """
        cam_feat = self._build_camera_feature(camera_images)
        lidar_feat = self._build_lidar_feature(lidar_points)
        status_feat = self._build_status_feature(velocity, acceleration, command)

        features = {
            "camera_feature": cam_feat.unsqueeze(0).to(self.device),
            "status_feature": status_feat.unsqueeze(0).to(self.device),
        }
        if not self.config.latent:
            features["lidar_feature"] = lidar_feat.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(features)
        traj = output["trajectory"][0].cpu().numpy()  # [num_poses, 3]
        return [[float(p[0]), float(p[1])] for p in traj]

    # ------------------------------------------------------------------
    # Feature builders (match pipelines/container-navsim-transfuser/scripts/extract_features.py)
    # ------------------------------------------------------------------

    def _build_camera_feature(self, camera_images):
        """Stitch 3 cameras and resize to [3, 256, 1024]."""
        imgs = []
        for key in ["l0", "f0", "r0"]:
            img = camera_images.get(key)
            if img is not None:
                # CARLA gives BGR, convert to RGB
                imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not imgs:
            return torch.zeros(3, 256, 1024)
        stitched = np.concatenate(imgs, axis=1)
        resized = cv2.resize(stitched, (1024, 256), interpolation=cv2.INTER_AREA)
        return self._to_tensor(resized)  # [3, 256, 1024], float32 [0,1]

    def _build_lidar_feature(self, lidar_points):
        """Convert point cloud to BEV histogram [1, 256, 256].

        Matches _pointcloud_to_bev in extract_features.py.
        """
        if lidar_points is None or len(lidar_points) == 0:
            return torch.zeros(1, 256, 256)

        pc = lidar_points.copy()
        # CARLA coords: x=fwd, y=right, z=up
        # nuPlan coords: x=fwd, y=left, z=up
        pc[:, 1] = -pc[:, 1]

        resolution = 256
        x_range, y_range = 50.0, 50.0
        x, y = pc[:, 0], pc[:, 1]
        mask = (np.abs(x) < x_range) & (np.abs(y) < y_range)
        x, y = x[mask], y[mask]

        xi = ((x + x_range) / (2 * x_range) * resolution).astype(int).clip(0, resolution - 1)
        yi = ((y + y_range) / (2 * y_range) * resolution).astype(int).clip(0, resolution - 1)
        bev = np.zeros((resolution, resolution), dtype=np.float32)
        np.add.at(bev, (xi, yi), 1)
        bev = np.clip(bev / max(bev.max(), 1), 0, 1)
        return torch.tensor(bev, dtype=torch.float32).unsqueeze(0)  # [1, 256, 256]

    @staticmethod
    def _build_status_feature(velocity, acceleration, command):
        """Build [8] status vector: [cmd_onehot(4), vx, vy, ax, ay].

        Order matches extract_features.py (CORRECTED: command, velocity, accel).
        """
        vx, vy = velocity
        ax, ay = acceleration
        cmd_onehot = [0.0] * 4
        cmd_onehot[COMMAND_MAP.get(command.upper(), 3)] = 1.0
        return torch.tensor(cmd_onehot + [vx, vy, ax, ay], dtype=torch.float32)
