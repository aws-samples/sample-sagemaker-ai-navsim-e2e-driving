# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# This file contains code derived from NAVSIM
# (https://github.com/autonomousvision/navsim)
# Copyright: University of Tübingen, Tübingen AI Center, and contributors
# Original License: Apache License 2.0
#
# Changes from the original:
#   - Removed nuplan imports (TrackedObjectType, SemanticMapLayer)
#   - Replaced BEV semantic class definitions with integer constants

from dataclasses import dataclass


@dataclass
class TransfuserConfig:
    """Global TransFuser config."""

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"

    pretrained: bool = True

    latent: bool = False

    lidar_seq_len: int = 1
    use_ground_plane: bool = False

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp: int = 4
    n_layer: int = 2
    n_head: int = 4
    n_scale: int = 4
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    gpt_linear_layer_init_mean: float = 0.0
    gpt_linear_layer_init_std: float = 0.02
    gpt_layer_norm_init_weight: float = 1.0

    perspective_downsample_factor: int = 1
    transformer_decoder_join: bool = True
    detect_boxes: bool = True
    use_bev_semantic: bool = True
    use_semantic: bool = False
    use_depth: bool = False
    add_features: bool = True

    # Transformer decoder
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # Detection
    num_bounding_boxes: int = 30

    # Loss weights
    trajectory_weight: float = 10.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 10.0

    # BEV semantic (nuplan 依存を排除し、整数クラス ID で直接定義)
    num_bev_classes: int = 7
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2

    # Trajectory
    num_poses: int = 8
