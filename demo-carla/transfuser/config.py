# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""CARLA demo configuration for TransFuser."""

# CARLA server
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_TIMEOUT = 10.0

# Simulation
TOWN = "Town04"
FIXED_DELTA = 0.05  # 20 FPS simulation
WEATHER = "ClearNoon"
DURATION_SEC = 60

# Recording camera (third-person view)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOV = 110
CAMERA_TRANSFORM = {"x": -5.5, "z": 2.8, "pitch": -15, "yaw": 0}
VIDEO_FPS = 20

# Perception cameras (3 cameras for TransFuser input)
PERCEPTION_CAM_WIDTH = 1600
PERCEPTION_CAM_HEIGHT = 900
PERCEPTION_CAM_FOV = 70
PERCEPTION_CAM_X = 1.5
PERCEPTION_CAM_Z = 2.4
PERCEPTION_CAM_YAW_LEFT = -60
PERCEPTION_CAM_YAW_RIGHT = 60

# LiDAR
LIDAR_RANGE = 50.0
LIDAR_ROTATION_FREQ = 20
LIDAR_CHANNELS = 64
LIDAR_POINTS_PER_SEC = 600000
LIDAR_X = 1.5
LIDAR_Z = 2.4

# Vehicle (Tesla Model 3 approximate)
WHEELBASE = 2.875  # meters

# Pure pursuit lateral control
MIN_LOOKAHEAD = 3.5
MAX_LOOKAHEAD = 12.0
LOOKAHEAD_GAIN = 1.0

# Longitudinal PID
PID_LONGITUDINAL_KP = 0.8
PID_LONGITUDINAL_KI = 0.15
PID_LONGITUDINAL_KD = 0.05
TARGET_SPEED_MS = 5.5  # m/s (~20 km/h)

# Trajectory (NAVSIM TransFuser: 8 poses at 0.5s intervals = 4s horizon)
NUM_POSES = 8
TRAJ_DT = 0.5

# Road waypoint chain for fallback steering
ROAD_WAYPOINT_SPACING = 3.0
ROAD_WAYPOINT_COUNT = 10

# Domain Adaptation

