# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
CARLA simulation demo with NAVSIM TransFuser.

Spawns a vehicle with 3 RGB cameras and LiDAR in CARLA, drives it using
TransFuser trajectory predictions, and records a video.

Usage:
    python run.py --model model/model.pth
    python run.py --model model/model.pth --town Town04 --duration 30
"""

import argparse
import math
import os
import random
import subprocess
import sys

# NumPy 2.x と opencv-python の非互換を自動修正
try:
    import numpy as np
    if int(np.__version__.split(".")[0]) >= 2:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy<2"])
        print("Downgraded numpy to <2 for opencv compatibility. Re-run this script.", flush=True)
        sys.exit(0)
except ImportError:
    pass

import carla
import numpy as np

import config as cfg
from agent import TransfuserAgent
from pid_controller import PIDController
from recorder import Recorder


# ---------------------------------------------------------------------------
# Sensor data buffers
# ---------------------------------------------------------------------------
camera_data = {"l0": None, "f0": None, "r0": None}
lidar_points = None


def _make_cam_callback(key):
    def callback(image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        camera_data[key] = arr.reshape((image.height, image.width, 4))[:, :, :3]
    return callback


def _lidar_callback(data):
    global lidar_points
    pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
    lidar_points = pts[:, :3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_driving_command(vehicle, world_map, planner_distance=20.0):
    """Derive driving command from CARLA waypoint topology."""
    transform = vehicle.get_transform()
    wp = world_map.get_waypoint(transform.location)
    if wp is None:
        return "FORWARD"

    heading_change = 0.0
    prev_yaw = wp.transform.rotation.yaw
    cur = wp
    for _ in range(6):
        nexts = cur.next(planner_distance / 6.0)
        if not nexts:
            break
        cur = nexts[0]
        yaw = cur.transform.rotation.yaw
        diff = (yaw - prev_yaw + 180) % 360 - 180
        heading_change += diff
        prev_yaw = yaw

    if heading_change < -20:
        return "LEFT"
    elif heading_change > 20:
        return "RIGHT"
    return "FORWARD"


def find_straight_spawn(world_map, world, vehicle_bp, force_candidate=None):
    """Find a spawn point on a long straight road with clear path ahead.

    Strict criteria for reliable spawning:
    - Not inside a junction at spawn time
    - Spawn point direction matches waypoint direction (within 45 degrees)
    - At least 200m of straight road ahead (junction-free, <2 degree curvature)
    - No blocking obstacle within 15m ahead (raycast at two heights)

    Args:
        force_candidate: If specified, skip to this candidate number (1-indexed)

    Returns the carla.Transform to use for spawning.
    """
    candidates = []

    for sp in world_map.get_spawn_points():
        wp = world_map.get_waypoint(sp.location,
                                     project_to_road=True,
                                     lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue

        # Check spawn point direction matches waypoint direction
        sp_yaw = sp.rotation.yaw
        wp_yaw = wp.transform.rotation.yaw
        yaw_diff = abs((sp_yaw - wp_yaw + 180) % 360 - 180)
        if yaw_diff > 45:
            continue

        # Measure straight distance ahead using waypoints
        # Strict criteria: 2.0m intervals, 2 degree tolerance, 200m minimum
        cur, dist, prev_yaw = wp, 0, wp.transform.rotation.yaw
        for _ in range(100):  # 100 * 2.0m = 200m
            nexts = cur.next(2.0)
            if not nexts:
                break
            cur = nexts[0]
            if cur.is_junction or cur.lane_type != carla.LaneType.Driving:
                break
            yaw = cur.transform.rotation.yaw
            if abs((yaw - prev_yaw + 180) % 360 - 180) > 2:
                break
            prev_yaw = yaw
            dist += 2.0

        if dist < 200:  # Require at least 200m straight
            continue

        # Score by straight distance (longer is better)
        candidates.append((dist, sp))

    candidates.sort(reverse=True, key=lambda x: x[0])

    # If force_candidate specified, try that one directly
    if force_candidate is not None and 1 <= force_candidate <= len(candidates):
        dist, sp = candidates[force_candidate - 1]
        print(f"Forcing candidate {force_candidate}: {dist:.0f}m straight ahead", flush=True)
        return sp

    # Try candidates in order; skip ones with blocked path ahead
    for i, (dist, sp) in enumerate(candidates[:20], 1):
        try:
            test_vehicle = world.spawn_actor(vehicle_bp, sp)
        except RuntimeError:
            continue
        world.tick()
        clear = _is_front_clear(world, test_vehicle, distance=15.0)
        test_vehicle.destroy()
        world.tick()
        if clear:
            print(f"Selected spawn: {dist:.0f}m straight ahead "
                  f"(candidate {i}/{min(20, len(candidates))})", flush=True)
            return sp

    # Fallback: use first recommended spawn point
    print("Warning: no clear spawn found, falling back to first spawn point", flush=True)
    return world_map.get_spawn_points()[0]


def _is_front_clear(world, vehicle, distance=15.0):
    """Check if the forward path is clear using two-height raycasts.

    Raycasts are cheap - keep full obstacle detection to avoid curbs/poles.
    """
    veh_tf = vehicle.get_transform()
    fwd = veh_tf.get_forward_vector()
    front_x = veh_tf.location.x + fwd.x * 2.5
    front_y = veh_tf.location.y + fwd.y * 2.5

    # Cast rays at two heights: low (0.3m, for curbs/low walls) and mid (1.0m, for structures)
    blocking = {1, 2, 3, 4, 5, 11, 12, 13}  # Building, Fence, Other, Pedestrian, Pole, Wall, Sign, Vegetation
    for z_offset in (0.3, 1.0):
        start = carla.Location(x=front_x, y=front_y, z=veh_tf.location.z + z_offset)
        end = carla.Location(x=start.x + fwd.x * distance,
                             y=start.y + fwd.y * distance, z=start.z)
        hits = world.cast_ray(start, end)
        if not hits:
            continue
        for h in hits:
            label = getattr(h, 'label', None)
            if label is not None and int(label) in blocking:
                return False
    return True


def spawn_perception_sensors(world, vehicle, bp_lib):
    """Attach 3 RGB cameras and 1 LiDAR to the vehicle."""
    actors = []

    # --- RGB cameras ---
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(cfg.PERCEPTION_CAM_WIDTH))
    cam_bp.set_attribute("image_size_y", str(cfg.PERCEPTION_CAM_HEIGHT))
    cam_bp.set_attribute("fov", str(cfg.PERCEPTION_CAM_FOV))

    cam_configs = [
        ("l0", cfg.PERCEPTION_CAM_YAW_LEFT),
        ("f0", 0),
        ("r0", cfg.PERCEPTION_CAM_YAW_RIGHT),
    ]
    for key, yaw in cam_configs:
        t = carla.Transform(
            carla.Location(x=cfg.PERCEPTION_CAM_X, z=cfg.PERCEPTION_CAM_Z),
            carla.Rotation(yaw=yaw),
        )
        cam = world.spawn_actor(cam_bp, t, attach_to=vehicle)
        cam.listen(_make_cam_callback(key))
        actors.append(cam)

    # --- LiDAR ---
    lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range", str(cfg.LIDAR_RANGE))
    lidar_bp.set_attribute("rotation_frequency", str(cfg.LIDAR_ROTATION_FREQ))
    lidar_bp.set_attribute("channels", str(cfg.LIDAR_CHANNELS))
    lidar_bp.set_attribute("points_per_second", str(cfg.LIDAR_POINTS_PER_SEC))
    t = carla.Transform(carla.Location(x=cfg.LIDAR_X, z=cfg.LIDAR_Z))
    lidar = world.spawn_actor(lidar_bp, t, attach_to=vehicle)
    lidar.listen(_lidar_callback)
    actors.append(lidar)

    return actors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CARLA demo with TransFuser")
    parser.add_argument("--model", default="model/model.pth", help="Path to model.pth")
    parser.add_argument("--town", default=cfg.TOWN)
    parser.add_argument("--duration", type=int, default=cfg.DURATION_SEC)
    parser.add_argument("--output", default="outputs/transfuser_demo.mp4")
    args = parser.parse_args()

    pid = PIDController()
    actors = []

    # --- Connect ---
    print("--- Connecting to CARLA ---", flush=True)
    client = carla.Client(cfg.CARLA_HOST, cfg.CARLA_PORT)
    client.set_timeout(120.0)

    # リトライ付き接続 (run_demo.py 経由の場合、サーバー起動直後で不安定なことがある)
    import time as _time
    for attempt in range(6):
        try:
            world = client.get_world()
            break
        except RuntimeError:
            if attempt == 5:
                raise
            print(f"  Waiting for CARLA... (retry {attempt + 1})", flush=True)
            _time.sleep(10)
    current_map = world.get_map().name.split("/")[-1]
    print(f"Current map: {current_map}", flush=True)

    if current_map != args.town:
        print(f"Loading {args.town} (this may take 30-60 seconds)...", flush=True)
        world = client.load_world(args.town)
        print(f"{args.town} loaded.", flush=True)

    world_map = world.get_map()

    # --- Load model (after CARLA connection) ---
    agent = TransfuserAgent(model_path=args.model)

    # --- Synchronous mode ---
    print("\n--- Setting up synchronous mode ---", flush=True)
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg.FIXED_DELTA
    world.apply_settings(settings)
    world.tick()

    try:
        # --- Spawn vehicle + sensors ---
        print("\n--- Spawning vehicle and sensors ---", flush=True)
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        spawn = find_straight_spawn(world_map, world, vehicle_bp)
        vehicle = world.spawn_actor(vehicle_bp, spawn)
        actors.append(vehicle)
        world.tick()

        veh_tf = vehicle.get_transform()
        print(f"Vehicle spawned at {veh_tf.location}, yaw={veh_tf.rotation.yaw:.1f}", flush=True)

        recorder = Recorder(world, vehicle)
        sensor_actors = spawn_perception_sensors(world, vehicle, bp_lib)
        actors.extend(sensor_actors)
        print(f"Sensors attached: 3 cameras + 1 LiDAR", flush=True)

        collision_flag = [False]
        collision_bp = bp_lib.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        actors.append(collision_sensor)
        collision_sensor.listen(lambda _: collision_flag.__setitem__(0, True))

        # --- Warm-up (wait for sensor data) ---
        print("\n--- Warm-up phase (40 ticks) ---", flush=True)
        for i in range(40):
            if collision_flag[0]:
                for _ in range(20):
                    vehicle.apply_control(carla.VehicleControl(
                        throttle=0.5, steer=0.0, reverse=True, hand_brake=False))
                    world.tick()
                collision_flag[0] = False
                break
            vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=0.0, hand_brake=False))
            world.tick()
        print("Warm-up complete.", flush=True)

        # Debug: verify vehicle heading matches road direction
        tf_dbg = vehicle.get_transform()
        wp_dbg = world_map.get_waypoint(tf_dbg.location)
        if wp_dbg:
            veh_yaw = tf_dbg.rotation.yaw
            road_yaw = wp_dbg.transform.rotation.yaw
            diff = (road_yaw - veh_yaw + 180) % 360 - 180
            print(f"Post-warmup: vehicle_yaw={veh_yaw:.1f}, road_yaw={road_yaw:.1f}, diff={diff:.1f}")
            vel_dbg = vehicle.get_velocity()
            fwd_dbg = tf_dbg.get_forward_vector()
            vx_dbg = vel_dbg.x * fwd_dbg.x + vel_dbg.y * fwd_dbg.y
            print(f"Post-warmup: forward_speed={vx_dbg:.2f} m/s (negative=backward)")

        # --- Main driving loop ---
        total_ticks = int(args.duration / cfg.FIXED_DELTA)
        print(f"\n--- Driving: {args.duration}s ({total_ticks} ticks) ---", flush=True)
        stuck_count = 0
        recovery_cooldown = 0

        for tick_i in range(total_ticks):
            world.tick()

            vel = vehicle.get_velocity()
            acc = vehicle.get_acceleration()
            tf = vehicle.get_transform()
            fwd = tf.get_forward_vector()
            right = tf.get_right_vector()

            # Ego-local velocity/acceleration (nuPlan convention: x=fwd, y=left)
            vx = vel.x * fwd.x + vel.y * fwd.y
            vy = -(vel.x * right.x + vel.y * right.y)
            ax = acc.x * fwd.x + acc.y * fwd.y
            ay = -(acc.x * right.x + acc.y * right.y)
            speed = math.sqrt(vel.x ** 2 + vel.y ** 2)

            if recovery_cooldown > 0:
                recovery_cooldown -= 1

            # Collision recovery
            if collision_flag[0] and recovery_cooldown == 0:
                print(f"  Collision at tick {tick_i}, reversing...", flush=True)
                for _ in range(25):
                    vehicle.apply_control(carla.VehicleControl(
                        throttle=0.5, steer=0.0, reverse=True, hand_brake=False))
                    world.tick()
                collision_flag[0] = False
                recovery_cooldown = 40
                stuck_count = 0
                pid = PIDController()
                continue

            # Stuck recovery
            if speed < 0.3 and tick_i > 60:
                stuck_count += 1
            else:
                stuck_count = 0

            if stuck_count > 30:
                print(f"  Stuck detected at tick {tick_i}, recovering...", flush=True)
                for _ in range(30):
                    vehicle.apply_control(carla.VehicleControl(
                        throttle=0.5, steer=0.0, reverse=True, hand_brake=False))
                    world.tick()
                for _ in range(40):
                    vehicle.apply_control(carla.VehicleControl(
                        throttle=0.8, steer=0.0, hand_brake=False))
                    world.tick()
                stuck_count = 0
                pid = PIDController()
                continue

            # Predict & control (only when all sensors have data)
            command = get_driving_command(vehicle, world_map)
            sensors_ready = all(v is not None for v in camera_data.values()) and lidar_points is not None
            if sensors_ready:
                trajectory = agent.predict(
                    camera_data, lidar_points, [vx, vy], [ax, ay], command,
                )
            else:
                # Fallback: straight trajectory
                trajectory = [[float(i + 1) * 2.0, 0.0] for i in range(cfg.NUM_POSES)]

            current_wp = world_map.get_waypoint(tf.location)
            control = pid.run(trajectory, tf, speed, current_wp=current_wp)
            vehicle.apply_control(control)

            if tick_i % 20 == 0:
                pct = tick_i / total_ticks * 100
                d = getattr(pid, "debug", {})
                loc = tf.location
                traj_end = d.get('traj_end', [0, 0])
                traj_str = f"traj=[{trajectory[0][0]:.1f},{trajectory[0][1]:.1f}]...[{traj_end[0]:.1f},{traj_end[1]:.1f}]" if trajectory else "none"
                print(f"  [{pct:5.1f}%] spd={speed*3.6:.0f}km/h vx={vx:.1f}m/s cmd={command:7s} "
                      f"steer={control.steer:+.2f} "
                      f"(wp={d.get('wp_steer',0):+.2f} "
                      f"model={d.get('model_steer',0):+.2f} "
                      f"latOff={d.get('lat_offset',0):+.2f}m) "
                      f"{traj_str} "
                      f"sensors={'OK' if sensors_ready else 'WAIT'} "
                      f"pos=({loc.x:.0f},{loc.y:.0f})",
                      flush=True)

        # --- Save video ---
        print(f"\n--- Saving video ({len(recorder._frames)} frames) ---", flush=True)
        world.tick()
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        recorder.save(args.output)

    finally:
        recorder.destroy()
        for a in actors:
            a.destroy()
        world.apply_settings(original_settings)


if __name__ == "__main__":
    main()
