# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Waypoint-primary lateral control with model trajectory blending.

Uses CARLA road waypoints as the primary steering source (reliable on all
roads), and blends in the TransFuser model's predicted trajectory with a
configurable weight. When the model is trained on real data, increase
MODEL_BLEND to let the model drive more.
"""

import math

import carla

import config as cfg

# How much to trust the model vs CARLA waypoints.
# 0.0 = pure waypoint following (like CARLA BasicAgent)
# 1.0 = pure model prediction
#
# Domain gap exists between OpenScene (real-world) and CARLA (synthetic).
# Waypoint provides stable road-following, model provides command awareness.
#
# Experimental results (see MODEL_BLEND_EXPERIMENT_RESULTS.md):
# - 0.3: Excellent stability (1 collision in Town03 curves)
# - 0.5: Best balance (2 collisions in Town04, 1 in Town03) ← RECOMMENDED
# - 0.7: Acceptable (2 collisions, larger lateral offset)
# - 1.0: Uncontrollable (21-29 collisions, fails to recover from curves)
MODEL_BLEND = 0.5  # waypoint 50% + model 50%


class PIDController:
    """Pure-pursuit on CARLA waypoints, blended with model trajectory."""

    def __init__(self):
        self._steer_prev = 0.0
        self._lon_error_integral = 0.0
        self._lon_error_prev = 0.0

    def run(self, trajectory, vehicle_transform, current_speed,
            current_wp=None, **_kwargs):
        # --- Steering from CARLA waypoints (primary) ---
        wp_steer = self._waypoint_steer(vehicle_transform, current_wp, current_speed)

        # --- Steering from model trajectory ---
        ld = max(cfg.MIN_LOOKAHEAD,
                 cfg.LOOKAHEAD_GAIN * current_speed + cfg.MIN_LOOKAHEAD)
        ld = min(ld, cfg.MAX_LOOKAHEAD)
        model_steer = self._pure_pursuit_on_trajectory(trajectory, ld)
        if model_steer is None:
            model_steer = 0.0

        # --- Blend ---
        steer = (1.0 - MODEL_BLEND) * wp_steer + MODEL_BLEND * model_steer

        # Smooth
        steer = 0.7 * steer + 0.3 * self._steer_prev
        steer = max(-1.0, min(1.0, steer))
        self._steer_prev = steer

        # --- Debug ---
        self.debug = {
            "model_steer": model_steer,
            "wp_steer": wp_steer,
            "lat_offset": self._last_lat_offset,
            "traj_end": trajectory[-1] if trajectory else [0, 0],
        }

        # --- Longitudinal ---
        target_speed = cfg.TARGET_SPEED_MS
        # Slow down on curves (detected from waypoint heading change)
        curv = self._road_curvature(current_wp)
        if curv > 0.01:
            target_speed = min(target_speed, 2.5 / math.sqrt(curv))
            target_speed = max(target_speed, 2.0)

        control = self._longitudinal_pid(current_speed, target_speed)
        control.steer = steer
        return control

    # ------------------------------------------------------------------
    # Waypoint-based steering (like CARLA BasicAgent)
    # ------------------------------------------------------------------

    _last_lat_offset = 0.0

    def _waypoint_steer(self, vehicle_transform, current_wp, current_speed):
        """Pure pursuit on CARLA road waypoints."""
        if current_wp is None:
            return 0.0

        # Pick a target waypoint ahead
        ld = max(4.0, current_speed * 1.0 + 3.0)
        target_wp = current_wp
        dist_accum = 0.0
        for _ in range(20):
            nexts = target_wp.next(2.0)
            if not nexts:
                break
            target_wp = nexts[0]
            dist_accum += 2.0
            if dist_accum >= ld:
                break

        # Compute lateral offset for debug
        ego_loc = vehicle_transform.location
        wp_loc = current_wp.transform.location
        wp_yaw = math.radians(current_wp.transform.rotation.yaw)
        dx = ego_loc.x - wp_loc.x
        dy = ego_loc.y - wp_loc.y
        right_x = math.cos(wp_yaw + math.pi / 2)
        right_y = math.sin(wp_yaw + math.pi / 2)
        self._last_lat_offset = dx * right_x + dy * right_y

        # Vector from ego to target waypoint
        tw = target_wp.transform.location
        fx = tw.x - ego_loc.x
        fy = tw.y - ego_loc.y
        dist = math.hypot(fx, fy)
        if dist < 0.5:
            return 0.0

        # Vehicle forward vector
        fwd = vehicle_transform.get_forward_vector()

        # Signed angle between forward and target direction
        # cross = fwd.x * fy - fwd.y * fx  (positive = target is to the left in CARLA)
        cross = fwd.x * fy - fwd.y * fx
        dot = fwd.x * fx + fwd.y * fy
        angle = math.atan2(cross, dot)

        # Pure pursuit: steer = atan2(2 * L * sin(alpha), dist)
        steer_rad = math.atan2(2.0 * cfg.WHEELBASE * math.sin(angle), dist)
        return max(-1.0, min(1.0, steer_rad / 1.1))

    # ------------------------------------------------------------------
    # Model trajectory steering
    # ------------------------------------------------------------------

    def _pure_pursuit_on_trajectory(self, trajectory, ld):
        """Pure pursuit on model's ego-local trajectory (nuPlan: x=fwd, y=left)."""
        if not trajectory or len(trajectory) < 2:
            return None

        goal = None
        for i in range(len(trajectory) - 1):
            p1, p2 = trajectory[i], trajectory[i + 1]
            d1 = math.hypot(p1[0], p1[1])
            d2 = math.hypot(p2[0], p2[1])
            if d2 >= ld:
                if d2 - d1 > 1e-6:
                    t = max(0.0, min(1.0, (ld - d1) / (d2 - d1)))
                    goal = [p1[0] + t * (p2[0] - p1[0]),
                            p1[1] + t * (p2[1] - p1[1])]
                else:
                    goal = list(p2)
                break
        if goal is None:
            goal = list(trajectory[-1])

        gx, gy = goal
        dist = math.hypot(gx, gy)
        if dist < 0.5:
            return None

        # nuPlan y=left → CARLA steer: left = negative steer
        alpha = math.atan2(-gy, gx)
        steer_rad = math.atan2(2.0 * cfg.WHEELBASE * math.sin(alpha), dist)
        return max(-1.0, min(1.0, steer_rad / 1.1))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _road_curvature(current_wp):
        """Estimate road curvature from waypoint heading changes."""
        if current_wp is None:
            return 0.0
        prev_yaw = current_wp.transform.rotation.yaw
        cur = current_wp
        heading_change = 0.0
        total_dist = 0.0
        for _ in range(8):
            nexts = cur.next(3.0)
            if not nexts:
                break
            cur = nexts[0]
            yaw = cur.transform.rotation.yaw
            diff = (yaw - prev_yaw + 180) % 360 - 180
            heading_change += abs(diff)
            total_dist += 3.0
            prev_yaw = yaw
        if total_dist < 1e-3:
            return 0.0
        return math.radians(heading_change) / total_dist

    def _longitudinal_pid(self, current_speed, target_speed):
        error = target_speed - current_speed
        self._lon_error_integral = max(-5.0, min(5.0,
            self._lon_error_integral + error))
        derivative = error - self._lon_error_prev
        self._lon_error_prev = error

        accel = (cfg.PID_LONGITUDINAL_KP * error
                 + cfg.PID_LONGITUDINAL_KI * self._lon_error_integral
                 + cfg.PID_LONGITUDINAL_KD * derivative)

        control = carla.VehicleControl()
        control.reverse = False
        if accel >= 0:
            control.throttle = min(accel, 1.0)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(-accel, 1.0)
        control.hand_brake = False
        return control
