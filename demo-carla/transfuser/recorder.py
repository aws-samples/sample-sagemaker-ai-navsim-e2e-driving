# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Video recorder: attach CARLA camera sensor and save frames to mp4."""

import os

import carla
import cv2
import numpy as np

import config as cfg


class Recorder:
    """Attach an RGB camera to a vehicle and record frames."""

    def __init__(self, world, vehicle):
        self._frames = []
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(cfg.CAMERA_WIDTH))
        bp.set_attribute("image_size_y", str(cfg.CAMERA_HEIGHT))
        bp.set_attribute("fov", str(cfg.CAMERA_FOV))
        t = cfg.CAMERA_TRANSFORM
        transform = carla.Transform(
            carla.Location(x=t["x"], z=t["z"]),
            carla.Rotation(pitch=t["pitch"], yaw=t.get("yaw", 0)),
        )
        self._camera = world.spawn_actor(bp, transform, attach_to=vehicle)
        self._camera.listen(self._on_frame)

    def _on_frame(self, image):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]  # BGRA → BGR
        self._frames.append(arr)

    def save(self, output_path="output.mp4"):
        """Write collected frames to mp4 (H.264 if ffmpeg available)."""
        import shutil
        import subprocess

        frames = list(self._frames)
        if not frames:
            print("No frames recorded.")
            return

        has_ffmpeg = shutil.which("ffmpeg") is not None
        write_path = output_path + ".tmp.mp4" if has_ffmpeg else output_path
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(write_path, fourcc, cfg.VIDEO_FPS,
                              (cfg.CAMERA_WIDTH, cfg.CAMERA_HEIGHT))
        total = len(frames)
        for i, f in enumerate(frames):
            out.write(f)
            if (i + 1) % 200 == 0 or i == total - 1:
                print(f"  Writing frames: {i+1}/{total}", flush=True)
        out.release()

        if has_ffmpeg:
            print(f"  Converting to H.264 ({total} frames)...", flush=True)
            subprocess.run(
                ["ffmpeg", "-y", "-i", write_path, "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-preset", "fast", output_path],
                capture_output=True,
            )
            os.remove(write_path)

        print(f"Saved {total} frames to {output_path}"
              + (" (H.264)" if has_ffmpeg else " (mp4v)"), flush=True)

    def destroy(self):
        if self._camera is not None:
            try:
                self._camera.stop()
            except RuntimeError:
                pass
            self._camera.destroy()
