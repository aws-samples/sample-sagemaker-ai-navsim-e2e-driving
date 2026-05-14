#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""Run CARLA TransFuser demo end-to-end (equivalent to carla-transfuser-demo.ipynb)."""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description="CARLA TransFuser demo runner")
    parser.add_argument("--model", default=None, help="Path to model.pth (auto-download if omitted)")
    parser.add_argument("--town", default="Town04")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--output", default="outputs/transfuser_demo.mp4")
    parser.add_argument("--skip-install", action="store_true", help="Skip CARLA/deps install")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    print("=" * 60, flush=True)
    print("  CARLA TransFuser Demo", flush=True)
    print("=" * 60, flush=True)

    # --- 1. Model ---
    model_path = args.model or os.path.join(root, "model", "model.pth")
    if os.path.exists(model_path):
        print(f"\n[1/4] Model check: {os.path.relpath(model_path)} found", flush=True)
    else:
        print(f"\n[1/4] Model check: not found, downloading from S3...", flush=True)
        model_path = download_model(region, os.path.join(root, "model"))

    # --- 2. CARLA install ---
    carla_dir = os.path.expanduser("~/SageMaker/carla")
    carla_bin = os.path.join(carla_dir, "CarlaUE4.sh")
    if not args.skip_install:
        install_carla(carla_dir, carla_bin)
        install_deps()
    else:
        print("\n[2/4] Dependencies: skipped (--skip-install)", flush=True)

    # --- 3. Start CARLA server ---
    print("\n[3/4] Starting CARLA server...", flush=True)
    subprocess.run(["pkill", "-9", "-f", "CarlaUE4"], capture_output=True)
    time.sleep(2)
    carla_log = open(os.path.join(tempfile.gettempdir(), "carla_server.log"), "w", encoding="utf-8")
    carla_proc = subprocess.Popen(
        [carla_bin, "-RenderOffScreen", "--world-port=2000"],
        stdout=carla_log, stderr=carla_log,
    )

    try:
        wait_for_carla()

        # --- 4. Run simulation ---
        print(f"\n[4/4] Running simulation ({args.town}, {args.duration}s)...", flush=True)
        cmd = [sys.executable, "-u", "run.py",
               "--model", model_path, "--town", args.town,
               "--duration", str(args.duration), "--output", args.output]

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        result = subprocess.run(cmd, cwd=root, env=env)
        if result.returncode != 0:
            sys.exit(f"\nSimulation failed (exit code {result.returncode})")

        print(f"\n{'=' * 60}")
        print(f"  ✅ Done — video saved to {os.path.join(root, args.output)}")
        print(f"{'=' * 60}", flush=True)

    finally:
        carla_proc.kill()
        carla_proc.wait()
        carla_log.close()
        print("\nCARLA server stopped.", flush=True)


def download_model(region, model_dir):
    """Download latest TransFuser model.pth from S3."""
    import boto3
    sm = boto3.client("sagemaker", region_name=region)
    paginator = sm.get_paginator("list_training_jobs")
    found = None
    for page in paginator.paginate(
        SortBy="CreationTime", SortOrder="Descending", StatusEquals="Completed",
    ):
        for summary in page["TrainingJobSummaries"]:
            job = sm.describe_training_job(TrainingJobName=summary["TrainingJobName"])
            image = job.get("AlgorithmSpecification", {}).get("TrainingImage", "")
            if "container-navsim-transfuser" in image:
                found = job
                break
        if found:
            break
    if not found:
        sys.exit("No completed TransFuser training job found. Run the pipeline first.")

    s3_uri = found["ModelArtifacts"]["S3ModelArtifacts"]
    print(f"  Training Job: {found['TrainingJobName']}", flush=True)
    print(f"  Downloading from: {s3_uri}", flush=True)

    s3 = boto3.client("s3", region_name=region)
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    tar_path = "/tmp/model.tar.gz"
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except s3.exceptions.ClientError:
        job_name = found['TrainingJobName']
        print(f"\n  ❌ Model artifact not found on S3.", flush=True)
        print(f"     The training job may still be running, or the artifact was deleted.", flush=True)
        print(f"\n     Job:    {job_name}", flush=True)
        print(f"     S3:     {s3_uri}", flush=True)
        print(f"     Check:  aws sagemaker describe-training-job --training-job-name {job_name} --region {region}", flush=True)
        sys.exit(1)
    s3.download_file(bucket, key, tar_path)
    os.makedirs(model_dir, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(model_dir, filter="data")
    model_path = os.path.join(model_dir, "model.pth")
    print(f"  Model saved to: {model_path}", flush=True)
    return model_path


def install_carla(carla_dir, carla_bin):
    """Install CARLA if not present or version changed."""
    version = "0.9.16"
    version_file = os.path.join(carla_dir, ".carla_version")
    installed_version = ""
    if os.path.exists(version_file):
        installed_version = open(version_file, encoding="utf-8").read().strip()
    if os.path.exists(carla_bin) and installed_version == version:
        print(f"\n[2/4] CARLA {version} and dependencies: already installed", flush=True)
        return
    tar_path = os.path.expanduser(f"~/SageMaker/CARLA_{version}.tar.gz")
    print(f"\n[2/4] Installing CARLA {version} and dependencies...", flush=True)
    if os.path.exists(carla_dir):
        shutil.rmtree(carla_dir)
    subprocess.run([
        "wget", "-q",
        "https://carla-releases.b-cdn.net/Linux/CARLA_0.9.16.tar.gz",
        "-O", tar_path,
    ], check=True)
    os.makedirs(carla_dir, exist_ok=True)
    subprocess.run(["tar", "xzf", tar_path, "-C", carla_dir], check=True)
    os.remove(tar_path)
    with open(version_file, "w", encoding="utf-8") as f:
        f.write(version)
    print(f"  CARLA installed to {carla_dir}", flush=True)


def install_deps():
    """Install system and Python dependencies."""
    subprocess.run(
        "sudo yum install -y -q vulkan-loader vulkan-tools mesa-vulkan-drivers 2>/dev/null || true",
        shell=True)
    subprocess.run(
        "conda install -y -q -c conda-forge ffmpeg 2>/dev/null || true",
        shell=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "carla==0.9.16", "opencv-python", "numpy<2", "timm", "torchvision"])


def wait_for_carla(timeout=300):
    """Wait for CARLA server to be ready."""
    import carla
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(5)
        try:
            client = carla.Client("localhost", 2000)
            client.set_timeout(5.0)
            ver = client.get_server_version()
            print(f"CARLA server ready (v{ver})", flush=True)
            return
        except Exception:
            elapsed = int(time.time() - start)
            print(f"  Waiting for CARLA server... ({elapsed}s)", flush=True)
    sys.exit("CARLA server failed to start within 5 minutes")


if __name__ == "__main__":
    main()
