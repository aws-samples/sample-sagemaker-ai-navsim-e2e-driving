# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
NAVSIM EgoStatusMLP Endpoint テストスクリプト。

Streamlit アプリ (app/main.py) と同等の推論リクエストを送信してテストする。
"""

import json
import os
from pathlib import Path
import boto3

# .env ファイルがあれば読み込む (既存の環境変数は上書きしない)
_env_file = Path(__file__).resolve().parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _v = _v.strip().strip('"')
        os.environ.setdefault(_k.strip(), _v)

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
ENDPOINT_NAME = "navsim-ego-mlp-endpoint"

client = boto3.client("sagemaker-runtime", region_name=REGION)


def invoke(payload: dict) -> dict:
    resp = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(resp["Body"].read().decode())


# --- テストケース ---

# 1. 直進 (vx=10 m/s)
print("=== Test 1: Forward at 10 m/s ===")
result = invoke({"velocity": [10.0, 0.0], "acceleration": [0.0, 0.0], "command": "FORWARD"})
print(f"Request:  velocity=[10, 0], accel=[0, 0], command=FORWARD")
print(f"Trajectory (x, y):")
for i, (x, y) in enumerate(result["trajectory"]):
    print(f"  step {i}: x={x:+.3f}, y={y:+.3f}")

# 2. 左折
print("\n=== Test 2: Left turn ===")
result = invoke({"velocity": [8.0, 0.0], "acceleration": [0.0, 0.5], "command": "LEFT"})
print(f"Request:  velocity=[8, 0], accel=[0, 0.5], command=LEFT")
print(f"Trajectory (x, y):")
for i, (x, y) in enumerate(result["trajectory"]):
    print(f"  step {i}: x={x:+.3f}, y={y:+.3f}")

# 3. 右折
print("\n=== Test 3: Right turn ===")
result = invoke({"velocity": [8.0, 0.0], "acceleration": [0.0, -0.5], "command": "RIGHT"})
print(f"Request:  velocity=[8, 0], accel=[0, -0.5], command=RIGHT")
print(f"Trajectory (x, y):")
for i, (x, y) in enumerate(result["trajectory"]):
    print(f"  step {i}: x={x:+.3f}, y={y:+.3f}")

# 4. 停車中
print("\n=== Test 4: Stationary ===")
result = invoke({"velocity": [0.0, 0.0], "acceleration": [0.0, 0.0], "command": "FORWARD"})
print(f"Request:  velocity=[0, 0], accel=[0, 0], command=FORWARD")
print(f"Trajectory (x, y):")
for i, (x, y) in enumerate(result["trajectory"]):
    print(f"  step {i}: x={x:+.3f}, y={y:+.3f}")

# 5. Full response (heading 含む)
print("\n=== Test 5: Full response with heading ===")
result = invoke({"velocity": [10.0, 0.0], "acceleration": [1.0, 0.0], "command": "FORWARD"})
print(f"Full response: {json.dumps(result, indent=2)}")
