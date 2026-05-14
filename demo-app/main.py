# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
"""
SageMaker Endpoint 推論デモアプリ (Streamlit)

NAVSIM EgoStatusMLP の推論エンドポイントに対して
リクエストを送信し、予測軌跡を可視化する。
スライダー操作で自動推論、プリセットシナリオでアニメーション再生が可能。

使い方:
    pip install streamlit boto3 pandas plotly
    streamlit run demo-app/main.py

    # Mock モード (エンドポイント不要)
    streamlit run demo-app/main.py -- --mock
"""

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
from pathlib import Path

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
MOCK_MODE = "--mock" in sys.argv

# ---------------------------------------------------------------------------
# Mock 推論
# ---------------------------------------------------------------------------
COMMAND_MAP = {"FORWARD": 0, "LEFT": 1, "RIGHT": 2}


def mock_inference(payload: dict) -> dict:
    """物理ベースの簡易軌跡生成 (エンドポイント不要)。"""
    vx, vy = payload.get("velocity", [10, 0])
    ax, ay = payload.get("acceleration", [0, 0])
    cmd = payload.get("command", "FORWARD")

    if cmd == "LEFT":
        curvature = -(0.008 + abs(ay) * 0.002)
    elif cmd == "RIGHT":
        curvature = 0.008 + abs(ay) * 0.002
    else:
        curvature = -ay * 0.001

    dt = 0.5
    trajectory = []
    x, y, theta = 0.0, 0.0, 0.0
    v = math.sqrt(vx ** 2 + vy ** 2)

    for i in range(8):
        v = max(0, v + ax * dt)
        theta += curvature * v * dt
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        noise = np.random.normal(0, 0.05, 2)
        trajectory.append([round(x + noise[0], 3), round(y + noise[1], 3)])

    return {"trajectory": trajectory, "mock": True}


# ---------------------------------------------------------------------------
# Endpoint 呼び出し
# ---------------------------------------------------------------------------
if not MOCK_MODE:
    import boto3
    sm_runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    cfn_client = boto3.client("cloudformation", region_name=REGION)

MODEL_TYPE_MAP = {"ego-mlp": "NAVSIM EgoStatusMLP"}


@st.cache_data(ttl=60)
def discover_endpoints():
    if MOCK_MODE:
        return {}
    endpoints = {}
    try:
        paginator = cfn_client.get_paginator("list_stacks")
        for page in paginator.paginate(StackStatusFilter=["CREATE_COMPLETE", "UPDATE_COMPLETE"]):
            for stack in page["StackSummaries"]:
                if "sagemaker-ai-inference" not in stack["StackName"]:
                    continue
                outputs = cfn_client.describe_stacks(StackName=stack["StackName"])[
                    "Stacks"][0].get("Outputs", [])
                for o in outputs:
                    if o["OutputKey"] == "EndpointName":
                        name = o["OutputValue"]
                        model_type = "NAVSIM EgoStatusMLP"
                        for key, label in MODEL_TYPE_MAP.items():
                            if key in name:
                                model_type = label
                                break
                        endpoints[name] = model_type
    except Exception:
        pass
    return endpoints


def invoke_endpoint(endpoint_name: str, payload: dict) -> dict:
    if MOCK_MODE or endpoint_name == "🎭 Mock Demo":
        return mock_inference(payload)
    resp = sm_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(resp["Body"].read().decode())


# ---------------------------------------------------------------------------
# シナリオ定義
# ---------------------------------------------------------------------------
SCENARIOS = {
    "🚀 Acceleration (0 → 20 m/s)": {"road_offset": -1.75, "frames": [
        {"velocity": [v, 0], "acceleration": [2.0, 0], "command": "FORWARD"}
        for v in [float(x) for x in range(0, 21, 1)]
    ]},
    "🛑 Braking (20 → 0 m/s)": {"road_offset": -1.75, "frames": [
        {"velocity": [v, 0], "acceleration": [-2.0, 0], "command": "FORWARD"}
        for v in [float(x) for x in range(20, -1, -1)]
    ]},
    "↩️ Left Turn": {"road_offset": -1.75, "frames": [
        {"velocity": [10, 0], "acceleration": [0, ay], "command": "LEFT"}
        for ay in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
                   2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.0]
    ]},
    "↪️ Right Turn": {"road_offset": -1.75, "frames": [
        {"velocity": [10, 0], "acceleration": [0, ay], "command": "RIGHT"}
        for ay in [0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75, -2.0, -2.25, -2.5, -2.75, -3.0,
                   -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0]
    ]},
    "🔄 Lane Change (Right → Left)": {"road_offset": -1.75, "frames": [
        {"velocity": [15, vy], "acceleration": [0, ay], "command": cmd}
        for vy, ay, cmd in [
            (0, 0.3, "LEFT"), (0, 0.6, "LEFT"), (0.15, 0.8, "LEFT"), (0.3, 0.9, "LEFT"),
            (0.5, 0.9, "LEFT"), (0.7, 0.8, "LEFT"), (0.85, 0.6, "LEFT"), (0.95, 0.4, "LEFT"),
            (1.0, 0.2, "FORWARD"), (1.0, 0, "FORWARD"),
            (0.95, -0.2, "FORWARD"), (0.85, -0.4, "FORWARD"), (0.7, -0.6, "FORWARD"),
            (0.5, -0.6, "FORWARD"), (0.3, -0.4, "FORWARD"), (0.15, -0.3, "FORWARD"),
            (0.05, -0.15, "FORWARD"), (0, 0, "FORWARD"),
        ]
    ]},
    "🏎️ Cornering (Accel into Turn)": {"road_offset": -1.75, "frames": [
        {"velocity": [v, 0], "acceleration": [a, ay], "command": "LEFT"}
        for v, a, ay in [
            (8, 1.0, 0), (9, 0.75, 0.25), (10, 0.5, 0.5), (10.5, 0.25, 0.75),
            (11, 0, 1.0), (11.5, 0, 1.25), (12, 0, 1.5), (12, 0, 1.75),
            (12, 0, 2.0), (12, 0, 2.25), (12, 0, 2.5), (12, -0.25, 2.25),
            (12, -0.5, 2.0), (11.5, -0.5, 1.75), (11, -0.5, 1.5), (10.5, -0.25, 1.25),
            (10, 0, 1.0), (10, 0, 0.75), (10, 0, 0.5), (10, 0.25, 0.25), (10, 0.5, 0),
        ]
    ]},
}

# ---------------------------------------------------------------------------
# 周囲車両の生成
# ---------------------------------------------------------------------------
SURROUNDING_VEHICLES = [
    # (lateral_offset, forward_offset, color, label)
    (1.75, 15.0, "#5577AA", "Vehicle A"),
    (1.75, 35.0, "#AA5577", "Vehicle B"),
    (-1.75, 25.0, "#77AA55", "Vehicle C"),
]


# ---------------------------------------------------------------------------
# 可視化
# ---------------------------------------------------------------------------
GRADIENT_COLORS = [
    "#3366FF", "#4477EE", "#5588DD", "#6699CC",
    "#CC8844", "#DD7733", "#EE5522", "#FF3311",
]


def _car_polygon(cx, cy, angle_deg, length=2.5, width=1.2):
    """トップダウンの車ポリゴン。"""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    hw, hl = width / 2, length / 2
    pts = [
        (-hl, -hw), (-hl, hw),
        (hl * 0.7, hw), (hl, hw * 0.5),
        (hl, -hw * 0.5), (hl * 0.7, -hw),
        (-hl, -hw),
    ]
    rx = [cx + p[0] * cos_a - p[1] * sin_a for p in pts]
    ry = [cy + p[0] * sin_a + p[1] * cos_a for p in pts]
    return rx, ry


def _add_surrounding_vehicles(fig, ego_fwd, ego_lat, road_offset):
    """周囲車両を描画する。"""
    for lat_off, fwd_off, color, label in SURROUNDING_VEHICLES:
        vx = road_offset + lat_off
        vy = ego_fwd + fwd_off
        rx, ry = _car_polygon(vx, vy, 90, length=2.2, width=1.0)
        fig.add_trace(go.Scatter(
            x=rx, y=ry, fill="toself",
            fillcolor=color, line=dict(color="white", width=1),
            opacity=0.6, name=label, hoverinfo="name",
            showlegend=False,
        ))
        # ヘッドライト風のアクセント
        fig.add_trace(go.Scatter(
            x=[vx], y=[vy + 2.0],
            mode="markers",
            marker=dict(size=4, color="#FFEE88", symbol="diamond"),
            showlegend=False, hoverinfo="skip",
        ))


def _add_road_arrows(fig, y_lo, y_hi, road_offset):
    """道路上の進行方向矢印を描画する。"""
    for yy in range(int(y_lo) + 8, int(y_hi), 20):
        for lane_x in [-1.75, 1.75]:
            x0 = road_offset + lane_x
            fig.add_shape(type="line", x0=x0, x1=x0, y0=yy, y1=yy + 1.5,
                          line=dict(color="#444444", width=1.5), layer="below")


def _add_confidence_band(fig, traj, ego_fwd, ego_lat):
    """軌跡の不確実性を半透明の帯で表示する。"""
    if len(traj) < 2:
        return
    upper_x, upper_y, lower_x, lower_y = [], [], [], []
    for i, p in enumerate(traj):
        spread = 0.15 + i * 0.12  # 遠い未来ほど不確実性が大きい
        px = ego_lat + p[1]
        py = ego_fwd + p[0]
        upper_x.append(px + spread)
        upper_y.append(py)
        lower_x.append(px - spread)
        lower_y.append(py)

    fig.add_trace(go.Scatter(
        x=upper_x + lower_x[::-1],
        y=upper_y + lower_y[::-1],
        fill="toself", fillcolor="rgba(51, 102, 255, 0.12)",
        line=dict(width=0),
        name="Confidence", showlegend=True, hoverinfo="skip",
    ))


def render_trajectory(traj: list, title_suffix: str = "", road_offset: float = 0.0,
                      ego_pos: tuple = (0.0, 0.0), past_path: list = None,
                      show_vehicles: bool = True, view_range: dict = None,
                      payload: dict = None):
    """軌跡プロットを描画。上方向 = 前進 (X)、右方向 = 横 (Y)。
    view_range: {"x_lo", "x_hi", "y_lo", "y_hi"} 固定表示範囲 (アニメーション用)。
    payload: 入力パラメータ (タイトルにステータス表示)。
    """
    fig = go.Figure()

    ego_fwd, ego_lat = ego_pos
    plot_x = [ego_lat + p[1] for p in traj]
    plot_y = [ego_fwd + p[0] for p in traj]

    if view_range:
        x_lo, x_hi = view_range["x_lo"], view_range["x_hi"]
        y_lo, y_hi = view_range["y_lo"], view_range["y_hi"]
    else:
        all_px = [ego_lat] + plot_x
        all_py = [ego_fwd] + plot_y
        if past_path:
            all_px += [p[1] for p in past_path]
            all_py += [p[0] for p in past_path]
        px_range = max(all_px) - min(all_px)
        py_range = max(all_py) - min(all_py)
        px_margin = max(px_range * 0.3, (20 - px_range) / 2, 3)
        py_margin = max(py_range * 0.15, (40 - py_range) / 2, 2)
        x_lo, x_hi = min(all_px) - px_margin, max(all_px) + px_margin
        y_lo, y_hi = min(all_py) - py_margin, max(all_py) + py_margin

    # --- 道路背景 ---
    fig.add_shape(type="rect", x0=x_lo, x1=x_hi, y0=y_lo, y1=y_hi,
                  fillcolor="#2a2a2a", line_width=0, layer="below")
    # 路肩 (グレーの帯)
    for side in [-5.25, 5.25]:
        fig.add_shape(type="rect",
                      x0=road_offset + side - 0.3, x1=road_offset + side + 0.3,
                      y0=y_lo, y1=y_hi,
                      fillcolor="#3d3d3d", line_width=0, layer="below")
    # 車線区分線 (白の破線、同方向レーン)
    for yy in range(int(y_lo), int(y_hi) + 1, 3):
        fig.add_shape(type="line", x0=road_offset, x1=road_offset, y0=yy, y1=yy + 1.5,
                      line=dict(color="#FFFFFF", width=2, dash="solid"), layer="below")
    # 車線境界 (白の実線)
    for lane_x in [-3.5, 3.5]:
        fig.add_shape(type="line",
                      x0=road_offset + lane_x, x1=road_offset + lane_x,
                      y0=y_lo, y1=y_hi,
                      line=dict(color="#FFFFFF", width=1.5, dash="solid"), layer="below")

    # --- 道路矢印 ---
    _add_road_arrows(fig, y_lo, y_hi, road_offset)

    # --- 周囲車両 ---
    if show_vehicles:
        _add_surrounding_vehicles(fig, ego_fwd, ego_lat, road_offset)

    # --- 不確実性バンド ---
    _add_confidence_band(fig, traj, ego_fwd, ego_lat)

    # --- 軌跡 (グラデーション + グロー効果) ---
    for i in range(len(traj) - 1):
        color = GRADIENT_COLORS[min(i, len(GRADIENT_COLORS) - 1)]
        # グロー (太い半透明の線)
        fig.add_trace(go.Scatter(
            x=[plot_x[i], plot_x[i + 1]], y=[plot_y[i], plot_y[i + 1]],
            mode="lines", line=dict(width=12, color=color),
            opacity=0.2, showlegend=False, hoverinfo="skip",
        ))
        # メインの線
        fig.add_trace(go.Scatter(
            x=[plot_x[i], plot_x[i + 1]], y=[plot_y[i], plot_y[i + 1]],
            mode="lines", line=dict(width=4, color=color),
            showlegend=False, hoverinfo="skip",
        ))

    # --- 軌跡ポイント ---
    fig.add_trace(go.Scatter(
        x=plot_x, y=plot_y, mode="markers",
        marker=dict(
            size=[8 + i * 1.5 for i in range(len(traj))],
            color=GRADIENT_COLORS[:len(traj)],
            line=dict(width=1.5, color="white"),
        ),
        name="Predicted Path",
        text=[f"Step {i+1} ({0.5*(i+1):.1f}s)<br>fwd={traj[i][0]:.1f}m lat={traj[i][1]:.1f}m"
              for i in range(len(traj))],
        hovertemplate="%{text}<extra></extra>",
    ))

    # --- 過去の軌跡 ---
    if past_path and len(past_path) > 1:
        # 間引いて表示 (最大20点)
        step = max(1, len(past_path) // 20)
        sampled = past_path[::step]
        if past_path[-1] not in sampled:
            sampled.append(past_path[-1])
        fig.add_trace(go.Scatter(
            x=[p[1] for p in sampled], y=[p[0] for p in sampled],
            mode="lines+markers",
            line=dict(width=2.5, color="#66BB6A"),
            marker=dict(size=5, color="#66BB6A", symbol="circle"),
            name="Past Path", hoverinfo="skip",
        ))

    # --- 自車 (マーカー) ---
    fig.add_trace(go.Scatter(
        x=[ego_lat], y=[ego_fwd],
        mode="markers",
        marker=dict(size=14, color="#00CC66", symbol="triangle-up",
                    line=dict(width=2, color="#00FF88")),
        name="Ego Vehicle", hoverinfo="skip",
    ))

    # --- 到達点マーカー ---
    fig.add_trace(go.Scatter(
        x=[plot_x[-1]], y=[plot_y[-1]], mode="markers",
        marker=dict(size=16, symbol="x", color="#FF3311",
                    line=dict(width=2, color="white")),
        name=f"Destination ({0.5 * len(traj):.1f}s)",
    ))

    fig.update_layout(
        xaxis=dict(title="← Left — Lateral (m) — Right →",
                   range=[x_lo, x_hi], gridcolor="#444444",
                   zerolinecolor="#666666", color="white"),
        yaxis=dict(title="Forward (m) →",
                   range=[y_lo, y_hi], gridcolor="#444444",
                   zerolinecolor="#666666", color="white"),
        height=500, margin=dict(t=80, b=80, l=50, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="left", x=0.0,
                    font=dict(color="white", size=11)),
        plot_bgcolor="#2a2a2a", paper_bgcolor="#1a1a1a",
        font=dict(color="white"),
        title=dict(text=title_suffix, font=dict(size=13, color="#cccccc"),
                   x=1.0, xanchor="right", y=0.99, yanchor="top") if title_suffix else None,
    )
    return fig


def render_gauges(payload: dict, key_prefix: str = "gauge"):
    """スピードメーター風ゲージ + メトリクス表示。"""
    vx = payload.get("velocity", [0, 0])[0]
    vy = payload.get("velocity", [0, 0])[1]
    ax = payload.get("acceleration", [0, 0])[0]
    ay = payload.get("acceleration", [0, 0])[1]
    speed_kmh = math.sqrt(vx ** 2 + vy ** 2) * 3.6
    cmd = payload.get("command", "FORWARD")
    cmd_emoji = {"FORWARD": "⬆️", "LEFT": "↩️", "RIGHT": "↪️"}.get(cmd, "⬆️")

    col_gauge, col_metrics = st.columns([1, 2])

    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=speed_kmh,
            number=dict(suffix=" km/h", font=dict(size=28, color="white")),
            gauge=dict(
                axis=dict(range=[0, 120], tickcolor="white",
                          tickfont=dict(color="white")),
                bar=dict(color="#00CC66"),
                bgcolor="#333333",
                bordercolor="#555555",
                steps=[
                    dict(range=[0, 40], color="#1a3a1a"),
                    dict(range=[40, 80], color="#3a3a1a"),
                    dict(range=[80, 120], color="#3a1a1a"),
                ],
                threshold=dict(line=dict(color="#FF3311", width=3),
                               thickness=0.8, value=speed_kmh),
            ),
        ))
        fig.update_layout(
            height=250, margin=dict(t=40, b=20, l=30, r=30),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_speed")

    with col_metrics:
        cols = st.columns(4)
        cols[0].metric("Vx (m/s)", f"{vx:.1f}")
        cols[1].metric("Vy (m/s)", f"{vy:.1f}")
        cols[2].metric("Accel X/Y", f"{ax:+.1f} / {ay:+.1f}")
        cols[3].metric(f"{cmd_emoji} Command", cmd)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="NAVSIM Trajectory Prediction Demo", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetric"] {
        border-radius: 10px; padding: 10px 14px;
        border: 1px solid #444; background: #1e1e1e;
    }
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    [data-testid="stMetricLabel"] { color: #aaa; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px; border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚗 NAVSIM Trajectory Prediction Demo")

with st.sidebar:
    st.header("⚙️ Settings")
    use_mock = st.toggle("🎭 Mock Demo Mode", value=MOCK_MODE,
                         help="Enable to use simulated predictions without an endpoint")

    if not use_mock:
        discovered = discover_endpoints()
        if discovered:
            endpoint_options = list(discovered.keys())
            endpoint_name = st.selectbox("Endpoint", endpoint_options)
            model_type = discovered[endpoint_name]
            st.caption(f"Model: {model_type}")
        else:
            st.warning("No deployed endpoints found.")
            endpoint_name = st.text_input("Endpoint Name (manual)")
            model_type = "NAVSIM EgoStatusMLP"
            st.caption(f"Model: {model_type}")
    else:
        endpoint_name = "🎭 Mock Demo"
        model_type = "NAVSIM EgoStatusMLP"
        st.info("🎭 Mock mode: predictions are physics-based simulations.")

    show_vehicles = st.toggle("🚙 Show Surrounding Vehicles", value=True)

    with st.expander("ℹ️ About"):
        st.markdown(
            "**Predicted Trajectory** shows the model's predicted future path "
            "(4 seconds, 8 steps at 0.5s intervals) in the ego vehicle's "
            "local coordinate system.\n\n"
            "⬆️ Up = forward (X), ➡️ Right = lateral right (Y).\n\n"
            "The blue shaded band shows prediction uncertainty "
            "(wider = less certain).\n\n"
            "**予測軌跡**は自車ローカル座標系での将来 4 秒間の予測経路です。"
        )

if not endpoint_name:
    st.info(
        "👈 Deploy an inference endpoint first, or enable Mock Demo Mode.\n\n"
        "`./infra/sagemaker-ai-inference/scripts/deploy.sh -c navsim-ego-mlp`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# タブ
# ---------------------------------------------------------------------------
tab_scenario, tab_manual = st.tabs(["🎬 Scenario", "🎛️ Manual"])

with tab_manual:
    col_input, col_viz = st.columns([1, 2])
    with col_input:
        st.subheader(f"{model_type}")
        vx = st.slider("Velocity X (m/s)", 0.0, 30.0, 10.0, 0.5)
        vy = st.slider("Velocity Y (m/s)", -5.0, 5.0, 0.0, 0.1)
        ax = st.slider("Accel X (m/s²)", -5.0, 5.0, 0.0, 0.1)
        ay = st.slider("Accel Y (m/s²)", -5.0, 5.0, 0.0, 0.1)
        command = st.selectbox("Driving Command", ["FORWARD", "LEFT", "RIGHT"])
        payload = {"velocity": [vx, vy], "acceleration": [ax, ay],
                   "command": command}

    with col_viz:
        render_gauges(payload, key_prefix="manual")
        try:
            result = invoke_endpoint(endpoint_name, payload)
            if "trajectory" in result:
                st.plotly_chart(
                    render_trajectory(result["trajectory"],
                                     show_vehicles=show_vehicles),
                    use_container_width=True, key="manual_chart")
            with st.expander("📋 Details", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.json(payload)
                with c2:
                    st.json(result)
        except Exception as e:
            st.error(f"Endpoint error: {e}")

with tab_scenario:
    col_ctrl, col_viz2 = st.columns([1, 2])
    with col_ctrl:
        st.subheader("🎬 Scenario")
        scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()))
        speed = st.slider("Speed (sec/step)", 0.05, 1.0, 0.15, 0.05)
        play = st.button("🔮 Predict", type="primary", use_container_width=True)
        frames = SCENARIOS[scenario_name]["frames"]
        road_offset = SCENARIOS[scenario_name]["road_offset"]
        st.caption(f"{len(frames)} steps")
        if not play:
            st.dataframe(pd.DataFrame(frames), use_container_width=True,
                         height=350)

    with col_viz2:
        if play:
            # --- 全フレームの推論を先に実行 ---
            with st.spinner("Computing all frames..."):
                all_frames_data = []
                ego_fwd, ego_lat = 0.0, 0.0
                past_path = [(0.0, 0.0)]
                dt_frame = 0.5

                for i, frame_payload in enumerate(frames):
                    result = invoke_endpoint(endpoint_name, frame_payload)
                    all_frames_data.append({
                        "payload": frame_payload,
                        "result": result,
                        "ego_pos": (ego_fwd, ego_lat),
                        "past_path": list(past_path),
                    })
                    vx_cur = frame_payload["velocity"][0]
                    vy_cur = (frame_payload["velocity"][1]
                              if len(frame_payload["velocity"]) > 1 else 0)
                    ax_cur = frame_payload["acceleration"][0]
                    ay_cur = (frame_payload["acceleration"][1]
                              if len(frame_payload["acceleration"]) > 1 else 0)
                    ego_fwd += vx_cur * dt_frame + 0.5 * ax_cur * dt_frame ** 2
                    ego_lat -= (vy_cur * dt_frame + 0.5 * ay_cur * dt_frame ** 2)
                    past_path.append((ego_fwd, ego_lat))

            # --- 全フレーム共通の表示範囲を計算 ---
            all_x, all_y = [0.0], [0.0]
            for fd in all_frames_data:
                ef, el = fd["ego_pos"]
                all_x.append(el)
                all_y.append(ef)
                for p in fd["result"].get("trajectory", []):
                    all_x.append(el + p[1])
                    all_y.append(ef + p[0])
            # 道路の範囲も含める
            all_x.extend([road_offset - 6, road_offset + 6])
            margin_x = max((max(all_x) - min(all_x)) * 0.15, 4)
            margin_y = max((max(all_y) - min(all_y)) * 0.1, 5)
            view_range = {
                "x_lo": min(all_x) - margin_x, "x_hi": max(all_x) + margin_x,
                "y_lo": min(all_y) - margin_y, "y_hi": max(all_y) + margin_y,
            }

            # --- Plotly frames でアニメーション ---
            base = all_frames_data[0]
            base_fig = render_trajectory(
                base["result"]["trajectory"], "", road_offset,
                ego_pos=base["ego_pos"], past_path=base["past_path"],
                show_vehicles=show_vehicles, view_range=view_range)

            anim_frames = []
            for i, fd in enumerate(all_frames_data):
                traj = fd["result"].get("trajectory", [])
                if not traj:
                    continue
                ef, el = fd["ego_pos"]
                frame_fig = render_trajectory(
                    traj, "", road_offset,
                    ego_pos=(ef, el), past_path=fd["past_path"],
                    show_vehicles=show_vehicles, view_range=view_range)

                vx_val = fd["payload"]["velocity"][0]
                vy_val = fd["payload"]["velocity"][1] if len(fd["payload"]["velocity"]) > 1 else 0
                ax_val = fd["payload"]["acceleration"][0]
                ay_val = fd["payload"]["acceleration"][1]
                cmd_val = fd["payload"].get("command", "")
                spd = math.sqrt(vx_val**2 + vy_val**2) * 3.6
                cmd_icon = {"FORWARD": "⬆", "LEFT": "↩", "RIGHT": "↪"}.get(cmd_val, "")
                label = (f"{i+1}/{len(frames)} | {spd:.0f}km/h | "
                         f"{cmd_icon}{cmd_val} | "
                         f"Ax={ax_val:+.1f} Ay={ay_val:+.1f}")

                anim_frames.append(go.Frame(
                    data=frame_fig.data,
                    layout=go.Layout(
                        title=dict(text=label, font=dict(size=16, color="white"), x=0.5),
                        xaxis=frame_fig.layout.xaxis,
                        yaxis=frame_fig.layout.yaxis,
                    ),
                    name=str(i),
                ))

            base_fig.frames = anim_frames
            frame_duration = int(speed * 1000)
            base_fig.update_layout(
                updatemenus=[dict(
                    type="buttons", showactive=False,
                    x=0.0, y=-0.12, xanchor="left",
                    buttons=[
                        dict(label="▶ Play", method="animate",
                             args=[None, dict(
                                 frame=dict(duration=frame_duration, redraw=True),
                                 fromcurrent=True, mode="immediate",
                                 transition=dict(duration=frame_duration // 2),
                             )]),
                        dict(label="⏸ Pause", method="animate",
                             args=[[None], dict(
                                 frame=dict(duration=0, redraw=False),
                                 mode="immediate",
                             )]),
                    ],
                )],
                sliders=[dict(
                    active=0, steps=[
                        dict(args=[[str(i)], dict(
                            frame=dict(duration=frame_duration, redraw=True),
                            mode="immediate",
                        )], label=str(i + 1), method="animate")
                        for i in range(len(anim_frames))
                    ],
                    x=0.15, len=0.85, y=-0.05,
                    currentvalue=dict(prefix="Step: ", font=dict(color="white")),
                    font=dict(color="white"),
                    tickcolor="white",
                )],
            )

            st.plotly_chart(base_fig, use_container_width=True, key="scenario_anim")
            st.success(f"✅ '{scenario_name}' — press ▶ Play to animate")

            with st.expander("📋 Last Step", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    st.json(all_frames_data[-1]["payload"])
                with c2:
                    st.json(all_frames_data[-1]["result"])
