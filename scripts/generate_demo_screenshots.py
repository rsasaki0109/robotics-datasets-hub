#!/usr/bin/env python3
"""Generate demo screenshots using synthetic CoVLA-like data for README."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "docs" / "images"
DEMO_DATA_DIR = REPO_ROOT / "data" / "covla_demo"


def generate_synthetic_covla_scenes(n_scenes: int = 9) -> None:
    """Generate synthetic CoVLA-like JSONL data for demo purposes."""
    np.random.seed(42)
    states_dir = DEMO_DATA_DIR / "states"
    captions_dir = DEMO_DATA_DIR / "captions"
    states_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    scene_types = [
        ("straight_highway", "highway driving at high speed"),
        ("left_turn", "turning left at intersection"),
        ("right_turn", "turning right at intersection"),
        ("s_curve", "navigating an S-curve road"),
        ("roundabout", "entering a roundabout"),
        ("stop_and_go", "stop-and-go traffic"),
        ("parking", "parking maneuver"),
        ("lane_change_left", "changing lanes to the left"),
        ("lane_change_right", "changing lanes to the right"),
    ]

    for i in range(min(n_scenes, len(scene_types))):
        scene_name, desc = scene_types[i]
        scene_id = f"2024-demo-{scene_name}"
        n_frames = 600  # 30 sec @ 20 fps

        t = np.linspace(0, 30, n_frames)
        x, y = _generate_trajectory(scene_name, t)

        # Compute derived signals
        dx = np.gradient(x, t)
        dy = np.gradient(y, t)
        speed = np.sqrt(dx**2 + dy**2)
        heading = np.arctan2(dy, dx)
        steering = np.degrees(np.gradient(heading, t)) * 2

        accel = np.gradient(speed, t)
        gas = np.clip(accel * 0.3, 0, 1)
        brake = np.clip(-accel * 0.3, 0, 1)

        # ECEF-like positions (offset from origin)
        base_ecef = np.array([3950000.0, 3350000.0, 3700000.0])

        states = []
        for j in range(n_frames):
            states.append({
                "frame_id": j,
                "timestamp": int((1700000000 + t[j]) * 1000),
                "vEgo": float(speed[j]),
                "aEgo": float(accel[j]),
                "steeringAngleDeg": float(steering[j]),
                "brake": float(brake[j]),
                "gas": float(gas[j]),
                "positions_ecef": [
                    float(base_ecef[0] + x[j]),
                    float(base_ecef[1] + y[j]),
                    float(base_ecef[2]),
                ],
            })

        captions_data = []
        weather = np.random.choice(["sunny", "cloudy", "rainy"])
        traffic = np.random.choice(["light", "moderate", "heavy"])
        for j in range(0, n_frames, 20):
            spd_kmh = speed[j] * 3.6
            if spd_kmh < 5:
                motion = "stopped"
            elif spd_kmh < 30:
                motion = "moving slowly"
            elif spd_kmh < 60:
                motion = "moving at moderate speed"
            else:
                motion = "moving at high speed"

            caption = (
                f"The ego vehicle is {motion} ({spd_kmh:.0f} km/h), {desc}. "
                f"Weather is {weather} with {traffic} traffic. "
                f"Steering angle: {steering[j]:.1f} degrees."
            )
            captions_data.append({"frame_id": j, "caption": caption})

        with open(states_dir / f"{scene_id}.jsonl", "w") as f:
            for s in states:
                f.write(json.dumps(s) + "\n")

        with open(captions_dir / f"{scene_id}.jsonl", "w") as f:
            for c in captions_data:
                f.write(json.dumps(c) + "\n")

    print(f"Generated {n_scenes} synthetic scenes in {DEMO_DATA_DIR}")


def _generate_trajectory(scene_type: str, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate trajectory coordinates based on scene type."""
    n = len(t)
    if scene_type == "straight_highway":
        speed = 25.0 + np.random.randn(n) * 0.5
        x = np.cumsum(speed * np.gradient(t))
        y = np.cumsum(np.random.randn(n) * 0.02)
    elif scene_type == "left_turn":
        theta = np.linspace(0, np.pi / 2, n)
        r = 30
        x = r * np.sin(theta) + np.cumsum(np.random.randn(n) * 0.01)
        y = r * (1 - np.cos(theta)) + np.cumsum(np.random.randn(n) * 0.01)
    elif scene_type == "right_turn":
        theta = np.linspace(0, np.pi / 2, n)
        r = 25
        x = r * np.sin(theta)
        y = -r * (1 - np.cos(theta))
    elif scene_type == "s_curve":
        x = np.linspace(0, 200, n)
        y = 20 * np.sin(x / 200 * 2 * np.pi)
    elif scene_type == "roundabout":
        theta = np.linspace(0, 1.5 * np.pi, n)
        r = 20
        x = r * np.cos(theta) - r
        y = r * np.sin(theta)
    elif scene_type == "stop_and_go":
        speed = 8 * (1 + np.sin(t * 0.8)) / 2
        x = np.cumsum(speed * np.gradient(t))
        y = np.cumsum(np.random.randn(n) * 0.01)
    elif scene_type == "parking":
        x = np.linspace(0, 15, n)
        y = 3 * np.tanh((t - 15) / 5)
    elif scene_type == "lane_change_left":
        x = np.linspace(0, 300, n)
        y = 3.5 * (1 / (1 + np.exp(-(t - 15) * 2)))
    elif scene_type == "lane_change_right":
        x = np.linspace(0, 300, n)
        y = -3.5 * (1 / (1 + np.exp(-(t - 15) * 2)))
    else:
        x = np.linspace(0, 100, n)
        y = np.zeros(n)

    return x, y


def generate_screenshots() -> None:
    """Generate all demo screenshots."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Add src to path
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from rdh.datasets.covla import viz_multi_scene_trajectories, viz_scene

    # 1. Multi-scene trajectory overview
    print("Generating multi-scene trajectory overview ...")
    viz_multi_scene_trajectories(
        DEMO_DATA_DIR, n_scenes=9,
        save_path=ARTIFACTS_DIR / "covla_multi_trajectories.png",
    )

    # 2. Single scene detail
    from rdh.datasets.covla import list_scenes
    scenes = list_scenes(DEMO_DATA_DIR)
    if scenes:
        print(f"Generating single-scene detail for {scenes[0]} ...")
        viz_scene(
            DEMO_DATA_DIR, scenes[0],
            save_path=ARTIFACTS_DIR / "covla_scene_detail.png",
        )

    # 3. CLI demo screenshot (text-based)
    _generate_cli_screenshot()

    print(f"\nAll screenshots saved to {ARTIFACTS_DIR}")


def _generate_cli_screenshot() -> None:
    """Generate a figure showing CLI usage example."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor("#1e1e2e")
    fig.patch.set_facecolor("#1e1e2e")
    ax.axis("off")

    cli_text = """$ rdh list
┌───────────┬──────────────────────────────────┬──────────────────┬───────────────────┬────────────┐
│ Name      │ Description                      │ Modalities       │ Tasks             │ License    │
├───────────┼──────────────────────────────────┼──────────────────┼───────────────────┼────────────┤
│ covla     │ Vision-Language-Action Dataset    │ vision, language  │ autonomous-driving│ Non-comm.  │
│ hm3d_ovon │ Open-Vocab Object Goal Nav       │ RGB-D, 3D-mesh   │ object-goal-nav   │ Non-comm.  │
│ polaris   │ Maritime Detection/Tracking      │ RGB, TIR, radar   │ object-detection  │ N/A        │
│ mcd       │ Multi-Campus Robot Perception    │ multi-LiDAR, cam  │ SLAM, segmentation│ CC BY-NC   │
│ ggrt      │ Pose-free 3D Gaussian Splatting  │ RGB               │ novel-view-synth. │ N/A        │
│ slabim    │ SLAM coupled with BIM            │ LiDAR, camera     │ SLAM, BIM-reg.    │ N/A        │
│ hk_mems   │ Multi-LiDAR Extreme Mapping     │ MEMS-LiDAR, cam   │ LiDAR-SLAM        │ N/A        │
│ geode     │ Geometric Degeneracy SLAM        │ multi-LiDAR       │ LiDAR-SLAM        │ N/A        │
└───────────┴──────────────────────────────────┴──────────────────┴───────────────────┴────────────┘

$ rdh info covla
  CoVLA: Comprehensive Vision-Language-Action Dataset
  Paper      : https://arxiv.org/abs/2408.10845
  Modalities : vision, language, action
  Tasks      : autonomous-driving, vision-language-action
  Size       : Mini: 50 scenes, Full: 10k scenes

$ rdh download covla --split metadata"""

    ax.text(
        0.02, 0.98, cli_text,
        transform=ax.transAxes, fontsize=8, fontfamily="monospace",
        color="#cdd6f4", verticalalignment="top",
    )

    plt.savefig(
        ARTIFACTS_DIR / "cli_demo.png", dpi=150, bbox_inches="tight",
        facecolor="#1e1e2e",
    )
    plt.close()
    print(f"Saved CLI demo: {ARTIFACTS_DIR / 'cli_demo.png'}")


if __name__ == "__main__":
    generate_synthetic_covla_scenes(9)
    generate_screenshots()
