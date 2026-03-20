"""CoVLA dataset-specific loader and visualizer."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

console = Console()


def list_scenes(data_dir: Path) -> list[str]:
    """List available scene IDs from the states directory."""
    states_dir = data_dir / "states"
    if not states_dir.exists():
        console.print("[yellow]No states/ directory found. Download metadata first.[/]")
        return []
    return sorted(p.stem for p in states_dir.glob("*.jsonl"))


def load_scene(data_dir: Path, scene_id: str) -> dict:
    """Load states and captions for a single scene."""
    states = _load_jsonl(data_dir / "states" / f"{scene_id}.jsonl")
    captions = _load_jsonl(data_dir / "captions" / f"{scene_id}.jsonl")

    caption_map = {c["frame_id"]: c.get("caption", "") for c in captions}
    for s in states:
        s["caption"] = caption_map.get(s["frame_id"], "")

    return {"scene_id": scene_id, "frames": states, "n_frames": len(states)}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def viz_scene(data_dir: Path, scene_id: str, save_path: Path | None = None) -> None:
    """Visualize a CoVLA scene: trajectory, speed, steering, and captions."""
    scene = load_scene(data_dir, scene_id)
    frames = scene["frames"]
    if not frames:
        console.print(f"[red]No frames found for scene {scene_id}[/]")
        return

    timestamps = np.array([f.get("timestamp", i) for i, f in enumerate(frames)])
    t_sec = (timestamps - timestamps[0]) / 1000.0

    speeds = np.array([f.get("vEgo", 0) for f in frames]) * 3.6  # m/s -> km/h
    steerings = np.array([f.get("steeringAngleDeg", 0) for f in frames])
    accels = np.array([f.get("aEgo", 0) for f in frames])
    brakes = np.array([f.get("brake", 0) for f in frames])
    gases = np.array([f.get("gas", 0) for f in frames])

    # Extract trajectory from ECEF positions
    positions = [f.get("positions_ecef", [0, 0, 0]) for f in frames]
    pos = np.array(positions)
    # Convert to local frame (relative to first position)
    pos_local = pos - pos[0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"CoVLA Scene: {scene_id}", fontsize=14, fontweight="bold")

    # 1. Trajectory (XY)
    ax = axes[0, 0]
    ax.plot(pos_local[:, 0], pos_local[:, 1], "b-", linewidth=1.2, alpha=0.8)
    ax.scatter(pos_local[0, 0], pos_local[0, 1], c="green", s=100, zorder=5, label="Start")
    ax.scatter(pos_local[-1, 0], pos_local[-1, 1], c="red", s=100, zorder=5, label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Trajectory (ECEF local)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 2. Speed
    ax = axes[0, 1]
    ax.plot(t_sec, speeds, "b-", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Vehicle Speed")
    ax.grid(True, alpha=0.3)

    # 3. Steering
    ax = axes[0, 2]
    ax.plot(t_sec, steerings, "r-", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Steering Angle (deg)")
    ax.set_title("Steering Angle")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # 4. Acceleration
    ax = axes[1, 0]
    ax.plot(t_sec, accels, "g-", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("Acceleration")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # 5. Brake / Gas pedals
    ax = axes[1, 1]
    ax.plot(t_sec, gases, "g-", linewidth=1, label="Gas")
    ax.plot(t_sec, brakes, "r-", linewidth=1, label="Brake")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pedal Value")
    ax.set_title("Gas / Brake Pedals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Caption samples
    ax = axes[1, 2]
    ax.axis("off")
    ax.set_title("Language Annotations (samples)")
    captions_with_frame = [
        (f["frame_id"], f["caption"]) for f in frames if f.get("caption")
    ]
    if captions_with_frame:
        samples = captions_with_frame[:: max(1, len(captions_with_frame) // 4)][:4]
        text = ""
        for fid, cap in samples:
            cap_short = cap[:120] + "..." if len(cap) > 120 else cap
            text += f"[Frame {fid}]\n{cap_short}\n\n"
        ax.text(
            0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )
    else:
        ax.text(0.5, 0.5, "No captions available", transform=ax.transAxes,
                ha="center", fontsize=12, color="gray")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        console.print(f"[green]Saved:[/] {save_path}")
    plt.show()


def viz_multi_scene_trajectories(
    data_dir: Path, n_scenes: int = 9, save_path: Path | None = None
) -> None:
    """Plot trajectories of multiple scenes in a grid."""
    scenes = list_scenes(data_dir)[:n_scenes]
    if not scenes:
        return

    cols = min(3, len(scenes))
    rows = (len(scenes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("CoVLA Mini - Scene Trajectories", fontsize=14, fontweight="bold")

    for idx, scene_id in enumerate(scenes):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        scene = load_scene(data_dir, scene_id)
        positions = [f.get("positions_ecef", [0, 0, 0]) for f in scene["frames"]]
        pos = np.array(positions)
        pos_local = pos - pos[0]
        speeds = np.array([f.get("vEgo", 0) for f in scene["frames"]]) * 3.6

        sc = ax.scatter(pos_local[:, 0], pos_local[:, 1], c=speeds, s=2, cmap="RdYlGn_r")
        ax.scatter(pos_local[0, 0], pos_local[0, 1], c="blue", s=40, zorder=5, marker="^")
        ax.set_title(scene_id[:25] + "...", fontsize=7)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)

    # Remove empty subplots
    for idx in range(len(scenes), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    fig.colorbar(sc, ax=axes, label="Speed (km/h)", shrink=0.6)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        console.print(f"[green]Saved:[/] {save_path}")
    plt.show()
