"""MCD (Multi-Campus Dataset) specific loader and visualizer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

console = Console()


def list_sequences(data_dir: Path) -> list[str]:
    """List available sequences."""
    return sorted(
        p.name for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def load_ground_truth(gt_path: Path) -> np.ndarray:
    """Load ground truth poses from TUM or KITTI format.

    TUM format: timestamp tx ty tz qx qy qz qw
    KITTI format: 3x4 transformation matrix per line
    """
    data = np.loadtxt(str(gt_path))
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] == 8:  # TUM format
        return data
    elif data.shape[1] == 12:  # KITTI format
        poses = []
        for row in data:
            T = row.reshape(3, 4)
            poses.append([0, T[0, 3], T[1, 3], T[2, 3], 0, 0, 0, 1])
        return np.array(poses)
    else:
        return data


def find_ground_truth_files(data_dir: Path) -> list[Path]:
    """Find ground truth pose files."""
    patterns = [
        "**/*ground_truth*.txt", "**/*gt*.txt", "**/*poses*.txt",
        "**/*ground_truth*.csv", "**/*gt*.csv",
    ]
    files = []
    for pat in patterns:
        files.extend(data_dir.glob(pat))
    return sorted(set(files))


def viz_trajectories(
    data_dir: Path,
    save_path: Path | None = None,
) -> None:
    """Visualize all ground truth trajectories found in the dataset."""
    gt_files = find_ground_truth_files(data_dir)

    if not gt_files:
        console.print("[yellow]No ground truth files found.[/]")
        console.print("Looking for *gt*.txt, *ground_truth*.txt, *poses*.txt")
        console.print(f"Contents of {data_dir}:")
        for p in sorted(data_dir.iterdir()):
            console.print(f"  {p.name}")
        return

    n = len(gt_files)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("MCD: Multi-Campus Trajectories", fontsize=14, fontweight="bold")

    for idx, gt_file in enumerate(gt_files):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        try:
            data = load_ground_truth(gt_file)
            x, y = data[:, 1], data[:, 2]
            ax.plot(x, y, "b-", linewidth=0.8, alpha=0.8)
            ax.scatter(x[0], y[0], c="green", s=60, zorder=5, label="Start")
            ax.scatter(x[-1], y[-1], c="red", s=60, zorder=5, label="End")
            ax.set_title(gt_file.parent.name + "/" + gt_file.name, fontsize=8)
            ax.legend(fontsize=7)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("X (m)", fontsize=8)
            ax.set_ylabel("Y (m)", fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", transform=ax.transAxes)
            ax.set_title(gt_file.name, fontsize=8)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        console.print(f"[green]Saved:[/] {save_path}")
    plt.show()


def viz_sequence_stats(
    data_dir: Path,
    save_path: Path | None = None,
) -> None:
    """Visualize statistics across sequences (distance, duration)."""
    gt_files = find_ground_truth_files(data_dir)
    if not gt_files:
        console.print("[yellow]No ground truth files found.[/]")
        return

    names = []
    distances = []
    durations = []

    for gt_file in gt_files:
        try:
            data = load_ground_truth(gt_file)
            x, y, z = data[:, 1], data[:, 2], data[:, 3]
            dx = np.diff(x)
            dy = np.diff(y)
            dz = np.diff(z)
            dist = np.sum(np.sqrt(dx**2 + dy**2 + dz**2))
            duration = data[-1, 0] - data[0, 0]

            names.append(gt_file.parent.name)
            distances.append(dist)
            durations.append(duration)
        except Exception:
            continue

    if not names:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("MCD: Sequence Statistics", fontsize=14, fontweight="bold")

    ax1.barh(names, distances, color="steelblue")
    ax1.set_xlabel("Total Distance (m)")
    ax1.set_title("Trajectory Length")
    ax1.grid(True, alpha=0.3, axis="x")

    ax2.barh(names, durations, color="coral")
    ax2.set_xlabel("Duration (s)")
    ax2.set_title("Sequence Duration")
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        console.print(f"[green]Saved:[/] {save_path}")
    plt.show()
