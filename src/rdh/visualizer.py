"""Lightweight visualization helpers for common robotics data formats."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

console = Console()


def viz_images(data_dir: Path, n_samples: int = 5, pattern: str = "**/*.png") -> None:
    """Display a grid of sample images from a dataset directory."""
    paths = sorted(data_dir.glob(pattern))
    if not paths:
        paths = sorted(data_dir.glob("**/*.jpg"))
    if not paths:
        console.print("[yellow]No images found.[/]")
        return

    paths = paths[:n_samples]
    fig, axes = plt.subplots(1, len(paths), figsize=(4 * len(paths), 4))
    if len(paths) == 1:
        axes = [axes]
    for ax, p in zip(axes, paths):
        img = plt.imread(str(p))
        ax.imshow(img)
        ax.set_title(p.name, fontsize=8)
        ax.axis("off")
    fig.suptitle(f"Samples from {data_dir.name}", fontsize=12)
    plt.tight_layout()
    plt.show()


def viz_trajectory_csv(csv_path: Path, x_col: int = 0, y_col: int = 1) -> None:
    """Plot a 2D trajectory from a CSV file."""
    data = np.loadtxt(str(csv_path), delimiter=",", skiprows=1)
    plt.figure(figsize=(8, 8))
    plt.plot(data[:, x_col], data[:, y_col], "b-", linewidth=0.8)
    plt.scatter(data[0, x_col], data[0, y_col], c="green", s=80, zorder=5, label="Start")
    plt.scatter(data[-1, x_col], data[-1, y_col], c="red", s=80, zorder=5, label="End")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Trajectory: {csv_path.name}")
    plt.legend()
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def viz_point_cloud(pcd_path: Path) -> None:
    """Visualize a point cloud file using open3d (optional dependency)."""
    try:
        import open3d as o3d
    except ImportError:
        console.print("[yellow]open3d not installed. Install with: pip install rdh[viz3d][/]")
        return

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    console.print(f"  Points: {len(pcd.points)}")
    o3d.visualization.draw_geometries([pcd], window_name=pcd_path.name)
