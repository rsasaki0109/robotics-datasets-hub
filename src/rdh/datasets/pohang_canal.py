"""Pohang Canal Dataset (PoLaRIS base data) loader and visualizer for real data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

console = Console()

# WGS84 constants
A_WGS84 = 6378137.0
F_WGS84 = 1 / 298.257223563


def load_gps(gps_path: Path) -> np.ndarray:
    """Load GPS data: timestamp, lat, lon, alt, etc."""
    rows = []
    with open(gps_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            ts = float(parts[0])
            lat = float(parts[2])
            if parts[3] == "S":
                lat = -lat
            lon = float(parts[4])
            if parts[5] == "W":
                lon = -lon
            alt = float(parts[6])
            rows.append([ts, lat, lon, alt])
    return np.array(rows)


def load_ahrs(ahrs_path: Path) -> np.ndarray:
    """Load AHRS data: timestamp, qx, qy, qz, qw, wx, wy, wz, ax, ay, az."""
    rows = []
    with open(ahrs_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 11:
                rows.append([float(p) for p in parts[:11]])
    return np.array(rows)


def load_extrinsics(calib_path: Path) -> dict:
    """Load sensor extrinsic calibration."""
    with open(calib_path) as f:
        return json.load(f)


def latlon_to_local(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert lat/lon to local ENU coordinates (meters) relative to first point."""
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat0 = lat_rad[0]
    lon0 = lon_rad[0]

    e2 = 2 * F_WGS84 - F_WGS84**2
    N = A_WGS84 / np.sqrt(1 - e2 * np.sin(lat0) ** 2)

    x = (lon_rad - lon0) * N * np.cos(lat0)
    y = (lat_rad - lat0) * N
    return x, y


def viz_pohang_canal(
    data_dir: Path,
    save_path: Path | None = None,
) -> None:
    """Full visualization of Pohang Canal real data."""
    gps_path = data_dir / "navigation" / "gps.txt"
    ahrs_path = data_dir / "navigation" / "ahrs.txt"
    calib_path = data_dir / "calibration" / "extrinsics.json"

    if not gps_path.exists():
        console.print(f"[red]GPS data not found: {gps_path}[/]")
        return

    gps = load_gps(gps_path)
    console.print(f"[green]GPS:[/] {len(gps)} points, "
                  f"lat [{gps[:, 1].min():.6f}, {gps[:, 1].max():.6f}], "
                  f"lon [{gps[:, 2].min():.6f}, {gps[:, 2].max():.6f}]")

    x, y = latlon_to_local(gps[:, 1], gps[:, 2])
    t_gps = gps[:, 0] - gps[0, 0]

    has_ahrs = ahrs_path.exists()
    has_calib = calib_path.exists()

    n_plots = 2 + (2 if has_ahrs else 0) + (1 if has_calib else 0)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    fig.suptitle(
        "Pohang Canal Dataset (pohang00) - Real Sensor Data", fontsize=14, fontweight="bold"
    )
    plot_idx = 0

    # 1. GPS trajectory
    ax = axes_flat[plot_idx]
    plot_idx += 1
    speed_ms = np.sqrt(np.gradient(x, t_gps)**2 + np.gradient(y, t_gps)**2)
    speed_ms = np.clip(speed_ms, 0, 15)
    sc = ax.scatter(x, y, c=speed_ms, s=2, cmap="RdYlGn_r", alpha=0.8)
    ax.scatter(x[0], y[0], c="blue", s=80, zorder=5, marker="^", label="Start")
    ax.scatter(x[-1], y[-1], c="red", s=80, zorder=5, marker="s", label="End")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("GPS Trajectory (speed-colored)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Speed (m/s)", shrink=0.8)

    # 2. GPS altitude over time
    ax = axes_flat[plot_idx]
    plot_idx += 1
    ax.plot(t_gps, gps[:, 3], "b-", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("GPS Altitude")
    ax.grid(True, alpha=0.3)

    # 3-4. AHRS data
    if has_ahrs:
        ahrs = load_ahrs(ahrs_path)
        console.print(f"[green]AHRS:[/] {len(ahrs)} samples")
        t_ahrs = ahrs[:, 0] - ahrs[0, 0]

        # Angular velocity
        ax = axes_flat[plot_idx]
        plot_idx += 1
        ax.plot(t_ahrs, ahrs[:, 5], "r-", linewidth=0.3, alpha=0.7, label="wx")
        ax.plot(t_ahrs, ahrs[:, 6], "g-", linewidth=0.3, alpha=0.7, label="wy")
        ax.plot(t_ahrs, ahrs[:, 7], "b-", linewidth=0.3, alpha=0.7, label="wz")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angular Velocity (rad/s)")
        ax.set_title("AHRS: Gyroscope")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Acceleration
        ax = axes_flat[plot_idx]
        plot_idx += 1
        ax.plot(t_ahrs, ahrs[:, 8], "r-", linewidth=0.3, alpha=0.7, label="ax")
        ax.plot(t_ahrs, ahrs[:, 9], "g-", linewidth=0.3, alpha=0.7, label="ay")
        ax.plot(t_ahrs, ahrs[:, 10], "b-", linewidth=0.3, alpha=0.7, label="az")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Acceleration (m/s²)")
        ax.set_title("AHRS: Accelerometer")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # 5. Sensor configuration
    if has_calib:
        extr = load_extrinsics(calib_path)
        ax = axes_flat[plot_idx]
        plot_idx += 1

        sensor_names = []
        sensor_x = []
        sensor_y = []
        for name, data in extr.items():
            t = data["translation"]
            sensor_names.append(name)
            sensor_x.append(t[0])
            sensor_y.append(t[1])

        colors_map = {
            "gps": "red", "stereo_left": "blue", "stereo_right": "cyan",
            "infrared": "orange", "radar_high": "green", "radar_low": "lime",
            "lidar_front": "purple", "lidar_port": "magenta", "lidar_starboard": "pink",
        }
        for name, sx, sy in zip(sensor_names, sensor_x, sensor_y):
            c = "gray"
            for key, col in colors_map.items():
                if key in name:
                    c = col
                    break
            ax.scatter(sx, sy, c=c, s=60, zorder=5)
            ax.annotate(name, (sx, sy), fontsize=5, rotation=30,
                        xytext=(3, 3), textcoords="offset points")

        ax.set_xlabel("X (m, forward)")
        ax.set_ylabel("Y (m, left)")
        ax.set_title("Sensor Configuration (extrinsics)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(plot_idx, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        console.print(f"[green]Saved:[/] {save_path}")
    else:
        plt.show()
    plt.close()
