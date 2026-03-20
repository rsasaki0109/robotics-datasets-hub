"""HM3D-OVON dataset-specific loader and visualizer."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

console = Console()


def find_episode_files(data_dir: Path) -> list[Path]:
    """Find episode JSON/JSONL files."""
    files = []
    for pattern in ["**/*.json", "**/*.json.gz"]:
        files.extend(data_dir.glob(pattern))
    return sorted(files)


def load_episodes(episode_path: Path) -> list[dict]:
    """Load episodes from a JSON file."""
    import gzip

    if episode_path.suffix == ".gz":
        with gzip.open(episode_path, "rt") as f:
            data = json.load(f)
    else:
        with open(episode_path) as f:
            data = json.load(f)

    if isinstance(data, dict) and "episodes" in data:
        return data["episodes"]
    elif isinstance(data, list):
        return data
    return [data]


def viz_episode_overview(
    data_dir: Path,
    n_episodes: int = 20,
    save_path: Path | None = None,
) -> None:
    """Visualize episode statistics: goal objects, start/goal positions."""
    ep_files = find_episode_files(data_dir)
    if not ep_files:
        console.print("[yellow]No episode files found.[/]")
        return

    all_episodes = []
    for ep_file in ep_files:
        try:
            episodes = load_episodes(ep_file)
            all_episodes.extend(episodes)
        except Exception as e:
            console.print(f"[yellow]Skipping {ep_file.name}: {e}[/]")

    if not all_episodes:
        console.print("[yellow]No episodes loaded.[/]")
        return

    console.print(f"Loaded {len(all_episodes)} episodes from {len(ep_files)} files")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("HM3D-OVON: Episode Overview", fontsize=14, fontweight="bold")

    # 1. Goal object distribution
    ax = axes[0]
    goal_objects = []
    for ep in all_episodes:
        obj = ep.get("object_category") or ep.get("goal_object") or ep.get("target", "")
        if obj:
            goal_objects.append(str(obj))

    if goal_objects:
        from collections import Counter
        counts = Counter(goal_objects).most_common(15)
        labels, values = zip(*counts)
        ax.barh(list(labels), list(values), color="steelblue")
        ax.set_xlabel("Count")
        ax.set_title("Top Goal Objects")
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, "No goal objects found", ha="center", transform=ax.transAxes)
        ax.set_title("Goal Objects")

    # 2. Start positions (top-down view)
    ax = axes[1]
    start_positions = []
    for ep in all_episodes[:n_episodes * 10]:
        pos = ep.get("start_position")
        if pos and len(pos) >= 2:
            start_positions.append(pos)

    if start_positions:
        pos_arr = np.array(start_positions)
        ax.scatter(pos_arr[:, 0], pos_arr[:, 2] if pos_arr.shape[1] > 2 else pos_arr[:, 1],
                   s=10, alpha=0.5, c="blue")
        ax.set_title(f"Start Positions ({len(pos_arr)} episodes)")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
    else:
        ax.text(0.5, 0.5, "No start positions", ha="center", transform=ax.transAxes)
        ax.set_title("Start Positions")

    # 3. Geodesic distances
    ax = axes[2]
    distances = []
    for ep in all_episodes:
        d = ep.get("geodesic_distance") or ep.get("info", {}).get("geodesic_distance")
        if d is not None:
            distances.append(float(d))

    if distances:
        ax.hist(distances, bins=30, color="coral", edgecolor="white", alpha=0.8)
        ax.set_xlabel("Geodesic Distance (m)")
        ax.set_ylabel("Count")
        ax.set_title(f"Goal Distance Distribution (n={len(distances)})")
        ax.axvline(np.mean(distances), color="red", linestyle="--",
                   label=f"Mean: {np.mean(distances):.1f}m")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No distance data", ha="center", transform=ax.transAxes)
        ax.set_title("Goal Distances")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        console.print(f"[green]Saved:[/] {save_path}")
    plt.show()


def viz_episode_detail(
    data_dir: Path,
    episode_idx: int = 0,
    save_path: Path | None = None,
) -> None:
    """Show details of a single episode."""
    ep_files = find_episode_files(data_dir)
    if not ep_files:
        console.print("[yellow]No episode files found.[/]")
        return

    all_episodes = []
    for ep_file in ep_files:
        try:
            all_episodes.extend(load_episodes(ep_file))
        except Exception:
            continue

    if episode_idx >= len(all_episodes):
        console.print(f"[red]Episode {episode_idx} not found. Total: {len(all_episodes)}[/]")
        return

    ep = all_episodes[episode_idx]

    console.print(f"\n[bold cyan]Episode {episode_idx}[/]")
    for key, val in ep.items():
        if isinstance(val, (list, dict)) and len(str(val)) > 100:
            console.print(f"  {key}: [{type(val).__name__}, len={len(val)}]")
        else:
            console.print(f"  {key}: {val}")
