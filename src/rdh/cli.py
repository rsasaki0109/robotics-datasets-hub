"""CLI entry point for robotics-datasets-hub (rdh)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from rdh import __version__
from rdh.registry import Registry

app = typer.Typer(
    name="rdh",
    help="robotics-datasets-hub: One-command download & visualize curated AI/Robotics datasets.",
    add_completion=False,
)
console = Console()


def _registry() -> Registry:
    return Registry()


@app.command()
def list(query: Optional[str] = typer.Argument(None, help="Search query to filter datasets")):
    """List available datasets."""
    reg = _registry()
    entries = reg.search(query) if query else reg.all()

    if not entries:
        console.print("[yellow]No datasets found.[/]")
        raise typer.Exit()

    table = Table(title="Available Datasets", show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", max_width=50)
    table.add_column("Modalities", style="green")
    table.add_column("Tasks", style="magenta")
    table.add_column("License", style="yellow")

    for e in entries:
        table.add_row(
            e.name,
            e.description[:80],
            ", ".join(e.modalities),
            ", ".join(e.tasks),
            e.license,
        )

    console.print(table)


@app.command()
def info(name: str = typer.Argument(..., help="Dataset name")):
    """Show detailed information about a dataset."""
    reg = _registry()
    entry = reg.get(name)
    if not entry:
        console.print(f"[red]Dataset '{name}' not found.[/]")
        console.print(f"Available: {', '.join(reg.list_names())}")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]{entry.display_name}[/]")
    console.print(f"  [dim]{entry.description}[/]\n")
    console.print(f"  Paper      : {entry.paper_url}")
    console.print(f"  Project    : {entry.project_url}")
    if entry.github_url:
        console.print(f"  GitHub     : {entry.github_url}")
    if entry.huggingface_id:
        console.print(f"  HuggingFace: https://huggingface.co/datasets/{entry.huggingface_id}")
    console.print(f"  Modalities : {', '.join(entry.modalities)}")
    console.print(f"  Tasks      : {', '.join(entry.tasks)}")
    console.print(f"  License    : {entry.license}")
    console.print(f"  Size       : {entry.size_hint}")
    console.print(f"  DL Method  : {entry.download.get('method', 'wget')}")
    if entry.tags:
        console.print(f"  Tags       : {', '.join(entry.tags)}")
    console.print()


@app.command()
def download(
    name: str = typer.Argument(..., help="Dataset name"),
    split: Optional[str] = typer.Option(None, "--split", "-s", help="Dataset split (e.g. mini)"),
    output: Path = typer.Option(Path("./data"), "--output", "-o", help="Output directory"),
):
    """Download a dataset."""
    from rdh.downloader import download_dataset

    reg = _registry()
    entry = reg.get(name)
    if not entry:
        console.print(f"[red]Dataset '{name}' not found.[/]")
        raise typer.Exit(1)

    download_dataset(entry, output, split)


@app.command()
def viz(
    name: str = typer.Argument(..., help="Dataset name"),
    data_dir: Path = typer.Option(Path("./data"), "--data-dir", "-d", help="Data directory"),
    n_samples: int = typer.Option(5, "--samples", "-n", help="Number of samples to visualize"),
):
    """Visualize dataset samples."""
    from rdh.visualizer import viz_images

    dataset_dir = data_dir / name
    if not dataset_dir.exists():
        console.print(f"[red]Data directory not found: {dataset_dir}[/]")
        console.print("Run [cyan]rdh download {name}[/] first.")
        raise typer.Exit(1)

    viz_images(dataset_dir, n_samples=n_samples)


@app.command()
def demo(
    name: str = typer.Argument(..., help="Dataset name (e.g. covla)"),
    data_dir: Path = typer.Option(Path("./data"), "--data-dir", "-d", help="Data directory"),
    scene: Optional[str] = typer.Option(None, "--scene", "-s", help="Scene ID"),
    save: Optional[Path] = typer.Option(None, "--save", help="Save figure to path"),
):
    """Run dataset-specific demo visualization."""
    dataset_dir = data_dir / name
    if not dataset_dir.exists():
        # Fall back to data_dir itself if it looks like the dataset directory
        if data_dir.exists() and data_dir.name != name:
            dataset_dir = data_dir
        else:
            console.print(f"[red]Data directory not found: {dataset_dir}[/]")
            console.print(f"Run [cyan]rdh download {name} --split metadata[/] first.")
            raise typer.Exit(1)

    if name == "covla":
        from rdh.datasets.covla import list_scenes, viz_multi_scene_trajectories, viz_scene

        scenes = list_scenes(dataset_dir)
        if not scenes:
            raise typer.Exit(1)
        console.print(f"[green]Found {len(scenes)} scenes[/]")

        if scene:
            viz_scene(dataset_dir, scene, save_path=save)
        else:
            console.print("Plotting multi-scene trajectory overview ...")
            viz_multi_scene_trajectories(dataset_dir, save_path=save)
    elif name == "polaris":
        nav_dir = dataset_dir / "navigation"
        if nav_dir.exists():
            from rdh.datasets.pohang_canal import viz_pohang_canal

            console.print("PoLaRIS/Pohang Canal: real sensor data visualization ...")
            viz_pohang_canal(dataset_dir, save_path=save)
        else:
            from rdh.datasets.polaris import viz_detection_annotations, viz_sensor_comparison

            console.print("PoLaRIS: multi-sensor maritime visualization ...")
            viz_sensor_comparison(dataset_dir, save_path=save)
            viz_detection_annotations(dataset_dir, save_path=save)
    elif name == "mcd":
        from rdh.datasets.mcd import viz_sequence_stats, viz_trajectories

        console.print("MCD: multi-campus trajectory visualization ...")
        viz_trajectories(dataset_dir, save_path=save)
        viz_sequence_stats(dataset_dir)
    elif name == "hm3d_ovon":
        from rdh.datasets.hm3d_ovon import viz_episode_overview

        console.print("HM3D-OVON: episode overview ...")
        viz_episode_overview(dataset_dir, save_path=save)
    else:
        console.print(f"[yellow]No specific demo for '{name}'. Using generic viz.[/]")
        from rdh.visualizer import viz_images

        viz_images(dataset_dir)


@app.command()
def compare(
    names: Optional[str] = typer.Argument(
        None, help="Comma-separated dataset names (default: all)"
    ),
    save: Optional[Path] = typer.Option(None, "--save", help="Save comparison chart"),
):
    """Compare datasets side-by-side with radar chart and summary table."""
    import matplotlib

    matplotlib.use("Agg" if save else matplotlib.get_backend())
    import matplotlib.pyplot as plt
    import numpy as np

    reg = _registry()
    if names:
        entries = [reg.get(n.strip()) for n in names.split(",")]
        entries = [e for e in entries if e is not None]
    else:
        entries = reg.all()

    if not entries:
        console.print("[yellow]No datasets to compare.[/]")
        raise typer.Exit()

    # Summary table
    table = Table(title="Dataset Comparison", show_lines=True)
    table.add_column("", style="bold")
    for e in entries:
        table.add_column(e.name, style="cyan", no_wrap=True)

    rows = {
        "Modalities": [str(len(e.modalities)) for e in entries],
        "Tasks": [str(len(e.tasks)) for e in entries],
        "License": [e.license[:20] for e in entries],
        "DL Method": [e.download.get("method", "?") for e in entries],
        "Size": [e.size_hint[:30] if e.size_hint else "N/A" for e in entries],
        "Has Paper": ["Yes" if e.paper_url else "No" for e in entries],
        "Has GitHub": ["Yes" if e.github_url else "No" for e in entries],
        "HuggingFace": ["Yes" if e.huggingface_id else "No" for e in entries],
    }
    for label, vals in rows.items():
        table.add_row(label, *vals)
    console.print(table)

    # Radar chart
    all_modalities = sorted({m for e in entries for m in e.modalities})
    all_tasks = sorted({t for e in entries for t in e.tasks})

    categories = ["Modalities", "Tasks", "Paper", "GitHub", "HuggingFace", "Tags"]
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(entries)))

    for idx, entry in enumerate(entries):
        values = [
            len(entry.modalities),
            len(entry.tasks),
            1 if entry.paper_url else 0,
            1 if entry.github_url else 0,
            1 if entry.huggingface_id else 0,
            len(entry.tags),
        ]
        max_vals = [len(all_modalities), len(all_tasks), 1, 1, 1, 10]
        normalized = [v / m if m > 0 else 0 for v, m in zip(values, max_vals)]
        normalized += normalized[:1]
        ax.plot(angles, normalized, "o-", color=colors[idx], linewidth=2, label=entry.name)
        ax.fill(angles, normalized, color=colors[idx], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title("Dataset Feature Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout()
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save, dpi=150, bbox_inches="tight")
        console.print(f"[green]Saved:[/] {save}")
    else:
        plt.show()
    plt.close()


@app.command()
def dashboard():
    """Launch the Streamlit web dashboard."""
    import subprocess
    import sys

    app_path = Path(__file__).parent / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


@app.command()
def version():
    """Show version."""
    console.print(f"robotics-datasets-hub v{__version__}")


if __name__ == "__main__":
    app()
