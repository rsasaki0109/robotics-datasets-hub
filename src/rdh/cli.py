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
