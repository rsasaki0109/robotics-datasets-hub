"""Unified download engine supporting HuggingFace Hub, Google Drive (gdown), wget, and git clone."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rich.console import Console

from rdh.registry import DatasetEntry

console = Console()


def download_dataset(entry: DatasetEntry, output_dir: Path, split: str | None = None) -> Path:
    """Download a dataset using the appropriate backend."""
    dest = output_dir / entry.name
    dest.mkdir(parents=True, exist_ok=True)

    method = entry.download.get("method", "wget")
    url = _resolve_url(entry, split)

    console.print(f"[bold green]Downloading[/] {entry.display_name} via {method} ...")
    console.print(f"  URL : {url}")
    console.print(f"  Dest: {dest}")

    if method == "huggingface":
        _download_huggingface(entry, dest, split)
    elif method == "gdown":
        _download_gdown(url, dest)
    elif method == "git":
        _download_git(url, dest)
    else:
        _download_wget(url, dest)

    console.print(f"[bold green]Done![/] Files saved to {dest}")
    return dest


def _resolve_url(entry: DatasetEntry, split: str | None) -> str:
    dl = entry.download
    if split and "splits" in dl and split in dl["splits"]:
        return dl["splits"][split]
    return dl.get("url", "")


def _download_huggingface(entry: DatasetEntry, dest: Path, split: str | None) -> None:
    from huggingface_hub import snapshot_download

    repo_id = entry.huggingface_id or entry.download.get("url", "")
    kwargs: dict = {"repo_id": repo_id, "repo_type": "dataset", "local_dir": str(dest)}
    if split and "splits" in entry.download and split in entry.download["splits"]:
        patterns = entry.download["splits"][split]
        if isinstance(patterns, str):
            kwargs["allow_patterns"] = [p.strip() for p in patterns.split(",")]
        else:
            kwargs["allow_patterns"] = patterns
    snapshot_download(**kwargs)


def _download_gdown(url: str, dest: Path) -> None:
    import gdown

    if "drive.google.com/drive/folders" in url:
        gdown.download_folder(url, output=str(dest), quiet=False)
    else:
        gdown.download(url, output=str(dest / "download"), fuzzy=True, quiet=False)


def _download_git(url: str, dest: Path) -> None:
    if (dest / ".git").exists():
        console.print("  Repository already cloned, pulling latest ...")
        subprocess.run(["git", "-C", str(dest), "pull"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def _download_wget(url: str, dest: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "wget", "-o", str(dest / Path(url).name), url],
        check=False,
    )
    # Fallback to requests if python-wget is not available
    try:
        import requests
        from tqdm import tqdm

        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        fname = dest / (Path(url).name or "download")
        with open(fname, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    except Exception as e:
        console.print(f"[yellow]wget fallback failed: {e}[/]")
