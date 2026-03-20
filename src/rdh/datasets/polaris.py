"""PoLaRIS maritime dataset-specific loader and visualizer."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rich.console import Console

console = Console()


def list_sequences(data_dir: Path) -> list[str]:
    """List available sequences/folders."""
    candidates = sorted(
        p.name for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    return candidates


def find_images(data_dir: Path, pattern: str = "**/*.png") -> list[Path]:
    """Find all images in the dataset directory."""
    paths = sorted(data_dir.glob(pattern))
    if not paths:
        paths = sorted(data_dir.glob("**/*.jpg"))
    return paths


def find_annotations(data_dir: Path) -> list[Path]:
    """Find annotation files (JSON/XML/txt)."""
    annots = []
    for ext in ["**/*.json", "**/*.xml", "**/*.txt"]:
        annots.extend(data_dir.glob(ext))
    return sorted(annots)


def viz_sensor_comparison(
    data_dir: Path,
    n_samples: int = 3,
    save_path: Path | None = None,
) -> None:
    """Visualize RGB vs thermal infrared (TIR) image pairs side by side."""
    rgb_dir = None
    tir_dir = None

    for d in data_dir.iterdir():
        if not d.is_dir():
            continue
        name_lower = d.name.lower()
        if "rgb" in name_lower or "visible" in name_lower or "camera" in name_lower:
            rgb_dir = d
        elif "tir" in name_lower or "thermal" in name_lower or "infrared" in name_lower:
            tir_dir = d

    if rgb_dir and tir_dir:
        _viz_paired_sensors(rgb_dir, tir_dir, "RGB", "Thermal IR", n_samples, save_path)
    else:
        console.print("[yellow]Could not find RGB/TIR subdirectories.[/]")
        console.print("Available directories:")
        for d in sorted(data_dir.iterdir()):
            if d.is_dir():
                n_imgs = len(list(d.glob("*.png"))) + len(list(d.glob("*.jpg")))
                console.print(f"  {d.name}: {n_imgs} images")
        # Fallback: show any images found
        images = find_images(data_dir)
        if images:
            _viz_image_grid(images[:n_samples], "PoLaRIS Samples", save_path)


def _viz_paired_sensors(
    dir_a: Path, dir_b: Path,
    label_a: str, label_b: str,
    n_samples: int, save_path: Path | None,
) -> None:
    """Show paired images from two sensor directories."""
    imgs_a = sorted(dir_a.glob("*.png")) + sorted(dir_a.glob("*.jpg"))
    imgs_b = sorted(dir_b.glob("*.png")) + sorted(dir_b.glob("*.jpg"))

    n = min(n_samples, len(imgs_a), len(imgs_b))
    if n == 0:
        console.print("[yellow]No paired images found.[/]")
        return

    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle("PoLaRIS: Multi-Sensor Maritime Data", fontsize=14, fontweight="bold")

    for i in range(n):
        img_a = Image.open(imgs_a[i])
        img_b = Image.open(imgs_b[i])

        axes[0, i].imshow(img_a)
        axes[0, i].set_title(f"{label_a}: {imgs_a[i].name}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(img_b, cmap="inferno" if "thermal" in label_b.lower() else None)
        axes[1, i].set_title(f"{label_b}: {imgs_b[i].name}", fontsize=8)
        axes[1, i].axis("off")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        console.print(f"[green]Saved:[/] {save_path}")
    plt.show()


def _viz_image_grid(
    images: list[Path], title: str, save_path: Path | None,
) -> None:
    """Show a simple grid of images."""
    n = len(images)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(title, fontsize=14, fontweight="bold")

    for idx, img_path in enumerate(images):
        r, c = divmod(idx, cols)
        img = Image.open(img_path)
        axes[r, c].imshow(img)
        axes[r, c].set_title(img_path.name, fontsize=7)
        axes[r, c].axis("off")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        console.print(f"[green]Saved:[/] {save_path}")
    plt.show()


def viz_detection_annotations(
    data_dir: Path,
    n_samples: int = 4,
    save_path: Path | None = None,
) -> None:
    """Visualize object detection annotations overlaid on images (COCO format)."""
    annot_files = list(data_dir.glob("**/*annotations*.json"))
    if not annot_files:
        annot_files = list(data_dir.glob("**/*instances*.json"))
    if not annot_files:
        console.print("[yellow]No COCO-format annotation files found.[/]")
        return

    annot_path = annot_files[0]
    console.print(f"Loading annotations from {annot_path.name} ...")

    with open(annot_path) as f:
        coco = json.load(f)

    images_info = {img["id"]: img for img in coco.get("images", [])}
    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    annotations = coco.get("annotations", [])

    # Group annotations by image
    img_annots: dict[int, list] = {}
    for ann in annotations:
        img_annots.setdefault(ann["image_id"], []).append(ann)

    # Pick samples
    sample_img_ids = list(img_annots.keys())[:n_samples]
    if not sample_img_ids:
        console.print("[yellow]No annotated images found.[/]")
        return

    fig, axes = plt.subplots(1, len(sample_img_ids), figsize=(6 * len(sample_img_ids), 6))
    if len(sample_img_ids) == 1:
        axes = [axes]

    fig.suptitle("PoLaRIS: Object Detection Annotations", fontsize=14, fontweight="bold")
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(categories), 1)))

    for ax, img_id in zip(axes, sample_img_ids):
        img_info = images_info[img_id]
        img_path = data_dir / img_info["file_name"]
        if not img_path.exists():
            # Try finding in subdirs
            candidates = list(data_dir.glob(f"**/{img_info['file_name']}"))
            if candidates:
                img_path = candidates[0]

        if img_path.exists():
            img = Image.open(img_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f"Image not found:\n{img_info['file_name']}",
                    ha="center", transform=ax.transAxes)

        for ann in img_annots.get(img_id, []):
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                cat_id = ann.get("category_id", 0)
                color = colors[cat_id % len(colors)]
                rect = patches.Rectangle(
                    (x, y), w, h, linewidth=2, edgecolor=color, facecolor="none",
                )
                ax.add_patch(rect)
                label = categories.get(cat_id, str(cat_id))
                ax.text(x, y - 2, label, fontsize=7, color="white",
                        bbox=dict(facecolor=color, alpha=0.7, pad=1))

        ax.set_title(img_info.get("file_name", ""), fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        console.print(f"[green]Saved:[/] {save_path}")
    plt.show()
