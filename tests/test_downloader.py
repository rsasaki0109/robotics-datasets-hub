"""Tests for the download engine."""

from pathlib import Path

from rdh.registry import Registry

REGISTRY_DIR = Path(__file__).resolve().parents[1] / "registry"


def test_resolve_url_with_split():
    reg = Registry(REGISTRY_DIR)
    entry = reg.get("covla")
    assert entry is not None
    assert entry.download["method"] == "huggingface"
    splits = entry.download.get("splits", {})
    assert "metadata" in splits
    assert "*.jsonl" in splits["metadata"]


def test_resolve_url_s3():
    reg = Registry(REGISTRY_DIR)
    entry = reg.get("polaris")
    assert entry is not None
    assert entry.download["method"] == "s3"
    assert entry.download["url"].startswith("s3://")
    splits = entry.download.get("splits", {})
    assert "nav-only" in splits


def test_resolve_url_git():
    reg = Registry(REGISTRY_DIR)
    entry = reg.get("ggrt")
    assert entry is not None
    assert entry.download["method"] == "git"
    assert entry.download["url"].endswith(".git")


def test_all_datasets_have_download():
    reg = Registry(REGISTRY_DIR)
    for entry in reg.all():
        assert "method" in entry.download, f"{entry.name} missing download method"
        assert "url" in entry.download, f"{entry.name} missing download url"
