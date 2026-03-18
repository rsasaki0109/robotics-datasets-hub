"""Tests for the dataset registry."""

from pathlib import Path

from rdh.registry import DatasetEntry, Registry

REGISTRY_DIR = Path(__file__).resolve().parents[1] / "registry"


def test_registry_loads():
    reg = Registry(REGISTRY_DIR)
    names = reg.list_names()
    assert len(names) >= 5
    assert "covla" in names


def test_get_entry():
    reg = Registry(REGISTRY_DIR)
    entry = reg.get("covla")
    assert entry is not None
    assert entry.display_name == "CoVLA: Comprehensive Vision-Language-Action Dataset"
    assert "vision" in entry.modalities


def test_search():
    reg = Registry(REGISTRY_DIR)
    results = reg.search("maritime")
    assert len(results) >= 1
    assert results[0].name == "polaris"


def test_search_no_results():
    reg = Registry(REGISTRY_DIR)
    results = reg.search("nonexistent_dataset_xyz")
    assert len(results) == 0


def test_all():
    reg = Registry(REGISTRY_DIR)
    entries = reg.all()
    assert len(entries) >= 5


def test_entry_from_yaml():
    yaml_path = REGISTRY_DIR / "covla.yaml"
    entry = DatasetEntry.from_yaml(yaml_path)
    assert entry.name == "covla"
    assert entry.huggingface_id == "turing-motors/CoVLA-Dataset"
    assert entry.download["method"] == "huggingface"
