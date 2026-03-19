"""Dataset registry: loads YAML definitions and provides lookup/search."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

REGISTRY_DIR = Path(__file__).resolve().parents[2] / "registry"


@dataclass
class DatasetEntry:
    name: str
    display_name: str
    description: str
    paper_url: str
    project_url: str
    modalities: list[str]
    tasks: list[str]
    license: str
    download: dict  # method, url, splits, ...
    tags: list[str] = field(default_factory=list)
    github_url: str = ""
    huggingface_id: str = ""
    size_hint: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> DatasetEntry:
        data = yaml.safe_load(path.read_text())
        return cls(**data)


class Registry:
    def __init__(self, registry_dir: Path | None = None):
        self._dir = registry_dir or REGISTRY_DIR
        self._entries: dict[str, DatasetEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self._dir.exists():
            return
        for p in sorted(self._dir.glob("*.yaml")):
            entry = DatasetEntry.from_yaml(p)
            self._entries[entry.name] = entry

    def list_names(self) -> list[str]:
        return list(self._entries.keys())

    def get(self, name: str) -> DatasetEntry | None:
        return self._entries.get(name)

    def search(self, query: str) -> list[DatasetEntry]:
        q = query.lower()
        results = []
        for entry in self._entries.values():
            searchable = " ".join(
                [entry.name, entry.display_name, entry.description]
                + entry.modalities
                + entry.tasks
                + entry.tags
            ).lower()
            if q in searchable:
                results.append(entry)
        return results

    def all(self) -> list[DatasetEntry]:
        return list(self._entries.values())
