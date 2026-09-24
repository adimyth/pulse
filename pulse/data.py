"""Pinned acquisition and preparation for the official GoEmotions data files."""

from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .labels import DIMENSION_SOURCES, GOEMOTIONS, dimension_targets


SOURCE_REPOSITORY = "https://github.com/google-research/google-research/tree/master/goemotions"
SOURCE_BASE = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data"
SOURCE_URLS = {name: f"{SOURCE_BASE}/{name}" for name in ("train.tsv", "dev.tsv", "test.tsv", "emotions.txt")}


@dataclass(frozen=True)
class Example:
    """A single text record with independent target values for Pulse Local's documented dimensions."""

    text: str
    targets: tuple[float, ...]


def sha256(path: Path) -> str:
    """Compute a stable source hash without retaining another copy of the file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_sources(destination: str | Path, timeout_s: int = 60) -> dict[str, Path]:
    """Fetch the official GoEmotions split files only when they are missing from the ignored runtime directory."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, url in SOURCE_URLS.items():
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            request = urllib.request.Request(url, headers={"User-Agent": "pulse-local/0.1"})
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                path.write_bytes(response.read())
        paths[name] = path
    label_order = tuple(paths["emotions.txt"].read_text(encoding="utf-8").splitlines())
    if label_order != GOEMOTIONS:
        raise ValueError("official GoEmotions label order has changed; refuse to train with an unrecorded mapping")
    return paths


def read_split(path: str | Path) -> list[Example]:
    """Parse an official agreement-filtered split into one multi-label target per text."""
    examples: list[Example] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"unexpected GoEmotions row: {line[:80]!r}")
        text, raw_ids, _ = fields
        emotion_ids = {int(value) for value in raw_ids.split(",") if value}
        if not emotion_ids or any(index < 0 or index >= len(GOEMOTIONS) for index in emotion_ids):
            raise ValueError(f"invalid GoEmotions labels: {raw_ids!r}")
        examples.append(Example(text=text.strip(), targets=dimension_targets({GOEMOTIONS[index] for index in emotion_ids})))
    if not examples:
        raise ValueError("GoEmotions split is empty")
    return examples


def stt_normalise(text: str) -> str:
    """Make a deterministic text-only STT-style variant without creating new semantic labels."""
    return " ".join(text.lower().replace("?", "").replace("!", "").replace(",", "").replace(".", "").split())


def source_revision() -> str | None:
    """Record the current public source commit when it is reachable; all downloaded bytes are hashed regardless."""
    try:
        return subprocess.check_output(["git", "ls-remote", "https://github.com/google-research/google-research.git", "HEAD"], text=True, timeout=20).split()[0]
    except Exception:
        return None


def write_manifest(path: str | Path, sources: dict[str, Path], seed: int) -> None:
    """Persist provenance and the dimension mapping beside artifacts without copying supervised records into git."""
    body = {
        "dataset": "GoEmotions",
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": source_revision(),
        "source_urls": SOURCE_URLS,
        "source_sha256": {name: sha256(source) for name, source in sorted(sources.items())},
        "goemotions_label_order": list(GOEMOTIONS),
        "dimension_mapping": {name: sorted(values) for name, values in DIMENSION_SOURCES.items()},
        "training_seed": seed,
        "attribution": "Demszky et al., GoEmotions: A Dataset of Fine-Grained Emotions, ACL 2020.",
        "data_not_committed": True,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
