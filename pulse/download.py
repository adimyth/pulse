"""Download the published Pulse artifact and its base encoder for offline local use."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model import ENCODER


MODEL_REPOSITORY = "adimyth/pulse-goemotions"
RELEASE_FILES = ("model.safetensors", "metadata.json", "dataset-manifest.json")


def download_artifact(output: str | Path, repository: str = MODEL_REPOSITORY) -> Path:
    """Fetch the public artifact into Pulse's normal local artifact directory and cache its encoder."""
    from huggingface_hub import snapshot_download

    destination = Path(output)
    snapshot_download(repo_id=repository, local_dir=destination, allow_patterns=[*RELEASE_FILES, "README.md"])
    snapshot_download(repo_id=ENCODER)
    missing = [name for name in RELEASE_FILES if not (destination / name).is_file()]
    if missing:
        raise RuntimeError(f"published Pulse artifact is incomplete: {', '.join(missing)}")
    return destination


def parse_args() -> argparse.Namespace:
    """Accept an alternative artifact directory or a development model repository without changing defaults."""
    parser = argparse.ArgumentParser(description="Download the published Pulse model for local use.")
    parser.add_argument("--output", type=Path, default=Path("var/pulse-model"))
    parser.add_argument("--repository", default=MODEL_REPOSITORY)
    return parser.parse_args()


def main() -> None:
    """Download the frozen artifact and report the local directory that Pulse will load."""
    destination = download_artifact(**vars(parse_args()))
    print(f"Pulse model ready at {destination}")


if __name__ == "__main__":
    main()
