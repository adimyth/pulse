"""Publish the frozen Pulse artifact and model card to the public Hugging Face Hub repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

from pulse.download import MODEL_REPOSITORY, RELEASE_FILES


def parse_args() -> argparse.Namespace:
    """Accept a locally trained artifact while preserving the public release destination by default."""
    parser = argparse.ArgumentParser(description="Publish a Pulse artifact to Hugging Face.")
    parser.add_argument("--model-dir", type=Path, default=Path("var/pulse-model"))
    parser.add_argument("--model-card", type=Path, default=Path("docs/huggingface-model-card.md"))
    parser.add_argument("--repository", default=MODEL_REPOSITORY)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def validate_artifact(directory: Path, card: Path) -> None:
    """Refuse to publish incomplete or mismatched artifacts so the Hub release remains reproducible."""
    missing = [name for name in RELEASE_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing release files: {', '.join(missing)}")
    if not card.is_file():
        raise FileNotFoundError(f"missing model card: {card}")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_id") != "pulse-local-goemotions-0.1":
        raise ValueError("refuse to publish an unexpected Pulse model artifact")


def publish(args: argparse.Namespace) -> str:
    """Create or update the model repository and upload only weights, metadata, provenance, and the model card."""
    validate_artifact(args.model_dir, args.model_card)
    api = HfApi()
    url = api.create_repo(repo_id=args.repository, repo_type="model", private=args.private, exist_ok=True)
    for filename in RELEASE_FILES:
        api.upload_file(path_or_fileobj=args.model_dir / filename, path_in_repo=filename, repo_id=args.repository, repo_type="model", commit_message="Release Pulse GoEmotions model")
    api.upload_file(path_or_fileobj=args.model_card, path_in_repo="README.md", repo_id=args.repository, repo_type="model", commit_message="Add Pulse model card")
    return url


def main() -> None:
    """Publish the requested model artifact and print the resulting public URL."""
    print(publish(parse_args()))


if __name__ == "__main__":
    main()
