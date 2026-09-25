"""Refit calibration for an already selected Pulse checkpoint without changing its learned weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file
from transformers import AutoTokenizer

from .data import download_sources, read_split
from .model import PulseModel, device
from .train import fit_temperatures, logits_for, metrics


def main() -> None:
    """Use the official development split for temperatures, then update the artifact with one final test record."""
    parser = argparse.ArgumentParser(description="Refit Pulse calibration without retraining weights.")
    parser.add_argument("--artifacts", type=Path, default=Path("var/pulse-model"))
    parser.add_argument("--data-dir", type=Path, default=Path("var/goemotions"))
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    metadata_path = args.artifacts / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target = device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(metadata["encoder"], local_files_only=True)
    model = PulseModel(metadata["encoder"], local_files_only=True).to(target).eval()
    model.load_state_dict(load_file(str(args.artifacts / "model.safetensors")))
    paths = download_sources(args.data_dir)
    development, test = read_split(paths["dev.tsv"]), read_split(paths["test.tsv"])
    dev_logits, dev_targets = logits_for(model, tokenizer, development, target, args.batch_size)
    temperatures = fit_temperatures(dev_logits, dev_targets)
    test_logits, test_targets = logits_for(model, tokenizer, test, target, args.batch_size)
    report = {"development": metrics(dev_logits, dev_targets, temperatures), "test": metrics(test_logits, test_targets, temperatures), "temperatures": temperatures.tolist()}
    metadata["temperatures"] = temperatures.tolist()
    metadata["report"]["development"] = report["development"]
    metadata["report"]["test"] = report["test"]
    metadata["report"]["temperatures"] = report["temperatures"]
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["test"]["macro_f1"] < 0.40 or report["test"]["mean_ece"] > 0.10:
        raise SystemExit("quality gate failed: require test macro-F1 >= 0.40 and mean ECE <= 0.10")


if __name__ == "__main__":
    main()
