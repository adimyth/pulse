"""Fine-tune a compact local multi-label text-sentiment model on the official GoEmotions splits."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from .data import Example, download_sources, read_split, stt_normalise, write_manifest
from .labels import DIMENSIONS
from .model import ENCODER, MODEL_ID, PulseModel, device, normalise_transcript


DEFAULT_SEED = 20260925


def batches(items: list[Example], size: int) -> Iterable[list[Example]]:
    """Yield fixed-size batches without a hidden data-loader worker or background data persistence."""
    for start in range(0, len(items), size):
        yield items[start:start + size]


def encode(tokenizer, items: list[Example], target: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Tokenize normalized text and preserve independent targets for every display dimension."""
    tokens = tokenizer([normalise_transcript(item.text) for item in items], return_tensors="pt", padding=True, truncation=True, max_length=128)
    labels = torch.tensor([item.targets for item in items], dtype=torch.float32)
    return {key: value.to(target) for key, value in tokens.items()}, labels.to(target)


@torch.inference_mode()
def logits_for(model: PulseModel, tokenizer, items: list[Example], target: torch.device, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate in batches and return CPU tensors so calibration is device-independent and reproducible."""
    logits, targets = [], []
    for group in batches(items, batch_size):
        tokens, labels = encode(tokenizer, group, target)
        logits.append(model(**tokens).detach().float().cpu())
        targets.append(labels.detach().float().cpu())
    return torch.cat(logits), torch.cat(targets)


def metrics(logits: torch.Tensor, targets: torch.Tensor, temperatures: torch.Tensor | None = None) -> dict[str, float]:
    """Report thresholded macro-F1 and independent-dimension calibration rather than a misleading single-label accuracy."""
    if temperatures is None:
        temperatures = torch.ones(len(DIMENSIONS))
    probabilities = torch.sigmoid(logits / temperatures)
    predicted = probabilities >= 0.5
    truth = targets >= 0.5
    f1s = []
    eces = []
    for index in range(len(DIMENSIONS)):
        tp = (predicted[:, index] & truth[:, index]).sum().item()
        fp = (predicted[:, index] & ~truth[:, index]).sum().item()
        fn = (~predicted[:, index] & truth[:, index]).sum().item()
        f1s.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
        probabilities_i, truth_i = probabilities[:, index], truth[:, index].float()
        eces.append(binary_ece(probabilities_i, truth_i))
    exact_match = predicted.eq(truth).all(dim=1).float().mean().item()
    return {"macro_f1": sum(f1s) / len(f1s), "mean_ece": sum(eces) / len(eces), "exact_match": exact_match}


def binary_ece(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate one fixed-bin ECE exactly as the dashboard's independent scores are reported."""
    ece = 0.0
    boundaries = torch.linspace(0, 1, 16)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        membership = (probabilities > lower) & (probabilities <= upper)
        if membership.any():
            ece += membership.float().mean().item() * abs(probabilities[membership].mean().item() - targets[membership].mean().item())
    return ece


def fit_temperatures(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Choose a separate temperature by direct development-set ECE minimization, not surrogate likelihood alone."""
    calibration_logits = torch.tensor(logits.detach().cpu().tolist(), dtype=torch.float32)
    calibration_targets = torch.tensor(targets.detach().cpu().tolist(), dtype=torch.float32)
    candidates = [0.25 + index * 0.025 for index in range(111)]
    temperatures = []
    for index in range(len(DIMENSIONS)):
        values, truth = calibration_logits[:, index], calibration_targets[:, index]
        temperatures.append(min(candidates, key=lambda temperature: binary_ece(torch.sigmoid(values / temperature), truth)))
    return torch.tensor(temperatures, dtype=torch.float32)


def train(args: argparse.Namespace) -> dict:
    """Train on official train, choose on official development, calibrate, then touch test once for the frozen artifact."""
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    target = device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    source_paths = download_sources(args.data_dir)
    source_train, source_dev, source_test = (read_split(source_paths[name]) for name in ("train.tsv", "dev.tsv", "test.tsv"))
    train_items = source_train + [Example(stt_normalise(item.text), item.targets) for item in source_train]
    random.Random(args.seed).shuffle(train_items)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = PulseModel(args.encoder).to(target)
    positives = torch.tensor([sum(item.targets[index] for item in source_train) for index in range(len(DIMENSIONS))])
    pos_weight = ((len(source_train) - positives) / positives.clamp_min(1)).clamp(max=15).to(target)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    best_state, best_f1, stale, history = None, -1.0, 0, []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        random.Random(args.seed + epoch).shuffle(train_items)
        total_loss, count = 0.0, 0
        for group in batches(train_items, args.batch_size):
            tokens, labels = encode(tokenizer, group, target)
            optimiser.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(**tokens), labels, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += float(loss.item()) * len(group)
            count += len(group)
        model.eval()
        dev_logits, dev_targets = logits_for(model, tokenizer, source_dev, target, args.batch_size)
        dev_metrics = metrics(dev_logits, dev_targets)
        history.append({"epoch": epoch, "train_loss": total_loss / count, "development": dev_metrics})
        if dev_metrics["macro_f1"] > best_f1 + 1e-6:
            best_f1, stale = dev_metrics["macro_f1"], 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    dev_logits, dev_targets = logits_for(model, tokenizer, source_dev, target, args.batch_size)
    temperatures = fit_temperatures(dev_logits, dev_targets)
    test_logits, test_targets = logits_for(model, tokenizer, source_test, target, args.batch_size)
    report = {
        "training": {"seed": args.seed, "epochs": len(history), "history": history, "source_train_examples": len(source_train), "augmented_train_examples": len(train_items)},
        "development": metrics(dev_logits, dev_targets, temperatures),
        "test": metrics(test_logits, test_targets, temperatures),
        "temperatures": temperatures.tolist(),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(output / "model.safetensors"))
    metadata = {"model_id": MODEL_ID, "encoder": args.encoder, "dimensions": list(DIMENSIONS), "temperatures": temperatures.tolist(), "minimum_words": args.minimum_words, "report": report}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_manifest(output / "dataset-manifest.json", source_paths, args.seed)
    return report


def parse_args() -> argparse.Namespace:
    """Parse a small, reproducible local training configuration suitable for Apple Silicon."""
    parser = argparse.ArgumentParser(description="Train Pulse's text-sentiment model on GoEmotions.")
    parser.add_argument("--data-dir", type=Path, default=Path("var/goemotions"))
    parser.add_argument("--output", type=Path, default=Path("var/pulse-model"))
    parser.add_argument("--encoder", default=ENCODER)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--minimum-words", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    """Train and print the single frozen-configuration test report."""
    report = train(parse_args())
    print(json.dumps(report, indent=2))
    if report["test"]["macro_f1"] < 0.40 or report["test"]["mean_ece"] > 0.10:
        raise SystemExit("quality gate failed: require test macro-F1 >= 0.40 and mean ECE <= 0.10")


if __name__ == "__main__":
    main()
