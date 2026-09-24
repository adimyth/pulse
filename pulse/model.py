"""The compact local multi-label text-sentiment model used by the live Pulse dashboard."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .labels import DIMENSIONS, DISPLAY_NAMES, action_pressure, intensity_name


ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_ID = "pulse-local-goemotions-0.1"


def device(name: str | None = None) -> torch.device:
    """Prefer the local Apple GPU, then CUDA, then CPU without using a remote runtime."""
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalise_transcript(text: str) -> str:
    """Use the same safe normalisation for a live transcript and supervised text input."""
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class SentimentScore:
    """One independent, calibrated text-sentiment dimension for UI rendering."""

    key: str
    label: str
    value: float
    level: str


@dataclass(frozen=True)
class SentimentResult:
    """Typed, JSON-ready local classification output for one provisional or final transcript."""

    scores: tuple[SentimentScore, ...]
    action_pressure: float
    dominant: str | None
    abstained: bool
    latency_ms: float
    model: str = MODEL_ID

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scores"] = [asdict(score) for score in self.scores]
        return payload


class PulseModel(torch.nn.Module):
    """MiniLM mean pooling with six independent sentiment logits."""

    def __init__(self, encoder_name: str = ENCODER, local_files_only: bool = False) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(encoder_name, local_files_only=local_files_only)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, len(DIMENSIONS))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **encoder_kwargs: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **encoder_kwargs).last_hidden_state
        pooled = (hidden * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return self.classifier(pooled)


class PulseClassifier:
    """Load a trained local-only artifact once and score a transcript in the live loop."""

    def __init__(self, artifacts: str | Path, device_name: str | None = None) -> None:
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        root = Path(artifacts)
        metadata_path, weights_path = root / "metadata.json", root / "model.safetensors"
        if not metadata_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError("missing Pulse Local artifacts; run `uv run python -m pulse.train` first")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if tuple(self.metadata.get("dimensions", ())) != DIMENSIONS:
            raise ValueError("artifact dimensions do not match this Pulse Local release")
        # Whisper owns Metal in the live path. MiniLM is faster and more stable on the local CPU while Whisper is decoding, and callers can still explicitly request another backend.
        self.device = torch.device(device_name) if device_name else torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.metadata.get("encoder", ENCODER), local_files_only=True)
        self.model = PulseModel(self.metadata.get("encoder", ENCODER), local_files_only=True).to(self.device).eval()
        self.model.load_state_dict(load_file(str(weights_path)))
        self.temperatures = torch.tensor(self.metadata["temperatures"], dtype=torch.float32)
        if self.temperatures.shape != (len(DIMENSIONS),) or torch.any(self.temperatures <= 0):
            raise ValueError("artifact temperature metadata is invalid")
        self.minimum_words = int(self.metadata.get("minimum_words", 3))
        self.model_id = str(self.metadata.get("model_id", MODEL_ID))

    def warmup(self) -> None:
        """Complete local tokenizer and device initialization before a microphone session starts."""
        for _ in range(4):
            self.classify("I am checking a short local sentiment stream before the session begins")

    @torch.inference_mode()
    def classify(self, text: str) -> SentimentResult:
        """Return all independent scores so a mixed utterance is not forced into one sentiment bucket."""
        cleaned = normalise_transcript(text)
        started = time.perf_counter()
        if not cleaned:
            return SentimentResult(scores=tuple(), action_pressure=0.0, dominant=None, abstained=True, latency_ms=0.0, model=self.model_id)
        encoded = self.tokenizer([cleaned], return_tensors="pt", truncation=True, max_length=128).to(self.device)
        logits = self.model(**encoded)[0].detach().float().cpu()
        values = torch.sigmoid(logits / self.temperatures).tolist()
        scores = tuple(SentimentScore(key=key, label=DISPLAY_NAMES[key], value=round(float(value), 5), level=intensity_name(float(value))) for key, value in zip(DIMENSIONS, values, strict=True))
        non_neutral = [score for score in scores if score.key != "neutral"]
        leading = max(non_neutral, key=lambda score: score.value)
        abstained = len(cleaned.split()) < self.minimum_words
        return SentimentResult(scores=scores, action_pressure=round(action_pressure(cleaned), 5), dominant=None if abstained else leading.key, abstained=abstained, latency_ms=round((time.perf_counter() - started) * 1000, 3), model=self.model_id)
