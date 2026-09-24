"""Data-grounded sentiment dimensions derived from GoEmotions' human-annotated labels."""

from __future__ import annotations

import re
from collections.abc import Mapping


GOEMOTIONS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
)

DIMENSIONS = (
    "frustration",
    "positive",
    "surprise",
    "uncertainty",
    "low_mood",
    "neutral",
)

DISPLAY_NAMES = {
    "frustration": "Frustration",
    "positive": "Positive",
    "surprise": "Surprise",
    "uncertainty": "Uncertainty",
    "low_mood": "Low mood",
    "neutral": "Neutral",
}

DIMENSION_SOURCES: Mapping[str, frozenset[str]] = {
    "frustration": frozenset({"anger", "annoyance", "disapproval", "disgust"}),
    "positive": frozenset({"admiration", "amusement", "approval", "caring", "desire", "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief"}),
    "surprise": frozenset({"realization", "surprise"}),
    "uncertainty": frozenset({"confusion", "curiosity", "fear", "nervousness"}),
    "low_mood": frozenset({"disappointment", "embarrassment", "grief", "remorse", "sadness"}),
    "neutral": frozenset({"neutral"}),
}

_PRESSURE = re.compile(r"\b(asap|urgent(?:ly)?|immediately|right now|today|deadline|emergency|time-sensitive|cannot wait)\b", re.IGNORECASE)


def dimension_targets(emotions: set[str]) -> tuple[float, ...]:
    """Map one original multi-label GoEmotions record to the documented display dimensions."""
    if not emotions.issubset(GOEMOTIONS):
        raise ValueError(f"unknown GoEmotions labels: {sorted(emotions - set(GOEMOTIONS))}")
    return tuple(float(bool(emotions & DIMENSION_SOURCES[dimension])) for dimension in DIMENSIONS)


def action_pressure(text: str) -> float:
    """Expose an explicit lexical action-pressure cue without calling it an inferred emotion."""
    matches = len(_PRESSURE.findall(text))
    return min(1.0, matches / 2.0)


def intensity_name(score: float) -> str:
    """Give each independent dimension a legible level while keeping the underlying score visible."""
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "present"
    if score >= 0.20:
        return "emerging"
    return "low"
