#!/usr/bin/env python3
"""Replay a video through local STT and Pulse Local sentiment inference without retaining extracted audio."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import tempfile
from pathlib import Path

from pulse.audio import SAMPLE_RATE, WhisperClient, WhisperProcess, pcm_to_wav, wait_for_whisper
from pulse.model import PulseClassifier


def percentile(values: list[float], q: float) -> float:
    """Compute a simple deterministic percentile without another numerical dependency."""
    if not values:
        raise ValueError("no values to benchmark")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def pcm_from_video(source: Path) -> bytes:
    """Extract 16 kHz mono fixture audio into a temporary directory that is deleted immediately after reading."""
    with tempfile.TemporaryDirectory(prefix="pulse-local-fixture-") as temporary:
        output = Path(temporary) / "audio.pcm"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", str(output)], check=True)
        return output.read_bytes()


async def replay(args: argparse.Namespace) -> dict:
    """Use 300 ms windows in the same direction as the browser without saving microphone-like PCM elsewhere."""
    classifier = PulseClassifier(args.artifacts, args.device)
    classifier.warmup()
    whisper = WhisperProcess(args.whisper_binary, args.whisper_model, port=args.whisper_port)
    whisper.start()
    try:
        client = WhisperClient(f"http://127.0.0.1:{args.whisper_port}", language="auto")
        await wait_for_whisper(client.base_url)
        pcm = pcm_from_video(args.source)
        window_bytes, step_bytes = SAMPLE_RATE * 3 * 2, SAMPLE_RATE * 300 // 1000 * 2
        events = []
        for end in range(step_bytes, len(pcm) + 1, step_bytes):
            transcription = await client.transcribe(pcm_to_wav(pcm[max(0, end - window_bytes):end]))
            if not transcription.text:
                continue
            sentiment = classifier.classify(transcription.text)
            events.append({"transcript": transcription.text, "sentiment": sentiment.to_dict(), "timings_ms": {"stt": transcription.latency_ms, "classifier": sentiment.latency_ms, "audio_to_ui": transcription.latency_ms + sentiment.latency_ms}})
        transcript = " ".join(event["transcript"] for event in events).lower()
        frustration = max((next(score["value"] for score in event["sentiment"]["scores"] if score["key"] == "frustration") for event in events), default=0)
        positive = max((next(score["value"] for score in event["sentiment"]["scores"] if score["key"] == "positive") for event in events), default=0)
        totals = [event["timings_ms"]["audio_to_ui"] for event in events]
        classifier_times = [event["timings_ms"]["classifier"] for event in events]
        return {"events": events, "checks": {"multiple_live_updates": len(events) >= 3, "positive_language_transcribed": "loving" in transcript or "thank" in transcript, "frustration_language_transcribed": "charged" in transcript and "twice" in transcript, "positive_signal_observed": positive >= 0.20, "frustration_signal_observed": frustration >= 0.20}, "latency_ms": {"classifier_p95": percentile(classifier_times, .95), "audio_to_ui_p50": percentile(totals, .5), "audio_to_ui_p95": percentile(totals, .95)}}
    finally:
        whisper.stop()


def parse_args() -> argparse.Namespace:
    """Read the fixture and local runtime choices used by the automatic STT gate."""
    parser = argparse.ArgumentParser(description="Validate Pulse Local on a prerecorded video fixture.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--artifacts", type=Path, default=Path("var/pulse-model"))
    parser.add_argument("--whisper-binary", type=Path, default=Path("var/whisper.cpp/build-arm64/bin/whisper-server"))
    parser.add_argument("--whisper-model", type=Path, default=Path("var/whisper.cpp/models/ggml-small.bin"))
    parser.add_argument("--whisper-port", type=int, default=8178)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path, default=Path("var/pulse-model/fixture-report.json"))
    return parser.parse_args()


def main() -> None:
    """Persist a local-only summary and fail when speed or visible fixture behavior regresses."""
    args = parse_args()
    if not args.source.is_file():
        raise SystemExit(f"fixture does not exist: {args.source}")
    report = asyncio.run(replay(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": report["checks"], "latency_ms": report["latency_ms"]}, indent=2))
    latency = report["latency_ms"]
    if not all(report["checks"].values()) or latency["classifier_p95"] > 100 or latency["audio_to_ui_p50"] > 1000 or latency["audio_to_ui_p95"] > 1750:
        raise SystemExit("fixture or timing gate failed")


if __name__ == "__main__":
    main()
