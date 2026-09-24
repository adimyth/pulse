"""Loopback-only rolling audio, Whisper transport, and live-event mechanics for Pulse Local."""

from __future__ import annotations

import asyncio
import json
import math
import struct
import time
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from subprocess import DEVNULL, Popen
from typing import Any, Protocol
from urllib.parse import urlsplit

from .model import PulseClassifier, SentimentResult


SAMPLE_RATE = 16_000
PARTIAL_INTERVAL_MS = 300
FINAL_SILENCE_MS = 700
WINDOW_MS = 3_000


def loopback_url(url: str) -> bool:
    """Accept only a local HTTP endpoint, including for internal health checks and transcriptions."""
    parsed = urlsplit(url)
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class Transcription:
    """A local whisper.cpp response with only the timing needed by the UI."""

    text: str
    latency_ms: float


class Transcriber(Protocol):
    """The small transport interface makes the live state machine testable without a microphone or model."""

    async def transcribe(self, wav: bytes) -> Transcription: ...


def pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap 16 kHz mono signed PCM in an in-memory WAV so no temporary microphone file is necessary."""
    if len(pcm) % 2:
        raise ValueError("PCM data must contain whole 16-bit samples")
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm)
    return output.getvalue()


def speech_present(pcm: bytes, threshold: float = 350) -> bool:
    """Use a small local energy gate to avoid asking Whisper to decode pure silence."""
    if not pcm or len(pcm) % 2:
        return False
    values = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    return math.sqrt(sum(value * value for value in values) / len(values)) >= threshold


class WhisperClient:
    """Send only a temporary in-memory rolling window to the already-loaded local Whisper server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8178", timeout_s: float = 8.0) -> None:
        if not loopback_url(base_url):
            raise ValueError("Pulse Local only permits a loopback Whisper server")
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def transcribe(self, wav: bytes) -> Transcription:
        """Make one local multipart request and parse only Whisper's text response."""
        import httpx

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(f"{self.base_url}/inference", files={"file": ("window.wav", wav, "audio/wav")}, data={"response_format": "json", "language": "en", "temperature": "0.0", "no_speech_thold": "0.6"})
        response.raise_for_status()
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError("local Whisper server returned invalid JSON") from exc
        text = str(body.get("text", "")).strip()
        if text.upper() in {"[BLANK_AUDIO]", "[BLANK AUDIO]"}:
            text = ""
        return Transcription(text=text, latency_ms=(time.perf_counter() - started) * 1000)


class WhisperProcess:
    """Own one preloaded native whisper.cpp process that binds only to loopback."""

    def __init__(self, binary: str | Path, model: str | Path, host: str = "127.0.0.1", port: int = 8178) -> None:
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("Pulse Local refuses a non-loopback Whisper server")
        self.binary, self.model, self.host, self.port = Path(binary), Path(model), host, port
        self.process: Popen | None = None

    def start(self) -> None:
        """Start only when the exact local binary and verified model exist."""
        if self.process is not None:
            return
        if not self.binary.is_file() or not self.model.is_file():
            raise FileNotFoundError("Whisper runtime is absent; run scripts/bootstrap.sh first")
        self.process = Popen([str(self.binary), "--host", self.host, "--port", str(self.port), "--threads", "8", "--model", str(self.model)], stdout=DEVNULL, stderr=DEVNULL)

    def stop(self) -> None:
        """Terminate the child and release its local model memory without preserving audio."""
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()
            self.process.wait(timeout=5)
        finally:
            self.process = None


async def wait_for_whisper(base_url: str, timeout_s: float = 45) -> None:
    """Wait for the actual local inference route so a port collision cannot masquerade as a ready server."""
    import httpx

    if not loopback_url(base_url):
        raise ValueError("Pulse Local only permits a loopback Whisper server")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                response = await client.options(f"{base_url.rstrip('/')}/inference")
            if response.status_code < 400:
                return
        except httpx.HTTPError:
            await asyncio.sleep(0.1)
    raise TimeoutError("local Whisper server did not become ready")


class LiveSession:
    """Maintain one short in-memory microphone window and emit every partial sentiment update to the dashboard."""

    def __init__(self, classifier: PulseClassifier, transcriber: Transcriber, partial_interval_ms: int = PARTIAL_INTERVAL_MS, final_silence_ms: int = FINAL_SILENCE_MS, window_ms: int = WINDOW_MS) -> None:
        self.classifier, self.transcriber = classifier, transcriber
        self.partial_interval_ms, self.final_silence_ms = partial_interval_ms, final_silence_ms
        self.max_window_bytes = int(SAMPLE_RATE * window_ms / 1000) * 2
        self.window = bytearray()
        self.active = False
        self.sequence = 0
        self.last_tick_ms: float | None = None
        self.last_audio_ms: float | None = None
        self.last_speech_ms: float | None = None

    def start(self) -> None:
        """Start a fresh capture, explicitly clearing any preceding in-memory audio and utterance state."""
        self.window.clear()
        self.active, self.sequence = True, 0
        self.last_tick_ms = self.last_audio_ms = self.last_speech_ms = None

    def stop(self) -> None:
        """Immediately clear retained PCM and disable further transcription or inference."""
        self.window.clear()
        self.active = False
        self.last_audio_ms = self.last_speech_ms = None

    def push_pcm(self, pcm: bytes, now_ms: float) -> None:
        """Receive aligned browser PCM while retaining no more than the short rolling local window."""
        if not self.active:
            return
        if len(pcm) % 2:
            raise ValueError("browser PCM is not aligned to 16-bit samples")
        self.window.extend(pcm)
        if len(self.window) > self.max_window_bytes:
            del self.window[:-self.max_window_bytes]
        self.last_audio_ms = now_ms
        if speech_present(pcm):
            self.last_speech_ms = now_ms

    async def tick(self, now_ms: float) -> dict[str, Any] | None:
        """Refresh a transcript at the live cadence and classify every non-empty refresh before finalizing silence."""
        if not self.active or self.last_speech_ms is None:
            return None
        if self.last_tick_ms is None:
            self.last_tick_ms = now_ms
            return None
        final = now_ms - self.last_speech_ms >= self.final_silence_ms
        if not final and now_ms - self.last_tick_ms < self.partial_interval_ms:
            return None
        self.last_tick_ms = now_ms
        transcript = await self.transcriber.transcribe(pcm_to_wav(bytes(self.window)))
        text = " ".join(transcript.text.split())
        if not text:
            return None
        sentiment = self.classifier.classify(text)
        self.sequence += 1
        completed = time.perf_counter() * 1000
        event = {"type": "final" if final else "partial", "sequence": self.sequence, "transcript": text, "sentiment": sentiment.to_dict(), "timings_ms": {"stt": round(transcript.latency_ms, 3), "classifier": sentiment.latency_ms, "audio_to_ui": round(completed - (self.last_audio_ms or completed), 3)}}
        if final:
            self.window.clear()
            self.last_speech_ms = self.last_audio_ms = None
        return event
