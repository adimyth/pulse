"""Loopback-only rolling audio, Whisper transport, and live-event mechanics for Pulse Local."""

from __future__ import annotations

import asyncio
import json
import math
import re
import struct
import time
import wave
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from subprocess import DEVNULL, Popen
from typing import Any, Protocol
from urllib.parse import urlsplit

from .labels import DIMENSIONS, DISPLAY_NAMES
from .model import PulseClassifier, SentimentResult, SentimentScore


SAMPLE_RATE = 16_000
PARTIAL_INTERVAL_MS = 300
FINAL_SILENCE_MS = 700
WINDOW_MS = 3_000
MAX_UTTERANCE_MS = 45_000
NON_SPEECH_CAPTIONS = frozenset({"blank audio", "silence", "howling wind", "wind", "wind blowing", "crowd cheer", "crowd cheering", "cheering", "applause", "engine revving", "engine reving", "keyboard clicking", "typing", "background noise", "music", "laughter", "non english speech", "speaking in foreign language", "foreign language"})
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'”’\)\]])*\s+(?=[A-Z0-9])")


def loopback_url(url: str) -> bool:
    """Accept only a local HTTP endpoint, including for internal health checks and transcriptions."""
    parsed = urlsplit(url)
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def human_speech_text(value: object) -> str:
    """Discard pure sound captions from Whisper while preserving genuine spoken text unchanged."""
    text = " ".join(str(value or "").split())
    caption = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return "" if caption in NON_SPEECH_CAPTIONS else text


def merge_live_transcript(previous: str, candidate: str) -> str:
    """Keep all likely later rolling-window text while allowing the more accurate final decode to replace provisional wording."""
    old_words, new_words = previous.split(), candidate.split()
    if not old_words:
        return candidate
    old_keys = [re.sub(r"[^a-z0-9]+", "", word.lower()) for word in old_words]
    new_keys = [re.sub(r"[^a-z0-9]+", "", word.lower()) for word in new_words]
    if new_keys[:len(old_keys)] == old_keys:
        return candidate
    if old_keys[:len(new_keys)] == new_keys:
        return previous
    for overlap in range(min(len(old_words), len(new_words)), 0, -1):
        if old_keys[-overlap:] == new_keys[:overlap]:
            return " ".join([*old_words, *new_words[overlap:]])
    best_length, best_old_start, best_new_start = 0, 0, 0
    for old_start in range(max(0, len(old_keys) - 40), len(old_keys)):
        for new_start in range(len(new_keys)):
            length = 0
            while old_start + length < len(old_keys) and new_start + length < len(new_keys) and old_keys[old_start + length] == new_keys[new_start + length]:
                length += 1
            if length > best_length:
                best_length, best_old_start, best_new_start = length, old_start, new_start
    if best_length >= 2:
        return " ".join([*old_words[:best_old_start + best_length], *new_words[best_new_start + best_length:]])
    if len(new_words) >= 3:
        return " ".join([*old_words, *new_words])
    return previous


def choose_final_transcript(previous: str, candidate: str) -> str:
    """Prefer the accurate final decode when it covers the provisional draft, retaining a rolling draft only when the final audio window was truncated."""
    if not previous:
        return candidate
    if len(candidate.split()) >= max(4, math.floor(len(previous.split()) * .65)):
        return candidate
    return merge_live_transcript(previous, candidate)


def sentence_segments(value: str) -> tuple[str, ...]:
    text = " ".join(value.split())
    if not text:
        return ()
    return tuple(segment for segment in SENTENCE_BOUNDARY.split(text) if segment)


@dataclass(frozen=True)
class Transcription:
    """A local whisper.cpp response with only the timing needed by the UI."""

    text: str
    latency_ms: float
    language: str | None = None


@dataclass(frozen=True)
class TranscriptionRequest:
    """An immutable audio snapshot that can decode in the background while the WebSocket continues receiving microphone frames."""

    audio: bytes
    final: bool
    requested_at_ms: float
    utterance_bytes: int
    prior_text: str
    prior_language: str | None


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


def speech_present(pcm: bytes, threshold: float = 180) -> bool:
    """Use a small local energy gate to avoid asking Whisper to decode pure silence."""
    if not pcm or len(pcm) % 2:
        return False
    values = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    return math.sqrt(sum(value * value for value in values) / len(values)) >= threshold


class WhisperClient:
    """Send only a temporary in-memory rolling window to the already-loaded local Whisper server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8178", timeout_s: float = 12.0, language: str = "en") -> None:
        if not loopback_url(base_url):
            raise ValueError("Pulse Local only permits a loopback Whisper server")
        if language != "en":
            raise ValueError("Pulse Local is configured for English-only STT")
        self.base_url, self.timeout_s, self.language = base_url.rstrip("/"), timeout_s, language

    async def transcribe(self, wav: bytes) -> Transcription:
        """Make one local multipart request and parse only Whisper's text response."""
        import httpx

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            response = await client.post(f"{self.base_url}/inference", files={"file": ("window.wav", wav, "audio/wav")}, data={"response_format": "verbose_json", "language": self.language, "no_language_probabilities": "true", "temperature": "0.0", "no_speech_thold": "0.6"})
        response.raise_for_status()
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError("local Whisper server returned invalid JSON") from exc
        text = human_speech_text(body.get("text", ""))
        language = str(body.get("language", "")).strip().lower() or None
        return Transcription(text=text, latency_ms=(time.perf_counter() - started) * 1000, language=language)


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


def supports_sentiment(language: str | None) -> bool:
    """Allow calibrated sentiment only for the language represented by the current training data."""
    return language in {None, "english", "en"}


def unavailable_sentiment(language: str | None, model_id: str = "pulse-local") -> SentimentResult:
    """Return an explicit empty result instead of applying an English sentiment model to another language."""
    scores = tuple(SentimentScore(key=key, label=DISPLAY_NAMES[key], value=0.0, level="low") for key in DIMENSIONS)
    return SentimentResult(scores=scores, action_pressure=0.0, dominant=None, abstained=True, latency_ms=0.0, model=model_id)


class LiveSession:
    """Maintain one short in-memory microphone window and emit every partial sentiment update to the dashboard."""

    def __init__(self, classifier: PulseClassifier, transcriber: Transcriber, final_transcriber: Transcriber | None = None, partial_interval_ms: int = PARTIAL_INTERVAL_MS, final_silence_ms: int = FINAL_SILENCE_MS, window_ms: int = WINDOW_MS, max_utterance_ms: int = MAX_UTTERANCE_MS) -> None:
        self.classifier, self.transcriber, self.final_transcriber = classifier, transcriber, final_transcriber or transcriber
        self.partial_interval_ms, self.final_silence_ms = partial_interval_ms, final_silence_ms
        self.max_window_bytes = int(SAMPLE_RATE * window_ms / 1000) * 2
        self.max_utterance_bytes = int(SAMPLE_RATE * max_utterance_ms / 1000) * 2
        self.window = bytearray()
        self.utterance = bytearray()
        self.active = False
        self.sequence = 0
        self.last_tick_ms: float | None = None
        self.last_audio_ms: float | None = None
        self.last_speech_ms: float | None = None
        self.last_human_transcript: str | None = None
        self.last_human_language: str | None = None

    def start(self) -> None:
        """Start a fresh capture, explicitly clearing any preceding in-memory audio and utterance state."""
        self.window.clear()
        self.utterance.clear()
        self.active, self.sequence = True, 0
        self.last_tick_ms = self.last_audio_ms = self.last_speech_ms = None
        self.last_human_transcript = None
        self.last_human_language = None

    def stop(self) -> None:
        """Immediately clear retained PCM and disable further transcription or inference."""
        self.discard_current_utterance()
        self.active = False

    def discard_current_utterance(self) -> None:
        self.window.clear()
        self.utterance.clear()
        self.last_tick_ms = self.last_audio_ms = self.last_speech_ms = None
        self.last_human_transcript = None
        self.last_human_language = None

    def push_pcm(self, pcm: bytes, now_ms: float) -> None:
        """Receive aligned browser PCM while retaining a short live window and the active utterance in memory only."""
        if not self.active:
            return
        if len(pcm) % 2:
            raise ValueError("browser PCM is not aligned to 16-bit samples")
        self.window.extend(pcm)
        self.utterance.extend(pcm)
        if len(self.window) > self.max_window_bytes:
            del self.window[:-self.max_window_bytes]
        if len(self.utterance) > self.max_utterance_bytes:
            del self.utterance[:-self.max_utterance_bytes]
        self.last_audio_ms = now_ms
        if speech_present(pcm):
            self.last_speech_ms = now_ms

    def next_request(self, now_ms: float) -> TranscriptionRequest | None:
        """Snapshot the next decode without awaiting it so microphone transport never waits on Whisper inference."""
        if not self.active or self.last_speech_ms is None:
            return None
        if self.last_tick_ms is None:
            self.last_tick_ms = now_ms
            return None
        final = now_ms - self.last_speech_ms >= self.final_silence_ms
        if not final and now_ms - self.last_tick_ms < self.partial_interval_ms:
            return None
        self.last_tick_ms = now_ms
        audio = self.utterance if final else self.window
        return TranscriptionRequest(audio=bytes(audio), final=final, requested_at_ms=now_ms, utterance_bytes=len(self.utterance), prior_text=self.last_human_transcript or "", prior_language=self.last_human_language)

    def force_final_request(self, now_ms: float) -> TranscriptionRequest | None:
        """Flush received speech on Stop so the last phrase is retained even when the user does not pause first."""
        if not self.active or self.last_speech_ms is None or not self.utterance:
            return None
        self.last_tick_ms = now_ms
        return TranscriptionRequest(audio=bytes(self.utterance), final=True, requested_at_ms=now_ms, utterance_bytes=len(self.utterance), prior_text=self.last_human_transcript or "", prior_language=self.last_human_language)

    async def transcribe_request(self, request: TranscriptionRequest) -> Transcription:
        """Decode a detached request while new browser PCM continues entering the session buffers, reserving the accurate model for final text."""
        transcriber = self.final_transcriber if request.final else self.transcriber
        return await transcriber.transcribe(pcm_to_wav(request.audio))

    def complete_request(self, request: TranscriptionRequest, transcript: Transcription) -> dict[str, Any] | None:
        """Merge one completed decode, preserve later-arriving audio, and emit one ordered dashboard event."""
        candidate = " ".join(transcript.text.split())
        language = transcript.language
        if not candidate:
            if request.final and request.prior_text:
                text = request.prior_text
                language = request.prior_language
            elif request.final:
                self._clear_finished_request(request)
                return None
            else:
                return None
        elif request.final:
            text = choose_final_transcript(request.prior_text, candidate)
        else:
            text = merge_live_transcript(request.prior_text, candidate)
        sentiment_supported = supports_sentiment(language)
        if request.final:
            sentences = []
            for segment in sentence_segments(text):
                segment_sentiment = self.classifier.classify(segment) if sentiment_supported else unavailable_sentiment(language, getattr(self.classifier, "model_id", "pulse-local"))
                sentences.append({"text": segment, "sentiment": segment_sentiment.to_dict()})
            sentiment = segment_sentiment
            classifier_latency = round(sum(item["sentiment"]["latency_ms"] for item in sentences), 3)
        else:
            sentiment = self.classifier.classify(text) if sentiment_supported else unavailable_sentiment(language, getattr(self.classifier, "model_id", "pulse-local"))
            sentences = []
            classifier_latency = sentiment.latency_ms
        self.sequence += 1
        completed = time.perf_counter() * 1000
        event = {"type": "final" if request.final else "partial", "sequence": self.sequence, "transcript": text, "sentences": sentences, "language": language, "sentiment_supported": sentiment_supported, "sentiment": sentiment.to_dict(), "timings_ms": {"stt": round(transcript.latency_ms, 3), "classifier": classifier_latency, "audio_to_ui": round(completed - (self.last_audio_ms or completed), 3)}}
        if request.final:
            self._clear_finished_request(request)
        else:
            self.last_human_transcript = text
            self.last_human_language = language
        return event

    def _clear_finished_request(self, request: TranscriptionRequest) -> None:
        """Remove only the decoded prefix so frames that arrived during a final decode remain available for the next utterance."""
        del self.utterance[:request.utterance_bytes]
        if self.utterance:
            self.window = bytearray(self.utterance[-self.max_window_bytes:])
        else:
            self.window.clear()
        if self.last_speech_ms is None or self.last_speech_ms <= request.requested_at_ms:
            self.last_speech_ms = self.last_audio_ms = None
        self.last_human_transcript = None
        self.last_human_language = None

    async def tick(self, now_ms: float) -> dict[str, Any] | None:
        """Keep the simple awaitable API for unit tests while production dispatches the same work in a background task."""
        request = self.next_request(now_ms)
        if request is None:
            return None
        return self.complete_request(request, await self.transcribe_request(request))
