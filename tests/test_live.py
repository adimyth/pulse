import asyncio

import pytest

from pulse.audio import LiveSession, Transcription, WhisperClient, human_speech_text, pcm_to_wav, speech_present
from pulse.labels import DIMENSIONS, DISPLAY_NAMES
from pulse.model import SentimentResult, SentimentScore
from pulse.server import create_app


def pcm(amplitude: int, samples: int = 320) -> bytes:
    return b"".join(amplitude.to_bytes(2, "little", signed=True) for _ in range(samples))


class FakeClassifier:
    model_id = "pulse-test"

    def classify(self, text: str) -> SentimentResult:
        assert text == "I am happy but this is frustrating"
        scores = tuple(SentimentScore(key=key, label=DISPLAY_NAMES[key], value=.8 if key in {"positive", "frustration"} else .1, level="high" if key in {"positive", "frustration"} else "low") for key in DIMENSIONS)
        return SentimentResult(scores=scores, action_pressure=0.0, dominant="frustration", abstained=False, latency_ms=3.0)


class FakeTranscriber:
    async def transcribe(self, wav: bytes) -> Transcription:
        assert wav.startswith(b"RIFF")
        return Transcription("I am happy but this is frustrating", 11.0)


def test_pcm_stays_in_memory_and_loopback_client_rejects_external_hosts():
    assert pcm_to_wav(pcm(0)).startswith(b"RIFF")
    assert speech_present(pcm(10)) is False
    assert speech_present(pcm(2000)) is True
    with pytest.raises(ValueError):
        WhisperClient("https://example.com")


def test_pure_sound_captions_do_not_become_transcript_entries():
    assert human_speech_text("[BLANK_AUDIO]") == ""
    assert human_speech_text("Howling wind.") == ""
    assert human_speech_text("crowd cheer") == ""
    assert human_speech_text("engine revving") == ""
    assert human_speech_text("keyboard clicking") == ""
    assert human_speech_text("NON-ENGLISH SPEECH") == ""
    assert human_speech_text("speaking in foreign language") == ""
    assert human_speech_text("I need help with this charge") == "I need help with this charge"


def test_live_session_emits_every_partial_then_one_final_and_clears_pcm():
    async def run() -> None:
        session = LiveSession(FakeClassifier(), FakeTranscriber())
        session.start()
        session.push_pcm(pcm(2000), 0)
        assert await session.tick(0) is None
        first = await session.tick(300)
        assert first["type"] == "partial"
        session.push_pcm(pcm(2000), 350)
        second = await session.tick(600)
        assert second["type"] == "partial"
        assert second["sequence"] == 2
        session.push_pcm(pcm(0), 650)
        final = await session.tick(1100)
        assert final["type"] == "final"
        assert session.window == bytearray()
        assert session.utterance == bytearray()
        session.stop()
        assert session.active is False

    asyncio.run(run())


def test_final_transcription_receives_the_full_active_utterance_not_only_the_live_window():
    class CapturingTranscriber:
        def __init__(self) -> None:
            self.audio_sizes: list[int] = []

        async def transcribe(self, wav: bytes) -> Transcription:
            self.audio_sizes.append(len(wav) - 44)
            return Transcription("I am happy but this is frustrating", 11.0, "english")

    async def run() -> None:
        transcriber = CapturingTranscriber()
        session = LiveSession(FakeClassifier(), transcriber)
        session.start()
        session.push_pcm(pcm(2000, 16_000), 0)
        assert await session.tick(0) is None
        assert (await session.tick(300))["type"] == "partial"
        session.push_pcm(pcm(2000, 48_000), 350)
        session.push_pcm(pcm(0), 650)
        final = await session.tick(1_100)
        assert final["type"] == "final"
        assert transcriber.audio_sizes[-1] > session.max_window_bytes

    asyncio.run(run())


def test_dashboard_websocket_contract_is_local_and_testable():
    from fastapi.testclient import TestClient

    app = create_app(FakeClassifier(), FakeTranscriber())
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"ok": True, "local_only": True, "model": "pulse-test"}
        with client.websocket_connect("/ws/live") as socket:
            assert socket.receive_json()["type"] == "ready"
            socket.send_json({"type": "start"})
            assert socket.receive_json() == {"type": "started"}
            socket.send_json({"type": "stop"})
            assert socket.receive_json() == {"type": "stopped"}
