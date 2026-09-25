import asyncio

import pytest

from pulse.audio import LiveSession, Transcription, WhisperClient, choose_final_transcript, human_speech_text, merge_live_transcript, pcm_to_wav, sentence_segments, speech_present
from pulse.labels import DIMENSIONS, DISPLAY_NAMES
from pulse.model import SentimentResult, SentimentScore
from pulse.server import create_app


def pcm(amplitude: int, samples: int = 320) -> bytes:
    return b"".join(amplitude.to_bytes(2, "little", signed=True) for _ in range(samples))


class FakeClassifier:
    model_id = "pulse-test"

    def classify(self, text: str) -> SentimentResult:
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
    with pytest.raises(ValueError):
        WhisperClient(language="auto")


def test_pure_sound_captions_do_not_become_transcript_entries():
    assert human_speech_text("[BLANK_AUDIO]") == ""
    assert human_speech_text("Howling wind.") == ""
    assert human_speech_text("crowd cheer") == ""
    assert human_speech_text("engine revving") == ""
    assert human_speech_text("keyboard clicking") == ""
    assert human_speech_text("NON-ENGLISH SPEECH") == ""
    assert human_speech_text("speaking in foreign language") == ""
    assert human_speech_text("I need help with this charge") == "I need help with this charge"


def test_sentence_segments_preserve_complete_sentences_for_independent_coloring():
    assert sentence_segments("Hello. This charge is wrong! Please help") == ("Hello.", "This charge is wrong!", "Please help")


def test_live_transcript_retains_new_rolling_window_text_when_its_wording_changes():
    assert merge_live_transcript("Hello, thank you", "thank you for helping") == "Hello, thank you for helping"
    assert merge_live_transcript("Hello, thank you", "Hello thank you for helping") == "Hello thank you for helping"
    assert merge_live_transcript("Hello, thank you", "unrelated revision") == "Hello, thank you"
    assert merge_live_transcript("Hello, thank you", "the bank charged me") == "Hello, thank you the bank charged me"


def test_final_transcript_prefers_the_accurate_full_decode_over_the_provisional_draft():
    assert choose_final_transcript("Hello thank you the bank charged me twice", "Hello, thank you. The bank charged me twice.") == "Hello, thank you. The bank charged me twice."
    assert choose_final_transcript("One two three four five six seven eight nine ten", "eight nine ten") == "One two three four five six seven eight nine ten"


def test_transcription_recovery_discards_only_the_failed_utterance_and_keeps_listening():
    session = LiveSession(FakeClassifier(), FakeTranscriber())
    session.start()
    session.push_pcm(pcm(2000), 0)
    session.discard_current_utterance()
    assert session.active is True
    assert session.window == bytearray()
    assert session.utterance == bytearray()
    assert session.last_speech_ms is None


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
        assert final["sentences"] == [{"text": "I am happy but this is frustrating", "sentiment": final["sentiment"]}]
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


def test_final_flush_preserves_audio_arriving_during_its_background_decode():
    async def run() -> None:
        session = LiveSession(FakeClassifier(), FakeTranscriber())
        session.start()
        session.push_pcm(pcm(2000, 640), 0)
        request = session.force_final_request(700)
        assert request is not None
        later_audio = pcm(2000, 320)
        session.push_pcm(later_audio, 900)
        event = session.complete_request(request, await session.transcribe_request(request))
        assert event is not None
        assert event["type"] == "final"
        assert session.utterance == bytearray(later_audio)
        assert session.last_speech_ms == 900

    asyncio.run(run())


def test_live_session_uses_fast_transcriber_for_partials_and_accurate_transcriber_for_final_text():
    class NamedTranscriber:
        def __init__(self, text: str) -> None:
            self.text = text

        async def transcribe(self, wav: bytes) -> Transcription:
            return Transcription(self.text, 1.0, "english")

    async def run() -> None:
        session = LiveSession(FakeClassifier(), NamedTranscriber("fast draft"), NamedTranscriber("accurate final"))
        session.start()
        session.push_pcm(pcm(2000), 0)
        assert await session.tick(0) is None
        assert (await session.tick(300))["transcript"] == "fast draft"
        final_request = session.force_final_request(700)
        assert final_request is not None
        assert (await session.transcribe_request(final_request)).text == "accurate final"

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
            socket.send_bytes(pcm(2000, 640))
            socket.send_json({"type": "stop"})
            final = socket.receive_json()
            assert final["type"] == "final"
            assert socket.receive_json() == {"type": "stopped"}
