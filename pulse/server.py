"""The local-only FastAPI dashboard server for Pulse Local."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .audio import SAMPLE_RATE, LiveSession, WhisperClient, WhisperProcess, pcm_to_wav, wait_for_whisper
from .model import PulseClassifier


STATIC_DIR = Path(__file__).parent / "static"


def create_app(classifier, transcriber) -> FastAPI:
    """Create a testable loopback dashboard application without a cloud route or a microphone requirement."""
    app = FastAPI(title="Pulse Local", version="0.1")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "local_only": True, "model": getattr(classifier, "model_id", "pulse-local")}

    @app.websocket("/ws/live")
    async def live(socket: WebSocket) -> None:
        await socket.accept()
        session = LiveSession(classifier, transcriber)
        pending: tuple[object, asyncio.Task] | None = None
        stopping = False
        await socket.send_json({"type": "ready", "sample_rate": 16_000, "refresh_ms": 300, "final_silence_ms": 700, "local_only": True})
        try:
            while True:
                try:
                    message = await asyncio.wait_for(socket.receive(), timeout=0.02)
                except TimeoutError:
                    message = None
                if message is not None:
                    if message["type"] == "websocket.disconnect":
                        return
                    if message.get("bytes") is not None:
                        session.push_pcm(message["bytes"], time.perf_counter() * 1000)
                    elif message.get("text"):
                        try:
                            command = json.loads(message["text"])
                        except json.JSONDecodeError:
                            await socket.send_json({"type": "error", "message": "controls must be JSON"})
                            continue
                        if command == {"type": "start"}:
                            session.start()
                            await socket.send_json({"type": "started"})
                        elif command == {"type": "stop"}:
                            if pending is not None:
                                pending[1].cancel()
                                pending = None
                            final_request = session.force_final_request(time.perf_counter() * 1000)
                            if final_request is None:
                                session.stop()
                                await socket.send_json({"type": "stopped"})
                            else:
                                pending = (final_request, asyncio.create_task(session.transcribe_request(final_request)))
                                stopping = True
                        elif command == {"type": "discard"}:
                            if pending is not None:
                                pending[1].cancel()
                                pending = None
                            stopping = False
                            session.stop()
                            await socket.send_json({"type": "stopped", "discarded": True})
                        else:
                            await socket.send_json({"type": "error", "message": "only start, stop, and discard are supported"})
                if pending is not None and pending[1].done():
                    request, task = pending
                    pending = None
                    try:
                        event = session.complete_request(request, task.result())
                    except asyncio.CancelledError:
                        event = None
                    except Exception as exc:
                        session.discard_current_utterance()
                        await socket.send_json({"type": "error", "message": str(exc)})
                        event = None
                    if event:
                        await socket.send_json(event)
                    if stopping:
                        session.stop()
                        stopping = False
                        await socket.send_json({"type": "stopped"})
                if pending is None and not stopping:
                    request = session.next_request(time.perf_counter() * 1000)
                    if request is not None:
                        pending = (request, asyncio.create_task(session.transcribe_request(request)))
        except WebSocketDisconnect:
            return
        finally:
            if pending is not None:
                pending[1].cancel()
            session.stop()

    return app


def parse_args() -> argparse.Namespace:
    """Read loopback-only runtime paths with a non-conflicting local Whisper port."""
    parser = argparse.ArgumentParser(description="Run the Pulse Local speech-to-sentiment dashboard.")
    parser.add_argument("--artifacts", type=Path, default=Path("var/pulse-model"))
    parser.add_argument("--whisper-binary", type=Path, default=Path("var/whisper.cpp/build-arm64/bin/whisper-server"))
    parser.add_argument("--whisper-model", type=Path, default=Path("var/whisper.cpp/models/ggml-medium.en.bin"))
    parser.add_argument("--whisper-port", type=int, default=8178)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    """Preload local STT, then serve the browser UI only on loopback until interrupted."""
    import uvicorn

    args = parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Pulse Local refuses a non-loopback dashboard host")
    classifier = PulseClassifier(args.artifacts, args.device)
    classifier.warmup()
    whisper = WhisperProcess(args.whisper_binary, args.whisper_model, port=args.whisper_port)
    whisper.start()
    try:
        asyncio.run(wait_for_whisper(f"http://127.0.0.1:{args.whisper_port}"))
        transcriber = WhisperClient(f"http://127.0.0.1:{args.whisper_port}", language="en")
        asyncio.run(transcriber.transcribe(pcm_to_wav(b"\0" * SAMPLE_RATE * 2)))
        uvicorn.run(create_app(classifier, transcriber), host=args.host, port=args.port, workers=1)
    finally:
        whisper.stop()


if __name__ == "__main__":
    main()
