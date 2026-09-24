# Pulse Local

Pulse Local is a separate, local-only live speech-to-sentiment demonstration. A browser microphone stream becomes a live transcript and independently updating text-sentiment dimensions on the same Mac.

It is deliberately not a banking-intent demo, a cloud service, a customer-scoring system, or a claim to infer a speaker’s hidden feelings from their voice. The first release classifies the text in the live transcript. It does not use pitch, stress, or facial information.

## The dimensions

The model reports independent scores for Frustration, Positive, Surprise, Uncertainty, Low mood, and Neutral. They are derived transparently from human-annotated GoEmotions categories rather than invented labels. The UI also shows Action pressure as a separate lexical cue for phrases such as “urgent”, “right now”, or “cannot wait”; it is not presented as an emotion.

## Local architecture

`whisper.cpp` provides locally preloaded, Metal-accelerated English speech-to-text. A compact fine-tuned `all-MiniLM-L6-v2` multi-label classifier scores each refreshed transcript on the local CPU, leaving Metal available to Whisper and avoiding contention in the live loop. The browser sends PCM only over a loopback WebSocket, and the backend sends each rolling-window WAV only to its own loopback Whisper server. Audio stays in memory and is discarded when recording stops.

The project does not invoke Compass’s Qwen decision server in the 300 ms live loop. That model would add avoidable latency. This project keeps the useful Compass discipline—local inference, calibrated typed outputs, provenance, and abstention—while using a dedicated lightweight model for live interaction.

## First run

```sh
scripts/bootstrap.sh
uv run python -m pulse.train
scripts/validate_fixture.sh /absolute/path/to/video.mp4
scripts/run.sh
```

Open `http://127.0.0.1:8050`, start the microphone, and speak naturally. The fixture command prefers `small.en` and retries `base.en` only when the preferred model misses the transcript or timing gate.

## Data and limitations

Training uses the official GoEmotions train/dev/test files, with the source revision, hashes, exact mapping, training seed, and attribution written into the local artifact manifest. Dataset rows, audio, transcripts, and trained weights are ignored by git.

GoEmotions contains English Reddit comments and its labels reflect its annotators and source population. Pulse Local is English-only and suitable for a local UI demo, not for employment, health, credit, safety, or automated customer decisions.

See [the validation record](docs/validation.md) for the frozen model’s split metrics and the local video replay timings.
