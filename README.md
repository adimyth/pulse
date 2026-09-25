# Pulse

Pulse is a separate, local-only live speech-to-sentiment demonstration. A browser microphone stream becomes a live transcript and independently updating text-sentiment dimensions on the same Mac.

It is deliberately not a banking-intent demo, a cloud service, a customer-scoring system, or a claim to infer a speaker’s hidden feelings from their voice. The first release classifies the text in the live transcript. It does not use pitch, stress, or facial information.

## Origin and recording

Pulse was inspired by this [ElevenLabs Developers post](https://x.com/ElevenLabsDevs/status/2102884507078791484). I wanted to build real-time sentiment analysis myself, with the whole experience running locally on a Mac.

The included [Pulse recording](assets/pulse.mov) uses the same audio as the inspiration for an apples-to-apples demonstration of the live transcription and sentence-level sentiment flow. It is a reproducible demo fixture, not training data.

## The dimensions

The model reports independent scores for Frustration, Positive, Surprise, Uncertainty, Low mood, and Neutral. They are derived transparently from human-annotated GoEmotions categories rather than invented labels. The UI also shows Action pressure as a separate lexical cue for phrases such as “urgent”, “right now”, or “cannot wait”; it is not presented as an emotion.

## Local architecture

`whisper.cpp` provides locally preloaded, Metal-accelerated English speech-to-text through two pinned local models: `small.en` produces fast provisional text and `medium.en` produces the retained final transcript after a pause. A compact fine-tuned `all-MiniLM-L6-v2` multi-label classifier scores refreshed transcripts on the local CPU, leaving Metal available to Whisper and avoiding contention in the live loop. The browser sends PCM only over a loopback WebSocket, and the backend sends each rolling-window WAV only to its own loopback Whisper server. Audio stays in memory and is discarded when recording stops.

The project does not invoke Compass’s Qwen decision server in the 300 ms live loop. That model would add avoidable latency. This project keeps the useful Compass discipline—local inference, calibrated typed outputs, provenance, and abstention—while using a dedicated lightweight model for live interaction.

## Run Pulse locally

### Requirements

- An Apple Silicon Mac running macOS. The bootstrap script builds `whisper.cpp` with Metal support for arm64.
- `git`, `cmake`, and [`uv`](https://docs.astral.sh/uv/). Install Apple’s command-line tools with `xcode-select --install` if they are not already installed; Homebrew users can run `brew install cmake uv`.
- A current browser that can grant microphone permission to `http://127.0.0.1:8050`.
- An internet connection for first-time setup only, to fetch the pinned Whisper source and models, Python dependencies, the MiniLM base model, and GoEmotions. Runtime audio and inference remain on the Mac.

### From a fresh clone

```sh
git clone <repository-url> pulse
cd pulse

# Builds the pinned Metal-enabled whisper.cpp server and downloads medium.en + small.en.
scripts/bootstrap.sh

# Downloads the public training source, trains the local text model, and writes the local artifact.
uv run python -m pulse.train

# Optional but recommended: replay the included recording before using the microphone.
scripts/validate_fixture.sh assets/pulse.mov

# Starts Pulse and both loopback-only Whisper servers. Keep this terminal open.
scripts/run.sh
```

Open `http://127.0.0.1:8050`, select **Start listening**, allow microphone access, and speak naturally in English. The footer reports when audio input is arriving. `small.en` supplies fast provisional text; after a natural pause, `medium.en` replaces it with the retained final transcript. The fixture command prefers `medium.en`, then retries `small.en` and `base.en` only when the prior model misses the transcript or timing gate.

If the dashboard says it cannot reach the local server, confirm that `scripts/run.sh` is still running and reload the page. If it reports no microphone signal, verify that the browser granted `127.0.0.1` microphone access and that the correct input device is selected. Pulse is English-only in this release.

## Data and limitations

Training uses the official GoEmotions train/dev/test files, with the source revision, hashes, exact mapping, training seed, and attribution written into the local artifact manifest. Dataset rows, audio, transcripts, and trained weights are ignored by git.

GoEmotions contains English Reddit comments and its labels reflect its annotators and source population. Pulse uses English-only transcription and sentiment labels. It is suitable for a local UI demo, not for employment, health, credit, safety, or automated customer decisions.

See [the validation record](docs/validation.md) for the frozen model’s split metrics and the local video replay timings.
