# Pulse

> A fully local, real-time speech-to-text and text-sentiment demo for English conversation on Apple Silicon.

## Demo

![](assets/pulse.mov)

- Inspired by [this ElevenLabs Developers post](https://x.com/ElevenLabsDevs/status/2102884507078791484).
- I wanted to build the real-time sentiment layer myself and keep the entire experience on a Mac.
- The recording deliberately uses the same audio as the inspiration, so it demonstrates the transcription and sentence-level sentiment flow against the same input.

## What Pulse does

1. Captures microphone audio in the browser.
2. Produces provisional English transcripts while someone is speaking.
3. Scores each completed sentence independently, so mixed sentiment is visible as it arrives.
4. Replaces the provisional text with a more accurate final transcript after a natural pause.
5. Keeps the colored transcript and speech waveform on screen for the session.

## Signals

| Signal | Source labels | Interpretation |
| --- | --- | --- |
| Frustration | anger, annoyance, disapproval, disgust | Negative or dissatisfied language |
| Positive | admiration, gratitude, joy, optimism, and related labels | Positive language |
| Surprise | realization, surprise | Unexpected information or reaction |
| Uncertainty | confusion, curiosity, fear, nervousness | Doubt, questions, or hesitation |
| Low mood | disappointment, embarrassment, grief, remorse, sadness | Downbeat language |
| Neutral | neutral | No stronger text signal |
| Action pressure | Explicit urgency phrases | A separate lexical cue, not an emotion |

The six emotion signals are independent multi-label scores. A sentence can be both positive and urgent, or frustrated and uncertain.

## How it works locally

| Stage | Runs on | Job |
| --- | --- | --- |
| Browser capture | Browser | Converts microphone input to 16 kHz mono PCM and sends it only to `127.0.0.1`. |
| Fast transcription | `whisper.cpp` + `small.en` | Refreshes a rolling three-second window every 300 ms for responsive provisional text. |
| Final transcription | `whisper.cpp` + `medium.en` | Re-decodes the completed utterance after a 700 ms pause to retain better text. |
| Sentiment | Fine-tuned `all-MiniLM-L6-v2` | Scores each completed sentence and the changing live tail with six calibrated probabilities. |
| Dashboard | Browser | Preserves stable sentence spans, colored history, and the session waveform without re-rendering the entire transcript. |

- `whisper.cpp` uses Metal on Apple Silicon.
- The text classifier runs locally on CPU, leaving Metal available for speech recognition.
- Audio stays in memory and is discarded when the session stops.
- No runtime request leaves the Mac.

## Model and data

- **Public model:** [adimyth/pulse-goemotions](https://huggingface.co/adimyth/pulse-goemotions)
- **Base encoder:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Training data:** [Google Research’s GoEmotions dataset](https://huggingface.co/datasets/google-research-datasets/go_emotions), a 58k-comment English dataset with 27 emotion categories plus Neutral.
- **Training:** official train split plus deterministic unpunctuated STT-style augmentation; model selection and temperature fitting on the official development split; untouched official test split evaluated once after freezing.
- **Frozen result:** test macro-F1 `0.601`, mean ECE `0.090`, exact multi-label match `0.454`.
- **Release files:** `model.safetensors`, label/temperature metadata, and a source-hash manifest; the dataset itself is never bundled.

> [!NOTE]
> Pulse is English-only in this release.

## Run it

### Requirements

- Apple Silicon Mac
- Xcode Command Line Tools (`xcode-select --install`)
- `git`, `cmake`, and [`uv`](https://docs.astral.sh/uv/) (`brew install cmake uv` if you use Homebrew)
- A browser that can grant microphone permission to `http://127.0.0.1:8050`

### Start with the published model

```sh
git clone https://github.com/adimyth/pulse.git
cd pulse
scripts/bootstrap.sh
uv run python -m pulse.download
scripts/run.sh
```

Open `http://127.0.0.1:8050`, select **Start listening**, and allow microphone access.

### Verify the demo recording

```sh
scripts/validate_fixture.sh assets/pulse.mov
```

### Train a new local artifact

```sh
uv run python -m pulse.train
```

## Reproducibility

- The trained artifact records the GoEmotions source revision, SHA-256 hashes, original label order, derived label mapping, seed, temperatures, and split metrics.
- The video is a demo fixture only; it is not training data.
- See [the validation record](docs/validation.md) for fixture replay timings and the frozen evaluation record.
