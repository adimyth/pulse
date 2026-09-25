# Validation record

This record applies to `pulse-local-goemotions-0.1`, trained locally on 24 September 2026. The model was selected using the official GoEmotions development split, then evaluated once on its official test split.

| Split | Macro F1 | Mean ECE | Exact multi-label match |
| --- | ---: | ---: | ---: |
| Train | not reported | not reported | not reported |
| Development | 0.600 | 0.086 | 0.461 |
| Official test | 0.601 | 0.090 | 0.454 |

Training used 43,410 official source-train examples plus one deterministic punctuation-normalised transcript variant per example, for 86,820 training examples. The seed was `20260925`; five epochs completed. Temperatures are fit per displayed dimension on development data and are stored with the local artifact. The project does not add dataset rows, transcripts, audio, or weights to git.

The supplied video was replayed through the locally built English `small.en` Whisper runtime and the frozen classifier on this M4 Pro. It emitted multiple transcript updates, recovered both the positive and duplicate-charge/frustration cues, and met the latency gate after warm-up:

| Measure | Result | Gate |
| --- | ---: | ---: |
| Text classifier p95 | 34.6 ms | <=100 ms |
| Audio-to-dashboard p50 | 157 ms | <=1,000 ms |
| Audio-to-dashboard p95 | 231 ms | <=1,750 ms |

The latency split is deliberate: Whisper uses Metal and MiniLM uses the local CPU, avoiding contention between two models on the same Apple GPU. These results validate the local demonstration path; they do not establish production sentiment accuracy for a particular population or business workflow.
