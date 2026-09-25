---
language:
- en
license: apache-2.0
pipeline_tag: text-classification
base_model: sentence-transformers/all-MiniLM-L6-v2
datasets:
- google-research-datasets/go_emotions
tags:
- pulse
- sentiment-analysis
- emotion
- multi-label-classification
- pytorch
---

# Pulse GoEmotions

Pulse GoEmotions is the compact multi-label text-sentiment classifier used by [Pulse](https://github.com/adimyth/pulse), a local real-time speech-to-text and sentiment demo for Apple Silicon.

## Output

- `frustration`: anger, annoyance, disapproval, disgust
- `positive`: admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief
- `surprise`: realization, surprise
- `uncertainty`: confusion, curiosity, fear, nervousness
- `low_mood`: disappointment, embarrassment, grief, remorse, sadness
- `neutral`: neutral

Scores are independent calibrated probabilities. `action_pressure` is a separate transparent lexical cue and is not a learned emotion output.

## Training and evaluation

- Base encoder: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- Data: [Google Research’s GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions), Apache-2.0
- Training: 43,410 official train examples plus deterministic unpunctuated STT-style augmentation
- Selection and calibration: official development split
- Frozen test result: macro-F1 `0.601`, mean ECE `0.090`, exact multi-label match `0.454`
- Seed: `20260925`

The release includes the state dict, temperatures, dimension mapping, source revision, source hashes, and split metrics. It does not include dataset rows or audio.

## Use with Pulse

```sh
git clone https://github.com/adimyth/pulse.git
cd pulse
uv run python -m pulse.download
uv run python -m pulse.server
```

For direct local use:

```python
from pulse.model import PulseClassifier

classifier = PulseClassifier("var/pulse-model")
print(classifier.classify("I am really disappointed and need help right now.").to_dict())
```

## Limitations

- English text only.
- The model classifies what is expressed in the transcript; it does not infer emotion from vocal prosody, facial expression, or a speaker’s hidden state.
- GoEmotions consists of Reddit comments and carries the dataset’s population and annotation limitations.

## Citation

```bibtex
@inproceedings{demszky2020goemotions,
  title={GoEmotions: A Dataset of Fine-Grained Emotions},
  author={Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020}
}
```
