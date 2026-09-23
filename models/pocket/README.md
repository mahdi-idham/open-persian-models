# Model pocket (Persian, light, CPU)

Engine: [pocket-tts](https://github.com/kyutai-labs/pocket-tts) + weights
[pocket-tts-farsi v1](https://huggingface.co/mehdi-hf/pocket-tts-farsi)
(MIT; CC0 training data — fine for commercial use).

## Why pocket?

- CPU only (no graphics card), about 1 GB RAM, faster than the sample GPU model.
- Single voice (`example_voice.wav`); accepts `instruct` and `seed` but ignores them (a fresh take every time).

## Limits

- Text max 1500 characters; long texts are split into ~18-token chunks.
- Latin letters inside Persian text are dropped (training normalizer) — ideal for pure Persian text.
- Concurrency 1 (lock); output is always mono 64kbps MP3 + `X-Model: pocket-fa-v1` header.

## Model weights

Weights are NOT in the image; download them and put them in `/models/tts/pocket-fa/`:

- [pocket-tts-farsi v1 on HuggingFace](https://huggingface.co/mehdi-hf/pocket-tts-farsi)

Files needed: `model.safetensors` + `tokenizer.model` + `farsi.yaml` + `normalize_fa.py` + `example_voice.wav`.
