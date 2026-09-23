# Model pocket-v2 (Persian, two voices, CPU)

Engine: [pocket-tts (fork)](https://github.com/mallahyari/pocket-tts) + weights
[pocket-tts-farsi v2](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2)
(**CC-BY-NC-4.0 — non-commercial only**; trained on 973 hours from 2,978 speakers).

> Licence: this model's weights are non-commercial — personal/research use with
> attribution ([model page](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2)).
> For commercial use pick the `pocket` model (MIT).

## Why pocket-v2?

- CPU only (no graphics card), about 1 GB RAM, ~0.7 s per sentence.
- **Two voices**: female and male (two voice prompts); picked with `"instruct":"female"` or `"male"` (default female).
- More accurate than v1 (WER about 0.58 vs 2.05) and mid-generation stalls (runaway) are nearly gone.

## Limits

- Text max 1500 characters; long texts are split into ~18-token chunks.
- This model works on phonemes — the worker converts your Persian text with the
  [G2P](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF) model itself; you send normal Persian text.
- Concurrency with lock; output is always mono 64kbps MP3 + `X-Model: pocket-fa-v2` and `X-Voice` headers.

## Model weights

Weights are NOT in the image; download them and keep this layout under `/models`:

- [pocket-tts-farsi v2 on HuggingFace](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2) →
  `/models/tts/pocket-fa-v2/` (`model.safetensors` + `tokenizer_ph.model` + `model.yaml` + `normalize_fa.py` + `samples/`)
- [Homo-GE2PE-Persian-HF on HuggingFace](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF) →
  `/models/g2p/Homo-GE2PE-Persian-HF/`
