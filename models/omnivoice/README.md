# Model omnivoice (sample-grade Persian)

Engine: [omnivoice.cpp](https://github.com/ServeurpersoCom/omnivoice.cpp) at pinned
revision `040c8b3` (reproducible build — never follow a moving branch).

## Voices

The `instruct` parameter accepts only these values (anything else is a 422):

`female` · `male` · `child` · `teenager` · `young adult` · `middle-aged` · `elderly` ·
`high pitch` · `moderate pitch` · `low pitch` · `very high pitch` · `very low pitch` · `whisper`

Example: `{"text":"...","instruct":"female, young adult","seed":777}` — a fixed seed
gives a fixed result; no seed means random.

## Limits

- Text max 1500 characters (very short text gives poor quality).
- Concurrency 1 (GPU lock); simultaneous requests queue up.
- Output is always mono 64kbps MP3 + `X-Gen-Ms` and `X-Audio-Seconds` headers.

## Model weights

Weights are NOT in the image; download them and put them in `/models/tts/omnivoice-q4/`:

- [OmniVoice-GGUF on HuggingFace](https://huggingface.co/Serveurperso/OmniVoice-GGUF)

Files needed: `omnivoice-base-Q4_K_M.gguf` (model) + `omnivoice-tokenizer-Q4_K_M.gguf` (tokenizer).
