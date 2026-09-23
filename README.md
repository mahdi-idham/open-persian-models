# Open Persian Models with Docker

Ready-to-build Dockerfiles for running Persian models: Persian **text-to-speech**
(TTS) on GPU or CPU. Live version: [Farsihoosh](https://farsihoosh.ir)

## What is this repo?

If you have a TTS model and want to run it inside Docker, you do not need to
start from zero. Each model has its own folder with three things: `Dockerfile` +
Python worker + that model's guide. `models/_template` is an empty template
for the next model.

## Available models (TTS)

| Model | What it does | Weight licence | Folder |
|---|---|---|---|
| omnivoice | Persian text to speech with several voices (female, male, child, ...) | Non-commercial (CC-BY-NC-4.0) | [models/omnivoice](models/omnivoice) |
| pocket | Persian text to speech with one voice, CPU-only (no graphics card), fast | Permissive (MIT) | [models/pocket](models/pocket) |
| pocket-v2 | Persian text to speech with two voices (female and male), CPU-only | Non-commercial (CC-BY-NC-4.0) | [models/pocket-v2](models/pocket-v2) |

When a new model is added, it appears in this table. The add-model guide is in the [template](models/_template).

## Image text extraction (OCR)

Farsihoosh also extracts text from photos (model
[Bina-0.2-Rizeh](https://huggingface.co/Reza2kn/Bina-0.2-Rizeh), permissive
Apache-2.0 licence): try-it page [persian-ocr](https://farsihoosh.ir/persian-ocr)
and docs [api-docs-ocr](https://farsihoosh.ir/api-docs-ocr).
There is no ready OCR docker in this repo yet — for now only the weights link above.

## Prerequisites

- Linux (for the omnivoice model: NVIDIA graphics card with at least 8 GB VRAM; pocket and pocket-v2 need CPU only)
- NVIDIA driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (omnivoice only)
- Docker + Docker Compose
- The model weight files (never baked into the image; attached with a volume)

## Quickstart (omnivoice model)

```bash
# 1. Put the weights somewhere on the host, e.g.:
/models/tts/omnivoice-q4/

# 2. Put a static ffmpeg next to the Dockerfile (77 MB, once):
cd models/omnivoice && bash get_ffmpeg.sh && cd ../..

# 3. Build and run:
docker compose up -d --build

# 4. Health check and speech generation:
curl localhost:8300/health
curl -s localhost:8300/tts -H 'Content-Type: application/json' \
  -d '{"text":"سلام دنیا","instruct":"female, young adult"}' -o out.mp3
```

## Quickstart (pocket model, CPU only)

```bash
# 1. Put the weights somewhere on the host, e.g.:
/models/tts/pocket-fa/

# 2. Put a static ffmpeg next to the Dockerfile (77 MB, once):
cd models/pocket && bash get_ffmpeg.sh && cd ../..

# 3. Build and run (no GPU needed):
docker compose up -d --build tts-pocket

# 4. Health check and speech generation:
curl localhost:8302/health
curl -s localhost:8302/tts -H 'Content-Type: application/json' \
  -d '{"text":"سلام دنیا"}' -o out.mp3
```

## Quickstart (pocket-v2 model, CPU only, two voices)

```bash
# 1. Put the weights somewhere on the host, e.g.:
/models/tts/pocket-fa-v2/
# And the G2P model (text to phonemes):
/models/g2p/Homo-GE2PE-Persian-HF/

# 2. Put a static ffmpeg next to the Dockerfile (77 MB, once):
cd models/pocket-v2 && bash get_ffmpeg.sh && cd ../..

# 3. Build and run (no GPU needed):
docker compose up -d --build tts-pocketv2

# 4. Health check and speech generation (female and male):
curl localhost:8303/health
curl -s localhost:8303/tts -H 'Content-Type: application/json' \
  -d '{"text":"سلام دنیا","voice":"female"}' -o out-f.mp3
curl -s localhost:8303/tts -H 'Content-Type: application/json' \
  -d '{"text":"سلام دنیا","voice":"male"}' -o out-m.mp3
```

## How it works

```
POST /tts {text} → worker → TTS engine (GPU or CPU) → WAV → ffmpeg → 64k MP3 → answer
GET /health → {"ok": true}
```

- The worker is plain Python (no framework) with one lock: one synthesis at a time.
- Persian text is normalized before generation (Arabic letters to Persian, extra Tashkeel removed).
- Output is mono 64kbps MP3, optimized for slow connections.

## Adding a new model

1. `cp -r models/_template models/<model-name>`
2. Fill in the Dockerfile (engine source + pinned SHA + build command).
3. Write the worker (two routes are enough: `POST /tts` and `GET /health`).
4. Add a service with a different port in `docker-compose.yml`.
5. Build and test like the quickstarts.

The full guide is in `models/_template/README.md`.

## Note for filtered networks

If `git clone` inside the build cannot reach GitHub, build with a SOCKS proxy:

```bash
docker build --network=host --build-arg GIT_PROXY_URL=socks5h://127.0.0.1:10808 models/omnivoice/
```

## Licence

This repo's files are MIT — use and modify freely. Note that **model weights
have their own licences**: pocket is permissive (MIT), but omnivoice and
pocket-v2 are **non-commercial only** (CC-BY-NC-4.0) and the OCR model is
permissive (Apache-2.0).
