# TEMPLATE for a new TTS model image. Copy this folder to models/<your-model>/
# and fill the TODOs. Keep the pattern: builder compiles, runtime stays small.

# TODO 1: pick builder base with compiler + CUDA dev libs (example below).
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04 AS builder
# TODO 2: clone your TTS engine at a PINNED revision (reproducible builds).
#   Never track a moving branch: pin a commit SHA and record it here.
# RUN git clone <ENGINE_REPO> /opt/engine && cd /opt/engine && git checkout <SHA> && ./build.sh

# TODO 3: slim runtime base + only runtime deps (python? ffmpeg? libs?).
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04
# RUN apt-get update && apt-get install -y --no-install-recommends python3 ... && rm -rf /var/lib/apt/lists/*
# COPY --from=builder /opt/engine/build /opt/engine/build
# COPY worker.py /app/worker.py
# EXPOSE 8300
# HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -sf localhost:8300/health || exit 1
# CMD ["python3", "/app/worker.py"]
