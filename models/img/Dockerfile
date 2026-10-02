# qwen-image-2.1-gguf worker (512x512, single GPU).
#
# WEIGHTS ARE NOT IN THIS REPO. The GGUF build lives on HuggingFace at
# gguf-org/qwen-image-2.1-gguf and carries the upstream Qwen licence
# ("other"). Mount it read-only at /models (see README.md).
#
# Engine: `ggk` (MIT, github.com/gguf-org/ggk) instead of diffusers - it runs
# the GGUF quant directly, so no CUDA source build of the model is needed.
#
# NOTE: the text-encoder dependency is NOT pinned here on purpose. The fast
# config uses a small (0.6B) encoder whose package name is not recorded in
# this repo; until it is pinned and verified, install it into the image
# yourself and record the exact command in README.md before publishing.
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

# ggk itself is MIT and on PyPI.
RUN python3 -m pip install --no-cache-dir --upgrade pip \
 && python3 -m pip install --no-cache-dir ggk

WORKDIR /work
COPY worker_img_ggk.py /work/worker_img_ggk.py

ENV LISTEN=127.0.0.1 \
    PORT=9412
EXPOSE 9412
CMD ["python3", "/work/worker_img_ggk.py"]
