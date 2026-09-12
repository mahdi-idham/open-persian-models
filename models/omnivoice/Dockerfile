# Persian TTS GPU worker image. Built from PUBLIC bases + public source only.
# First build needs internet (Docker Hub base + GitHub clone); afterwards the
# Docker cache makes code-change rebuilds take seconds.
#
#   bash get_ffmpeg.sh                             # vendor static ffmpeg (once)
#   docker build -t tts-worker:latest .
#
# Pinned engine revision (reproducible builds - never track a moving branch):
#   040c8b344d8c670ce1475194751d119b5ef82c78
#
# On filtered networks pass the GitHub proxy (git has native SOCKS support):
#   docker build --network=host --build-arg GIT_PROXY_URL=socks5h://127.0.0.1:10808 .

ARG CUDA_VERSION=12.6.0
ARG UBUNTU_VERSION=22.04
ARG OMNIVOICE_CPP_REPO=https://github.com/ServeurpersoCom/omnivoice.cpp.git
ARG OMNIVOICE_CPP_REVISION=040c8b344d8c670ce1475194751d119b5ef82c78

# ---- builder: compile omnivoice.cpp CUDA backend (nvcc; no GPU device needed) --
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder
ARG OMNIVOICE_CPP_REPO
ARG OMNIVOICE_CPP_REVISION
ARG GIT_PROXY_URL=""
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends git cmake g++ make ca-certificates \
 && rm -rf /var/lib/apt/lists/*
# GitHub-only proxy (git has native SOCKS support; apt/pip come from Ubuntu mirrors).
RUN if [ -n "${GIT_PROXY_URL}" ]; then \
      export https_proxy="${GIT_PROXY_URL}" http_proxy="${GIT_PROXY_URL}"; \
    fi \
 && git clone --recurse-submodules "${OMNIVOICE_CPP_REPO}" /opt/omnivoice.cpp \
 && cd /opt/omnivoice.cpp \
 && git checkout "${OMNIVOICE_CPP_REVISION}" \
 && git rev-parse HEAD > /opt/OMNIVOICE_REVISION \
 # No GPU at link time: NVIDIA ships link-only stubs for libcuda.so.1; ld needs
 # them on the default search path for the TRANSITIVE DT_NEEDED of libggml-cuda.
 && export LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH} \
 && ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
 && cp /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
 && ldconfig \
 && ./buildcuda.sh

# ---- runtime: official NVIDIA CUDA runtime base (has libcudart/libcublas/
#      libcublasLt/libnccl). Python3 via apt. No private images anywhere. --------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}
ENV PYTHONUNBUFFERED=1 GGML_BACKEND=CUDA0 \
    MODEL_DIR=/models TTS_BIN=/opt/omnivoice.cpp/build/omnivoice-tts \
    BITRATE=64k TIMEOUT_S=120 PORT=8300 TTS_LANG=Persian
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 libgomp1 ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/omnivoice.cpp/build /opt/omnivoice.cpp/build
COPY --from=builder /opt/OMNIVOICE_REVISION /app/OMNIVOICE_REVISION
COPY ffmpeg /usr/local/bin/ffmpeg
COPY worker_tts.py /app/worker_tts.py
EXPOSE 8300
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -sf localhost:8300/health || exit 1
CMD ["python3", "/app/worker_tts.py"]
