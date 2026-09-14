# pocket model: pocket-tts-farsi v1 CPU worker (MIT weights + CC0 data).
# Built from PUBLIC artefacts: python:3.12-slim + PyPI (torch CPU, pocket-tts)
# + repo files. NO git clone, NO CUDA (100M params, CPU several-x-realtime).
# Model + voice + yaml come from a READ-ONLY host mount (never baked in).
#
# First build needs internet (PyPI torch CPU + pocket-tts); on filtered
# networks build via a SOCKS proxy (see main README):
#   cd models/pocket && bash get_ffmpeg.sh   # vendored static ffmpeg (gitignored, once)
#   docker build -t tts-pocket-worker:latest --network=host \
#     --build-arg PIP_PROXY=socks5h://127.0.0.1:10808 .
#
# Pins: torch from the CPU index (latest), pocket-tts from PyPI (latest).
# Model + voice + yaml come from a READ-ONLY host mount (never baked in).

ARG PYTHON_IMAGE=python:3.12-slim
ARG PIP_PROXY=""

FROM ${PYTHON_IMAGE}
ENV PYTHONUNBUFFERED=1 \
    MODEL_YAML=/models/tts/pocket-fa/farsi.yaml \
    VOICE_WAV=/models/tts/pocket-fa/example_voice.wav \
    POCKET_TEMP=0.3 POCKET_EOS=-2.0 \
    BITRATE=64k TIMEOUT_S=120 PORT=8302 MAX_CHARS=1500 \
    ARCHIVE_DIR=/archive
# NOTE: offline env (HF_HUB_OFFLINE=1) is set in docker-compose.yml, not
# here, so it does not invalidate the cached pip layer. worker_tts.py localizes
# the hf:// URIs in farsi.yaml to the /models mount either way.
ARG PIP_PROXY
# curl for the healthcheck; pinned socks wheels so pip can use the proxy.
# Pinned proxy wheels: PySocks 1.7.0 (requests/urllib3) + socksio 1.0.0
# (pip25/httpx — 0.1.1 breaks httpcore). Public PyPI URLs below.
RUN printf '%s\n' 'Acquire::http::Proxy "socks5h://127.0.0.1:10808/";' \
    'Acquire::https::Proxy "socks5h://127.0.0.1:10808/";' \
    > /etc/apt/apt.conf.d/99proxy \
 && apt-get update -qq \
 && apt-get install -y -qq curl \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /w && cd /w \
 && curl -s -L -O "https://files.pythonhosted.org/packages/cd/18/102cc70347486e75235a29a6543f002cf758042189cb063ec25334993e36/PySocks-1.7.0-py3-none-any.whl" \
 && curl -s -L -O "https://files.pythonhosted.org/packages/37/c3/6eeb6034408dac0fa653d126c9204ade96b819c936e136c5e8a6897eee9c/socksio-1.0.0-py3-none-any.whl" \
 && pip install -q --no-index /w/*.whl \
 && python -c "import socks, socksio" \
 && if [ -n "${PIP_PROXY}" ]; then export HTTPS_PROXY="${PIP_PROXY}" HTTP_PROXY="${PIP_PROXY}" ALL_PROXY="${PIP_PROXY}"; PIP_ARGS="--proxy ${PIP_PROXY}"; else PIP_ARGS=""; fi \
 && pip install -q $PIP_ARGS --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -q $PIP_ARGS pocket-tts soundfile numpy \
 && python -c "import torch, pocket_tts, soundfile; print('stack ok, torch', torch.__version__)" \
 && rm -rf /w /root/.cache/pip
COPY ffmpeg /usr/local/bin/ffmpeg
COPY worker_tts.py farsi_tts.py normalize_fa.py /app/
EXPOSE 8302
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -sf localhost:8302/health || exit 1
CMD ["python3", "/app/worker_tts.py"]
