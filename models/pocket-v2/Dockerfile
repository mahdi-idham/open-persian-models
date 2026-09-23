# pocket-v2 model: pocket-tts-farsi-v2 CPU worker (CC-BY-NC-4.0 weights).
# Built from PUBLIC artefacts: python:3.12-slim + PyPI (torch CPU) + the
# pocket-tts fork tarball + repo files. NO git clone, NO CUDA (109M params,
# CPU several-x-realtime). Model + G2P + voices come from a READ-ONLY host
# mount (never baked in).
#
# Weights are NOT in this repo (non-commercial licence); download them from
# HuggingFace (see README.md) and mount at /models (layout below).
#
# First build needs internet (PyPI torch CPU + deps + fork tarball); on
# filtered networks build via a SOCKS proxy (see main README):
#   cd models/pocket-v2 && bash get_ffmpeg.sh   # vendored static ffmpeg (gitignored, once)
#   docker build -t tts-pocketv2-worker:latest --network=host \
#     --build-arg PIP_PROXY=socks5h://127.0.0.1:10808 .
#
# Pins: torch from the CPU index; transformers==5.17.0; pocket-tts fork
# commit 3807c20 (PyPI pocket-tts 3.1.0 REJECTS the v2 model.yaml:
# `capitalize_first_letter: Extra inputs are not permitted` — the fork
# carries the extra config fields; same version string, so --force-reinstall
# is required, and the build asserts the fork afterwards).
ARG PYTHON_IMAGE=python:3.12-slim
ARG PIP_PROXY=""

FROM ${PYTHON_IMAGE}
ENV PYTHONUNBUFFERED=1 \
    PORT=8303 THREADS=2 \
    VOICE_WAV=/models/tts/pocket-fa-v2/samples/prompt_short_sentence.wav \
    VOICES=female=/models/tts/pocket-fa-v2/samples/prompt_short_sentence.wav,male=/models/tts/pocket-fa-v2/samples/prompt_news_paragraph.wav \
    MODEL_DIR=/models/tts/pocket-fa-v2 G2P_DIR=/models/g2p/Homo-GE2PE-Persian-HF \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
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
 && pip install -q $PIP_ARGS "transformers==5.17.0" sentencepiece scipy safetensors einops \
 && pip install -q --force-reinstall --no-deps $PIP_ARGS \
      "https://github.com/mallahyari/pocket-tts/archive/3807c204babb5fe54be8fe18a362a58315e870d6.tar.gz" \
 && python -c "import torch, pocket_tts, transformers, sentencepiece, soundfile, scipy; from pocket_tts.utils.config import Config; assert 'capitalize_first_letter' in Config.model_fields, 'wrong pocket-tts (PyPI 3.1.0, not the fork)'; print('stack ok', torch.__version__)" \
 && rm -rf /w /root/.cache/pip
COPY ffmpeg /usr/local/bin/ffmpeg
COPY worker_tts_v2.py /app/worker_tts_v2.py
EXPOSE 8303
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -sf localhost:8303/health || exit 1
CMD ["python3", "/app/worker_tts_v2.py"]
