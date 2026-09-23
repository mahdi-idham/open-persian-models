# ocr-bina model: Persian OCR HTTP worker (Bina-0.2-Rizeh, Apache-2.0 weights).
# Built from PUBLIC artefacts: NVIDIA CUDA runtime + paddle + PaddleOCR + repo
# file. Model code + weights come from a READ-ONLY host mount (never baked in).
#
# Weights are NOT in this repo; download them from HuggingFace (see README.md)
# and mount at /models (layout below).
#
# GPU wheel exists ONLY as paddlepaddle-gpu==3.3.0 on the cu126 index (PyPI
# hosts no GPU wheel for 3.x) + paddleocr==3.7.0 (earlier 3.3.x registers only
# up to PP-OCRv5, and Bina ships PP-OCRv6 weights).
#
# First build needs internet; on filtered networks build via a SOCKS proxy
# (see main README):
#   docker build -t ocr-bina-worker:latest --network=host \
#     --build-arg PIP_PROXY=socks5h://127.0.0.1:10808 .
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
ARG PIP_PROXY=""
RUN printf '%s\n' 'Acquire::http::Proxy "socks5h://127.0.0.1:10808/";' \
    'Acquire::https::Proxy "socks5h://127.0.0.1:10808/";' \
    > /etc/apt/apt.conf.d/99proxy \
 && apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends python3 python3-pip libgomp1 libgl1 libglib2.0-0 curl poppler-utils \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /w && cd /w \
 && curl -s -L -O "https://files.pythonhosted.org/packages/cd/18/102cc70347486e75235a29a6543f002cf758042189cb063ec25334993e36/PySocks-1.7.0-py3-none-any.whl" \
 && curl -s -L -O "https://files.pythonhosted.org/packages/37/c3/6eeb6034408dac0fa653d126c9204ade96b819c936e136c5e8a6897eee9c/socksio-1.0.0-py3-none-any.whl" \
 && python3 -m pip install -q --no-index /w/*.whl \
 && python3 -c "import socks, socksio" \
 && if [ -n "${PIP_PROXY}" ]; then export HTTPS_PROXY="${PIP_PROXY}" HTTP_PROXY="${PIP_PROXY}" ALL_PROXY="${PIP_PROXY}"; PIP_ARGS="--proxy ${PIP_PROXY}"; else PIP_ARGS=""; fi \
 && python3 -m pip install -q --no-cache-dir --upgrade pip \
 && python3 -m pip install -q --no-cache-dir $PIP_ARGS paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
 && python3 -m pip install -q --no-cache-dir $PIP_ARGS paddleocr==3.7.0 Pillow numpy fastapi uvicorn python-multipart \
 && python3 -c "import paddle, paddleocr; print('ocr stack ok')" \
 && rm -rf /w /root/.cache/pip
WORKDIR /work
COPY worker_ocr.py /work/worker_ocr.py
EXPOSE 8340
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -sf http://127.0.0.1:8340/health || exit 1
CMD ["python3", "/work/worker_ocr.py"]
