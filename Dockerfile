# ------------------------------------------------------------
# asr_ocr — Media Transcription & OCR Server
# CPU by default; select CUDA 12.1 reqs via REQ_FILE build-arg
#
# Build (CPU, default requirements.txt):
#   docker build -t ghcr.io/knowusuboaky/asr_ocr:latest .
#
# Build (CUDA 12.1, requirements.cuda121.txt):
#   docker build --build-arg REQ_FILE=requirements.cuda121.txt \
#     -t ghcr.io/knowusuboaky/asr_ocr:cuda .
#
# Run (CPU):
#   docker run --rm -p 9002:9002 ghcr.io/knowusuboaky/asr_ocr:latest
#
# Run (CUDA):
#   docker run --rm --gpus all -p 9002:9002 ghcr.io/knowusuboaky/asr_ocr:cuda
#
# Env (optional):
#   ASR_MODEL=large-v3|medium|small|...
#   VAD_MIN_SIL_MS=500
#   KEEP_WAV=false
#   KEEP_WAV_ON_ERROR=true
#   OCR_LANGS="en,fr"
#   OCR_UPSCALE=1.5
#   OCR_RECOG_HEAD="english_g2"
#   OCR_ALLOWLIST="0123...ABC..."
# ------------------------------------------------------------

# ====== Dockerfile ============================================================
# Build args (override at build time)
# syntax=docker/dockerfile:1
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TOKENIZERS_PARALLELISM=false

# Optional system feature: 1 = install LibreOffice for DOC/PPT/XLSX -> PDF
ARG ENABLE_OFFICE=0

WORKDIR /app

# Runtime libs for Pillow/PyMuPDF/Paddle + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libgomp1 \
      fonts-dejavu \
      curl ca-certificates \
    && if [ "$ENABLE_OFFICE" = "1" ]; then \
         apt-get install -y --no-install-recommends libreoffice; \
       fi \
    && rm -rf /var/lib/apt/lists/*

# App code + both requirement sets
COPY ./transcribe_server.py /app/transcribe_server.py
COPY ./requirements.txt ./requirements.cuda121.txt /tmp/

# Choose which requirements to install at build time
#   REQ_FILE=requirements.txt          (CPU; default)
#   REQ_FILE=requirements.cuda121.txt  (CUDA 12.1)
ARG REQ_FILE=requirements.txt

# Optional extra index for torch wheels (pass at build time)
ARG TORCH_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=$TORCH_INDEX_URL

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r /tmp/${REQ_FILE}

EXPOSE 9002

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS http://127.0.0.1:9002/health || exit 1

CMD ["uvicorn", "transcribe_server:app", "--host", "0.0.0.0", "--port", "9002"]
# ============================================================================
