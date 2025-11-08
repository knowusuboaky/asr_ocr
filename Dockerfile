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
# Run (CPU) using baked snapshots (recommended if you copied ./models below):
#   docker run --rm -p 9002:9002 ghcr.io/knowusuboaky/asr_ocr:latest
#
# Optional: mount an external cache instead (NOTE: this masks baked snapshots):
#   docker run --rm -p 9002:9002 \
#     -v "$(pwd)/models:/models/hf" \
#     ghcr.io/knowusuboaky/asr_ocr:latest
#
# Useful env (optional):
#   ASR_MODEL=large-v3|medium|small|...
#   VAD_MIN_SIL_MS=500
#   KEEP_WAV=false
#   KEEP_WAV_ON_ERROR=true
#   OCR_LANGS="en,fr"
#   OCR_UPSCALE=1.5
#   OCR_RECOG_HEAD="english_g2"
#   OCR_ALLOWLIST="0123...ABC..."
#   FLORENCE_MAX_CONCURRENCY=1
#   # Snapshot logic (single knob each):
#   FLORENCE_ALLOW_DOWNLOAD=false   # prefer local snapshot; only download if missing
#   WHISPER_ALLOW_DOWNLOAD=false    # same behavior for faster-whisper
#   # Override locations (defaults point under MODELS_ROOT=/models/hf):
#   FLORENCE_LOCAL_DIR=/models/hf/Florence-2-large
#   FLORENCE_REVISION=main
#   WHISPER_REVISION=main
# ------------------------------------------------------------

# ====== Dockerfile ============================================================
# syntax=docker/dockerfile:1
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TOKENIZERS_PARALLELISM=false \
    # Default to single-worker uvicorn & in-model concurrency guard
    UVICORN_WORKERS=1 \
    FLORENCE_MAX_CONCURRENCY=1 \
    # Persisted HF cache + our snapshots live here
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    MODELS_ROOT=/models/hf \
    # Snapshot behavior defaults (prefer local, avoid network)
    FLORENCE_ALLOW_DOWNLOAD=false \
    WHISPER_ALLOW_DOWNLOAD=false \
    FLORENCE_REVISION=main \
    WHISPER_REVISION=main \
    # Point loaders to baked snapshots (can be overridden at runtime)
    WHISPER_SNAPSHOT=true \
    WHISPER_SNAPSHOT_DIR=/models/hf/faster-whisper-large-v3 \
    FLORENCE_LOCAL_DIR=/models/hf/Florence-2-large

# Optional system feature: 1 = install LibreOffice for DOC/PPT/XLSX -> PDF
ARG ENABLE_OFFICE=0

WORKDIR /app

# Runtime libs:
# - ffmpeg: robust ASR media conversion
# - libgl1/libglib2.0-0/libgomp1/fonts: Pillow, PyMuPDF, Paddle deps
# - curl/ca-certificates: healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libgl1 libglib2.0-0 libgomp1 \
      fonts-dejavu \
      curl ca-certificates \
    && if [ "$ENABLE_OFFICE" = "1" ]; then \
         apt-get install -y --no-install-recommends libreoffice; \
       fi \
    && rm -rf /var/lib/apt/lists/*

# Ensure cache/snapshot root exists
RUN mkdir -p /models/hf

# ---- Bake local model snapshots into the image (if present on build host) ----
# Expecting host paths:
#   ./models/faster-whisper-large-v3/**
#   ./models/Florence-2-large/**
RUN mkdir -p /models/hf/faster-whisper-large-v3 /models/hf/Florence-2-large

# Copy them explicitly so paths line up with env defaults
COPY ./models/faster-whisper-large-v3/ /models/hf/faster-whisper-large-v3/
COPY ./models/Florence-2-large/       /models/hf/Florence-2-large/

# App code + both requirement sets
COPY ./transcribe_server.py /app/transcribe_server.py
COPY ./requirements.txt ./requirements.cuda121.txt /tmp/

# Choose which requirements to install at build time
#   REQ_FILE=requirements.txt          (CPU; default)
#   REQ_FILE=requirements.cuda121.txt  (CUDA 12.1)
ARG REQ_FILE=requirements.txt

# Optional extra index for torch wheels (pass at build time)
ARG TORCH_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=${TORCH_INDEX_URL}

RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install -r /tmp/${REQ_FILE}

# Optional: run as non-root (uncomment if desired)
# RUN useradd -m appuser && chown -R appuser /app /models
# USER appuser

EXPOSE 9002

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS http://127.0.0.1:9002/health || exit 1

# Pin workers=1 explicitly to avoid multi-process model duplication
CMD ["sh","-c","uvicorn transcribe_server:app --host 0.0.0.0 --port 9002 --workers ${UVICORN_WORKERS}"]
# ============================================================================
