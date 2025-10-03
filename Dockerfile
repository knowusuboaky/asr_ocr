# ------------------------------------------------------------
# asr_ocr — Media Transcription & OCR Server
# CPU by default, optional CUDA via build arg TORCH_INDEX_URL
#
# Build (CPU):
#   docker build -t ghcr.io/knowusuboaky/asr_ocr:latest .
#
# Build (CUDA 12.1 example, amd64):
#   docker build \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
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

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Runtime libs for EasyOCR/OpenCV + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./transcribe_server.py /app/transcribe_server.py

# ---- Python deps ----
# If TORCH_INDEX_URL is set (CUDA), install matching torch wheels; otherwise CPU torch.
ARG TORCH_INDEX_URL=
RUN python -m pip install --upgrade pip \
 && if [ -n "$TORCH_INDEX_URL" ]; then \
        python -m pip install "torch==2.*" --index-url "$TORCH_INDEX_URL" ; \
    else \
        python -m pip install torch ; \
    fi \
 && python -m pip install \
      fastapi "uvicorn[standard]" httpx \
      faster-whisper imageio-ffmpeg \
      easyocr pillow \
      beautifulsoup4 lxml \
      yt-dlp \
 && true

EXPOSE 9002

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS http://127.0.0.1:9002/health || exit 1

# Start API (single worker)
CMD ["uvicorn", "transcribe_server:app", "--host", "0.0.0.0", "--port", "9002"]
