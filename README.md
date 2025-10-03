# Automatic Speech Recognition and Optical Character Recognition

Media **Transcription & OCR** API powered by **faster-whisper** (ASR) and **EasyOCR** (OCR).
Also supports single-page HTML extraction and optional **yt-dlp** video download → audio transcription.

* **ASR (audio/video → text):** faster-whisper with automatic CPU/GPU selection
* **OCR (images → text):** EasyOCR with configurable languages and model head
* **HTML extract (1 page):** BeautifulSoup + lxml → clean Markdown
* **Video pages:** optional yt-dlp to fetch, extract audio, and transcribe
* **FastAPI** with a small, friendly HTTP surface

---

## Quick start

### Pull & run (CPU)

```bash
docker run --rm -p 9002:9002 ghcr.io/knowusuboaky/asr_ocr:latest
# docs: (not enabled)   | health: http://localhost:9002/health
```

### Pull & run (CUDA / NVIDIA)

```bash
docker run --rm --gpus all -p 9002:9002 ghcr.io/knowusuboaky/asr_ocr:cuda
```

> The server listens on **port 9002**.

---

## Endpoints

### `GET /health`

Health probe.

```
ok
```

### `GET /`

Plain text summary of active ASR/OCR setup (model/device, langs).

### `POST /transcribe_file`

* **Body:** raw bytes of an audio/video file (any ffmpeg-readable format)
* **Headers:** `X-Filename: yourfile.ext` (optional)
* **Returns:** Markdown transcript

```bash
curl -X POST http://localhost:9002/transcribe_file \
  -H "X-Filename: clip.mp3" \
  --data-binary @clip.mp3
```

### `POST /transcribe_image`

* **Body:** raw bytes of an image (png/jpg, etc.)
* **Headers:** `X-Filename: image.png` (optional)
* **Returns:** Markdown OCR text block

```bash
curl -X POST http://localhost:9002/transcribe_image \
  -H "X-Filename: slide.png" \
  --data-binary @slide.png
```

### `POST /transcribe_url`

Attempts, in order:

1. **yt-dlp** (if installed) → download video(s) → extract audio → transcribe
2. **HTML** page → extract clean text
3. **Image** → OCR
4. **Audio/Video bytes** → transcribe

* **JSON body:**

```json
{ "url": "https://example.com/...", "user_agent": "Optional UA string" }
```

```bash
curl -X POST http://localhost:9002/transcribe_url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://youtu.be/xyz"}'
```

Returns Markdown.

---

## Environment variables (optional)

| Variable            |             Default | Description                                                  |
| ------------------- | ------------------: | ------------------------------------------------------------ |
| `ASR_MODEL`         |          `large-v3` | faster-whisper model size (`large-v3`, `medium`, `small`, …) |
| `VAD_MIN_SIL_MS`    |               `500` | VAD min silence (ms) between segments                        |
| `KEEP_WAV`          |             `false` | Keep temporary WAV after processing                          |
| `KEEP_WAV_ON_ERROR` |              `true` | Keep WAV if an error occurs                                  |
| `OCR_LANGS`         |                `en` | Comma-sep langs for EasyOCR (e.g., `en,fr`)                  |
| `OCR_UPSCALE`       |               `1.5` | Pre-resize factor for OCR input                              |
| `OCR_RECOG_HEAD`    |        `english_g2` | EasyOCR recognition head                                     |
| `OCR_ALLOWLIST`     | alnum + punctuation | Characters allowed by OCR                                    |

Set via `docker run -e NAME=value …`.

---

## Docker

### Dockerfile (summary)

Exposes **9002**, installs:

* `fastapi`, `uvicorn[standard]`, `httpx`
* `faster-whisper`, `imageio-ffmpeg`
* `easyocr`, `pillow`
* `beautifulsoup4`, `lxml`
* `yt-dlp` (optional but included by default)

FFmpeg is provided via `imageio-ffmpeg` (bundled binary).

### Build locally (CPU)

```bash
docker build -t ghcr.io/knowusuboaky/asr_ocr:latest .
```

### Build locally (CUDA 12.1 example, amd64)

```bash
docker build \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
  -t ghcr.io/knowusuboaky/asr_ocr:cuda .
```

> Run GPU images with `--gpus all`. Ensure NVIDIA drivers + `nvidia-container-toolkit`.

---

## Compose (optional)

```yaml
# docker-compose.yml
services:
  asr_ocr:
    image: ghcr.io/knowusuboaky/asr_ocr:latest    # or :cuda
    ports:
      - "9002:9002"
    environment:
      - ASR_MODEL=large-v3
      - OCR_LANGS=en
    restart: unless-stopped
```

```bash
docker compose up -d
# http://localhost:9002/health
```

---

## Notes & tips

* **CPU vs GPU**: The server auto-detects CUDA via ctranslate2 for ASR; EasyOCR uses `torch.cuda.is_available()`. GPU builds significantly speed up large models (e.g., `large-v3`).
* **Model downloads**: faster-whisper will download models on first use; keep a Docker volume for cache if you rebuild often.
* **Large inputs**: For long videos, yt-dlp + ffmpeg conversion can take time. Using a GPU image is recommended.
* **HTML extraction**: Designed for single pages; not a crawler. Returns a cleaned, readable Markdown body with basic metadata.
* **Friendly network errors**: URL fetch returns a plain 502 message for DNS/connect errors.

---

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install fastapi "uvicorn[standard]" httpx faster-whisper imageio-ffmpeg \
            easyocr pillow beautifulsoup4 lxml yt-dlp
uvicorn transcribe_server:app --host 0.0.0.0 --port 9002
```

---
