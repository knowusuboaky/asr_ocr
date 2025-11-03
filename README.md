# Automatic Speech Recognition and Optical Character Recognition

Media **Transcription & OCR** + **Document Ingest** API powered by:
- **ASR:** `faster-whisper` (auto CPU/GPU via CTranslate2)
- **Vision OCR + captions:** **Microsoft Florence-2** (local VLM) with **PaddleOCR** fallback
- **Docs:** PDF / PPTX / DOCX / DOC / XLSX / CSV → clean Markdown (front-matter + grouped contents)
- **HTML (single page):** BeautifulSoup + lxml
- **Video pages:** optional `yt-dlp` → audio → ASR
- **FastAPI** with a compact, friendly HTTP surface (`/__docs__`)

---

## Quick start

### Pull & run (CPU)

```bash
docker run --rm -p 9002:9002 \
  -e ASR_MODEL=large-v3 \
  -e OCR_LANGS=en \
  -e VLM_MODEL_ID=microsoft/Florence-2-large \
  ghcr.io/knowusuboaky/asr_ocr:latest

# Docs:   http://localhost:9002/__docs__
# Health: http://localhost:9002/health
````

### Pull & run (CUDA / NVIDIA)

```bash
docker run --rm --gpus all -p 9002:9002 \
  -e ASR_MODEL=large-v3 \
  -e OCR_LANGS=en \
  -e VLM_MODEL_ID=microsoft/Florence-2-large \
  ghcr.io/knowusuboaky/asr_ocr:cuda
```

> The server listens on **port 9002**. GPU builds accelerate ASR and Florence-2.

---

## Endpoints

### `GET /health`

Liveness probe. Returns:

```
ok
```

### `GET /`

Plain-text summary of active ASR/VLM/OCR setup (model, device, langs).

### `POST /transcribe_file`

ASR for audio/video (any ffmpeg-readable format).

* **Body:** raw media bytes
* **Headers:** `X-Filename: yourfile.ext` (optional)
* **Returns:** Markdown transcript with timestamps

```bash
curl -X POST http://localhost:9002/transcribe_file \
  -H "X-Filename: clip.mp3" \
  --data-binary @clip.mp3
```

### `POST /transcribe_image`

OCR + optional captioning for a single image.

* **Body:** raw image bytes (png/jpg/…)
* **Headers:** `X-Filename: image.png` (optional)
* **Returns:** Markdown with global caption, region captions, and OCR text

```bash
curl -X POST http://localhost:9002/transcribe_image \
  -H "X-Filename: slide.png" \
  --data-binary @slide.png
```

### `POST /extract_document`

Full document ingest to Markdown (front-matter + grouped contents).
Supported: **.pdf, .pptx, .docx, .doc, .xlsx, .csv, .txt**

Query params (all optional – shown with defaults):

* `ocr=paddle|none` (default `paddle`)
* `lang=en` (primary OCR language)
* `dpi=300` (page OCR raster DPI)
* `ocr_threshold=50` (page OCR if native text < N chars)
* `vlm=hf_local|none` (default `hf_local`)
* `vlm_model=microsoft/Florence-2-large`
* `vlm_max_tokens=128`
* `vlm_prompt=<MORE_DETAILED_CAPTION>`
* `caption_pages=false` (page-level captions)
* `vlm_dpi=256` (page render DPI for captions)
* `image_columns=` (CSV: comma-sep column names that hold image refs)
* `fetch_http=false` (allow HTTP fetch for CSV/XLSX image links)
* `image_root=.` (resolve relative paths for linked images)
* `caption_xlsx_charts=false` (try page-captions via PDF export if LibreOffice present)
* `soffice_bin=` (path to LibreOffice/soffice if installed)
* `max_rows=-1` (tables → show **all** rows; set to N to truncate)
* `vlm_only=false` (emit only captions; skip native/OCR text)

```bash
curl -X POST "http://localhost:9002/extract_document?ocr=paddle&vlm=hf_local&lang=en" \
  -H "X-Filename: deck.pptx" \
  --data-binary @deck.pptx
```

### `POST /transcribe_url`

Ordered handling:

1. **yt-dlp** (if present): download video(s) → extract audio → ASR
2. **HTML**: single-page extract → Markdown
3. **Image**: OCR (+ captions)
4. **Audio/Video**: ASR

Body:

```json
{ "url": "https://example.com/...", "user_agent": "Optional UA string" }
```

Example:

```bash
curl -X POST http://localhost:9002/transcribe_url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://youtu.be/xyz"}'
```

---

## Environment variables

### ASR / General

| Variable            | Default             | Description                                            |
| ------------------- | ------------------- | ------------------------------------------------------ |
| `ASR_MODEL`         | `large-v3`          | faster-whisper size (`large-v3`, `medium`, `small`, …) |
| `VAD_MIN_SIL_MS`    | `500`               | VAD min silence (ms) between segments                  |
| `KEEP_WAV`          | `false`             | Keep temp WAV after processing                         |
| `KEEP_WAV_ON_ERROR` | `true`              | Keep WAV if an error occurs                            |
| `FETCH_UA`          | Chrome-like UA      | Default User-Agent for `/transcribe_url`               |
| `MAX_FETCH_BYTES`   | `26214400` (25 MiB) | Cap for single URL fetch                               |
| `HTTP_TIMEOUT_S`    | `30`                | Per-request timeout for URL fetch                      |

### Vision VLM (Florence-2)

| Variable                 | Default                      | Description               |
| ------------------------ | ---------------------------- | ------------------------- |
| `VLM_MODEL_ID`           | `microsoft/Florence-2-large` | HF model id               |
| `VLM_CAPTION_TASK`       | `<MORE_DETAILED_CAPTION>`    | Caption task token        |
| `VLM_MAX_TOKENS_CAPTION` | `768`                        | Max tokens for captioning |
| *Device/dtype*           | auto                         | CUDA→`float16`, else CPU  |

### OCR (PaddleOCR fallback)

| Variable         | Default      | Description                                   |
| ---------------- | ------------ | --------------------------------------------- |
| `OCR_LANGS`      | `en`         | Primary OCR lang (first in list is used)      |
| `OCR_UPSCALE`    | `1.5`        | Pre-resize factor for images                  |
| `OCR_RECOG_HEAD` | `english_g2` | Kept for compatibility (not used by Florence) |
| `OCR_ALLOWLIST`  | alnum+punct  | Characters allowed by OCR                     |

---

## Docker

### Caching models

Mount caches to avoid repeated downloads:

* Hugging Face: `/root/.cache/huggingface`
* PaddleOCR: `/root/.paddleocr`

### Compose (CPU)

```yaml
services:
  asr_ocr:
    image: ghcr.io/knowusuboaky/asr_ocr:latest
    ports: ["9002:9002"]
    environment:
      - ASR_MODEL=large-v3
      - OCR_LANGS=en
      - VLM_MODEL_ID=microsoft/Florence-2-large
    volumes:
      - hf-cache:/root/.cache/huggingface
      - paddle-cache:/root/.paddleocr
    restart: unless-stopped

volumes:
  hf-cache:
  paddle-cache:
```

### Compose (GPU)

```yaml
services:
  asr_ocr:
    image: ghcr.io/knowusuboaky/asr_ocr:cuda
    ports: ["9002:9002"]
    environment:
      - ASR_MODEL=large-v3
      - OCR_LANGS=en
      - VLM_MODEL_ID=microsoft/Florence-2-large
    gpus: all
    volumes:
      - hf-cache:/root/.cache/huggingface
      - paddle-cache:/root/.paddleocr
    restart: unless-stopped

volumes:
  hf-cache:
  paddle-cache:
```

> Health: `http://localhost:9002/health` • Docs: `http://localhost:9002/__docs__`

---

## Notes & tips

* **CPU vs GPU:** Device auto-selected; GPU strongly recommended for long media and Florence-2.
* **Documents:** Output Markdown includes front-matter (checksum, meta, pipeline, timestamps) and grouped sections (pages/slides/sheets). Images (incl. tiny icons) are OCR’d and captioned; CSV/XLSX can follow image links and caption them (`fetch_http=true`).
* **Legacy `.doc`:** If LibreOffice is available, the server converts `.doc → .docx/.pdf` to recover text + images; otherwise it falls back to CLI text extractors.
* **XLSX charts/canvas:** Charts are vector; to caption them, enable export via LibreOffice (`caption_xlsx_charts=true`) so pages can be VLM-captioned as images.
* **Friendly URL errors:** DNS/connect timeouts return a clear 502 message.
* **Docs UI:** FastAPI docs exposed at `/__docs__`.

---

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install fastapi "uvicorn[standard]" httpx faster-whisper imageio-ffmpeg \
            pillow transformers accelerate safetensors \
            paddleocr paddlepaddle \
            beautifulsoup4 lxml pandas openpyxl python-docx python-pptx pymupdf \
            tabulate chardet yt-dlp
uvicorn transcribe_server:app --host 0.0.0.0 --port 9002
```

