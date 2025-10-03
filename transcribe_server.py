# transcribe_server.py
#
# ASR + OCR deps
# pip install fastapi uvicorn[standard] httpx faster-whisper imageio-ffmpeg
# pip install easyocr pillow torch  # (GPU? install CUDA-enabled torch from pytorch.org)
#
# SINGLE-PAGE HTML extraction deps
# pip install beautifulsoup4 lxml
#
# Optional (recommended) for robust video URLs:
# pip install yt-dlp
#
# run (pick a non-800x port, e.g. 9001)
# uvicorn transcribe_server:app --host 127.0.0.1 --port 9001 --workers 1
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

import os, math, tempfile, subprocess, datetime, io, re, unicodedata, textwrap
from pathlib import Path
from typing import Optional, Tuple, List
import httpx
from httpx import ConnectError

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

# ---------- ASR: faster-whisper + ffmpeg ----------
from faster_whisper import WhisperModel
from imageio_ffmpeg import get_ffmpeg_exe

# ---------- OCR: EasyOCR + Pillow ----------
import torch
import numpy as np
from PIL import Image
import easyocr

# ---------- HTML parse ----------
from bs4 import BeautifulSoup

# ================== ASR CONFIG ====================
MODEL_SIZE = os.getenv("ASR_MODEL", "large-v3")
VAD_MIN_SIL_MS = int(os.getenv("VAD_MIN_SIL_MS", "500"))
KEEP_WAV = os.getenv("KEEP_WAV", "false").lower() == "true"
KEEP_WAV_ON_ERROR = os.getenv("KEEP_WAV_ON_ERROR", "true").lower() == "true"

def detect_device_and_precision():
    try:
        import ctranslate2
        has_cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        has_cuda = False
    if has_cuda:
        return "cuda", "float16"
    return "cpu", "int8_float32"

DEVICE, COMPUTE_TYPE = detect_device_and_precision()
ASR_MODEL = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

def fmt_ts(sec: float) -> str:
    if sec is None or sec < 0 or math.isinf(sec) or math.isnan(sec):
        sec = 0.0
    ms = int(round((sec - int(sec)) * 1000))
    s  = int(sec) % 60
    m  = (int(sec) // 60) % 60
    h  = int(sec) // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def ffmpeg_convert_to_wav(input_media: Path, output_wav: Path):
    ffmpeg = get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-i", str(input_media), "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(output_wav)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "ignore"))

def asr_transcribe_wav(wav_path: Path):
    segments, info = ASR_MODEL.transcribe(
        str(wav_path),
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=VAD_MIN_SIL_MS),
        beam_size=5,
        condition_on_previous_text=True,
    )
    return list(segments), info

def segments_to_markdown(segments, src_name: str, language: str) -> str:
    header = (
        f"---\n"
        f"title: Transcript of {src_name}\n"
        f"generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        f"model: faster-whisper/{MODEL_SIZE} ({DEVICE},{COMPUTE_TYPE})\n"
        f"language: {language}\n"
        f"---\n\n# Audio Transcript\n\n"
    )
    body = []
    for seg in segments:
        body.append(f"{fmt_ts(seg.start)} --> {fmt_ts(seg.end)}\n{(seg.text or '').strip()}\n")
    return header + "\n".join(body)

def run_asr_pipeline_to_markdown(data: bytes, original_name: str) -> str:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        media_path = td / (original_name or "input.bin")
        wav_path   = td / "audio.wav"
        media_path.write_bytes(data)

        wav_created = False
        info = None
        try:
            ffmpeg_convert_to_wav(media_path, wav_path)
            wav_created = wav_path.exists()

            # Pass 1
            segments, info = asr_transcribe_wav(wav_path)
            segs = list(segments)

            # Pass 2 (softer) if needed
            if not segs:
                segments2, info2 = ASR_MODEL.transcribe(
                    str(wav_path),
                    vad_filter=False,
                    beam_size=1,
                    temperature=0.0,
                    no_speech_threshold=0.2,
                    condition_on_previous_text=False,
                )
                segs = list(segments2)
                info = info2 or info

        finally:
            if wav_created and not KEEP_WAV:
                try: wav_path.unlink(missing_ok=True)
                except Exception: pass

    if not segs:
        language = getattr(info, "language", "auto") if info else "auto"
        return (
            f"---\n"
            f"title: Transcript of {original_name}\n"
            f"generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"model: faster-whisper/{MODEL_SIZE} ({DEVICE},{COMPUTE_TYPE})\n"
            f"language: {language}\n"
            f"---\n\n# Audio Transcript\n\n"
            f"(No speech detected, or audio was too short/quiet.)\n"
        )

    language = getattr(info, "language", "auto")
    return segments_to_markdown(segs, original_name, language)

# ================== OCR CONFIG ====================
def _parse_csv_env(name: str, default: str):
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]

OCR_LANGS = _parse_csv_env("OCR_LANGS", "en")
OCR_UPSCALE = float(os.getenv("OCR_UPSCALE", "1.5"))
OCR_RECOG_HEAD = os.getenv("OCR_RECOG_HEAD", "english_g2")
DEFAULT_ALLOWLIST = (
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    " .,;:!?@#%&()[]{}+-/*=_'\"\\|<>~$^`"
)
OCR_ALLOWLIST = os.getenv("OCR_ALLOWLIST", DEFAULT_ALLOWLIST)

USE_GPU = torch.cuda.is_available()
OCR_READER = easyocr.Reader(
    OCR_LANGS,
    gpu=USE_GPU,
    recog_network=OCR_RECOG_HEAD,
    quantize=False
)

def load_image_from_bytes(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if OCR_UPSCALE != 1.0:
        w, h = img.size
        img = img.resize((int(w * OCR_UPSCALE), int(h * OCR_UPSCALE)), Image.LANCZOS)
    return img

def ocr_image_bytes_to_markdown(data: bytes, src_name: str) -> str:
    img = load_image_from_bytes(data)
    img_np = np.array(img)
    blocks = OCR_READER.readtext(
        img_np,
        detail=0,
        paragraph=True,
        allowlist=OCR_ALLOWLIST,
        text_threshold=0.7,
        low_text=0.3,
        link_threshold=0.5
    )
    header = (
        f"---\n"
        f"title: OCR of {src_name}\n"
        f"generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        f"model: easyocr/{OCR_RECOG_HEAD} ({'cuda' if USE_GPU else 'cpu'},{','.join(OCR_LANGS)})\n"
        f"language: {','.join(OCR_LANGS)}\n"
        f"---\n\n# OCR Text\n\n"
    )
    body = "\n\n".join(b.strip() for b in blocks if b.strip())
    return header + body + ("\n" if body else "")

# ================== SINGLE-PAGE HTML HELPERS ====================
def _meta(soup: BeautifulSoup, key: str) -> Optional[str]:
    tag = (
        soup.find("meta", attrs={"name": key})
        or soup.find("meta", attrs={"property": f"og:{key}"})
        or soup.find("meta", attrs={"property": f"twitter:{key}"})
    )
    return tag.get("content", "").strip() if tag and tag.has_attr("content") else None

def _clean(
    text: str,
    *,
    normalize_unicode: bool = True,
    collapse_whitespace: bool = True,
    strip_line_ends: bool = True,
    dedent: bool = True,
) -> str:
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u00A0", " ")
        text = re.sub(r"[\u200B-\u200D\u2060\uFEFF]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if strip_line_ends:
        text = "\n".join(line.rstrip() for line in text.splitlines())
    if dedent:
        text = textwrap.dedent(text)
    if collapse_whitespace:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def single_page_markdown(url: str, html_bytes: bytes) -> str:
    soup = BeautifulSoup(html_bytes, "lxml")
    title = (soup.title.string.strip() if soup.title else url)
    description = _meta(soup, "description")
    author = _meta(soup, "author")
    content = _clean(soup.get_text(" ", strip=True))

    header = (
        f"---\n"
        f"title: Page extract of {title}\n"
        f"generated: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
        f"url: {url}\n"
        f"pages: 1\n"
        f"---\n\n# Page Content\n\n"
    )
    meta_block = []
    if author: meta_block.append(f"**Author:** {author}")
    if description: meta_block.append(f"**Description:** {description}")
    meta_str = ("\n\n" + "\n\n".join(meta_block) + "\n\n") if meta_block else "\n"
    return header + meta_str + content + "\n"

# ================== URL FETCH HELPERS (with friendly 502) ====================
async def fetch_head_or_get(url: str, user_agent: Optional[str] = None) -> Tuple[bytes, str]:
    """
    Try HEAD to read content-type; if not allowed, GET the content.
    Returns (content_bytes, lowercase_content_type).
    On DNS/connect errors, returns HTTP 502 with a clear message.
    """
    headers = {"User-Agent": user_agent} if user_agent else {}
    async with httpx.AsyncClient(follow_redirects=True, timeout=None, headers=headers) as client:
        if "://" not in url:
            url = "https://" + url
        try:
            # Try HEAD first
            hr = await client.head(url)
            ctype = (hr.headers.get("Content-Type") or "").lower()
            if hr.status_code == 200 and ctype and not ctype.startswith("text/html"):
                gr = await client.get(url)
                return gr.content, (gr.headers.get("Content-Type") or ctype).lower()

            # Fallback: GET
            gr = await client.get(url)
            if gr.status_code != 200:
                raise HTTPException(400, f"Failed to fetch URL (status {gr.status_code})")
            return gr.content, (gr.headers.get("Content-Type") or "").lower()

        except ConnectError as e:
            # DNS / connect errors → clearer message (handled by the app's exception handler)
            raise HTTPException(502, f"Network error while fetching URL (DNS/connect): {e}") from e
        except Exception as e:
            raise HTTPException(502, f"Network error while fetching URL: {e}") from e

def is_image_content_type(ctype: str) -> bool:
    return ctype.startswith("image/")

def is_audio_or_video_content_type(ctype: str) -> bool:
    return ctype.startswith("audio/") or ctype.startswith("video/") or ctype in (
        "application/octet-stream",
        "application/x-mpegurl",
        "application/vnd.apple.mpegurl",
    )

def is_probably_html(ctype: str, data: bytes) -> bool:
    if "text/html" in ctype:
        return True
    head = data[:256].lower()
    return head.startswith(b"<!doctype html") or b"<html" in head

# ================== yt-dlp DOWNLOAD + TRANSCRIBE ====================
def try_ytdlp_download_and_transcribe(url: str) -> Optional[str]:
    """
    Try to download one or more videos with yt-dlp, extract audio, and transcribe.
    Returns Markdown on success; None if yt-dlp isn't available or download fails.
    All temp files are deleted automatically.
    """
    try:
        import yt_dlp
    except Exception:
        return None  # yt-dlp not installed; let caller fallback

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        ydl_opts = {
            "outtmpl": str(tdir / "%(title)s.%(ext)s"),
            "format": "best[is_dash=false][ext=mp4]/best[is_dash=false]/best",
            "noprogress": True,
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception:
            return None

        video_exts = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
        videos = [p for p in tdir.iterdir() if p.is_file() and p.suffix.lower() in video_exts]
        if not videos:
            return None

        md_sections: List[str] = []
        for vid in videos:
            wav_path = tdir / (vid.stem + ".wav")
            try:
                ffmpeg_convert_to_wav(vid, wav_path)
                segments, info = asr_transcribe_wav(wav_path)
                if not segments:
                    continue
                language = getattr(info, "language", "auto")
                md = segments_to_markdown(segments, vid.name, language)
                md_sections.append(md)
            finally:
                try:
                    if wav_path.exists(): wav_path.unlink()
                except Exception:
                    pass
                try:
                    if vid.exists(): vid.unlink()
                except Exception:
                    pass

        return "\n\n".join(md_sections) if md_sections else None

# ================== FastAPI APP ===================
app = FastAPI(
    title="Media Transcription & OCR Server",
    version="0.1",
    docs_url="/__docs__"
)

# ---------- Friendly error handler ----------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = str(exc.detail or "")
    # Friendly message for DNS/connect 502s
    if exc.status_code == 502 and "Network error while fetching URL" in detail:
        return PlainTextResponse(
            "We couldn’t reach that URL. Please confirm it’s typed correctly, accessible from your network, and try again.",
            status_code=502,
            media_type="text/plain",
        )
    # Default JSON for other HTTPExceptions
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# ---------- Models ----------
class UrlBody(BaseModel):
    url: str
    user_agent: Optional[str] = None  # optional UA for the single GET

# ================== Routes ===================
@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

@app.get("/", response_class=PlainTextResponse)
def root():
    return (
        f"ASR: faster-whisper/{MODEL_SIZE} ({DEVICE},{COMPUTE_TYPE}) | "
        f"OCR: easyocr/{OCR_RECOG_HEAD} ({'cuda' if USE_GPU else 'cpu'},{','.join(OCR_LANGS)}) | "
        f"URL: single-page extract + yt-dlp media"
    )

@app.post("/transcribe_file")
async def transcribe_file(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(400, "Empty request body")
    name = request.headers.get("X-Filename", "input.bin")
    md = run_asr_pipeline_to_markdown(data, name)
    return PlainTextResponse(md, media_type="text/markdown")

@app.post("/transcribe_image")
async def transcribe_image(request: Request):
    data = await request.body()
    if not data:
        raise HTTPException(400, "Empty request body")
    name = request.headers.get("X-Filename", "image.bin")
    md = ocr_image_bytes_to_markdown(data, name)
    return PlainTextResponse(md, media_type="text/markdown")

@app.post("/transcribe_url")
async def transcribe_url(body: UrlBody):
    url = body.url.strip()
    if not url:
        raise HTTPException(400, "Missing 'url'")

    # Try yt-dlp first (video pages)
    md_from_ytdlp = try_ytdlp_download_and_transcribe(url)
    if md_from_ytdlp:
        return PlainTextResponse(md_from_ytdlp, media_type="text/markdown")

    # Else: single fetch with clear 502 on DNS/connect errors
    data, ctype = await fetch_head_or_get(url, body.user_agent)

    if is_probably_html(ctype, data):
        md = single_page_markdown(url if "://" in url else "https://" + url, data)
        return PlainTextResponse(md, media_type="text/markdown")

    if is_image_content_type(ctype):
        try:
            Image.open(io.BytesIO(data)).verify()
            name = Path(httpx.URL(url if "://" in url else "https://" + url).path).name or "remote_image"
            md = ocr_image_bytes_to_markdown(data, name)
            return PlainTextResponse(md, media_type="text/markdown")
        except Exception:
            pass

    if is_audio_or_video_content_type(ctype) or data:
        name = Path(httpx.URL(url if "://" in url else "https://" + url).path).name or "remote_media"
        md = run_asr_pipeline_to_markdown(data, name)
        return PlainTextResponse(md, media_type="text/markdown")

    raise HTTPException(415, f"Unsupported Content-Type for URL: {ctype or 'unknown'}")
