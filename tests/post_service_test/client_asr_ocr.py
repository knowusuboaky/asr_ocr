# client_asr_ocr.py
# pip install requests

import os
from pathlib import Path
from typing import Optional
import requests

API_HOST = os.getenv("ASR_HOST", "localhost")
API_PORT = os.getenv("ASR_PORT", "9002")  # default 9002
API_BASE = f"http://{API_HOST}:{API_PORT}"

# -------------------- Friendly messages --------------------

FRIENDLY_BY_STATUS = {
    400: "That link didn’t work. Please check the URL and try again.",
    404: "We couldn’t find anything at that address.",
    415: "This URL’s content type isn’t supported. Try a direct audio/video/image link.",
    422: "The server couldn’t process that input. Make sure the URL/file is valid.",
    429: "Too many requests. Please slow down and try again.",
    502: "We couldn’t reach that URL. Please confirm it’s typed correctly, accessible from your network, and try again.",
    503: "Service temporarily unavailable. Please try again in a moment.",
    504: "The request took too long and timed out. Please try again.",
}

def _extract_server_detail(resp: Optional[requests.Response]) -> Optional[str]:
    if not resp:
        return None
    # Try JSON {"detail": "..."}
    try:
        data = resp.json()
        if isinstance(data, dict) and isinstance(data.get("detail"), str):
            return data["detail"]
    except Exception:
        pass
    # Fallback: plain text
    txt = (resp.text or "").strip()
    return txt or None

def _friendly_from_http_error(e: requests.HTTPError) -> str:
    resp = e.response
    status = resp.status_code if resp is not None else None
    server_detail = _extract_server_detail(resp)

    # Prefer server's specific detail when present (except generic 5xx)
    if server_detail and (status is None or status < 500 or status == 502):
        if "Network error while fetching URL" in server_detail:
            return FRIENDLY_BY_STATUS.get(502)  # DNS/connect style
        return server_detail

    return FRIENDLY_BY_STATUS.get(status, f"Request failed ({status}). Please try again.")

def _friendly_from_request_exc(e: Exception) -> str:
    if isinstance(e, requests.ConnectionError):
        return "Couldn’t connect to the server. Is it running and reachable?"
    if isinstance(e, requests.Timeout):
        return "The request timed out. Please try again."
    return "Something went wrong while contacting the server. Please try again."

# -------------------- Helpers --------------------

def _save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")
    print(f"✓ Wrote {path.resolve()}")

def _post_bytes(url: str, data: bytes, filename: str) -> Optional[requests.Response]:
    try:
        r = requests.post(
            url,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Filename": filename or "input.bin",
            },
            data=data,
            timeout=None,
        )
        r.raise_for_status()
        return r
    except requests.HTTPError as e:
        print(_friendly_from_http_error(e))
        return None
    except requests.RequestException as e:
        print(_friendly_from_request_exc(e))
        return None

def _post_json(url: str, payload: dict) -> Optional[requests.Response]:
    try:
        r = requests.post(url, json=payload, timeout=None)
        r.raise_for_status()
        return r
    except requests.HTTPError as e:
        print(_friendly_from_http_error(e))
        return None
    except requests.RequestException as e:
        print(_friendly_from_request_exc(e))
        return None

# -------------------- Transcribe (ASR) --------------------

def transcribe_file(media_path: Path, out_md: Path = Path("transcript.md")) -> bool:
    """
    Send local video/audio (raw bytes) to /transcribe_file → Markdown.
    Returns True on success (file written), False on failure.
    """
    data = media_path.read_bytes()
    r = _post_bytes(f"{API_BASE}/transcribe_file", data, media_path.name)
    if r is None:
        return False
    _save_text(out_md, r.text)
    return True

# -------------------- OCR (images) --------------------

def transcribe_image(img_path: Path, out_md: Path = Path("ocr.md")) -> bool:
    """
    Send local image (raw bytes) to /transcribe_image → Markdown.
    Returns True on success, False on failure.
    """
    data = img_path.read_bytes()
    r = _post_bytes(f"{API_BASE}/transcribe_image", data, img_path.name)
    if r is None:
        return False
    _save_text(out_md, r.text)
    return True

# -------------------- URL (media page, direct media, image, or single HTML page) --------------------

def transcribe_url(url: str, out_md: Path = Path("transcript_from_url.md"), *, user_agent: str | None = None) -> bool:
    """
    Send a URL to /transcribe_url → Markdown.

    Server logic:
      - If the page is a video page supported by yt-dlp (YouTube/Vimeo/TikTok/etc.):
          * downloads video(s), extracts audio, transcribes, deletes temps
      - If direct image: OCR
      - If direct audio/video/binary: ASR
      - If HTML page: single-page text extract (no crawling)

    Returns True on success (file written), False on failure.
    """
    payload: dict = {"url": url}
    if user_agent:
        payload["user_agent"] = user_agent

    r = _post_json(f"{API_BASE}/transcribe_url", payload)
    if r is None:
        return False
    _save_text(out_md, r.text)
    return True

# -------------------- Quick examples (comment out if importing) --------------------
# Ensure your server is running, e.g.:
# uvicorn transcribe_server:app --host 0.0.0.0 --port 9002
#
# -------------------- Quick examples (comment out if importing) --------------------
# Ensure your server is running, e.g.:
#  uvicorn transcribe_server:app --host 0.0.0.0 --port 9002

# Examples:
ok = transcribe_file(Path("video.mp4"), Path("video_transcript.md"))
print("file->md:", "OK" if ok else "FAILED")

ok = transcribe_image(Path("example.png"), Path("example_ocr.md"))
print("image->md:", "OK" if ok else "FAILED")

# Direct media URL (audio/video):
ok = transcribe_url("https://download.samplelib.com/mp4/sample-5s.mp4", Path("remote_media.md"))
print("url(media)->md:", "OK" if ok else "FAILED")

# Direct image URL:
ok = transcribe_url("https://upload.wikimedia.org/wikipedia/commons/8/81/Stop_sign.png",
                    Path("remote_image_ocr.md"))
print("url(image)->md:", "OK" if ok else "FAILED")

# Single HTML page (no crawl) — extracts page text into Markdown:
ok = transcribe_url("https://www.python.org", Path("python_org.md"))
print("url(html)->md:", "OK" if ok else "FAILED")

# Optional: custom user-agent if a site blocks default:
ok = transcribe_url("https://example.com/some-media", Path("custom_ua.md"))
print("url(with UA)->md:", "OK" if ok else "FAILED")
