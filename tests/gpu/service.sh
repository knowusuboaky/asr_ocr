#!/usr/bin/env bash
# tests/gpu/service.sh
# Usage:
#   bash tests/gpu/service.sh up        # start (checks NVIDIA) & wait until healthy
#   bash tests/gpu/service.sh down      # stop & remove containers + volumes
#   bash tests/gpu/service.sh smoke     # start, wait, run basic checks, then down (default)
#   bash tests/gpu/service.sh pull      # docker compose pull
#   bash tests/gpu/service.sh ps        # show compose ps
#   bash tests/gpu/service.sh logs      # tail service logs
#   bash tests/gpu/service.sh restart   # up -d and re-check health
#   bash tests/gpu/service.sh health    # poll /health until success (or timeout)
#
# Notes:
#   - Requires NVIDIA drivers + nvidia-container-toolkit on the host,
#     unless you set SKIP_NVIDIA_CHECK=1 to bypass the check.
#   - API lives at http://localhost:9002

set -euo pipefail

ACTION="${1:-smoke}"

# ---------- Config (overridable via env) ----------
COMPOSE_FILE="${COMPOSE_FILE:-tests/gpu/docker-compose.yml}"
SERVICE_NAME="${SERVICE_NAME:-asr_ocr}"
BASE_URL="${BASE_URL:-http://localhost:9002}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
LOG_TAIL="${LOG_TAIL:-200}"

# Prefer Docker Compose v2; fallback to docker-compose
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose -f ${COMPOSE_FILE}"
else
  COMPOSE="docker-compose -f ${COMPOSE_FILE}"
fi

JQ="${JQ:-$(command -v jq || echo cat)}"

# ---------- Helpers ----------
check_nvidia() {
  if [[ "${SKIP_NVIDIA_CHECK:-0}" == "1" ]]; then
    echo "SKIP_NVIDIA_CHECK=1 → skipping NVIDIA runtime check."
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    # GPU present; still verify docker sees runtime
    :
  fi

  if ! docker info 2>/dev/null | grep -qiE 'Runtimes:.*nvidia|Default Runtime:\s*nvidia|NVIDIA'; then
    echo "ERROR: NVIDIA runtime not detected. Install nvidia-container-toolkit and ensure 'docker info' shows a 'nvidia' runtime."
    exit 1
  fi
}

wait_healthy() {
  echo "Waiting for service to become healthy (timeout ${TIMEOUT_SEC}s)…"
  local start end
  start=$(date +%s)
  while true; do
    if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
      echo "Healthy!"
      return 0
    fi
    sleep 3
    end=$(date +%s)
    if (( end - start >= TIMEOUT_SEC )); then
      echo "ERROR: Service did not become healthy in time."
      $COMPOSE logs --no-color --tail "${LOG_TAIL}" "${SERVICE_NAME}" || true
      exit 1
    fi
  done
}

print_root_summary() {
  echo "Root summary (/):"
  curl -fsS "${BASE_URL}/" | sed -e 's/^/  /' || echo "  (failed to fetch /)"
}

assert_cuda_hint_in_root() {
  echo "Checking root banner for CUDA hint…"
  if curl -fsS "${BASE_URL}/" | grep -qi 'cuda'; then
    echo "CUDA hint detected."
  else
    echo "WARN: No explicit CUDA hint in banner. Service may be running on CPU (informational)."
  fi
}

test_transcribe_url() {
  echo "HTML extract test (example.com):"
  curl -fsS -X POST "${BASE_URL}/transcribe_url" \
    -H "Content-Type: application/json" \
    -d '{"url":"https://example.com"}' | head -n 30 || echo "(request failed)"
}

test_extract_txt() {
  echo "TXT ingest test (/extract_document with inline TXT):"
  local tmp
  tmp="$(mktemp)"
  printf "Hello from GPU smoke test." > "$tmp"
  curl -fsS -X POST "${BASE_URL}/extract_document?ocr=none&vlm=none" \
    -H "X-Filename: note.txt" \
    --data-binary @"$tmp" | head -n 25 || echo "(request failed)"
  rm -f "$tmp"
}

new_tiny_png() {
  # 1x1 PNG
  local out b64
  out="$(mktemp --suffix=.png)"
  b64='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAuMBg6i1r6kAAAAASUVORK5CYII='
  printf "%s" "$b64" | base64 -d > "$out"
  echo "$out"
}

test_transcribe_image() {
  echo "Image OCR test (/transcribe_image with tiny PNG):"
  local png
  png="$(new_tiny_png)"
  curl -fsS -X POST "${BASE_URL}/transcribe_image" \
    -H "X-Filename: tiny.png" \
    --data-binary @"$png" | head -n 25 || echo "(request failed)"
  rm -f "$png"
}

# ---------- Actions ----------
case "$ACTION" in
  up)
    check_nvidia
    $COMPOSE up -d
    wait_healthy
    echo "Ready. API at ${BASE_URL}"
    ;;

  down)
    $COMPOSE down -v
    echo "Service stopped and volumes removed."
    ;;

  pull)
    $COMPOSE pull
    ;;

  ps)
    $COMPOSE ps
    ;;

  logs)
    $COMPOSE logs -f --tail "${LOG_TAIL}" "${SERVICE_NAME}"
    ;;

  restart)
    check_nvidia
    $COMPOSE up -d
    wait_healthy
    echo "Restarted and healthy. API at ${BASE_URL}"
    ;;

  health)
    wait_healthy
    ;;

  smoke)
    check_nvidia
    # ensure cleanup even on failure
    trap '$COMPOSE down -v || true' EXIT
    $COMPOSE pull
    $COMPOSE up -d
    wait_healthy

    print_root_summary
    assert_cuda_hint_in_root
    test_transcribe_url
    test_extract_txt
    test_transcribe_image

    echo "GPU smoke test complete!"
    ;;

  *)
    echo "Unknown action: $ACTION"
    echo "Usage: bash tests/gpu/service.sh [up|down|smoke|pull|ps|logs|restart|health]"
    exit 2
    ;;
esac
