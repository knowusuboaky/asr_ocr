#!/usr/bin/env bash
# tests/cpu/service.sh
# Usage:
#   bash tests/cpu/service.sh up         # start & wait until healthy
#   bash tests/cpu/service.sh down       # stop & remove containers + volumes
#   bash tests/cpu/service.sh smoke      # start, run basic checks, then down (default)
#   bash tests/cpu/service.sh pull       # docker compose pull
#   bash tests/cpu/service.sh health     # poll /health until success (or timeout)
#   bash tests/cpu/service.sh logs       # tail service logs
#   bash tests/cpu/service.sh ps         # show compose ps
#   bash tests/cpu/service.sh restart    # up -d and re-check health
set -euo pipefail

# -------- Config (overridable via env) --------
ACTION="${1:-smoke}"
COMPOSE_FILE="${COMPOSE_FILE:-tests/cpu/docker-compose.yml}"
BASE_URL="${BASE_URL:-http://localhost:9002}"   # asr_ocr listens on 9002
SERVICE_NAME="${SERVICE_NAME:-asr_ocr}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
LOG_TAIL="${LOG_TAIL:-200}"

# Prefer Docker Compose v2 plugin; fallback to docker-compose if needed
if docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose -f ${COMPOSE_FILE}"
else
  COMPOSE="docker-compose -f ${COMPOSE_FILE}"
fi

# Tools
CURL="$(command -v curl || true)"
SED="$(command -v sed || true)"
JQ="$(command -v jq || true)"
BASE64_BIN="$(command -v base64 || true)"

require() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: required tool '$1' not found in PATH" >&2
    exit 127
  fi
}

require docker
require ${COMPOSE%% *}   # first word (docker or docker-compose)
require curl

compose() {
  echo ">> $COMPOSE $*" >&2
  # shellcheck disable=SC2086
  $COMPOSE "$@"
}

wait_healthy() {
  echo "Waiting for service health at ${BASE_URL}/health (timeout ${TIMEOUT_SEC}s)..."
  local start now elapsed attempt sleep_s
  start=$(date +%s)
  attempt=0
  while true; do
    if ${CURL} -fsS --max-time 5 "${BASE_URL}/health" | grep -qx 'ok'; then
      echo "Healthy!"
      return 0
    fi
    now=$(date +%s)
    elapsed=$(( now - start ))
    if [ "${elapsed}" -ge "${TIMEOUT_SEC}" ]; then
      echo "ERROR: Service did not become healthy within ${TIMEOUT_SEC}s." >&2
      compose logs --no-color --tail="${LOG_TAIL}" "${SERVICE_NAME}" || true
      return 1
    fi
    attempt=$((attempt + 1))
    sleep_s=$(( 1 + attempt / 2 ))
    [ "${sleep_s}" -gt 5 ] && sleep_s=5
    sleep "${sleep_s}"
  done
}

print_root_summary() {
  echo "Root summary (/):"
  if [ -n "$SED" ]; then
    ${CURL} -fsS "${BASE_URL}/" | ${SED} -e 's/^/  /'
  else
    ${CURL} -fsS "${BASE_URL}/"
  fi
}

test_transcribe_url() {
  echo "HTML extract test (/transcribe_url → example.com):"
  ${CURL} -fsS -X POST "${BASE_URL}/transcribe_url" \
    -H "Content-Type: application/json" \
    --data '{"url":"https://example.com"}' \
    | head -n 30
}

test_extract_txt() {
  echo "TXT ingest test (/extract_document with inline TXT):"
  printf '%s\n' 'Hello from service smoke test.' \
    | ${CURL} -fsS -X POST "${BASE_URL}/extract_document?ocr=none&vlm=none" \
        -H "X-Filename: note.txt" \
        --data-binary @- \
    | head -n 25
}

test_transcribe_image_minimal() {
  # Optional: 1x1 PNG -> OCR/Caption path. Skipped if base64 is unavailable.
  [ -z "$BASE64_BIN" ] && { echo "Skipping image test (base64 not found)."; return 0; }
  echo "Image OCR test (/transcribe_image with 1x1 PNG):"
  tmp_png="$(mktemp -t tinyXXXX.png)"
  # 1x1 transparent PNG (base64)
  png_b64='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAuMBg6i1r6kAAAAASUVORK5CYII='
  # macOS uses -D, GNU uses -d; detect:
  if echo -n "" | $BASE64_BIN -d >/dev/null 2>&1; then DEC='-d'; else DEC='-D'; fi
  echo -n "$png_b64" | $BASE64_BIN "$DEC" > "$tmp_png"
  ${CURL} -fsS -X POST "${BASE_URL}/transcribe_image" \
      -H "X-Filename: tiny.png" \
      --data-binary @"$tmp_png" \
    | head -n 25
  rm -f "$tmp_png"
}

case "${ACTION}" in
  up)
    compose up -d
    wait_healthy
    echo "Ready. API at ${BASE_URL}"
    ;;

  down)
    compose down -v
    echo "Service stopped and volumes removed."
    ;;

  pull)
    compose pull
    ;;

  health)
    wait_healthy
    ;;

  logs)
    compose logs -f --tail="${LOG_TAIL}" "${SERVICE_NAME}"
    ;;

  ps)
    compose ps
    ;;

  restart)
    compose up -d
    wait_healthy
    echo "Restarted and healthy. API at ${BASE_URL}"
    ;;

  smoke)
    # Ensure cleanup even on failure
    cleanup() {
      echo "Bringing stack down..."
      compose down -v || true
    }
    trap cleanup EXIT

    compose pull || true
    compose up -d
    wait_healthy

    print_root_summary
    test_transcribe_url
    test_extract_txt
    test_transcribe_image_minimal

    echo "Smoke test complete!"
    ;;

  *)
    echo "Unknown action: ${ACTION}"
    echo "Usage: bash tests/cpu/service.sh [up|down|smoke|pull|health|logs|ps|restart]"
    exit 2
    ;;
esac
