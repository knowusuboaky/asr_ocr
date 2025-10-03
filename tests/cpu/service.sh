#!/usr/bin/env bash
# tests/cpu/service.sh
# Usage:
#   bash tests/cpu/service.sh up       # start & wait until healthy
#   bash tests/cpu/service.sh down     # stop & remove containers + volumes
#   bash tests/cpu/service.sh smoke    # start, wait, run basic checks, then down (default)
set -euo pipefail

ACTION="${1:-smoke}"
COMPOSE="docker compose -f tests/cpu/docker-compose.yml"
BASE_URL="http://localhost:9005"
JQ="$(command -v jq || echo cat)"

wait_healthy() {
  echo "Waiting for service to become healthy (timeout 300s)..."
  SECS=0
  until curl -fsS "$BASE_URL/healthz" >/dev/null 2>&1; do
    sleep 3
    SECS=$((SECS+3))
    if [ "$SECS" -ge 300 ]; then
      echo "ERROR: Service did not become healthy in time."
      $COMPOSE logs --no-color embeddings || true
      exit 1
    fi
  done
  echo "Healthy!"
}

case "$ACTION" in
  up)
    $COMPOSE up -d
    wait_healthy
    echo "Ready. API at $BASE_URL"
    ;;

  down)
    $COMPOSE down -v
    echo "Service stopped and volumes removed."
    ;;

  smoke)
    $COMPOSE up -d
    wait_healthy

    echo "Models:"
    curl -fsS "$BASE_URL/models" | $JQ .

    echo "Embedding test:"
    curl -fsS -X POST "$BASE_URL/embed?model=e5-large-v2" \
      -H "Content-Type: application/json" \
      -d '{"texts":["hello world"],"mode":"auto"}' | $JQ .

    # bring everything down after the test
    $COMPOSE down -v
    echo "Test complete!"
    ;;

  *)
    echo "Unknown action: $ACTION"
    echo "Usage: bash tests/cpu/service.sh [up|down|smoke]"
    exit 2
    ;;
esac
