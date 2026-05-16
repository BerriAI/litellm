#!/usr/bin/env bash
#
# End-to-end test for Tencent Hunyuan image generation via LiteLLM proxy.
# Starts the proxy, calls /v1/images/generations, downloads the returned URL,
# verifies the image is a valid PNG, then cleans up.
#
# Requirements:
#   - uv (Python package manager) available in PATH
#   - jq installed (brew install jq / apt-get install jq)
#   - curl available
#   - HUNYUAN_API_KEY and HUNYUAN_API_URL set (or sourced from .env)
#
# Usage:
#   # Run with credentials from .env (default):
#   ./scripts/test_hunyuan_image_generation.sh
#
#   # Run against an already-running proxy:
#   BASE_URL=http://localhost:4000 LITELLM_API_KEY=sk-1234 \
#       ./scripts/test_hunyuan_image_generation.sh --no-start-server
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

# ── Options ────────────────────────────────────────────────────────────────────
START_SERVER=true
for arg in "$@"; do
  case "$arg" in
    --no-start-server) START_SERVER=false ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# ── Load .env if present ───────────────────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip blanks and comments
    [[ -z "$line" || "$line" == \#* ]] && continue
    # Only process KEY=VALUE lines
    if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
      export "$line" 2>/dev/null || true
    fi
  done < "$ENV_FILE"
fi

# ── Validate prerequisites ─────────────────────────────────────────────────────
for cmd in curl jq; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "Error: '$cmd' is required but not found in PATH."
    exit 1
  fi
done

if [[ -z "${HUNYUAN_API_KEY:-}" ]]; then
  echo "Error: HUNYUAN_API_KEY is not set. Add it to .env or export it before running."
  exit 1
fi

# ── Proxy settings ─────────────────────────────────────────────────────────────
PROXY_PID=""
PROXY_LOG=""

if [[ "$START_SERVER" == "true" ]]; then
  PROXY_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
  BASE_URL="http://localhost:$PROXY_PORT"
  # Use the master key defined in proxy_config.yaml (sk-1234); clear the env-var
  # override so it doesn't shadow the config file value.
  LITELLM_API_KEY="sk-1234"
else
  BASE_URL="${BASE_URL:-http://localhost:4000}"
  LITELLM_API_KEY="${LITELLM_API_KEY:-sk-1234}"
fi

AUTH_HEADER="Authorization: Bearer $LITELLM_API_KEY"

# ── Cleanup on exit ────────────────────────────────────────────────────────────
cleanup() {
  if [[ -n "$PROXY_PID" ]] && kill -0 "$PROXY_PID" 2>/dev/null; then
    echo ""
    echo "Stopping proxy (PID $PROXY_PID)..."
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
  fi
  if [[ -n "$PROXY_LOG" && -f "$PROXY_LOG" ]]; then
    rm -f "$PROXY_LOG"
  fi
}
trap cleanup EXIT

# ── Start proxy ────────────────────────────────────────────────────────────────
if [[ "$START_SERVER" == "true" ]]; then
  PROXY_LOG=$(mktemp /tmp/litellm-proxy-XXXXXX.log)
  CONFIG_FILE="$REPO_ROOT/proxy_config.yaml"

  echo "Starting LiteLLM proxy on port $PROXY_PORT..."
  cd "$REPO_ROOT"
  # Start without database and without LITELLM_MASTER_KEY env override
  # so the master key in proxy_config.yaml (sk-1234) takes effect.
  env -u DATABASE_URL -u STORE_MODEL_IN_DB -u LITELLM_MASTER_KEY \
    uv run litellm --port "$PROXY_PORT" --config "$CONFIG_FILE" \
    >"$PROXY_LOG" 2>&1 &
  PROXY_PID=$!

  # Wait up to 60 s for the proxy to become healthy
  echo -n "Waiting for proxy to be ready"
  for i in $(seq 1 60); do
    if curl -sf "$BASE_URL/health/liveliness" -o /dev/null 2>/dev/null; then
      echo " ready (${i}s)"
      break
    fi
    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
      echo ""
      echo "Error: proxy process exited unexpectedly. Last log lines:"
      tail -20 "$PROXY_LOG"
      exit 1
    fi
    echo -n "."
    sleep 1
    if [[ "$i" -eq 60 ]]; then
      echo ""
      echo "Error: proxy did not start within 60 seconds. Last log lines:"
      tail -20 "$PROXY_LOG"
      exit 1
    fi
  done
fi

echo ""
echo "BASE_URL : $BASE_URL"
echo "Model    : hunyuan/gpt-image-2"
echo ""

# ── Step 1: Generate image ─────────────────────────────────────────────────────
echo "[1/3] POST /v1/images/generations"
GENERATE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/images/generations" \
  -H "$AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hunyuan/gpt-image-2",
    "prompt": "一只跳舞的小狗",
    "size":   "1024x1024"
  }')
GENERATE_HTTP=$(echo "$GENERATE_RESPONSE" | tail -n1)
GENERATE_BODY=$(echo "$GENERATE_RESPONSE" | sed '$d')

if [[ "$GENERATE_HTTP" -ne 200 ]]; then
  echo "  FAIL: expected HTTP 200, got $GENERATE_HTTP"
  echo "$GENERATE_BODY" | jq . 2>/dev/null || echo "$GENERATE_BODY"
  exit 1
fi
echo "  OK (HTTP $GENERATE_HTTP)"

IMAGE_URL=$(echo "$GENERATE_BODY" | jq -r '.data[0].url // empty')
if [[ -z "$IMAGE_URL" ]]; then
  echo "  FAIL: response did not contain .data[0].url"
  echo "$GENERATE_BODY" | jq . 2>/dev/null || echo "$GENERATE_BODY"
  exit 1
fi
echo "  Image URL: $IMAGE_URL"

# ── Step 2: Download the image ─────────────────────────────────────────────────
echo ""
echo "[2/3] Downloading generated image..."
TMP_IMAGE=$(mktemp /tmp/hunyuan-image-XXXXXX.png)
trap 'cleanup; rm -f "$TMP_IMAGE"' EXIT

HTTP_STATUS=$(curl -s -o "$TMP_IMAGE" -w "%{http_code}" "$IMAGE_URL")
if [[ "$HTTP_STATUS" -ne 200 ]]; then
  echo "  FAIL: image download returned HTTP $HTTP_STATUS"
  exit 1
fi
FILE_SIZE=$(wc -c <"$TMP_IMAGE")
echo "  OK (HTTP $HTTP_STATUS, ${FILE_SIZE} bytes)"

# ── Step 3: Verify PNG magic bytes ─────────────────────────────────────────────
echo ""
echo "[3/3] Verifying PNG signature..."
# PNG magic: 89 50 4e 47 (first 4 bytes)
MAGIC=$(python3 -c "
import sys
with open('$TMP_IMAGE','rb') as f:
    b = f.read(4)
print(b.hex())
")
if [[ "$MAGIC" == "89504e47" ]]; then
  echo "  OK — valid PNG"
else
  echo "  FAIL: unexpected file magic bytes: $MAGIC (expected 89504e47 for PNG)"
  exit 1
fi

# Cleanup temp image
rm -f "$TMP_IMAGE"
trap cleanup EXIT

echo ""
echo "All tests passed."
