#!/usr/bin/env bash
# One command to bring up the thin core and the Mistral OCR plugin.
#
# Usage:
#   MISTRAL_API_KEY=sk-... ./scripts/run.sh
#
# Tail both logs in foreground; Ctrl-C stops everything.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

CORE_PORT="${CORE_PORT:-8080}"
PLUGIN_PORT="${PLUGIN_PORT:-8081}"

if [[ -z "${MISTRAL_API_KEY:-}" ]]; then
    echo "MISTRAL_API_KEY is not set; the plugin will refuse to start" >&2
    exit 2
fi

cd "$ROOT/core"
echo "[build] cargo build --release"
cargo build --release

CORE_BIN="$ROOT/core/target/release/core"

cleanup() {
    if [[ -n "${PLUGIN_PID:-}" ]] && kill -0 "$PLUGIN_PID" 2>/dev/null; then
        kill "$PLUGIN_PID" || true
    fi
    if [[ -n "${CORE_PID:-}" ]] && kill -0 "$CORE_PID" 2>/dev/null; then
        kill "$CORE_PID" || true
    fi
}
trap cleanup EXIT INT TERM

cd "$ROOT"
echo "[plugin] starting Mistral OCR plugin on :$PLUGIN_PORT"
python3 -m plugins.mistral_ocr.server --host 127.0.0.1 --port "$PLUGIN_PORT" &
PLUGIN_PID=$!

echo "[core] starting on :$CORE_PORT"
"$CORE_BIN" --config "$ROOT/routes.toml" --bind "127.0.0.1:$CORE_PORT" &
CORE_PID=$!

wait -n "$CORE_PID" "$PLUGIN_PID"
