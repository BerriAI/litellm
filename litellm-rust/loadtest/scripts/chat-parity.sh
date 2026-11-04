#!/usr/bin/env bash
# Runs the chat completions parity check locally: writes one config pointed at chatparity's mock,
# starts the Python proxy and the Rust gateway on it, sends every case through both and writes
# out/report.md. Exits non-zero when there are unexplained differences.
#
#   scripts/chat-parity.sh                  # every case
#   FILTER=anthropic/ scripts/chat-parity.sh
#
# Needs: go, cargo, and a Python environment with the proxy and google-auth installed
# (`uv pip install -e '.[proxy]' google-auth` from the repo root). PYTHON picks the
# interpreter; GATEWAY skips the cargo build.
set -euo pipefail

here="$(cd "$(dirname "$0")/.." && pwd)"
repo="$(cd "$here/../.." && pwd)"
out="${OUT:-$here/out/chat-parity}"
mock_port="${MOCK_PORT:-9100}" python_port="${PYTHON_PORT:-4001}" rust_port="${RUST_PORT:-4002}"
python="${PYTHON:-python3}"
mkdir -p "$out"

(cd "$here" && go build -o "$out/chatparity" ./cmd/chatparity)
if [[ -z "${GATEWAY:-}" ]]; then
  (cd "$repo/litellm-rust" && cargo build -p litellm-gateway)
  GATEWAY="${CARGO_TARGET_DIR:-$repo/litellm-rust/target}/debug/litellm-gateway"
fi
"$out/chatparity" config -mock "http://127.0.0.1:$mock_port" -out "$out" >/dev/null

# Credentials for the bedrock-env deployment, which takes them from the environment. They only
# ever reach the mock.
export AWS_ACCESS_KEY_ID=AKIAPARITYPARITY AWS_SECRET_ACCESS_KEY=parity-secret AWS_REGION_NAME=us-east-1 AWS_REGION=us-east-1
unset AWS_SESSION_TOKEN AWS_PROFILE

pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true' EXIT
# Both price from the repo's cost map, so parameter support and defaults come from the same data.
LITELLM_LOCAL_MODEL_COST_MAP=True "$python" -m litellm.proxy.proxy_cli --config "$out/config.yaml" \
  --port "$python_port" >"$out/python.log" 2>&1 &
pids+=($!)
LITELLM_MODEL_COST_MAP_PATH="$repo/model_prices_and_context_window.json" LITELLM_CONFIG="$out/config.yaml" \
  HOST=127.0.0.1 PORT="$rust_port" "$GATEWAY" >"$out/rust.log" 2>&1 &
pids+=($!)

for url in "http://127.0.0.1:$python_port/health/liveliness" "http://127.0.0.1:$rust_port/health/liveliness"; do
  for _ in $(seq 120); do curl -sf "$url" >/dev/null && break; sleep 1; done
  curl -sf "$url" >/dev/null || { echo "not up: $url (see $out/*.log)" >&2; exit 1; }
done

"$out/chatparity" run -mock-addr "127.0.0.1:$mock_port" -python "http://127.0.0.1:$python_port" \
  -rust "http://127.0.0.1:$rust_port" -out "$out" -known "$here/cmd/chatparity/known.json" -filter "${FILTER:-}"
