#!/usr/bin/env bash
# E2E test for OAuth2 client_credentials MCP flow
# Usage: bash tests/mcp_tests/test_oauth2_e2e.sh
set -euo pipefail

MOCK_PORT=8765
PROXY_PORT=4000
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/test_oauth2_mcp_config.yaml"
MOCK_SERVER="$SCRIPT_DIR/mock_oauth2_mcp_server.py"

cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    kill "$MOCK_PID" 2>/dev/null || true
    kill "$PROXY_PID" 2>/dev/null || true
    wait "$MOCK_PID" 2>/dev/null || true
    wait "$PROXY_PID" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT

# ── 1. Start mock OAuth2 MCP server ──────────────────────────────────────────
echo "=== Starting mock OAuth2 MCP server on :$MOCK_PORT ==="
python "$MOCK_SERVER" &
MOCK_PID=$!
sleep 2

# Quick smoke test on the token endpoint
TOKEN_RESP=$(curl -sf http://localhost:$MOCK_PORT/oauth/token \
  -d "grant_type=client_credentials&client_id=test-client&client_secret=test-secret")
echo "Token endpoint OK: $TOKEN_RESP"


# ── 3. List tools ────────────────────────────────────────────────────────────
echo ""
echo "=== Request 1: List MCP tools ==="
curl -s http://localhost:$PROXY_PORT/mcp-rest/tools/list \
  -H "Authorization: Bearer sk-1234" | python3 -m json.tool

# ── 4. Call the echo tool ────────────────────────────────────────────────────
echo ""
echo "=== Request 2: Call echo tool ==="
# Get the server_id from health endpoint
SERVER_ID=$(curl -s http://localhost:$PROXY_PORT/v1/mcp/server/health \
  -H "Authorization: Bearer sk-1234" | python3 -c "import json,sys; print(json.load(sys.stdin)[0]['server_id'])")

curl -s http://localhost:$PROXY_PORT/mcp-rest/tools/call \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d "{\"name\": \"echo\", \"arguments\": {\"message\": \"Hello from OAuth2 client_credentials\"}, \"server_id\": \"$SERVER_ID\"}" | python3 -m json.tool

# ── 5. Call again (uses cached token) ────────────────────────────────────────
echo ""
echo "=== Request 3: Call echo again (cached token) ==="
curl -s http://localhost:$PROXY_PORT/mcp-rest/tools/call \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d "{\"name\": \"echo\", \"arguments\": {\"message\": \"Second call - token should be cached\"}, \"server_id\": \"$SERVER_ID\"}" | python3 -m json.tool

# ── 6. Show OAuth2-specific proxy logs ───────────────────────────────────────
echo ""
echo "=== Proxy OAuth2 logs ==="
grep -E "(Fetching OAuth2|Fetched OAuth2)" /tmp/litellm_oauth2_test.log || echo "(no OAuth2 log lines found)"

echo ""
echo "=== All requests succeeded ==="
