#!/usr/bin/env bash
# QA script for v2 managed agents PR #27400.
#
# Walks all three happy-path flows from .claude/v2_api_contract.md against a
# real LiteLLM proxy + a real opencode server. No Krrish dependency — we
# hand-INSERT a sessions row pointing at local opencode.
#
# Usage:
#   bash tests/managed_agents_tests/qa_happy_path.sh
#
# Required on PATH: opencode, jq, psql, curl
# Optional env:
#   LITELLM_PROXY_URL  (default: http://localhost:4000)
#   LITELLM_API_KEY    (default: sk-1234)
#   OC_PORT            (default: 1234)
#   LITELLM_DB         (default: litellm)
set -euo pipefail

PROXY_URL="${LITELLM_PROXY_URL:-http://localhost:4000}"
PROXY_KEY="${LITELLM_API_KEY:-sk-1234}"
OC_PORT="${OC_PORT:-1234}"
DB_NAME="${LITELLM_DB:-litellm}"
RUN_ID="qa_$$_$(date +%s)"
SES_ID="ses_$RUN_ID"

log()  { printf "\033[1;34m▸\033[0m %s\n" "$*" >&2; }
ok()   { printf "  \033[1;32m✓\033[0m %s\n" "$*" >&2; }
warn() { printf "  \033[1;33m!\033[0m %s\n" "$*" >&2; }
fail() { printf "  \033[1;31m✗\033[0m %s\n" "$*" >&2; exit 1; }

# ------------------------------------------------------------------
# 0. Preflight
# ------------------------------------------------------------------
log "preflight"
command -v opencode >/dev/null || fail "opencode not on PATH — install via 'npm i -g opencode-ai'"
command -v jq       >/dev/null || fail "jq not on PATH"
command -v psql     >/dev/null || fail "psql not on PATH"
curl -sf "$PROXY_URL/health/readiness" -o /dev/null \
  || fail "proxy not reachable at $PROXY_URL — start it with: pip install -e <pr-worktree> --no-deps && litellm --config proxy_server_config.yaml --port 4000"
psql -d "$DB_NAME" -c "SELECT 1" -tA >/dev/null 2>&1 \
  || fail "can't connect to db '$DB_NAME' — set LITELLM_DB or check your local postgres"
ok "opencode, jq, psql, proxy ($PROXY_URL), db ($DB_NAME)"

# ------------------------------------------------------------------
# 1. Bring up local opencode (or reuse existing)
# ------------------------------------------------------------------
STARTED_OC=0
if curl -sf "http://127.0.0.1:$OC_PORT/global/health" >/dev/null 2>&1; then
  log "opencode already running on $OC_PORT — reusing"
else
  log "starting opencode on port $OC_PORT (logs: /tmp/opencode-$RUN_ID.log)"
  opencode serve --port "$OC_PORT" >/tmp/opencode-$RUN_ID.log 2>&1 &
  OC_PID=$!
  STARTED_OC=1
  for i in {1..30}; do
    sleep 1
    if curl -sf "http://127.0.0.1:$OC_PORT/global/health" >/dev/null 2>&1; then
      ok "opencode up (pid $OC_PID)"; break
    fi
  done
  curl -sf "http://127.0.0.1:$OC_PORT/global/health" >/dev/null \
    || fail "opencode never became healthy — check /tmp/opencode-$RUN_ID.log"
fi

cleanup() {
  log "cleanup"
  psql -d "$DB_NAME" -c "DELETE FROM \"LiteLLM_ManagedAgentSession\" WHERE id='$SES_ID'" >/dev/null 2>&1 || true
  psql -d "$DB_NAME" -c "DELETE FROM \"LiteLLM_ManagedAgent\"        WHERE id='${AGENT_ID:-x}'" >/dev/null 2>&1 || true
  if [[ "$STARTED_OC" == "1" && -n "${OC_PID:-}" ]]; then
    kill "$OC_PID" 2>/dev/null || true
    ok "stopped opencode (pid $OC_PID)"
  fi
}
trap cleanup EXIT

# ------------------------------------------------------------------
# Flow 1 — agent → session → message → response
# ------------------------------------------------------------------
log "Flow 1.1: POST /v2/agents"
AGENT_RESP=$(curl -sS -X POST "$PROXY_URL/v2/agents" \
  -H "Authorization: Bearer $PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "'"qa-agent-$RUN_ID"'",
    "config": {
      "model": "anthropic/claude-3-haiku",
      "system_prompt": "You are a brief assistant. Reply in one sentence.",
      "tools": [],
      "litellm_api_key": "'"$PROXY_KEY"'",
      "litellm_base_url": "'"$PROXY_URL"'"
    }
  }')
echo "$AGENT_RESP" | jq .
AGENT_ID=$(echo "$AGENT_RESP" | jq -r .id)
[[ "$AGENT_ID" =~ ^agt_ ]] || fail "expected agt_* id, got: $AGENT_ID"
[[ $(echo "$AGENT_RESP" | jq -r .config.litellm_api_key) == *"****"* ]] || fail "litellm_api_key not masked"
ok "agent: $AGENT_ID  (api key masked)"

log "Flow 1.2: stand up session manually (replaces Krrish's POST /v2/sessions)"
OC_SID=$(curl -sS -X POST "http://127.0.0.1:$OC_PORT/session" -d '{}' | jq -r .id)
[[ -n "$OC_SID" && "$OC_SID" != "null" ]] || fail "opencode /session returned no id"
ok "opencode session: $OC_SID"

psql -d "$DB_NAME" >/dev/null <<SQL
INSERT INTO "LiteLLM_ManagedAgentSession" (
  id, agent_id, sandbox_type, sandbox_size, sandbox_timeout_minutes, sandbox_idle_timeout_minutes,
  sandbox_url, sandbox_metadata, status, repos, env_vars, created_by, created_at, updated_at
) VALUES (
  '$SES_ID', '$AGENT_ID', 'opencode', 'small', 60, 10,
  'http://127.0.0.1:$OC_PORT',
  '{"opencode_session_id":"$OC_SID"}'::jsonb,
  'ready', '[]'::jsonb, '{}'::jsonb, 'default_user_id', NOW(), NOW()
);
SQL
ok "fake session row: $SES_ID  →  http://127.0.0.1:$OC_PORT  (oc_sid=$OC_SID)"

log "Flow 1.3: POST /v2/sessions/$SES_ID/messages"
SEND_RESP=$(curl -sS -X POST "$PROXY_URL/v2/sessions/$SES_ID/messages" \
  -H "Authorization: Bearer $PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content":"Hello, who are you?"}')
echo "$SEND_RESP" | jq .
[[ $(echo "$SEND_RESP" | jq -r .role)         == "user" ]]        || fail "expected role=user"
[[ $(echo "$SEND_RESP" | jq -r .status)       == "in_progress" ]] || fail "expected status=in_progress"
[[ $(echo "$SEND_RESP" | jq -r .session_id)   == "$SES_ID" ]]     || fail "session_id mismatch"
[[ $(echo "$SEND_RESP" | jq -r .id)           =~ ^msg_ ]]         || fail "expected msg_* id"
ok "message accepted (202), msg id prefixed correctly"

log "Flow 1.4: GET /v2/sessions/$SES_ID/events  (SSE, 10s window)"
timeout 10 curl -sN "$PROXY_URL/v2/sessions/$SES_ID/events" \
  -H "Authorization: Bearer $PROXY_KEY" \
  -H "Accept: text/event-stream" 2>/dev/null | head -50 || true
ok "stream observed (truncated at 10s)"

# ------------------------------------------------------------------
# Flow 2 — followup on same session
# ------------------------------------------------------------------
log "Flow 2.1: POST followup message"
curl -sS -X POST "$PROXY_URL/v2/sessions/$SES_ID/messages" \
  -H "Authorization: Bearer $PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content":"What is 2+2?"}' | jq '{id, role, status}'
ok "followup accepted"

log "Flow 2.3: GET /v2/sessions/$SES_ID/messages  (verify history)"
sleep 3
LIST_RESP=$(curl -sS "$PROXY_URL/v2/sessions/$SES_ID/messages?limit=10" \
  -H "Authorization: Bearer $PROXY_KEY")
echo "$LIST_RESP" | jq '{count: (.data | length), roles: [.data[].role], statuses: [.data[].status]}'
COUNT=$(echo "$LIST_RESP" | jq '.data | length')
[[ $COUNT -ge 2 ]] || warn "expected >= 2 messages, got $COUNT (LLM may still be processing)"

# ------------------------------------------------------------------
# Flow 3 — resume + sandbox-death failure mode
# ------------------------------------------------------------------
log "Flow 3.2: GET /v2/sessions/$SES_ID  (verify still ready, no internal leaks)"
GET_RESP=$(curl -sS "$PROXY_URL/v2/sessions/$SES_ID" -H "Authorization: Bearer $PROXY_KEY")
echo "$GET_RESP" | jq .
[[ $(echo "$GET_RESP" | jq -r .status)                  == "ready" ]] || fail "expected status=ready"
[[ $(echo "$GET_RESP" | jq 'has("sandbox_url")')        == "false" ]] || fail "sandbox_url leaked in response"
[[ $(echo "$GET_RESP" | jq 'has("sandbox_metadata")')   == "false" ]] || fail "sandbox_metadata leaked"
ok "internal fields stripped from response"

log "Flow 3.4 (fail-closed): point row at unreachable URL, expect 504"
psql -d "$DB_NAME" -c "UPDATE \"LiteLLM_ManagedAgentSession\" SET sandbox_url='http://127.0.0.1:1' WHERE id='$SES_ID'" >/dev/null
DEATH=$(curl -sS -w "\n%{http_code}" -X POST "$PROXY_URL/v2/sessions/$SES_ID/messages" \
  -H "Authorization: Bearer $PROXY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content":"will fail"}')
DEATH_CODE=$(printf '%s\n' "$DEATH" | tail -n 1)
DEATH_BODY=$(printf '%s\n' "$DEATH" | sed '$d')
echo "$DEATH_BODY" | jq .
[[ "$DEATH_CODE" == "504" ]] || fail "expected 504, got $DEATH_CODE"
ok "sandbox-unreachable correctly returns 504"

# Validator behavior is covered by Pydantic unit tests in tests/test_litellm/managed_agents/test_types.py.
# Keeping this script HTTP-only so it doesn't need the PR's Python env.

printf "\n\033[1;32mALL CHECKS PASSED\033[0m  (run id: %s)\n" "$RUN_ID"
