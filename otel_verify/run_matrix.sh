#!/usr/bin/env bash
# OTEL fidelity verification matrix. Each case uses a deterministic traceparent
# so its spans can be looked up by trace_id. Prints "LABEL -> HTTP <code>".
set -u
BASE=http://localhost:4000
MASTER="sk-1234"
TEAM_KEY="$(cat "$(dirname "$0")/.team_key")"

tp() { echo "00-$1-0000000000000001-01"; }   # build a traceparent from a 32-hex trace id

run() {  # label  traceid  curl-args...
  local label="$1" tid="$2"; shift 2
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" -H "traceparent: $(tp "$tid")" "$@")
  printf "%-28s trace=%s -> HTTP %s\n" "$label" "$tid" "$code"
}

J="Content-Type: application/json"

# ---- #28405 http.response.status_code + #28273 team attrs (success/failure) ----
run "T01_chat_200_team"       00000000000000000000000000000001 -X POST $BASE/v1/chat/completions \
    -H "Authorization: Bearer $TEAM_KEY" -H "$J" \
    -d '{"model":"gpt-mock","messages":[{"role":"user","content":"hello"}],"mock_response":"hi there"}'

run "T02_chat_400_badjson"    00000000000000000000000000000002 -X POST $BASE/v1/chat/completions \
    -H "Authorization: Bearer $TEAM_KEY" -H "$J" -d '{ this is not valid json'

run "T03_chat_401_badkey"     00000000000000000000000000000003 -X POST $BASE/v1/chat/completions \
    -H "Authorization: Bearer sk-does-not-exist" -H "$J" \
    -d '{"model":"gpt-mock","messages":[{"role":"user","content":"hi"}],"mock_response":"x"}'

run "T04_chat_400_unknownmodel" 00000000000000000000000000000004 -X POST $BASE/v1/chat/completions \
    -H "Authorization: Bearer $MASTER" -H "$J" \
    -d '{"model":"definitely-not-a-real-model","messages":[{"role":"user","content":"hi"}]}'

run "T05_chat_429_team"       00000000000000000000000000000005 -X POST $BASE/v1/chat/completions \
    -H "Authorization: Bearer $TEAM_KEY" -H "$J" \
    -d '{"model":"gpt-mock","messages":[{"role":"user","content":"hi"}],"mock_response":"litellm.RateLimitError"}'

run "T06_chat_500_team"       00000000000000000000000000000006 -X POST $BASE/v1/chat/completions \
    -H "Authorization: Bearer $TEAM_KEY" -H "$J" \
    -d '{"model":"gpt-mock","messages":[{"role":"user","content":"hi"}],"mock_response":"litellm.InternalServerError"}'

run "T07_messages_200_team"   00000000000000000000000000000007 -X POST $BASE/v1/messages \
    -H "Authorization: Bearer $TEAM_KEY" -H "$J" \
    -d '{"model":"gpt-mock","max_tokens":50,"messages":[{"role":"user","content":"hi"}],"mock_response":"hi there"}'

# ---- #28405 admin-endpoint paths ----
run "T08_admin_200_keygen"    00000000000000000000000000000008 -X POST $BASE/key/generate \
    -H "Authorization: Bearer $MASTER" -H "$J" -d '{"models":["gpt-mock"]}'

run "T09_admin_500_keygen"    00000000000000000000000000000009 -X POST $BASE/key/generate \
    -H "Authorization: Bearer $MASTER" -H "$J" -d '{"duration":"not-a-valid-duration"}'

run "T10_admin_422_keygen"    00000000000000000000000000000010 -X POST $BASE/key/generate \
    -H "Authorization: Bearer $MASTER" -H "$J" -d '{"models":"should-be-a-list"}'

# ---- #28362 serialize guardrail_response + #28364 guardrail span on failure ----
run "T11_guardrail_block_400" 00000000000000000000000000000011 -X POST $BASE/v1/chat/completions \
    -H "Authorization: Bearer $TEAM_KEY" -H "$J" \
    -d '{"model":"gpt-mock","messages":[{"role":"user","content":"please blockme now"}]}'

run "T12_guardrail_allow_200" 00000000000000000000000000000012 -X POST $BASE/v1/chat/completions \
    -H "Authorization: Bearer $TEAM_KEY" -H "$J" \
    -d '{"model":"gpt-mock","messages":[{"role":"user","content":"please scanme now"}],"mock_response":"clean"}'
