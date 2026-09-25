#!/usr/bin/env bash
# QA: code interpreter sandbox stickiness via metadata.session_id
#   bash qa_sticky_session.sh
#   LITELLM_BASE_URL=http://localhost:4000 LITELLM_KEY=sk-1234 bash qa_sticky_session.sh

set -euo pipefail

BASE="${LITELLM_BASE_URL:-http://localhost:4000}"
KEY="${LITELLM_KEY:-sk-1234}"
MODEL="${LITELLM_MODEL:-gpt-4o-mini}"
# proxy running at http://localhost:4000 (master key: sk-1234)
SESSION_A="qa-session-$(date +%s)-A"
SESSION_B="qa-session-$(date +%s)-B"

content() {
  echo "$1" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('message',{}).get('content','<error>'))"
}

call() {
  local session="${1:-}" code="$2" meta=""
  [[ -n "$session" ]] && meta=", \"metadata\": {\"session_id\": \"$session\"}"
  curl -s -X POST "$BASE/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $KEY" \
    -d "{\"model\":\"$MODEL\"$meta,\"tools\":[{\"type\":\"code_interpreter\"}],\"messages\":[{\"role\":\"user\",\"content\":\"Run this Python code and tell me the result: $code\"}]}"
}

assert_match() {
  local label="$1" body="$2" pattern="$3"
  if echo "$body" | grep -qiE "$pattern"; then
    echo "PASS  $label"
  else
    echo "FAIL  $label (expected /$pattern/)"
    echo "      $(content "$body")"
    exit 1
  fi
}

echo "=== Sticky Session Sandbox QA ==="
echo "base: $BASE  session A: $SESSION_A  session B: $SESSION_B"
echo

R=$(call "$SESSION_A" "x = 42; print(x)")
assert_match "same session_id reuses sandbox (set x=42)" "$R" "42"

R=$(call "$SESSION_A" "print(x)")
assert_match "same session_id keeps state (x still 42)" "$R" "42"

R=$(call "$SESSION_B" "print(x)")
assert_match "different session_id is isolated" "$R" "not defined|NameError|undefined|error"

R=$(call "" "y = 99; print(y)")
assert_match "no session_id runs code" "$R" "99"

R=$(call "" "print(y)")
assert_match "no session_id gets fresh sandbox each request" "$R" "not defined|NameError|undefined|error"

echo
echo "All checks passed."
