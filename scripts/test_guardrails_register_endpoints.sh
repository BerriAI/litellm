#!/usr/bin/env bash
#
# Test guardrail register and submissions endpoints.
# Requires: proxy running with DB (migrations applied), valid admin API key.
#
# Usage:
#   export LITELLM_API_KEY="sk-..."   # required, use an admin key
#   ./scripts/test_guardrails_register_endpoints.sh
#   BASE_URL=http://localhost:4000 LITELLM_API_KEY="sk-..." ./scripts/test_guardrails_register_endpoints.sh
#
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:4000}"
API_KEY="${LITELLM_API_KEY:-}"

if ! command -v jq &>/dev/null; then
  echo "Error: jq is required. Install with: brew install jq (macOS) or apt-get install jq (Linux)"
  exit 1
fi

if [[ -z "$API_KEY" ]]; then
  echo "Error: LITELLM_API_KEY is not set. Use an admin key to test list/approve/reject."
  exit 1
fi

AUTH_HEADER="Authorization: Bearer $API_KEY"
TIMESTAMP=$(date +%s)
NAME_APPROVE="test-guardrail-approve-$TIMESTAMP"
NAME_REJECT="test-guardrail-reject-$TIMESTAMP"

echo "BASE_URL=$BASE_URL"
echo "Testing guardrail register and submissions endpoints..."
echo ""

# --- 1. Register a guardrail (will approve later) ---
echo "[1/6] POST /guardrails/register (guardrail: $NAME_APPROVE)"
REGISTER_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/guardrails/register" \
  -H "$AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d "{
    \"guardrail_name\": \"$NAME_APPROVE\",
    \"litellm_params\": {
      \"guardrail\": \"generic_guardrail_api\",
      \"mode\": \"pre_call\",
      \"api_base\": \"https://guardrails.example.com/validate\"
    },
    \"guardrail_info\": { \"description\": \"Test guardrail for approve flow\" }
  }")
REGISTER_HTTP=$(echo "$REGISTER_RESPONSE" | tail -n1)
REGISTER_BODY=$(echo "$REGISTER_RESPONSE" | sed '$d')
if [[ "$REGISTER_HTTP" -ne 200 ]]; then
  echo "  FAIL: expected 200, got $REGISTER_HTTP"
  echo "$REGISTER_BODY" | jq . 2>/dev/null || echo "$REGISTER_BODY"
  exit 1
fi
GUARDRAIL_ID_APPROVE=$(echo "$REGISTER_BODY" | jq -r '.guardrail_id')
echo "  OK (201/200) guardrail_id=$GUARDRAIL_ID_APPROVE"

# --- 2. Register a second guardrail (will reject later) ---
echo "[2/6] POST /guardrails/register (guardrail: $NAME_REJECT)"
REJECT_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/guardrails/register" \
  -H "$AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d "{
    \"guardrail_name\": \"$NAME_REJECT\",
    \"litellm_params\": {
      \"guardrail\": \"generic_guardrail_api\",
      \"mode\": \"post_call\",
      \"api_base\": \"https://guardrails.example.com/reject-test\"
    },
    \"guardrail_info\": { \"description\": \"Test guardrail for reject flow\" }
  }")
REJECT_HTTP=$(echo "$REJECT_RESPONSE" | tail -n1)
if [[ "$REJECT_HTTP" -ne 200 ]]; then
  echo "  FAIL: expected 200, got $REJECT_HTTP"
  echo "$REJECT_RESPONSE" | sed '$d' | jq . 2>/dev/null || echo "$REJECT_RESPONSE"
  exit 1
fi
GUARDRAIL_ID_REJECT=$(echo "$REJECT_RESPONSE" | sed '$d' | jq -r '.guardrail_id')
echo "  OK guardrail_id=$GUARDRAIL_ID_REJECT"

# --- 3. List submissions (admin) ---
echo "[3/6] GET /guardrails/submissions"
LIST_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "$BASE_URL/guardrails/submissions" -H "$AUTH_HEADER")
LIST_HTTP=$(echo "$LIST_RESPONSE" | tail -n1)
LIST_BODY=$(echo "$LIST_RESPONSE" | sed '$d')
if [[ "$LIST_HTTP" -ne 200 ]]; then
  echo "  FAIL: expected 200, got $LIST_HTTP"
  echo "$LIST_BODY" | jq . 2>/dev/null || echo "$LIST_BODY"
  exit 1
fi
echo "  OK summary: $(echo "$LIST_BODY" | jq -c '.summary' 2>/dev/null || echo "N/A")"

# --- 4. Get one submission by id ---
echo "[4/6] GET /guardrails/submissions/$GUARDRAIL_ID_APPROVE"
GET_RESPONSE=$(curl -s -w "\n%{http_code}" -X GET "$BASE_URL/guardrails/submissions/$GUARDRAIL_ID_APPROVE" -H "$AUTH_HEADER")
GET_HTTP=$(echo "$GET_RESPONSE" | tail -n1)
if [[ "$GET_HTTP" -ne 200 ]]; then
  echo "  FAIL: expected 200, got $GET_HTTP"
  exit 1
fi
echo "  OK status=$(echo "$GET_RESPONSE" | sed '$d' | jq -r '.status')"

# --- 5. Approve first submission ---
echo "[5/6] POST /guardrails/submissions/$GUARDRAIL_ID_APPROVE/approve"
APPROVE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/guardrails/submissions/$GUARDRAIL_ID_APPROVE/approve" -H "$AUTH_HEADER")
APPROVE_HTTP=$(echo "$APPROVE_RESPONSE" | tail -n1)
if [[ "$APPROVE_HTTP" -ne 200 ]]; then
  echo "  FAIL: expected 200, got $APPROVE_HTTP"
  echo "$APPROVE_RESPONSE" | sed '$d' | jq . 2>/dev/null || echo "$APPROVE_RESPONSE"
  exit 1
fi
echo "  OK $(echo "$APPROVE_RESPONSE" | sed '$d' | jq -c '.' 2>/dev/null)"

# --- 6. Reject second submission ---
echo "[6/6] POST /guardrails/submissions/$GUARDRAIL_ID_REJECT/reject"
REJECT_POST_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/guardrails/submissions/$GUARDRAIL_ID_REJECT/reject" -H "$AUTH_HEADER")
REJECT_POST_HTTP=$(echo "$REJECT_POST_RESPONSE" | tail -n1)
if [[ "$REJECT_POST_HTTP" -ne 200 ]]; then
  echo "  FAIL: expected 200, got $REJECT_POST_HTTP"
  exit 1
fi
echo "  OK $(echo "$REJECT_POST_RESPONSE" | sed '$d' | jq -c '.' 2>/dev/null)"

echo ""
echo "All 6 requests succeeded. Guardrail register and submissions endpoints are working."
