#!/usr/bin/env bash
#
# Creates a team, generates a team key, and submits a test guardrail with it.
# Requires: curl, jq
#
# Usage:
#   ADMIN_KEY=sk-your-admin-key ./scripts/create_team_key_and_submit_guardrail.sh
#   BASE_URL=http://localhost:4000 ADMIN_KEY=sk-your-admin-key ./scripts/create_team_key_and_submit_guardrail.sh

set -e

BASE_URL="${BASE_URL:-http://localhost:4000}"
BASE_URL="${BASE_URL%/}"

if [ -z "${ADMIN_KEY}" ]; then
  echo "Error: ADMIN_KEY is required (admin API key for the proxy)."
  echo "Usage: ADMIN_KEY=sk-your-admin-key $0"
  exit 1
fi

AUTH_HEADER="Authorization: Bearer ${ADMIN_KEY}"

echo "Using BASE_URL=${BASE_URL}"
echo "Creating team..."

TEAM_RESP=$(curl -s -X POST "${BASE_URL}/team/new" \
  -H "${AUTH_HEADER}" \
  -H "Content-Type: application/json" \
  -d '{
    "team_alias": "guardrail-test-team"
  }')

if ! echo "$TEAM_RESP" | jq -e .team_id >/dev/null 2>&1; then
  echo "Failed to create team. Response:"
  echo "$TEAM_RESP" | jq . 2>/dev/null || echo "$TEAM_RESP"
  exit 1
fi

TEAM_ID=$(echo "$TEAM_RESP" | jq -r .team_id)
echo "Created team_id: ${TEAM_ID}"

echo "Creating key for team..."

KEY_RESP=$(curl -s -X POST "${BASE_URL}/key/generate" \
  -H "${AUTH_HEADER}" \
  -H "Content-Type: application/json" \
  -d "{
    \"team_id\": \"${TEAM_ID}\"
  }")

if ! echo "$KEY_RESP" | jq -e .key >/dev/null 2>&1; then
  echo "Failed to create key. Response:"
  echo "$KEY_RESP" | jq . 2>/dev/null || echo "$KEY_RESP"
  exit 1
fi

TEAM_KEY=$(echo "$KEY_RESP" | jq -r .key)
echo "Created team key: ${TEAM_KEY}"

GUARDRAIL_NAME="test-guardrail-$(date +%s)"
echo "Submitting guardrail: ${GUARDRAIL_NAME}"

REGISTER_RESP=$(curl -s -X POST "${BASE_URL}/guardrails/register" \
  -H "Authorization: Bearer ${TEAM_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"guardrail_name\": \"${GUARDRAIL_NAME}\",
    \"litellm_params\": {
      \"guardrail\": \"generic_guardrail_api\",
      \"mode\": \"pre_call\",
      \"api_base\": \"https://example.com/guardrail\"
    },
    \"guardrail_info\": {
      \"description\": \"Test guardrail submitted via team key\"
    }
  }")

if ! echo "$REGISTER_RESP" | jq -e .guardrail_id >/dev/null 2>&1; then
  echo "Failed to register guardrail. Response:"
  echo "$REGISTER_RESP" | jq . 2>/dev/null || echo "$REGISTER_RESP"
  exit 1
fi

GUARDRAIL_ID=$(echo "$REGISTER_RESP" | jq -r .guardrail_id)
echo "Registered guardrail_id: ${GUARDRAIL_ID}"

echo ""
echo "Done."
echo "  team_id:      ${TEAM_ID}"
echo "  team_key:     ${TEAM_KEY}"
echo "  guardrail_id: ${GUARDRAIL_ID}"
echo "  guardrail_name: ${GUARDRAIL_NAME}"
