#!/usr/bin/env bash
#
# Test agent endpoint-level changes for MCP tool permissions (object_permission).
# Requires: proxy running, valid admin API key, curl, jq.
#
# Usage:
#   export LITELLM_PROXY_BASE_URL="http://localhost:4000"  # optional, default below
#   export LITELLM_API_KEY="sk-..."                        # required
#   ./scripts/test_agent_mcp_endpoints.sh
#
set -euo pipefail

BASE_URL="${LITELLM_PROXY_BASE_URL:-http://localhost:4000}"
API_KEY="${LITELLM_API_KEY:-}"

if ! command -v jq &>/dev/null; then
  echo "Error: jq is required. Install with: brew install jq (macOS) or apt install jq (Linux)"
  exit 1
fi
if [[ -z "$API_KEY" ]]; then
  echo "Error: LITELLM_API_KEY is not set. Export it or pass via env."
  exit 1
fi

AUTH_HEADER="Authorization: Bearer $API_KEY"
AGENT_NAME="test-agent-mcp-$(date +%s)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}PASS${NC}: $*"; }
fail() { echo -e "${RED}FAIL${NC}: $*"; exit 1; }
info() { echo -e "${YELLOW}INFO${NC}: $*"; }

# --- 1. Create agent with object_permission ---
info "Creating agent with object_permission (mcp_servers, mcp_tool_permissions)..."
CREATE_RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/agents" \
  -H "$AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "'"$AGENT_NAME"'",
    "agent_card_params": {
      "protocolVersion": "1.0",
      "name": "Test MCP Agent",
      "description": "Agent for endpoint tests",
      "url": "http://localhost:9999/",
      "version": "1.0.0",
      "defaultInputModes": ["text"],
      "defaultOutputModes": ["text"],
      "capabilities": {"streaming": true},
      "skills": []
    },
    "object_permission": {
      "mcp_servers": ["server_1", "server_2"],
      "mcp_access_groups": ["group_a"],
      "mcp_tool_permissions": {"server_1": ["tool_a", "tool_b"], "server_2": ["tool_c"]}
    }
  }')
HTTP_CODE=$(echo "$CREATE_RESP" | tail -n1)
BODY=$(echo "$CREATE_RESP" | sed '$d')
if [[ "$HTTP_CODE" != "200" ]]; then
  fail "POST /v1/agents returned $HTTP_CODE. Body: $BODY"
fi
AGENT_ID=$(echo "$BODY" | jq -r '.agent_id')
if [[ -z "$AGENT_ID" || "$AGENT_ID" == "null" ]]; then
  fail "POST /v1/agents did not return agent_id. Body: $BODY"
fi
pass "Created agent $AGENT_ID"

# Check create response includes object_permission
OP=$(echo "$BODY" | jq '.object_permission')
if [[ "$OP" == "null" || -z "$OP" ]]; then
  fail "POST /v1/agents response missing object_permission. Body: $BODY"
fi
SERVERS=$(echo "$OP" | jq -r '.mcp_servers | join(",")')
if [[ "$SERVERS" != "server_1,server_2" ]]; then
  fail "object_permission.mcp_servers unexpected: $SERVERS"
fi
pass "Create response includes object_permission with mcp_servers and mcp_tool_permissions"

# --- 2. GET /v1/agents (list) includes object_permission for our agent ---
info "GET /v1/agents and check one agent has object_permission..."
LIST_RESP=$(curl -s -w "\n%{http_code}" -X GET "$BASE_URL/v1/agents" -H "$AUTH_HEADER")
LIST_CODE=$(echo "$LIST_RESP" | tail -n1)
LIST_BODY=$(echo "$LIST_RESP" | sed '$d')
if [[ "$LIST_CODE" != "200" ]]; then
  fail "GET /v1/agents returned $LIST_CODE"
fi
AGENT_IN_LIST=$(echo "$LIST_BODY" | jq --arg id "$AGENT_ID" '.[] | select(.agent_id == $id)')
if [[ -z "$AGENT_IN_LIST" ]]; then
  fail "GET /v1/agents did not return agent $AGENT_ID (list might be key-scoped)"
fi
OP_LIST=$(echo "$AGENT_IN_LIST" | jq '.object_permission')
if [[ "$OP_LIST" == "null" || -z "$OP_LIST" ]]; then
  fail "GET /v1/agents list entry for agent missing object_permission"
fi
pass "GET /v1/agents list includes object_permission for agent"

# --- 3. GET /v1/agents/{agent_id} returns object_permission ---
info "GET /v1/agents/{agent_id}..."
GET_RESP=$(curl -s -w "\n%{http_code}" -X GET "$BASE_URL/v1/agents/$AGENT_ID" -H "$AUTH_HEADER")
GET_CODE=$(echo "$GET_RESP" | tail -n1)
GET_BODY=$(echo "$GET_RESP" | sed '$d')
if [[ "$GET_CODE" != "200" ]]; then
  fail "GET /v1/agents/$AGENT_ID returned $GET_CODE. Body: $GET_BODY"
fi
OP_GET=$(echo "$GET_BODY" | jq '.object_permission')
if [[ "$OP_GET" == "null" || -z "$OP_GET" ]]; then
  fail "GET /v1/agents/$AGENT_ID response missing object_permission"
fi
TOOL_PERMS=$(echo "$OP_GET" | jq -r '.mcp_tool_permissions.server_1 | join(",")')
if [[ "$TOOL_PERMS" != "tool_a,tool_b" ]]; then
  fail "object_permission.mcp_tool_permissions.server_1 unexpected: $TOOL_PERMS"
fi
pass "GET /v1/agents/{agent_id} returns object_permission with mcp_tool_permissions"

# --- 4. PATCH /v1/agents/{agent_id} with new object_permission ---
info "PATCH /v1/agents/{agent_id} with updated object_permission..."
PATCH_RESP=$(curl -s -w "\n%{http_code}" -X PATCH "$BASE_URL/v1/agents/$AGENT_ID" \
  -H "$AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{
    "object_permission": {
      "mcp_servers": ["server_3"],
      "mcp_tool_permissions": {"server_3": ["tool_x"]}
    }
  }')
PATCH_CODE=$(echo "$PATCH_RESP" | tail -n1)
PATCH_BODY=$(echo "$PATCH_RESP" | sed '$d')
if [[ "$PATCH_CODE" != "200" ]]; then
  fail "PATCH /v1/agents/$AGENT_ID returned $PATCH_CODE. Body: $PATCH_BODY"
fi
OP_PATCH=$(echo "$PATCH_BODY" | jq '.object_permission')
if [[ "$OP_PATCH" == "null" || -z "$OP_PATCH" ]]; then
  fail "PATCH response missing object_permission"
fi
PATCH_SERVERS=$(echo "$OP_PATCH" | jq -r '.mcp_servers | join(",")')
if [[ "$PATCH_SERVERS" != "server_3" ]]; then
  fail "PATCH object_permission.mcp_servers unexpected: $PATCH_SERVERS"
fi
pass "PATCH /v1/agents/{agent_id} updates and returns object_permission"

# --- 5. Create agent without object_permission; GET should still work ---
info "Creating agent without object_permission..."
AGENT_NAME_2="test-agent-no-mcp-$(date +%s)"
CREATE2_RESP=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/v1/agents" \
  -H "$AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "'"$AGENT_NAME_2"'",
    "agent_card_params": {
      "protocolVersion": "1.0",
      "name": "No MCP Agent",
      "description": "No object_permission",
      "url": "http://localhost:9999/",
      "version": "1.0.0",
      "defaultInputModes": ["text"],
      "defaultOutputModes": ["text"],
      "capabilities": {},
      "skills": []
    }
  }')
CODE2=$(echo "$CREATE2_RESP" | tail -n1)
BODY2=$(echo "$CREATE2_RESP" | sed '$d')
if [[ "$CODE2" != "200" ]]; then
  fail "POST /v1/agents (no object_permission) returned $CODE2. Body: $BODY2"
fi
AGENT_ID_2=$(echo "$BODY2" | jq -r '.agent_id')
# object_permission may be null or absent
pass "Created agent without object_permission: $AGENT_ID_2"

# --- 6. Cleanup: delete both agents ---
info "Deleting test agents..."
for AID in "$AGENT_ID" "$AGENT_ID_2"; do
  DEL_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "$BASE_URL/v1/agents/$AID" -H "$AUTH_HEADER")
  if [[ "$DEL_CODE" != "200" ]]; then
    info "DELETE /v1/agents/$AID returned $DEL_CODE (non-fatal)"
  fi
done
pass "Cleanup done"

echo ""
echo -e "${GREEN}All endpoint checks passed.${NC}"
