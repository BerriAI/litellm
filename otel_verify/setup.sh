#!/usr/bin/env bash
# Create a team + a virtual key (with allow_client_mock_response so the matrix
# can drive deterministic mock successes/errors). Writes .team_id / .team_key
# which run_matrix.sh and verify_spans.py read. Re-run after a fresh proxy/db.
set -euo pipefail
BASE="${BASE:-http://localhost:4000}"
MASTER="${MASTER_KEY:-sk-1234}"
HERE="$(dirname "$0")"

team=$(curl -s -X POST "$BASE/team/new" -H "Authorization: Bearer $MASTER" \
  -H "Content-Type: application/json" -d '{"team_alias":"otel-verify-team"}')
team_id=$(python3 -c "import json,sys; print(json.load(sys.stdin)['team_id'])" <<<"$team")

key=$(curl -s -X POST "$BASE/key/generate" -H "Authorization: Bearer $MASTER" \
  -H "Content-Type: application/json" \
  -d "{\"team_id\":\"$team_id\",\"models\":[\"gpt-mock\"],\"metadata\":{\"allow_client_mock_response\":true}}")
team_key=$(python3 -c "import json,sys; print(json.load(sys.stdin)['key'])" <<<"$key")

printf '%s' "$team_id"  > "$HERE/.team_id"
printf '%s' "$team_key" > "$HERE/.team_key"
echo "team_id=$team_id"
echo "team_key=$team_key (saved to .team_key)"
