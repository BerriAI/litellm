#!/bin/bash
set -eu
proof_dir=$(cd "$(dirname "$0")" && pwd)
label=$1
base_url=$2
mkdir -p "$proof_dir/$label"
set -a
source "$proof_dir/local.env"
set +a
cat > "$proof_dir/register.json" <<'JSON'
{"client_name":"LiteLLM local verification","redirect_uris":["http://localhost:9999/callback"],"grant_types":["authorization_code"],"response_types":["code"],"token_endpoint_auth_method":"none"}
JSON
for mode in true_passthrough oauth_delegate; do
  for bridge in false true; do
    curl --silent --show-error --max-time 15 --output "$proof_dir/$label/register-$mode-$bridge.json" --write-out '%{http_code}\n' "$base_url/figma_${mode}_${bridge}/register" -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' --data-binary "@$proof_dir/register.json" > "$proof_dir/$label/register-$mode-$bridge.status"
  done
done
curl --silent --show-error --max-time 15 --output "$proof_dir/$label/ui-register.json" --write-out '%{http_code}\n' "$base_url/v1/mcp/server/oauth/figma/register" -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' --data-binary "@$proof_dir/register.json" > "$proof_dir/$label/ui-register.status"
curl --silent --show-error --max-time 15 --output "$proof_dir/$label/ui-authorize.json" --write-out '%{http_code}\n' "$base_url/v1/mcp/server/oauth/figma/authorize?redirect_uri=http%3A%2F%2Flocalhost%3A9999%2Fcallback&code_challenge=abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG&code_challenge_method=S256&response_type=code" -H "Authorization: Bearer $LITELLM_MASTER_KEY" > "$proof_dir/$label/ui-authorize.status"
curl --silent --show-error --max-time 30 --output "$proof_dir/$label/tools-list.json" --write-out '%{http_code}\n' "$base_url/mcp-rest/tools/list?server_id=deepwiki" -H "Authorization: Bearer $LITELLM_MASTER_KEY" > "$proof_dir/$label/tools-list.status"
curl --silent --show-error --max-time 30 --output "$proof_dir/$label/tools-call.json" --write-out '%{http_code}\n' "$base_url/mcp-rest/tools/call" -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' --data '{"server_id":"deepwiki","name":"read_wiki_structure","arguments":{"repoName":"BerriAI/litellm"}}' > "$proof_dir/$label/tools-call.status"
curl --silent --show-error --max-time 15 --output "$proof_dir/$label/session.json" --write-out '%{http_code}\n' "$base_url/v1/mcp/server/oauth/session" -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' --data '{"server_name":"figma_discovery_test","url":"https://mcp.figma.com/mcp","transport":"http","auth_type":"true_passthrough"}' > "$proof_dir/$label/session.status"
server_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["server_id"])' "$proof_dir/$label/session.json")
curl --silent --show-error --max-time 15 --output "$proof_dir/$label/discovered-register.json" --write-out '%{http_code}\n' "$base_url/v1/mcp/server/oauth/$server_id/register" -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H 'Content-Type: application/json' --data-binary "@$proof_dir/register.json" > "$proof_dir/$label/discovered-register.status" 2> "$proof_dir/$label/discovered-register.stderr" || test "$label" = before
