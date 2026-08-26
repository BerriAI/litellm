# source: https://docs.litellm.ai/docs/mcp_rest_api "2. List tools" (One server)
curl -s "http://localhost:4000/mcp-rest/tools/list?server_id=places_api" \
  -H "Authorization: Bearer sk-1234" | jq .
