# source: https://docs.litellm.ai/docs/mcp_rest_api "2. List tools" (All servers)
curl -s http://localhost:4000/mcp-rest/tools/list \
  -H "Authorization: Bearer sk-1234" | jq .
