# source: https://docs.litellm.ai/docs/mcp_rest_api "1. List MCP servers"
curl -s http://localhost:4000/v1/mcp/server \
  -H "Authorization: Bearer sk-1234" | jq .
