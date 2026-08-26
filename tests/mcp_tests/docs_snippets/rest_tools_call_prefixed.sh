# source: https://docs.litellm.ai/docs/mcp_rest_api "3. Call a tool" (prefixed name, x-litellm-api-key header)
curl -s -X POST http://localhost:4000/mcp-rest/tools/call \
  -H "x-litellm-api-key: sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "server_id": "places_api",
    "name": "places_api-getPlaces",
    "arguments": { "query": "coffee" }
  }' | jq .
