# source: https://docs.litellm.ai/docs/mcp_rest_api "3. Call a tool" (unprefixed name + server name)
curl -s -X POST http://localhost:4000/mcp-rest/tools/call \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "server_id": "places_api",
    "name": "getPlaces",
    "arguments": { "query": "coffee" }
  }' | jq .
