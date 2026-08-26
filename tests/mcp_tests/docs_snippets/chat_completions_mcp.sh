# source: https://docs.litellm.ai/docs/mcp "Use MCP tools with /chat/completions"
curl --location '<your-litellm-proxy-base-url>/v1/chat/completions' \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $LITELLM_API_KEY" \
--data '{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "user", "content": "Find a well rated coffee place."}
  ],
  "tools": [
    {
      "type": "mcp",
      "server_url": "litellm_proxy/places_api/mcp",
      "server_label": "places_api",
      "require_approval": "never"
    }
  ]
}'
