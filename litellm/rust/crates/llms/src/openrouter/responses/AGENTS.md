# rules

- Send the OpenAI Responses payload to `/api/v1/responses` and relay the OpenAI SSE events
- Authenticate with a bearer token from `OPENROUTER_API_KEY`, falling back to `OR_API_KEY`
- Only `OPENROUTER_API_BASE` overrides the default base, never the OpenAI base variables
- No native WebSocket mode

# references

- https://openrouter.ai/docs/api_reference/responses/overview
