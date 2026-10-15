# rules

- Send the OpenAI Responses payload to `{api_base}/responses`, defaulting to `https://api.x.ai/v1`, with a bearer token from `api_key` or `XAI_API_KEY`. OAuth stays on Python
- Drop `metadata`, strip `container` from `code_interpreter`, reduce `web_search` and `x_search` to the fields xAI documents, and keep `max_output_tokens` as sent
- Restate `usage.cost` from `usage.cost_in_usd_ticks`
- No native WebSocket mode

# references

- https://docs.x.ai/docs/api-reference#create-new-response
- https://docs.x.ai/developers/cost-tracking
