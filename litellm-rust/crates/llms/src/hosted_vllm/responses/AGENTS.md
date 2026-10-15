# rules

- Send the OpenAI Responses payload to `{api_base}/v1/responses`, or `{api_base}/responses` when the base already ends with `/v1`. The base comes from `api_base` or `HOSTED_VLLM_API_BASE` and is required
- Authenticate with a bearer token from `api_key` or `HOSTED_VLLM_API_KEY`, falling back to `fake-api-key` since vLLM does not require one
- No native WebSocket mode

# references

- https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
