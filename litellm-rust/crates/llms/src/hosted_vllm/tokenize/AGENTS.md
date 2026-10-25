# rules

- Hosted vLLM tokenization uses `/tokenize` beside `/v1`, preserving any deployment path prefix
- Request token strings as well as IDs and preserve their separate normalized fields

# references

- https://docs.vllm.ai/en/v0.15.1/api/vllm/entrypoints/serve/tokenize/protocol/
- https://docs.vllm.ai/en/v0.12.0/api/vllm/entrypoints/openai/api_server/
