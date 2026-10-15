# rules

- Send the OpenAI Responses payload to `/openai/v1/responses` when the host passes `use_openai_responses_path` from the model cost map, else `/v1/responses`
- A standard `bedrock-mantle.<region>.api.aws` base is re-pinned to the resolved region, and any other base keeps its path
- Authenticate with a bearer from `api_key`, `BEDROCK_MANTLE_API_KEY` or `AWS_BEARER_TOKEN_BEDROCK`, otherwise sign with AWS SigV4 for service `bedrock`
- Keep only the tool types Mantle accepts, hoist `additional_tools` input items into `tools`, and rewrite Codex history items
- Models that serve only Chat Completions, and `service_tier` or `reasoning.summary` values Mantle rejects, stay on Python
- No native WebSocket mode

# references

- Python counterpart `litellm/llms/bedrock_mantle/responses/transformation.py`
