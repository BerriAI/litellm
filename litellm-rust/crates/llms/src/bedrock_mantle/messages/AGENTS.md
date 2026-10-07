# structure

- The `bedrock_mantle` provider's Messages config: a thin layer that supplies this provider's settings and auth to the shared adapter in `bedrock/messages/mantle`
- Only model names containing `claude` select it. The provider's other models serve OpenAI-compatible APIs and fall back to the chat-completion adapter

# boundaries

- Provider-wide mantle helpers (auth, region resolution) belong in `bedrock_mantle/` so the future chat and responses ports share them, not in this folder

# invariants

- Wire format, beta filtering and stream decoding come from `bedrock/messages/mantle`. Do not fork or override them here
- A bearer token wins when one resolves: the caller's `api_key`, then `BEDROCK_MANTLE_API_KEY`, then `AWS_BEARER_TOKEN_BEDROCK`. Otherwise sign with SigV4. The shared adapter removes any caller copy of a header the signer computes first
- The region comes from a public mantle host in `BEDROCK_MANTLE_API_BASE`, then `BEDROCK_MANTLE_REGION`, `AWS_REGION_NAME`, `AWS_REGION`, and defaults to `us-east-1`, unlike the `bedrock` spelling

# gotchas

- `bedrock_mantle` stays a separate LiteLLM provider because pricing, provider resolution and env var names key on the `bedrock_mantle/` prefix. It still gets no wire adapter of its own: the endpoint is the same one the `bedrock/mantle/` route calls
- Composition, not inheritance: this config is the `bedrock` adapter built with this provider's `MantleSettings`, and `bedrock` never depends on `bedrock_mantle`

# known gaps

- The `<region>/` model prefix Python reads in `split_mantle_region_prefix` does not pick the region yet. The core wiring has to strip it from the model and pass the region on
- `aws_region_name` and the per-call `aws_*` credentials from litellm params are not read, for the same reason as in `bedrock/messages/mantle`
- Python rewrites any `api_base` on a public mantle host to the bare `https://bedrock-mantle.<region>.api.aws` host, so an extra path or `http` scheme is dropped. The shared adapter keeps them

# references

## python

- `litellm/llms/bedrock_mantle/messages/transformation.py`
- `litellm/llms/bedrock_mantle/common_utils.py`

## docs

- https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.md
