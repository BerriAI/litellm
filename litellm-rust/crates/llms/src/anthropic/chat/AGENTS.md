# rules

- Chat Completions translated onto Anthropic Messages. Both format specs live under `llms-types/src/formats/`
- `top_k` is not forwarded, because Python gates it per model inside `transform_request`
- `build_messages_request` serves every host of the translation; `AnthropicHost` picks the host's cost-map reads, `service_tier`/`speed` support and whether the Anthropic API betas are added (Vertex AI sets its own in `vertex_ai/vertex_ai_partner_models/anthropic`)
