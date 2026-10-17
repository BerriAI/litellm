# rules

- Claude on Vertex AI for Chat Completions: the `anthropic/chat` translation built with `AnthropicHost::VertexAi`, addressed to the publisher `rawPredict` URL and authenticated with a Google access token
- The body drops `model`, carries `anthropic_version: vertex-2023-10-16`, sanitizes `output_config` and names its betas in `anthropic_beta`, filtered to the ones Vertex accepts
- Capability reads resolve the `vertex_ai/` cost map entry first; fast mode and `service_tier` are the Anthropic API's own

# references

- https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude/use-claude
- litellm/llms/vertex_ai/vertex_ai_partner_models/anthropic/transformation.py
- litellm/llms/vertex_ai/vertex_ai_partner_models/main.py
- litellm/llms/anthropic/chat/transformation.py
