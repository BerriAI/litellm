# rules

- Format spec lives in `llms-types/src/formats/messages/AGENTS.md`
- Payload shaping, metadata filtering, tool-ID rewriting, web-search replay, thinking translation and beta selection live here or in Anthropic-wide helpers
- Bedrock, Vertex, Azure and DeepSeek reuse these helpers for their Claude-compatible hosts. Keep the helpers free of those hosts' differences
