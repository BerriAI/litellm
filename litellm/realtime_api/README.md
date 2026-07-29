Abstraction / Routing logic for OpenAI's `/v1/realtime` endpoints.

Supported endpoints:
- WebSocket: `/v1/realtime` (with `intent=transcription` for transcription-only sessions)
- HTTP: `/v1/realtime/client_secrets`, `/v1/realtime/transcription_sessions`

Supported providers: OpenAI, Azure OpenAI, Bedrock, Vertex AI, xAI.

For user-facing documentation and usage examples, see the litellm-docs repo.