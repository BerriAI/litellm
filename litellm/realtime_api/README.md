Abstraction / Routing logic for OpenAI's `/v1/realtime` endpoints.

Supported endpoints:
- WebSocket: `/v1/realtime` (with `intent=transcription` for transcription-only sessions)
- HTTP: `/v1/realtime/client_secrets`, `/v1/realtime/transcription_sessions`

Supported providers: OpenAI, Azure OpenAI, Bedrock, Vertex AI, xAI.

Billing visibility:
- WebSocket sessions pass provider usage events through LiteLLM and support local spend tracking
- Client-secret and SDP call endpoints only proxy session setup; subsequent WebRTC media and usage events travel over the peer connection, so LiteLLM cannot record inference spend or enforce spend-based budgets for those sessions
- Use the proxied WebSocket transport when LiteLLM spend logs and budgets must include Realtime inference

Non-billable Realtime protocols are disabled by default. Operators who accept the billing and budget-enforcement limitation can opt in:

```yaml
general_settings:
  allow_non_billable_realtime_protocols: true
```

For user-facing documentation and usage examples, see the litellm-docs repo.
