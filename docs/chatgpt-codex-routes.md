# ChatGPT OAuth routes used by Codex

The ChatGPT provider supports image generation and editing, structured Responses output, Realtime WebSockets, and direct WebRTC offer exchange using the existing ChatGPT OAuth credentials configured with `CHATGPT_TOKEN_DIR` and `CHATGPT_AUTH_FILE`

## Models and transports

| Model | Proxy route | Upstream route |
| --- | --- | --- |
| `gpt-image-2` | `/v1/images/generations` | ChatGPT `/backend-api/codex/images/generations` |
| `gpt-image-2` | `/v1/images/edits` | ChatGPT `/backend-api/codex/images/edits` |
| `codex-auto-review` | `/v1/responses` | ChatGPT `/backend-api/codex/responses` |
| `gpt-realtime-1.5` | `/v1/realtime` WebSocket | OpenAI `/v1/realtime` with ChatGPT OAuth |
| `gpt-live-1-codex` | `/v1/realtime/calls`, then `/v1/live/{call_id}` | ChatGPT call signaling, OpenAI live sideband |
| `gpt-4o-mini-transcribe` | Nested in Realtime `session.audio.input.transcription.model` | Same Realtime session |

Codex uses `gpt-image-2` for both image routes. There is no separate `gpt-image-2-edit` model in that contract. Memory extraction and consolidation use ordinary Responses models and need no additional transport

Register deployments with `litellm_params.model: chatgpt/<model>`. These auxiliary models do not need to appear in the conversation model selector. This contribution uses the existing single-account ChatGPT authentication contract

## Images

Both synchronous and asynchronous SDK calls support image generation and editing. Edits accept the usual `image` file input or Codex's JSON `images` array, containing one to five `{ "image_url": "data:image/png;base64,..." }` references. PNG, JPEG, WEBP, and HTTPS references are accepted. Masks and mixing `image` with `images` are rejected

```python
await litellm.aimage_edit(
    model="chatgpt/gpt-image-2",
    prompt="Change the blue circle to red",
    images=[{"image_url": image_data_url}],
)
```

## Guardian

The Responses transformation preserves `text.format`, including strict JSON schemas. Custom Codex providers use `/responses` for `codex-auto-review`. The separate native `/guardian` route can be disabled by the upstream service; registering an alias does not enable that route

## Realtime and GPT-Live

Standard Realtime WebSockets use the OpenAI host with the selected ChatGPT access token and account header. Client protocol headers are forwarded without forwarding the client's proxy authorization header

Codex WebRTC offers can use JSON or multipart `sdp` and `session` fields with an ordinary LiteLLM key. Signaling preserves the session shape and the `intent` and `architecture` query parameters. Existing raw SDP requests authenticated with an encrypted ephemeral key retain their existing route

For GPT-Live, send `OpenAI-Alpha: quicksilver=v2`, `intent=quicksilver&architecture=avas`, and a Codex voice such as `sol`. Do not add the GA `session.type: realtime` field to a Frameless Bidi session. A generic `Voice session access denied` error can mean an unsupported voice; it is not sufficient evidence of missing account entitlement

The returned `Location` contains an encrypted call identifier. The sideband connection must use the same LiteLLM bearer key and retain access to the requested alias. The identifier expires after one hour and remains valid across proxy workers sharing the same salt key. OAuth tokens are never returned to clients

The direct `/v1/live?model=gpt-live-1-codex` WebSocket is forwarded, but upstream access can differ from WebRTC. A successful WebRTC call does not establish permission for direct live sessions. Upstream errors remain visible; the proxy does not replace the model, voice, or transport silently

## Verification

The integration was exercised with real image generation and JSON edits, strict Guardian JSON output, Realtime text and audio output, audio transcription, and a GPT-Live WebRTC session with a successful sideband context acknowledgment. Expired, malformed, tampered, and wrong-owner call identifiers have regression coverage

References: [Codex source](https://github.com/openai/codex), [OpenClaw voice authentication](https://docs.openclaw.ai/providers/openai/voice-and-speech), and [Pi Codex signaling implementation](https://github.com/monotykamary/pi-better-openai/blob/main/src/live/transport.ts). These upstream capabilities may change independently of LiteLLM
