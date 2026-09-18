# ChatGPT, Codex, and GPT-Live

The proxy exposes the public GPT-Live session routes and retains the Codex-compatible `POST /live` route. Clients authenticate to LiteLLM with a LiteLLM virtual key. A `chatgpt` deployment uses one proxy-wide ChatGPT OAuth record, while an `openai` deployment uses its configured OpenAI API key. Do not send an upstream OAuth token as the proxy key

## Configure LiteLLM deployments

Codex text, image, and voice requests need separate LiteLLM aliases because their backends have different capabilities. The canonical aliases below keep the primary Qwen model separate from the ChatGPT OAuth models

```yaml
model_list:
  - model_name: qwen3.8-flash-next-codex
    litellm_params:
      model: openai/qwen3.8-flash-next
      api_base: https://qwen.example.com/v1
      api_key: os.environ/QWEN_API_KEY

  - model_name: gpt-image-2
    litellm_params:
      model: chatgpt/gpt-image-2

  - model_name: gpt-image-2.5-flare
    litellm_params:
      model: chatgpt/gpt-image-2.5-flare

  - model_name: gpt-image-2.5-sunburst
    litellm_params:
      model: chatgpt/gpt-image-2.5-sunburst

  - model_name: gpt-realtime-1.5
    litellm_params:
      model: chatgpt/gpt-realtime-1.5

  - model_name: gpt-live-1-codex
    litellm_params:
      model: chatgpt/gpt-live-1-codex
```

If clients use shorter local aliases, publish separate aliases such as `qwen-codex`, `images`, and `voice` that point to the corresponding deployments. Authorize the exact alias sent by the client in the key or team policy, and for a restricted team member set `allowed_models` to contain the aliases used for text, image, or voice requests. Selecting the Qwen alias does not give it image-generation or voice capabilities. The Qwen endpoint in this example uses the OpenAI-compatible adapter and must expose `/v1/responses`; a native vLLM deployment can use `hosted_vllm/qwen3.8-flash-next` when that endpoint is available. LiteLLM does not promise provider-specific tool, reasoning, or stream behavior parity. For a deployment using the public OpenAI API instead, configure `model: openai/gpt-live-1` and `api_key: os.environ/OPENAI_API_KEY` under its own alias. The public API documentation uses `gpt-live-1`; the Codex alias and its backend capabilities are separate

### Configure global ChatGPT OAuth

The ChatGPT provider reads one auth file for the LiteLLM process. Set these environment variables before starting the proxy when the default location is not suitable

```bash
export CHATGPT_TOKEN_DIR=/var/lib/litellm/chatgpt
export CHATGPT_AUTH_FILE=auth.json
```

The defaults are `~/.config/litellm/chatgpt` and `auth.json`. Persist the directory and complete the provider's device-code OAuth flow. The provider refreshes the stored record when it expires. All `chatgpt/...` deployments in that process share this record. Live rejects per-deployment `chatgpt_auth_profile`, `chatgpt_token_dir`, and `chatgpt_auth_file` overrides

### Image generation and editing

Use the `gpt-image-2` alias for both `/v1/images/generations` and `/v1/images/edits`. Keep the exact image aliases requested by your Codex version; a shorter `images` alias only works for clients configured to request it. Codex sends image requests to its active provider's `base_url`, with no separate image URL override. LiteLLM then selects the image deployment independently of the primary text model. ChatGPT image generation and editing require a valid global ChatGPT OAuth login, but a successful login does not establish that the account or backend supports every image operation. Image editing accepts JSON reference images and multipart files, but not masks. Image 2.5 aliases preserve the requested model name; an accepted name does not prove which backend model executed

## Configure Codex through LiteLLM

Codex sends its Responses requests to LiteLLM's `/v1/responses` endpoint. Point the Codex provider at the proxy and select the primary Qwen alias (or the shorter alias you published)

```toml
model = "qwen3.8-flash-next-codex"
model_provider = "litellm"
experimental_realtime_ws_base_url = "https://litellm.example.com/v1"
experimental_realtime_webrtc_call_base_url = "https://litellm.example.com/v1"

[model_providers.litellm]
name = "LiteLLM"
base_url = "https://litellm.example.com/v1"
wire_api = "responses"
requires_openai_auth = true
experimental_bearer_token = "<LITELLM_VIRTUAL_KEY>"
```

Keep Codex signed in with ChatGPT for its client-side capability checks. `experimental_bearer_token` is the LiteLLM virtual key issued by the proxy. It must never contain the upstream ChatGPT OAuth access or refresh token. `requires_openai_auth = true` enables the Codex OpenAI-auth capability path while LiteLLM remains responsible for the upstream provider credentials

## Configure voice routing

The two realtime settings in the TOML example are root-level Codex settings, not fields inside `[model_providers.litellm]`. `experimental_realtime_ws_base_url` routes the Realtime WebSocket and its sideband through LiteLLM. `experimental_realtime_webrtc_call_base_url` is optional and separately routes HTTP WebRTC call creation. The optional root setting `experimental_realtime_ws_model` overrides the voice model; leave it unset to retain your client's default. An override must match the active protocol: `gpt-realtime-1.5` for legacy Realtime v1/v2 or `gpt-live-1-codex` for frameless Live v3. The base URLs end at `/v1`; LiteLLM adds the protocol-specific path

| Voice operation | LiteLLM path |
| --- | --- |
| Legacy Realtime WebSocket | `/v1/realtime` |
| HTTP WebRTC call creation | `/v1/realtime/calls` |
| Frameless Live v3 signaling | `/v1/live` and `/v1/live/{call_id}` |
| Public Live session APIs | `/v1/live/sessions...` |

WebSocket authentication uses `Authorization: Bearer <LITELLM_VIRTUAL_KEY>` by default. If the proxy sets `litellm_key_header_name`, send the virtual key in that configured header instead. Voice is experimental and pending retest: an observed mobile `POST /live` returned 201, but its sideband used the default `api.openai.com` and returned 404. The WebSocket base URL above addresses that routing gap; full bidirectional voice is not verified

## Public Live routes

Use the proxy host in place of `api.openai.com`. Send the configured LiteLLM alias in `session.model` when creating a session, or in the first `session.start` event for a primary WebSocket. Keep the returned session ID unchanged for subsequent operations

| Method | Path | Request and response |
| --- | --- | --- |
| POST | `/v1/live/sessions` | JSON `session` and `transport: {type: "webrtc", sdp: "<offer>"}`; returns 201 JSON with `session.id` and `transport.sdp` |
| POST | `/v1/live/sessions/{session_id}/fork` | JSON WebRTC `transport` and optional `session` overrides; returns 200 JSON with the new session ID and SDP answer |
| GET | `/v1/live/sessions/{session_id}/content` | Downloads stored recording content without converting it to JSON |
| POST | `/v1/live/sessions/{session_id}/accept` | JSON `session` with `type: "live"` and model; successful SIP acceptance returns an empty body |
| POST | `/v1/live/sessions/{session_id}/reject` | JSON with required integer `status_code` from 300 through 699 |
| POST | `/v1/live/sessions/{session_id}/refer` | JSON with `target_uri` for the SIP destination |
| POST | `/v1/live/sessions/{session_id}/hangup` | No request body |
| WebSocket | `/v1/live/sessions` | Start with `session.start`, then wait for `session.started` before sending audio or commands |
| WebSocket | `/v1/live/sessions/{session_id}/attach` | Attach to an existing session; do not send `session.start` or input audio |
| WebSocket | `/v1/live/sessions/{session_id}/fork` | Start with `session.start` and a required `session` overrides object, which may be empty |

Public WebRTC creation uses JSON, not the multipart or raw SDP formats used by the Codex compatibility route. `POST /live` and its existing aliases remain available for Codex clients using that format. WebRTC audio travels on media tracks; its data channel carries Live JSON events. Primary WebSocket audio uses base64 chunks in `session.input_audio.append` and `session.output_audio.delta`

The proxy preserves Live event payloads, including nested Responses events inside `response.event`, rather than translating them into Realtime events. Session routing rewrites the configured model alias to the selected upstream model. Audio, transcript, delegation and usage events retain their upstream format. Send `session.close` and wait for `session.closed` to obtain final usage; a disconnected socket alone does not confirm successful finalization

## Availability and verification

Route support does not establish that every configured backend or account supports every operation. The official API describes project API-key authentication; it does not guarantee equivalent capabilities for ChatGPT OAuth. An OAuth request reaching SDP validation proves only that the request reached that validation step. It does not prove a working audio session, recording, fork or SIP call. The routes listed here have not all been tested against a real upstream service

Session controls require a session known to the proxy and owned by the authenticated caller. Incoming SIP calls originate upstream. A proxy administrator can accept or reject the raw ID from a verified incoming-call webhook by supplying `x-litellm-live-model` with an alias that resolves to exactly one deployment. Successful acceptance returns the proxy-owned handle in `x-litellm-live-session-id`, preserving the API's empty response body. Use that handle for subsequent controls. Ordinary virtual keys cannot enroll arbitrary upstream session IDs; a trusted webhook-to-owner enrollment flow is still required for those keys

Live duration uses cumulative `usage.seconds`; legacy Codex milliseconds remain supported. WebRTC initialization has a 15-second minimum credited against running duration, not added to it. Nested terminal Responses usage is charged separately using its backend model and deduplicated by response ID. A failed observation connection cannot establish complete usage. Managed delegation also depends on receiving its backend usage events; the upstream sideband does not replay events emitted before attachment

Managed Responses delegation authorizes its backend model as well as the voice model. Use client delegation when budgets or request/token limits apply to the key, user, team, project, organization, team member, end user, or a model access group, since each backend invocation needs its own admission check. Keys scoped to access groups, projects, users, organizations, or teams are treated as model-restricted even if the key's own model list is empty. Managed WebRTC sessions with model restrictions must explicitly exclude `session.update` and wildcard events from frontend client events. That data channel connects directly to OpenAI and could otherwise change the backend model outside the proxy's checks. Client delegation does not need this restriction: the delegation type cannot change after startup or on a fork. Sparse sideband updates may omit the backend model to retain its current value

For both HTTP and WebSocket forks of managed sessions, restricted-model keys must explicitly provide an authorized `session.delegation.responses.model`. Empty overrides cannot safely authorize an inherited managed backend: the session handle records startup configuration, while later updates may have changed the upstream model. Client-delegation forks can use empty overrides because the delegation type is immutable

See the official [Live overview](https://developers.openai.com/api/docs/guides/live), [Live API reference](https://developers.openai.com/api/reference/resources/live), [session management](https://developers.openai.com/api/docs/guides/live-conversations), [WebRTC guide](https://developers.openai.com/api/docs/guides/voice-webrtc?api=live), [WebSocket guide](https://developers.openai.com/api/docs/guides/voice-websockets?api=live), [server controls](https://developers.openai.com/api/docs/guides/voice-server-controls?api=live) and [SIP guide](https://developers.openai.com/api/docs/guides/voice-sip?api=live) for the upstream contract. The voice guides also contain Realtime tabs with different routes and formats
