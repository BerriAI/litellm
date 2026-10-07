# structure

- GitHub Copilot's Messages adapter for the `github_copilot` provider: endpoint, Copilot token auth, Copilot headers and beta policy. Payload, response and SSE handling are Anthropic's

# boundaries

- Reuse `anthropic/messages` payload and response policy explicitly and keep only endpoint, auth, header and beta differences here

# invariants

- Only `github_copilot` models whose lowercased name contains `claude` reach this adapter. Every other Copilot model falls back to the chat completions bridge, and that gate lives in provider selection, not shared defaults
- The caller's `api_base` and `api_key` are always ignored. The base comes from `endpoints.api` in the cached Copilot token, so the Copilot bearer token is never sent to a caller-chosen host
- `openai-intent`, `x-interaction-type` and `x-github-api-version` are forced over caller values. All other Copilot defaults only fill keys the caller did not set
- A failure to obtain the Copilot token surfaces as an authentication error for provider `github_copilot`

# gotchas

- `anthropic-beta` values are never filtered against the provider beta allowlist, because Copilot forwards Messages natively and has no entry there
- Copilot does not execute `web_search` tools, so the adapter reports web search as not handled natively and the interception path short circuits web search only requests
- `X-Initiator` and `Copilot-Vision-Request` belong to the Copilot chat config. The Python Messages path does not send them, so do not add them here without a parity reason
- Copilot publishes no full HTTP spec for its `/v1/messages` endpoint, so the Anthropic Messages API is the working contract

- The Copilot token comes from an injected `CopilotTokenCache` (`github_copilot/common_utils.rs`). `cached()` reads `endpoints.api` for the URL without refreshing, like Python's `get_api_base`, and `acquire()` yields the bearer at send time through `AuthScheme::Token`
- The acquired Copilot bearer replaces any forwarded `authorization`. Python only fills it when the caller sent none, which would let a caller header stand in for the Copilot token
- The payload follows the first-party request policy (Claude thinking rules, billing metadata kept), matching Python's base `AnthropicMessagesConfig`

# known gaps

- No Rust `CopilotTokenCache` exists yet: the device flow, the token files under `GITHUB_COPILOT_TOKEN_DIR` and the refresh against `GITHUB_COPILOT_API_KEY_URL` are still Python only, so the config has no `const` instance to wire
- `BaseMessagesConfig` has no web search hook, so "not handled natively" is not expressed in Rust yet
- `x-request-id` is not generated per request, because `litellm-llms` has no UUID dependency

# references

## python

- `litellm/llms/github_copilot/messages/transformation.py` (`GithubCopilotAnthropicMessagesConfig`)
- `litellm/llms/github_copilot/authenticator.py`
- `litellm/llms/github_copilot/common_utils.py`

## docs

- https://docs.github.com/en/copilot/how-tos/copilot-sdk/features/streaming-events
- https://docs.github.com/api/article/body?pathname=/en/copilot/how-tos/copilot-sdk/features/streaming-events
- https://docs.github.com/llms.txt
- https://platform.claude.com/docs/en/api/http/messages/create
