# structure

- Tencent TokenHub's Messages adapter for provider `tencent`, which Python selects for every Tencent model with no model-name gating, so all of them go natively to `/v1/messages`
- TokenHub speaks the Anthropic wire format, so responses and streams need no Tencent-specific decoding

# boundaries

- Reuse Anthropic payload and response helpers explicitly and keep Tencent differences here. Sharing them does not make Anthropic policy a Messages-wide default

# invariants

- A missing Tencent key must fail, never fall back to `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`. Python's inherited Anthropic lookup does this and leaks an Anthropic credential to a third-party host
- Anthropic OAuth tokens and workload identity federation are Anthropic-org credentials and never apply here
- A caller-supplied credential header still wins over the resolved key
- One Tencent base must work for both chat and Messages, so URL construction accepts a base ending in `/v1` or `/v1/chat/completions` as well as a bare host

# gotchas

- Tencent does not use Anthropic thinking semantics: pass `thinking` through as sent and skip the Claude-specific thinking, effort and temperature rewrites
- `reasoning_effort` translation, the required `max_tokens` check, `context_management` mapping, and advisor and encrypted reasoning block stripping still apply as in the Anthropic config
- Since the provider is not `anthropic`, caller `anthropic-beta` values go through the non-Anthropic beta filter and `compaction` is not a supported request param
- Billing metadata system blocks are stripped, and `system` is dropped when nothing is left

- Python drops every `anthropic-beta` value for Tencent because its beta filter has no `tencent` column, so the adapter uses `BetaPolicy::Drop`
- `compaction` is removed from the body. Python lists it as unsupported for non-Anthropic providers but never applies that list, so it still forwards the field

# known gaps

- Python's last-resort fallback to the SDK-global `litellm.api_key` is not ported, since the gateway has no such global
- Not yet wired into `core` or `LlmProviders`

# references

## python

- `litellm/llms/tencent/messages/transformation.py` (`TencentAnthropicMessagesConfig`)
- `litellm/llms/anthropic/pass_through/messages/transformation.py` (the Anthropic config it subclasses)
- `litellm/utils.py` (`ProviderConfigManager._get_provider_anthropic_messages_config_cached`, provider selection)

## docs

- https://intl.cloud.tencent.com/document/product/1300/82347
- https://www.tencentcloud.com/llms.txt
