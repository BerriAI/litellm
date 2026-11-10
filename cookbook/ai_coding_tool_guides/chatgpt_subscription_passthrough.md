# Use your ChatGPT subscription through LiteLLM

Sign into ChatGPT in Codex or Pi and send model requests through your existing LiteLLM proxy. The client keeps your login and refreshes it. Each request uses that client's ChatGPT account, while a separate LiteLLM virtual key controls gateway access and usage attribution.

An administrator adds a `chatgpt/` model using the usual model configuration. No ChatGPT login needs to be added to the proxy. This uses the same separate-key convention as provider passthrough, including Claude Code subscription forwarding.

## Add a model to the proxy

Reuse your existing proxy. If you need one, follow the [proxy quickstart](https://docs.litellm.ai/docs/proxy/docker_quick_start). Merge this entry into its model list and restart the proxy:

```yaml
model_list:
  - model_name: gpt-6-luna
    litellm_params:
      model: chatgpt/gpt-6-luna
    model_info:
      mode: responses
```

Use a model available to the caller's subscription. The virtual key must allow the exposed model name. Leave saved ChatGPT logins and API credentials unset for a client-only deployment. Explicit cross-provider fallbacks can incur separate API charges; do not configure them for subscription-only traffic.

## Configure Codex

Use your existing LiteLLM URL and a virtual key assigned to your gateway user. Export that key as `LITELLM_API_KEY` in the environment that launches Codex. Keep the value out of committed configuration.

Add this provider to `~/.codex/config.toml`, preserving your other settings:

```toml
model = "gpt-6-luna"
model_provider = "litellm_subscription"

[model_providers.litellm_subscription]
name = "LiteLLM subscription"
base_url = "https://litellm.example.com/v1"
wire_api = "responses"
supports_websockets = false
requires_openai_auth = true
env_http_headers = { "x-litellm-api-key" = "LITELLM_API_KEY" }
```

Replace `https://litellm.example.com` with your proxy URL. A local proxy normally uses `http://localhost:4000/v1`. Launch Codex, sign into ChatGPT if needed, and send a prompt with a model available to your subscription. The selected model must match the name exposed by the gateway.

The `lite` CLI can set up this provider each time it starts Codex, with no `config.toml` edit. Export `LITELLM_PROXY_URL` and `LITELLM_PROXY_API_KEY`, then run `lite codex --subscription`. Run `lite config set codex_subscription true` to make it the default.

For a managed workspace rollout, use OpenAI's [gateway configuration guide](https://learn.chatgpt.com/docs/enterprise/sign-in-with-chatgpt-through-a-gateway).

## Native compatibility and Pi

For a native Codex model catalog and request format, change Codex's `base_url` to `https://litellm.example.com/backend-api/codex`. This route uses the same credentials. A ChatGPT model keeps its upstream name and needs no deployment of its own.

### Mix ChatGPT and LiteLLM models

On the native route, Codex's model picker shows OpenAI's catalog first and then the other chat and responses models the virtual key can use. Pick one of those and Codex's requests go through LiteLLM's `/v1/responses` with the key's usual routing, budgets, guardrails, and logging. The proxy removes the ChatGPT bearer and account header from that request, so the other provider and your callbacks never receive them. A name is routed this way when another provider serves it as an exact `model_name`, through a router `model_group_alias`, or as a team's public model name. A ChatGPT model, a name that only a wildcard deployment matches, or a name OpenAI's catalog lists for the caller's ChatGPT account keeps the passthrough, so a model OpenAI adds later still uses the subscription even when a deployment of another provider has the same name. The proxy learns the account's list from the catalog Codex fetches, or asks OpenAI with the caller's login. When it cannot get the list, for example because OpenAI is down, the request times out, or the client is not Codex, every request keeps the passthrough for the next minute, as it did before LiteLLM models were added to the picker. When the key can use many models, the picker lists the ones that fit in Codex's catalog size limit, and the proxy logs a warning with how many it left out.

Codex's picker sends a `none` reasoning effort for a LiteLLM model whose catalog entry lists no effort levels. With `litellm_settings: drop_params: true`, the proxy drops it for a model that does not support `none`, and the provider applies its own default. A model whose cost map entry or `model_info` sets `supports_none_reasoning_effort: true` keeps it. Without `drop_params`, the effort reaches the provider unchanged.

Codex also sends its ChatGPT app tools with every turn, which on a LiteLLM model adds a few hundred thousand prompt tokens to each turn. Set `apps = false` under `[features]` in Codex's `config.toml` to turn them off.

Only a key that may call `/v1/responses` gets these models. A key whose `allowed_routes` covers only the native routes sees OpenAI's catalog unchanged, and every request it sends uses the subscription.

Codex sends earlier reasoning items back with each turn. The proxy tags the reasoning that a LiteLLM model returns on this route, OpenAI and Azure deployments included. When a session switches to a ChatGPT model, the proxy drops those items before forwarding the request to OpenAI, which would reject them. Messages, tool calls, and ChatGPT's own reasoning are kept, and a LiteLLM model still gets its own reasoning back.

When the catalog adds LiteLLM models for a key, its `ETag` is a digest of OpenAI's ETag and those models, and that key's responses carry the same value in `X-Models-Etag`. Codex keeps its cached catalog while the two match and fetches it again when OpenAI's catalog or the key's models change. A key whose catalog adds no LiteLLM models keeps OpenAI's ETags.

### Restrict native models with an access group

A key with a model allowlist needs a rule for ChatGPT names that OpenAI adds over time. Put the subscription wildcards in an access group under `chatgpt/` names:

```yaml
model_list:
  - model_name: chatgpt/gpt-*
    litellm_params:
      model: chatgpt/gpt-*
    model_info:
      access_groups: ["chatgpt-native"]
  - model_name: chatgpt/codex-*
    litellm_params:
      model: chatgpt/codex-*
    model_info:
      access_groups: ["chatgpt-native"]
```

Then add `chatgpt-native` to the key's models, next to any other model names it should reach. On the native route, a key that holds only the `chatgpt/` form of a name is checked as that form, so a request for `gpt-5.6-sol` is allowed when the key may call `chatgpt/gpt-5.6-sol`. The key can call every ChatGPT model that matches the wildcards, including ones the proxy configuration does not name.

On `/v1` these wildcards match only a request that asks for a `chatgpt/` name, so a request for a plain `gpt-` or `codex-` name still goes where it went before, such as a `*` catch-all deployment. A `/v1` request for a `chatgpt/` name without a ChatGPT login falls back to the proxy's own saved login if it has one, so keep that login unset as described above.

### Configure Pi

Sign into the built-in `openai-codex` provider in Pi. Merge this override into `~/.pi/agent/models.json` and export your LiteLLM virtual key as `LITELLM_API_KEY` before launching Pi:

```json
{
  "providers": {
    "openai-codex": {
      "baseUrl": "https://litellm.example.com/backend-api/codex",
      "headers": {
        "x-litellm-api-key": "LITELLM_API_KEY"
      }
    }
  }
}
```

Keep Pi's subscription login as the provider credential. The header value names an environment variable, not a key to paste into the file.

## Verify account usage

Send a prompt and check the completed request in LiteLLM's Logs page. Its gateway user/key and token counts should match the caller. Give each person their own LiteLLM virtual key for separate attribution. The gateway credential and ChatGPT login are independently authenticated; LiteLLM does not establish that both identities belong to the same person.

The native routes are `GET /backend-api/codex/models` and `POST /backend-api/codex/responses`. Model requests use the upstream model name, so a restricted LiteLLM key needs the `chatgpt-native` access group above or that exact name. OpenAI's model catalog is returned in its native format; seeing a model there does not override gateway permissions.

The proxy forwards the ChatGPT bearer and account header to the fixed OpenAI Codex endpoint. It removes gateway credentials and cookies from the upstream request and redacts credential headers in logging callbacks. The client remains responsible for refresh and reauthentication. Native passthrough returns authentication and quota errors without switching subscriptions. Standard model routes retain the caller login across ChatGPT retries; explicitly configured cross-provider fallbacks use their own credentials. An explicit workspace header is preserved for OpenAI to validate.

This flow does not connect accounts through the dashboard or pool subscription allowances. Use a Codex build that supports ChatGPT authentication with a custom model provider. The integration forwards HTTP and SSE requests; it does not add native WebSocket or other Codex backend routes.
