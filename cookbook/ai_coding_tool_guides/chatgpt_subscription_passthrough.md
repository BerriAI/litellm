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

For a managed workspace rollout, use OpenAI's [gateway configuration guide](https://learn.chatgpt.com/docs/enterprise/sign-in-with-chatgpt-through-a-gateway).

## Native compatibility and Pi

For a native Codex model catalog and request format, change Codex's `base_url` to `https://litellm.example.com/backend-api/codex`. This route uses the same credentials but does not need a model deployment or resolve model aliases. Restricted keys must allow the upstream model name.

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

The native routes are `GET /backend-api/codex/models` and `POST /backend-api/codex/responses`. Model requests use the upstream model name, so a restricted LiteLLM key must allow that name. OpenAI's model catalog is returned in its native format; seeing a model there does not override gateway permissions.

The proxy forwards the ChatGPT bearer and account header to the fixed OpenAI Codex endpoint. It removes gateway credentials and cookies from the upstream request and redacts credential headers in logging callbacks. The client remains responsible for refresh and reauthentication. Native passthrough returns authentication and quota errors without switching subscriptions. Standard model routes retain the caller login across ChatGPT retries; explicitly configured cross-provider fallbacks use their own credentials. An explicit workspace header is preserved for OpenAI to validate.

This flow does not connect accounts through the dashboard or pool subscription allowances. Use a Codex build that supports ChatGPT authentication with a custom model provider. The integration forwards HTTP and SSE requests; it does not add native WebSocket or other Codex backend routes.
