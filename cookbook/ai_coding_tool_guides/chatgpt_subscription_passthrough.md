# Use your ChatGPT subscription through LiteLLM

Sign into ChatGPT in Codex or Pi and send model requests through your existing LiteLLM proxy. The client keeps your login and refreshes it. Each request uses that client's ChatGPT account, while a separate LiteLLM virtual key controls gateway access and usage attribution.

No ChatGPT credential or model deployment needs to be added to the proxy. This uses the same separate-key convention as provider passthrough, including Claude Code subscription forwarding.

## Configure Codex

Use your existing LiteLLM URL and a virtual key assigned to your gateway user. Export that key as `LITELLM_API_KEY` in the environment that launches Codex. Keep the value out of committed configuration.

Add this provider to `~/.codex/config.toml`, preserving your other settings:

```toml
model_provider = "litellm_subscription"

[model_providers.litellm_subscription]
name = "LiteLLM subscription"
base_url = "https://litellm.example.com/backend-api/codex"
wire_api = "responses"
requires_openai_auth = true
env_http_headers = { "x-litellm-api-key" = "LITELLM_API_KEY" }
```

Replace `https://litellm.example.com` with your proxy URL. A local proxy normally uses `http://localhost:4000`. Launch Codex, sign into ChatGPT if needed, and send a prompt with a model available to your subscription. Keep your existing model selection.

For a managed workspace rollout, use OpenAI's [gateway configuration guide](https://learn.chatgpt.com/docs/enterprise/sign-in-with-chatgpt-through-a-gateway).

## Configure Pi

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

The proxy forwards the ChatGPT bearer and account header to the fixed OpenAI Codex endpoint. It removes gateway credentials and cookies from the upstream request and redacts credential headers in logging callbacks. The client remains responsible for refresh and reauthentication. Provider authentication and quota errors are returned to that caller without switching subscriptions.

This flow does not connect accounts through the dashboard or pool subscription allowances. Use a Codex build that supports ChatGPT authentication with a custom model provider. The integration forwards HTTP and SSE requests; it does not add native WebSocket or other Codex backend routes.
