# Galileo (BETA)

Galileo logging is available via LiteLLM proxy logging callbacks.

## Requirements

```bash
export GALILEO_BASE_URL=""   # e.g. https://api.galileo... (console URL with `console` replaced by `api`)
export GALILEO_PROJECT_ID=""
export GALILEO_USERNAME=""
export GALILEO_PASSWORD=""
```

## Quick start (Proxy)

```yaml
model_list:
  - model_name: fake-openai-endpoint
    litellm_params:
      api_key: my-fake-key
      api_base: https://exampleopenaiendpoint-production.up.railway.app/
      model: openai/my-fake-model

litellm_settings:
  success_callback: ["galileo"] # 👈 KEY CHANGE
```

Start the proxy with this config and logs will be sent to Galileo.

For additional proxy-specific details, see the main [Proxy logging section](../proxy/logging.md#galileo).
