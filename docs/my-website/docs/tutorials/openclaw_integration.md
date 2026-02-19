---
sidebar_label: "OpenClaw"
---

# OpenClaw Integration

Use OpenClaw with LiteLLM Proxy through an OpenAI-compatible endpoint.

## 1. Create a simple LiteLLM config

Create `openclaw_proxy_config.yaml`:

```yaml showLineNumbers title="openclaw_proxy_config.yaml"
model_list:
  - model_name: openclaw-default
    litellm_params:
      model: openai/gpt-5-mini
      api_key: os.environ/OPENAI_API_KEY

general_settings:
  master_key: sk-1234
```

## 2. Start LiteLLM Proxy

```bash showLineNumbers
poetry run litellm --config openclaw_proxy_config.yaml --port 4000
```

## 3. Test the proxy

In another terminal:

```bash showLineNumbers
curl -s http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model":"openclaw-default",
    "messages":[{"role":"user","content":"hello"}]
  }'
```

## 4. Connect OpenClaw

Run:

```bash showLineNumbers
openclaw onboard --auth-choice litellm-api-key
```

Use these values:

- Base URL: `http://localhost:4000`
- API key: `sk-1234`
- API type: `openai-completions`
- Model: `openclaw-default`

## References

- OpenClaw docs: [https://docs.openclaw.ai/providers/litellm](https://docs.openclaw.ai/providers/litellm)
- LiteLLM proxy docs: [https://docs.litellm.ai/docs/proxy/docker_quick_start](https://docs.litellm.ai/docs/proxy/docker_quick_start)
