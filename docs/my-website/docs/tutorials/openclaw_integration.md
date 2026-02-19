---
sidebar_label: "OpenClaw"
---

# OpenClaw Integration

Use [OpenClaw](https://docs.openclaw.ai) with LiteLLM Proxy to route requests through 100+ LLM providers.

## Prerequisites

- LiteLLM Proxy installed (`pip install 'litellm[proxy]'`)
- OpenClaw installed (`npm install -g openclaw@latest`)
- At least one LLM provider API key (OpenAI, Anthropic, Gemini, etc.)

## 1. Create a LiteLLM config

Create `litellm_config.yaml`:

```yaml showLineNumbers title="litellm_config.yaml"
model_list:
  - model_name: gpt-5
    litellm_params:
      model: openai/gpt-5
      api_key: os.environ/OPENAI_API_KEY

general_settings:
  master_key: sk-1234
```

:::tip
You can add multiple models from different providers. See [LiteLLM proxy config docs](https://docs.litellm.ai/docs/proxy/configs) for all options.
:::

## 2. Start the proxy

```bash showLineNumbers
litellm --config litellm_config.yaml --port 4000
```

Verify it's running:

```bash showLineNumbers
curl http://localhost:4000/health
```

## 3. Connect OpenClaw

```bash showLineNumbers
openclaw onboard --auth-choice litellm-api-key
```

Enter these values when prompted:

| Field     | Value                    |
|-----------|--------------------------|
| Base URL  | `http://localhost:4000`  |
| API key   | `sk-1234`               |
| API type  | `openai-completions`    |
| Model     | `gpt-5`           |

## 4. Verify

```bash showLineNumbers
openclaw chat
```

Send a test message. If you get a response, the integration is working.

## Alternative: Manual config

If you prefer to configure OpenClaw directly instead of using `openclaw onboard`:

```bash showLineNumbers
export LITELLM_API_KEY="sk-1234"
```

Edit `~/.openclaw/openclaw.json`:

```json showLineNumbers title="~/.openclaw/openclaw.json"
{
  "models": {
    "providers": {
      "litellm": {
        "baseUrl": "http://localhost:4000",
        "apiKey": "${LITELLM_API_KEY}",
        "api": "openai-completions",
        "models": [
          {
            "id": "gpt-5",
            "name": "GPT-5",
            "reasoning": false,
            "input": ["text"],
            "contextWindow": 128000,
            "maxTokens": 16384
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": { "primary": "litellm/gpt-5" }
    }
  }
}
```

Then restart the gateway:

```bash showLineNumbers
openclaw gateway restart
```

## Troubleshooting

**OpenClaw uses a previous model (e.g., `Invalid model name`)**

```bash showLineNumbers
openclaw models set litellm/gpt-5
openclaw gateway restart
```

**Starting from scratch**

```bash showLineNumbers
openclaw gateway stop
rm -rf ~/.openclaw
openclaw onboard --auth-choice litellm-api-key
```

## References

- [OpenClaw LiteLLM provider docs](https://docs.openclaw.ai/providers/litellm)
- [LiteLLM proxy configuration](https://docs.litellm.ai/docs/proxy/configs)
