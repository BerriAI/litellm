# OpenClaw

Route [OpenClaw](https://openclaw.ai) — the open-source personal AI assistant — through LiteLLM for unified cost tracking, logging, and model management.

## Why use LiteLLM with OpenClaw?

- **Cost tracking** — See exactly what your AI assistant is spending
- **Model flexibility** — Switch between Claude, GPT-4, Gemini, or any provider without changing OpenClaw config
- **Virtual keys** — Give OpenClaw its own key with spend limits
- **Logging** — Full request/response logs for debugging

## Quick Start

OpenClaw uses Anthropic's `/v1/messages` protocol natively, which LiteLLM supports out of the box.

### 1. Start LiteLLM Proxy

```bash
litellm --model anthropic/claude-sonnet-4-20250514
```

### 2. Configure OpenClaw

Point OpenClaw to your LiteLLM proxy:

```bash
export ANTHROPIC_API_BASE="http://localhost:4000"
export ANTHROPIC_API_KEY="your-litellm-key"  # or sk-anything if no auth

openclaw
```

That's it. OpenClaw now routes through LiteLLM.

## Config File Setup

For a permanent setup, add to your OpenClaw config (`~/.openclaw/openclaw.json`):

```json5
{
  env: {
    ANTHROPIC_API_BASE: "http://localhost:4000",
    ANTHROPIC_API_KEY: "your-litellm-key"
  },
  agents: {
    defaults: {
      model: { primary: "anthropic/claude-sonnet-4-20250514" }
    }
  }
}
```

## Using Virtual Keys

Create a dedicated key for OpenClaw with a spending cap:

```bash
curl -X POST "http://localhost:4000/key/generate" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "key_alias": "openclaw",
    "max_budget": 50.00,
    "budget_duration": "monthly"
  }'
```

Use the generated key in OpenClaw:

```bash
export ANTHROPIC_API_KEY="sk-generated-key"
```

## Model Routing

Want OpenClaw to use a different model than what it requests? Configure model aliasing in LiteLLM:

```yaml title="config.yaml"
model_list:
  - model_name: claude-sonnet-4-20250514
    litellm_params:
      model: azure/gpt-4o  # Route to Azure instead
      api_key: os.environ/AZURE_API_KEY
```

OpenClaw keeps requesting `claude-sonnet-4-20250514` — LiteLLM handles the actual routing.

## Viewing Usage

Check the LiteLLM dashboard or API for OpenClaw's usage:

```bash
# Get key info
curl "http://localhost:4000/key/info" \
  -H "Authorization: Bearer your-litellm-key"

# View spend
curl "http://localhost:4000/spend/logs" \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY"
```

## Notes

- LiteLLM runs on `http://localhost:4000` by default
- The `/v1/messages` endpoint supports streaming, tools, prompt caching, and extended thinking
- All OpenClaw features work through LiteLLM — no limitations

## Related

- [/v1/messages endpoint docs](/docs/anthropic_unified/)
- [Virtual Keys](/docs/proxy/virtual_keys)
- [Cost Tracking](/docs/proxy/cost_tracking)
- [OpenClaw Documentation](https://docs.openclaw.ai)
