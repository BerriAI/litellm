# Using OpenClaw with LiteLLM

Run [OpenClaw](https://openclaw.ai) — the open-source AI assistant — through your LiteLLM proxy. Get unified logging, cost tracking, and model management while keeping OpenClaw's full capabilities.

## Why use LiteLLM with OpenClaw?

- **Unified API** — Route OpenClaw through any model provider (Anthropic, OpenAI, Azure, Bedrock, etc.)
- **Cost tracking** — Track spend across all your AI tools in one place
- **Virtual keys** — Give OpenClaw its own API key with spend limits
- **Model fallbacks** — Automatic failover if your primary model is down

## Quick Start

LiteLLM's native `/v1/messages` endpoint speaks Anthropic's protocol — perfect for OpenClaw.

### 1. Start LiteLLM Proxy

```bash
# Set your API key (works with any provider LiteLLM supports)
export ANTHROPIC_API_KEY="sk-ant-..."

# Start the proxy
litellm --model anthropic/claude-sonnet-4-5-20250929

# Running on http://0.0.0.0:4000
```

### 2. Configure OpenClaw

Point OpenClaw to your LiteLLM proxy:

```bash
# Set environment variables
export ANTHROPIC_API_BASE="http://localhost:4000"
export ANTHROPIC_API_KEY="sk-anything"  # or your LiteLLM virtual key

# Start OpenClaw
openclaw
```

That's it! OpenClaw now routes through LiteLLM via the native `/v1/messages` endpoint.

### Alternative: Config file

Add to your OpenClaw config (`~/.openclaw/config.json5`):

```json5
{
  env: {
    ANTHROPIC_API_BASE: "http://localhost:4000",
    ANTHROPIC_API_KEY: "your-litellm-key"
  },
  agents: {
    defaults: {
      model: { primary: "anthropic/claude-sonnet-4-5-20250929" }
    }
  }
}
```

## Using Virtual Keys

Want to track OpenClaw's spend separately? Create a virtual key:

```bash
# Generate a key for OpenClaw
curl -X POST 'http://localhost:4000/key/generate' \
  -H 'Authorization: Bearer your-master-key' \
  -H 'Content-Type: application/json' \
  -d '{
    "key_alias": "openclaw",
    "max_budget": 50.00,
    "models": ["claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101"]
  }'
```

Use the returned key as `ANTHROPIC_API_KEY` in OpenClaw.

## Using Different Models

LiteLLM lets you swap models without changing OpenClaw config. Update your `config.yaml`:

```yaml
model_list:
  # Map "claude-sonnet-4-5-20250929" to any model
  - model_name: claude-sonnet-4-5-20250929
    litellm_params:
      model: azure/gpt-4o  # or bedrock/claude, openai/gpt-4, vertex_ai/gemini-pro, etc.
      api_key: os.environ/AZURE_API_KEY
      api_base: https://your-resource.openai.azure.com
```

OpenClaw requests `claude-sonnet-4-5-20250929` → LiteLLM routes to your chosen model.

## Remote Access (Gateway Mode)

Running OpenClaw Gateway on a server? Point it to your LiteLLM instance:

```json5
// ~/.openclaw/config.json5 on your server
{
  env: {
    ANTHROPIC_API_BASE: "http://litellm-host:4000",
    ANTHROPIC_API_KEY: "your-litellm-key"
  },
  gateway: {
    bind: "lan",  // or "tailnet" for Tailscale
    port: 4321
  }
}
```

## Troubleshooting

### "Connection refused"

Make sure LiteLLM is running and accessible:
```bash
curl http://localhost:4000/health
```

### "Invalid API key"

If using virtual keys, ensure the key has access to the models OpenClaw uses:
```bash
curl http://localhost:4000/key/info \
  -H "Authorization: Bearer your-key"
```

### Model not found

Check your LiteLLM config includes the model OpenClaw is requesting:
```bash
curl http://localhost:4000/v1/models
```

## Learn More

- [OpenClaw Documentation](https://docs.openclaw.ai)
- [OpenClaw GitHub](https://github.com/openclaw/openclaw)
- [LiteLLM /v1/messages docs](/docs/anthropic_unified/)
