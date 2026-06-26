# Headroom prompt compression guardrail

Wraps [headroom-ai](https://github.com/chopratejas/headroom) as a LiteLLM `pre_call` guardrail. Compresses tool outputs, JSON arrays, and prose in the message history before the request reaches the LLM provider.

Headroom's `smart_crusher` deduplicates repeated JSON rows and converts verbose object arrays to a compact schema+CSV format. `kompress` handles prose and log compression. Typical savings: 40-90% on agentic workloads with large tool outputs.

## Running locally

```bash
cp .env.example .env  # set OPENAI_API_KEY (headroom uses it internally)
docker compose up
```

Service listens on `http://localhost:8100`.

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | API key headroom uses for compression |
| `HEADROOM_DEFAULT_MODEL` | No | `gpt-4o-mini` | Model used for token budget calculations |
| `GUARDRAIL_API_KEY` | No | — | If set, requires `x-api-key` header on requests |

## LiteLLM config

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o

guardrails:
  - guardrail_name: headroom-compression
    litellm_params:
      guardrail: generic_guardrail_api
      mode: pre_call
      api_base: http://localhost:8100
      # api_key: your-secret-key
```

## What gets compressed

- Tool results containing JSON (deduplication + schema compression)
- Tool results containing logs and prose
- User messages (opt-in via `compress_user_messages=True`, already enabled)

Conversational turns under 250 tokens are left unchanged. Error outputs are protected from compression.

## Deploying to Render

The included `render.yaml` deploys as a Docker web service. Set `OPENAI_API_KEY` as a secret environment variable in the Render dashboard after the first deploy.
