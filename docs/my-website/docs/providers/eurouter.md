import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# EUrouter

[EUrouter](https://eurouter.eu) is an EU-hosted, GDPR-compliant AI model routing service. It's a drop-in replacement for OpenRouter, routing requests to providers like OpenAI, Anthropic, Mistral, Google, and others — with guaranteed EU data residency.

EUrouter is OpenAI-compatible, so it works with LiteLLM out of the box.

## Quick Start

### Environment Variables

```bash
export EUROUTER_API_KEY="your-eurouter-api-key"
```

### Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python
import litellm

response = litellm.completion(
    model="eurouter/mistral/mistral-large-3",
    messages=[{"role": "user", "content": "Hello from the EU!"}],
)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add to your LiteLLM Proxy config.yaml:

```yaml
model_list:
  - model_name: mistral-large-3
    litellm_params:
      model: eurouter/mistral/mistral-large-3
      api_key: os.environ/EUROUTER_API_KEY
  - model_name: gpt-4o
    litellm_params:
      model: eurouter/openai/gpt-4o
      api_key: os.environ/EUROUTER_API_KEY
  - model_name: claude-sonnet-4-6
    litellm_params:
      model: eurouter/anthropic/claude-sonnet-4-6
      api_key: os.environ/EUROUTER_API_KEY
```

2. Start the proxy:

```bash
litellm --config config.yaml
```

3. Make a request:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-1234" \
  -d '{
    "model": "mistral-large-3",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

</TabItem>
</Tabs>

## Supported Models

EUrouter routes to 100+ models from 15+ providers. Use the `eurouter/` prefix followed by the provider and model name:

| Model | LiteLLM Model String |
|-------|---------------------|
| Mistral Large 3 | `eurouter/mistral/mistral-large-3` |
| GPT-4o | `eurouter/openai/gpt-4o` |
| GPT-4.1 | `eurouter/openai/gpt-4.1` |
| Claude Sonnet 4.6 | `eurouter/anthropic/claude-sonnet-4-6` |
| Claude Haiku 4.5 | `eurouter/anthropic/claude-haiku-4.5` |
| DeepSeek V3 | `eurouter/deepseek/deepseek-v3` |
| Llama 4 Maverick | `eurouter/meta/llama-4-maverick` |
| Llama 3.3 70B | `eurouter/meta/llama-3.3-70b-instruct` |

For a full list, visit the [EUrouter Models page](https://eurouter.eu/models) or call:

```bash
curl https://api.eurouter.ai/api/v1/models
```

## Why EUrouter?

- **EU Data Residency** — Requests are routed only to EU-hosted endpoints
- **GDPR Compliance** — Built for European data protection requirements
- **OpenAI Compatible** — Drop-in replacement, same API format
- **Multi-Provider** — Access OpenAI, Anthropic, Mistral, Google, and more through a single API
- **Transparent Pricing** — Model cost + 5% platform fee, all in EUR
