
# BlockRun

[BlockRun](https://blockrun.ai) provides access to 30+ LLM models (OpenAI, Anthropic, Google, DeepSeek, xAI, and more) through a single OpenAI-compatible API with [x402](https://www.x402.org/) micropayments.

**Key features:**
- 30+ models from multiple providers via one endpoint
- x402 micropayments — pay-per-request with a Base chain wallet (no API keys needed)
- Free models available (no wallet required)
- Optional smart routing via [ClawRouter](https://github.com/BlockRunAI/ClawRouter) for automatic cost optimization

## Quick Start — Free Models (No Setup)

BlockRun offers free models that require no authentication:

```python
from litellm import completion

# Free models work immediately — no API key needed
response = completion(
    model="blockrun/nvidia/gpt-oss-120b",
    messages=[{"role": "user", "content": "Write a short poem"}],
    api_key="free",  # placeholder for free models
)
print(response)
```

## Usage with Paid Models

For paid models (GPT-4o, Claude, Gemini, etc.), you need a funded Base chain wallet.

### 1. Set up your wallet

```bash
pip install blockrun-llm
blockrun-setup  # Creates wallet and shows funding instructions
```

### 2. Set environment variable

```bash
export BLOCKRUN_WALLET_KEY="0x..."  # Your Base chain wallet private key
```

### 3. Use with LiteLLM Python SDK

```python
import os
from litellm import completion

os.environ["BLOCKRUN_WALLET_KEY"] = "0x..."

messages = [{"role": "user", "content": "Explain quantum computing"}]
response = completion(model="blockrun/openai/gpt-4o", messages=messages)
print(response)
```

> **Note:** Paid models use the x402 payment protocol. The wallet private key is used for local signing only — it is never transmitted to any server.

## Usage with LiteLLM Proxy

### 1. Set BlockRun models in config.yaml

```yaml
model_list:
  - model_name: blockrun-free
    litellm_params:
      model: blockrun/nvidia/gpt-oss-120b
      api_key: "free"
  - model_name: blockrun-gpt4o
    litellm_params:
      model: blockrun/openai/gpt-4o
      api_key: "os.environ/BLOCKRUN_WALLET_KEY"
```

### 2. Start proxy

```bash
litellm --config config.yaml
```

### 3. Query proxy

```bash
curl http://localhost:4000/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_LITELLM_MASTER_KEY" \
  -d '{
    "model": "blockrun-free",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## Available Models

BlockRun provides access to models from multiple providers. Use the `blockrun/` prefix:

| Model | Example |
|-------|---------|
| OpenAI GPT-4o | `blockrun/openai/gpt-4o` |
| OpenAI GPT-4o Mini | `blockrun/openai/gpt-4o-mini` |
| Anthropic Claude Sonnet | `blockrun/anthropic/claude-sonnet-4-20250514` |
| Google Gemini Flash | `blockrun/google/gemini-2.5-flash` |
| DeepSeek Chat | `blockrun/deepseek/deepseek-chat` |
| NVIDIA GPT-OSS 120B (Free) | `blockrun/nvidia/gpt-oss-120b` |

For a full list of models and pricing, visit [blockrun.ai](https://blockrun.ai).

## Smart Routing with ClawRouter

[ClawRouter](https://github.com/BlockRunAI/ClawRouter) automatically routes each request to the cheapest capable model based on prompt complexity — saving 78-96% on inference costs.

```python
# Install: npm install clawrouter (Node.js) or use as OpenClaw plugin
# ClawRouter scores prompts across 15 dimensions locally (<1ms)
# and selects the optimal model tier (SIMPLE/MEDIUM/COMPLEX/REASONING)
```

## Supported Features

- Chat completions (streaming and non-streaming)
- Tool calling / function calling
- Multiple model providers through one endpoint
- Pay-per-request micropayments (no monthly subscriptions)
