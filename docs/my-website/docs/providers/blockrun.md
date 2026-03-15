# BlockRun

## Overview

| Property | Details |
|-------|-------|
| Description | BlockRun is a unified LLM gateway providing access to 41+ models across 7 providers (OpenAI, Anthropic, Google, DeepSeek, xAI, Moonshot, MiniMax) with pay-per-request micropayments via the x402 protocol. No API keys or monthly subscriptions needed. |
| Provider Route on LiteLLM | `blockrun/` |
| Link to Provider Doc | [BlockRun Website ↗](https://blockrun.ai) |
| Base URL | `https://blockrun.ai/api/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage), [`/images/generations`](#image-generation) |

<br />

## What is BlockRun?

BlockRun is an agent-native LLM gateway that enables:
- **41+ Models**: Access OpenAI, Anthropic, Google, DeepSeek, xAI, and more through a single endpoint
- **Pay-Per-Request**: USDC micropayments on Base chain via the [x402 protocol](https://www.x402.org/) — no API keys or subscriptions
- **Free Models**: Some models (e.g., `nvidia/gpt-oss-120b`) are available with zero authentication
- **Cost Savings**: Up to 92% cheaper than direct provider pricing
- **OpenAI-Compatible**: Standard OpenAI SDK and client compatibility

## Required Variables

### Free Models (No Setup)

Free models require no environment variables:

```python showLineNumbers title="No setup needed for free models"
# No environment variables required
# Use api_key="free" when calling free models
```

### Paid Models

```python showLineNumbers title="Environment Variables"
os.environ["BLOCKRUN_WALLET_KEY"] = ""  # your BlockRun wallet private key
```

Get a wallet key by visiting [blockrun.ai](https://blockrun.ai) and funding a Base chain wallet with USDC.

## Usage - LiteLLM Python SDK

### Free Models

```python showLineNumbers title="BlockRun Free Model"
import litellm
from litellm import completion

# No API key needed for free models
response = completion(
    model="blockrun/nvidia/gpt-oss-120b",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="free",
)

print(response)
```

### Paid Models

```python showLineNumbers title="BlockRun Paid Model"
import os
import litellm
from litellm import completion

os.environ["BLOCKRUN_WALLET_KEY"] = ""  # your wallet key

response = completion(
    model="blockrun/openai/gpt-4o",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(response)
```

### Streaming

```python showLineNumbers title="BlockRun Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["BLOCKRUN_WALLET_KEY"] = ""  # your wallet key

response = completion(
    model="blockrun/openai/gpt-4o",
    messages=[{"role": "user", "content": "Write a short poem about AI agents"}],
    stream=True,
)

for chunk in response:
    print(chunk)
```

## Usage - LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export BLOCKRUN_WALLET_KEY=""
```

### 2. Start the proxy

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: blockrun/openai/gpt-4o
      api_key: os.environ/BLOCKRUN_WALLET_KEY
  - model_name: claude-sonnet
    litellm_params:
      model: blockrun/anthropic/claude-sonnet-4-20250514
      api_key: os.environ/BLOCKRUN_WALLET_KEY
  - model_name: free-model
    litellm_params:
      model: blockrun/nvidia/gpt-oss-120b
      api_key: free
```

## Supported Models

BlockRun provides access to 41+ models. Here are some popular ones:

| Model | Model String |
|-------|-------------|
| GPT-5 | `blockrun/openai/gpt-5` |
| GPT-4o | `blockrun/openai/gpt-4o` |
| GPT-4o Mini | `blockrun/openai/gpt-4o-mini` |
| Claude Opus 4 | `blockrun/anthropic/claude-opus-4-20250514` |
| Claude Sonnet 4 | `blockrun/anthropic/claude-sonnet-4-20250514` |
| Gemini 2.5 Pro | `blockrun/google/gemini-2.5-pro` |
| Gemini 2.5 Flash | `blockrun/google/gemini-2.5-flash` |
| DeepSeek Chat | `blockrun/deepseek/deepseek-chat` |
| Grok 3 | `blockrun/xai/grok-3` |
| Grok 3 Mini | `blockrun/xai/grok-3-mini` |

### Free Models

| Model | Model String |
|-------|-------------|
| NVIDIA GPT-OSS 120B | `blockrun/nvidia/gpt-oss-120b` |

## Supported OpenAI Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | array | **Required**. Array of message objects with 'role' and 'content' |
| `model` | string | **Required**. Model ID in `provider/model-name` format |
| `stream` | boolean | Optional. Enable streaming responses |
| `temperature` | float | Optional. Sampling temperature (0-2) |
| `top_p` | float | Optional. Nucleus sampling parameter |
| `max_tokens` | integer | Optional. Maximum tokens to generate |
| `frequency_penalty` | float | Optional. Penalize frequent tokens |
| `presence_penalty` | float | Optional. Penalize tokens based on presence |
| `stop` | string/array | Optional. Stop sequences |
| `tools` | array | Optional. List of available tools/functions |
| `tool_choice` | string/object | Optional. Control tool/function calling |
| `response_format` | object | Optional. Response format specification |

## How Payments Work

BlockRun uses the [x402 protocol](https://www.x402.org/) for USDC micropayments on Base chain:

1. **Fund your wallet**: Send USDC to your Base chain wallet
2. **Set your wallet key**: Export `BLOCKRUN_WALLET_KEY` with your private key
3. **Make requests**: Each API call is a pay-per-request micropayment — no subscriptions

Free models bypass the payment system entirely.

## Additional Resources

- [BlockRun Documentation](https://blockrun.ai)
- [BlockRun GitHub](https://github.com/BlockRunAI)
- [x402 Protocol](https://www.x402.org/)
- [ClawRouter (Open Source)](https://github.com/BlockRunAI/clawrouter)
