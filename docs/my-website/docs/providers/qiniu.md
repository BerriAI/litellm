# Qiniu (`qiniu`)

## Overview

| Property | Details |
|-------|-------|
| Description | Qiniu AI (七牛云 AI) provides access to 60+ leading LLMs — including DeepSeek, Qwen, Doubao, MiniMax, Kimi, and GLM — via an OpenAI-compatible API. |
| Provider Route on LiteLLM | `qiniu/` |
| Link to Provider Doc | [Qiniu AI Docs ↗](https://www.qiniu.com/ai/models) |
| Base URL | `https://api.qnaigc.com/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage), [`/models`](#supported-models) |

<br />

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["QINIU_API_KEY"] = ""  # your Qiniu AI API key
```

Get your API key from the [Qiniu AI Console](https://www.qiniu.com/products/qnai).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Qiniu Non-streaming Completion"
import os
import litellm

os.environ["QINIU_API_KEY"] = ""  # your Qiniu AI API key

response = litellm.completion(
    model="qiniu/deepseek/deepseek-v3.1-terminus-thinking",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
)

print(response)
```

### Streaming

```python showLineNumbers title="Qiniu Streaming Completion"
import os
import litellm

os.environ["QINIU_API_KEY"] = ""  # your Qiniu AI API key

response = litellm.completion(
    model="qiniu/deepseek-v3",
    messages=[{"role": "user", "content": "Write a short poem about clouds"}],
    stream=True,
)

for chunk in response:
    print(chunk)
```

## Usage - LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export QINIU_API_KEY=""
```

### 2. Start the proxy

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: deepseek-v3
    litellm_params:
      model: qiniu/deepseek-v3
      api_key: os.environ/QINIU_API_KEY

  - model_name: deepseek-thinking
    litellm_params:
      model: qiniu/deepseek/deepseek-v3.1-terminus-thinking
      api_key: os.environ/QINIU_API_KEY

  - model_name: qwen3-32b
    litellm_params:
      model: qiniu/qwen3-32b
      api_key: os.environ/QINIU_API_KEY
```

```bash
litellm --config config.yaml
```

### 3. Call the proxy

```python showLineNumbers title="OpenAI SDK via LiteLLM Proxy"
from openai import OpenAI

client = OpenAI(api_key="sk-1234", base_url="http://0.0.0.0:4000")

response = client.chat.completions.create(
    model="deepseek-v3",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response)
```

## Supported Models

### Anthropic

| Model | Model ID | Context Length |
|-------|----------|----------------|
| Claude Opus 4.5 | `qiniu/claude-4.5-opus` | 200K |
| Claude Sonnet 4.6 | `qiniu/claude-4.6-sonnet` | 1M |
| Claude Opus 4.6 | `qiniu/claude-4.6-opus` | 200K |

### OpenAI

| Model | Model ID | Context Length |
|-------|----------|----------------|
| GPT-5.2 | `qiniu/openai/gpt-5.2` | 400K |
| GPT-5 Mini | `qiniu/openai/gpt-5-mini` | 400K |

### Google

| Model | Model ID | Context Length |
|-------|----------|----------------|
| Gemini 2.5 Flash | `qiniu/dj-gemini-2.5-flash` | 1M |
| Gemini 2.5 Pro | `qiniu/dj-gemini-2.5-pro` | 1M |

### xAI

| Model | Model ID | Context Length |
|-------|----------|----------------|
| Grok 4 Fast | `qiniu/x-ai/grok-4-fast-non-reasoning` | 2M |
| Grok 4 Fast (Reasoning) | `qiniu/x-ai/grok-4-fast-reasoning` | 2M |

### DeepSeek

| Model | Model ID | Context Length |
|-------|----------|----------------|
| DeepSeek V3.1 Terminus Thinking | `qiniu/deepseek/deepseek-v3.1-terminus-thinking` | 128K |
| DeepSeek V3.1 Terminus | `qiniu/deepseek/deepseek-v3.1-terminus` | 128K |
| DeepSeek V3 | `qiniu/deepseek-v3` | 128K |
| DeepSeek V3 0324 | `qiniu/deepseek-v3-0324` | 128K |
| DeepSeek R1 | `qiniu/deepseek-r1` | 80K |
| DeepSeek R1 0528 | `qiniu/deepseek-r1-0528` | 80K |

### Qwen (Aliyun)

| Model | Model ID | Context Length |
|-------|----------|----------------|
| Qwen3 235B A22B | `qiniu/qwen3-235b-a22b` | 128K |
| Qwen3 32B | `qiniu/qwen3-32b` | 40K |
| Qwen3 Max | `qiniu/qwen3-max` | 262K |
| Qwen Turbo | `qiniu/qwen-turbo` | 1M |
| Qwen2.5 VL 72B Instruct | `qiniu/qwen2.5-vl-72b-instruct` | 128K |

### Doubao (ByteDance)

| Model | Model ID | Context Length |
|-------|----------|----------------|
| Doubao 1.5 Pro 32K | `qiniu/doubao-1.5-pro-32k` | 128K |
| Doubao 1.5 Thinking Pro | `qiniu/doubao-1.5-thinking-pro` | 128K |
| Doubao Seed 1.6 | `qiniu/doubao-seed-1.6` | 256K |

### MiniMax

| Model | Model ID | Context Length |
|-------|----------|----------------|
| MiniMax M1 | `qiniu/MiniMax-M1` | 1M |
| MiniMax M2 | `qiniu/minimax/minimax-m2` | 200K |
| MiniMax M2.1 | `qiniu/minimax/minimax-m2.1` | 200K |

### Kimi (Moonshot)

| Model | Model ID | Context Length |
|-------|----------|----------------|
| Kimi K2 | `qiniu/kimi-k2` | 128K |
| Kimi K2 0905 | `qiniu/moonshotai/kimi-k2-0905` | 256K |
| Kimi K2 Thinking | `qiniu/moonshotai/kimi-k2-thinking` | 256K |

### GLM (zAI)

| Model | Model ID | Context Length |
|-------|----------|----------------|
| GLM-4.5 | `qiniu/glm-4.5` | 131K |
| GLM-4.5 Air | `qiniu/glm-4.5-air` | 131K |
| GLM-4.6 | `qiniu/z-ai/glm-4.6` | 200K |
| GLM-4.7 | `qiniu/z-ai/glm-4.7` | 200K |

### Others

| Model | Model ID | Context Length |
|-------|----------|----------------|
| GPT-OSS 120B (Qiniu) | `qiniu/gpt-oss-120b` | 128K |
| GPT-OSS 20B (Qiniu) | `qiniu/gpt-oss-20b` | 128K |
| Xiaomi MiMo V2 Flash | `qiniu/xiaomi/mimo-v2-flash` | 256K |
| Meituan Longcat Flash Lite | `qiniu/meituan/longcat-flash-lite` | 256K |
| Stepfun Step 3.5 Flash | `qiniu/stepfun/step-3.5-flash` | 64K |
| Arcee Trinity Large | `qiniu/arcee-ai/trinity-large-preview` | 128K |
| Arcee Trinity Mini | `qiniu/arcee-ai/trinity-mini` | 128K |

For the full list of 70+ supported models, see the [model pricing page](https://docs.litellm.ai/docs/pricing).

## Additional Resources

- [Qiniu AI Website](https://www.qiniu.com/products/qnai)
- [Qiniu AI API Documentation](https://developer.qiniu.com/ai-service)
