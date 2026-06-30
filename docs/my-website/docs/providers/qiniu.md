# Qiniu (`qiniu`)

## Overview

| Property | Details |
|-------|-------|
| Description | Qiniu AI (七牛云 AI) provides access to 70+ leading LLMs — including DeepSeek, Qwen, Doubao, MiniMax, Kimi, and GLM — via an OpenAI-compatible API. |
| Provider Route on LiteLLM | `qiniu/` |
| Link to Provider Doc | [Qiniu AI Docs ↗](https://developer.qiniu.com/aitokenapi) |
| Base URL | `https://api.qnaigc.com/v1` |
| Supported Operations | [`/chat/completions`](#usage---litellm-python-sdk), [`/models`](#supported-models) |

<br />

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["QINIU_API_KEY"] = ""  # your Qiniu AI API key
```

Get your API key from the [Qiniu AI Console](https://developer.qiniu.com/aitokenapi/12884/how-to-get-api-key).

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

| Model Name | Function Call |
|------------|---------------|
| `qiniu/openai/gpt-5.4-pro` | `completion('qiniu/openai/gpt-5.4-pro', messages)` |
| `qiniu/openai/gpt-5.4-mini` | `completion('qiniu/openai/gpt-5.4-mini', messages)` |
| `qiniu/openai/gpt-5.3-codex` | `completion('qiniu/openai/gpt-5.3-codex', messages)` |
| `qiniu/gemini-3.1-flash-lite-preview` | `completion('qiniu/gemini-3.1-flash-lite-preview', messages)` |
| `qiniu/gemini-3.1-flash-image-preview` | `completion('qiniu/gemini-3.1-flash-image-preview', messages)` |
| `qiniu/claude-4.6-opus` | `completion('qiniu/claude-4.6-opus', messages)` |
| `qiniu/claude-4.6-sonnet` | `completion('qiniu/claude-4.6-sonnet', messages)` |
| `qiniu/doubao-seed-2.0-pro` | `completion('qiniu/doubao-seed-2.0-pro', messages)` |
| `qiniu/doubao-seed-2.0-lite` | `completion('qiniu/doubao-seed-2.0-lite', messages)` |
| `qiniu/minimax/minimax-m2.5` | `completion('qiniu/minimax/minimax-m2.5', messages)` |
| `qiniu/z-ai/glm-4.7` | `completion('qiniu/z-ai/glm-4.7', messages)` |
| `qiniu/z-ai/glm-5` | `completion('qiniu/z-ai/glm-5', messages)` |
| `qiniu/stepfun/step-3.5-flash` | `completion('qiniu/stepfun/step-3.5-flash', messages)` |
| `qiniu/moonshotai/kimi-k2.5` | `completion('qiniu/moonshotai/kimi-k2.5', messages)` |
| `qiniu/x-ai/grok-4.1-fast-non-reasoning` | `completion('qiniu/x-ai/grok-4.1-fast-non-reasoning', messages)` |
| `qiniu/x-ai/grok-4.1-fast-reasoning` | `completion('qiniu/x-ai/grok-4.1-fast-reasoning', messages)` |
| `qiniu/deepseek/deepseek-v3.2-exp-thinking` | `completion('qiniu/deepseek/deepseek-v3.2-exp-thinking', messages)` |
| `qiniu/qwen3-max` | `completion('qiniu/qwen3-max', messages)` |

For the full list of 70+ supported models, see the [model pricing page](https://www.qiniu.com/ai/models).

## Additional Resources

- [Qiniu AI Website](https://www.qiniu.com/ai/agent)
- [Qiniu AI API Documentation](https://developer.qiniu.com/aitokenapi)
