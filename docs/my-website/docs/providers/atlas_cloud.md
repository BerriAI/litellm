import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Atlas Cloud

## Overview

| Property | Details |
|-------|-------|
| Description | [Atlas Cloud](https://www.atlascloud.ai/?utm_source=github&utm_medium=link&utm_campaign=litellm) is a full-modal AI inference platform that exposes a single OpenAI-compatible API for LLMs (DeepSeek, Qwen, GLM, Kimi, MiniMax, Claude, …). |
| Provider Route on LiteLLM | `atlas_cloud/` |
| Link to Provider Doc | [Atlas Cloud Documentation ↗](https://www.atlascloud.ai/docs) |
| Base URL | `https://api.atlascloud.ai/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />
<br />

**We support ALL Atlas Cloud chat models, just set `atlas_cloud/` as a prefix when sending completion requests.**

`atlas_cloud/deepseek-ai/deepseek-v4-pro` is a reasoning model — give it enough `max_tokens` (>= 512), otherwise tokens may be spent on the chain-of-thought and the response `content` can come back empty with `finish_reason="length"`.

## Available Models

A representative selection (full chat model list below):

| Model | Context Window |
|-------|----------------|
| `atlas_cloud/deepseek-ai/deepseek-v4-pro` | 1,048,576 tokens |
| `atlas_cloud/anthropic/claude-sonnet-4.6` | 204,800 tokens |
| `atlas_cloud/Qwen/Qwen3-235B-A22B-Instruct-2507` | 131,072 tokens |
| `atlas_cloud/zai-org/GLM-4.6` | 206,848 tokens |

<details>
<summary>All Atlas Cloud chat models (59), synced with <code>atlascloud.ai/models</code></summary>

- Anthropic (Claude): `atlas_cloud/anthropic/claude-haiku-4.5-20251001`, `atlas_cloud/anthropic/claude-opus-4.8`, `atlas_cloud/anthropic/claude-sonnet-4.6`
- OpenAI (GPT): `atlas_cloud/openai/gpt-5.4`, `atlas_cloud/openai/gpt-5.5`
- Google (Gemini): `atlas_cloud/google/gemini-3.1-flash-lite`, `atlas_cloud/google/gemini-3.1-pro-preview`, `atlas_cloud/google/gemini-3.5-flash`
- Alibaba (Qwen): `atlas_cloud/qwen/qwen2.5-7b-instruct`, `atlas_cloud/Qwen/Qwen3-235B-A22B-Instruct-2507`, `atlas_cloud/qwen/qwen3-235b-a22b-thinking-2507`, `atlas_cloud/qwen/qwen3-30b-a3b`, `atlas_cloud/Qwen/Qwen3-30B-A3B-Instruct-2507`, `atlas_cloud/qwen/qwen3-30b-a3b-thinking-2507`, `atlas_cloud/qwen/qwen3-32b`, `atlas_cloud/qwen/qwen3-8b`, `atlas_cloud/Qwen/Qwen3-Coder`, `atlas_cloud/qwen/qwen3-coder-next`, `atlas_cloud/qwen/qwen3-max-2026-01-23`, `atlas_cloud/Qwen/Qwen3-Next-80B-A3B-Instruct`, `atlas_cloud/Qwen/Qwen3-Next-80B-A3B-Thinking`, `atlas_cloud/Qwen/Qwen3-VL-235B-A22B-Instruct`, `atlas_cloud/qwen/qwen3-vl-235b-a22b-thinking`, `atlas_cloud/qwen/qwen3-vl-30b-a3b-instruct`, `atlas_cloud/qwen/qwen3-vl-30b-a3b-thinking`, `atlas_cloud/qwen/qwen3-vl-8b-instruct`, `atlas_cloud/qwen/qwen3.5-122b-a10b`, `atlas_cloud/qwen/qwen3.5-27b`, `atlas_cloud/qwen/qwen3.5-35b-a3b`, `atlas_cloud/qwen/qwen3.5-397b-a17b`, `atlas_cloud/qwen/qwen3.6-35b-a3b`, `atlas_cloud/qwen/qwen3.6-plus`
- DeepSeek: `atlas_cloud/deepseek-ai/deepseek-ocr`, `atlas_cloud/deepseek-ai/deepseek-r1-0528`, `atlas_cloud/deepseek-ai/DeepSeek-V3-0324`, `atlas_cloud/deepseek-ai/DeepSeek-V3.1`, `atlas_cloud/deepseek-ai/DeepSeek-V3.1-Terminus`, `atlas_cloud/deepseek-ai/deepseek-v3.2`, `atlas_cloud/deepseek-ai/DeepSeek-V3.2-Exp`, `atlas_cloud/deepseek-ai/deepseek-v4-flash`, `atlas_cloud/deepseek-ai/deepseek-v4-pro`
- Moonshot (Kimi): `atlas_cloud/moonshotai/Kimi-K2-Instruct`, `atlas_cloud/moonshotai/Kimi-K2-Instruct-0905`, `atlas_cloud/moonshotai/Kimi-K2-Thinking`, `atlas_cloud/moonshotai/kimi-k2.5`, `atlas_cloud/moonshotai/kimi-k2.6`
- Zhipu (GLM): `atlas_cloud/zai-org/GLM-4.6`, `atlas_cloud/zai-org/glm-4.7`, `atlas_cloud/zai-org/glm-5`, `atlas_cloud/zai-org/glm-5-turbo`, `atlas_cloud/zai-org/glm-5.1`, `atlas_cloud/zai-org/glm-5v-turbo`
- MiniMax: `atlas_cloud/MiniMaxAI/MiniMax-M2`, `atlas_cloud/minimaxai/minimax-m2.1`, `atlas_cloud/minimaxai/minimax-m2.5`, `atlas_cloud/minimaxai/minimax-m2.7`
- xAI: `atlas_cloud/xai/grok-4.3`
- Kuaishou (KAT): `atlas_cloud/kwaipilot/kat-coder-pro-v2`
- Other: `atlas_cloud/owl`

</details>

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["ATLASCLOUD_API_KEY"] = ""  # your Atlas Cloud API key
```

## Sample Usage

### Non-streaming

```python showLineNumbers title="Atlas Cloud Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["ATLASCLOUD_API_KEY"] = ""  # your Atlas Cloud API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Atlas Cloud call (deepseek-v4-pro is a reasoning model, give it enough max_tokens)
response = completion(
    model="atlas_cloud/deepseek-ai/deepseek-v4-pro",
    messages=messages,
    max_tokens=512,
)

print(response)
```

### Streaming

```python showLineNumbers title="Atlas Cloud Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["ATLASCLOUD_API_KEY"] = ""  # your Atlas Cloud API key

messages = [{"content": "Write a short story about AI", "role": "user"}]

response = completion(
    model="atlas_cloud/deepseek-ai/deepseek-v4-pro",
    messages=messages,
    max_tokens=512,
    stream=True,
)

for chunk in response:
    print(chunk)
```

### Function Calling

```python showLineNumbers title="Atlas Cloud Function Calling"
import os
from litellm import completion

os.environ["ATLASCLOUD_API_KEY"] = ""  # your Atlas Cloud API key

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                }
            },
            "required": ["location"]
        }
    }
}]

messages = [{"role": "user", "content": "What's the weather in Boston?"}]

response = completion(
    model="atlas_cloud/Qwen/Qwen3-235B-A22B-Instruct-2507",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

print(response)
```

## Usage - LiteLLM Proxy Server

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: deepseek-v4-pro
    litellm_params:
      model: atlas_cloud/deepseek-ai/deepseek-v4-pro
      api_key: os.environ/ATLASCLOUD_API_KEY
  - model_name: claude-sonnet
    litellm_params:
      model: atlas_cloud/anthropic/claude-sonnet-4.6
      api_key: os.environ/ATLASCLOUD_API_KEY
  - model_name: qwen3-235b
    litellm_params:
      model: atlas_cloud/Qwen/Qwen3-235B-A22B-Instruct-2507
      api_key: os.environ/ATLASCLOUD_API_KEY
  - model_name: glm-4.6
    litellm_params:
      model: atlas_cloud/zai-org/GLM-4.6
      api_key: os.environ/ATLASCLOUD_API_KEY
```

## Custom API Base

**Option 1: Environment variable**

```python showLineNumbers title="Custom API Base via env var"
import os
from litellm import completion

os.environ["ATLASCLOUD_API_BASE"] = "https://api.atlascloud.ai/v1"
os.environ["ATLASCLOUD_API_KEY"] = ""  # your API key

response = completion(
    model="atlas_cloud/deepseek-ai/deepseek-v4-pro",
    messages=[{"content": "Hello!", "role": "user"}],
    max_tokens=512,
)
```

**Option 2: Pass directly**

```python showLineNumbers title="Custom API Base via parameter"
from litellm import completion

response = completion(
    model="atlas_cloud/deepseek-ai/deepseek-v4-pro",
    messages=[{"content": "Hello!", "role": "user"}],
    api_base="https://api.atlascloud.ai/v1",
    api_key="your-api-key",
    max_tokens=512,
)
```

## Supported OpenAI Parameters

- `temperature`
- `max_tokens`
- `top_p`
- `frequency_penalty`
- `presence_penalty`
- `stop`
- `n`
- `stream`
- `tools`
- `tool_choice`
- `response_format`
- `seed`
- `logit_bias`
- `logprobs`
- `top_logprobs`
