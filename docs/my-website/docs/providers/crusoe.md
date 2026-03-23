import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Crusoe

## Overview

| Property | Details |
|-------|-------|
| Description | Crusoe Cloud provides GPU-accelerated inference for open-source large language models, optimized for performance and cost efficiency. |
| Provider Route on LiteLLM | `crusoe/` |
| Link to Provider Doc | [Crusoe Managed Inference Documentation ↗](https://docs.crusoecloud.com/managed-inference/overview/index.html) |
| Base URL | `https://managed-inference-api-proxy.crusoecloud.com/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />
<br />

**We support ALL Crusoe models, just set `crusoe/` as a prefix when sending completion requests**

## Available Models

| Model | Description | Context Window |
|-------|-------------|----------------|
| `crusoe/deepseek-ai/DeepSeek-R1-0528` | DeepSeek R1 reasoning model (May 2025) | 163,840 tokens |
| `crusoe/deepseek-ai/DeepSeek-V3-0324` | DeepSeek V3 chat model (March 2025) | 163,840 tokens |
| `crusoe/google/gemma-3-12b-it` | Google Gemma 3 12B instruction-tuned | 131,072 tokens |
| `crusoe/meta-llama/Llama-3.3-70B-Instruct` | Llama 3.3 70B instruction-tuned | 131,072 tokens |
| `crusoe/moonshotai/Kimi-K2-Thinking` | Kimi K2 extended thinking model | 262,144 tokens |
| `crusoe/openai/gpt-oss-120b` | OpenAI 120B open-source model | 131,072 tokens |
| `crusoe/Qwen/Qwen3-235B-A22B-Instruct-2507` | Qwen3 235B MoE instruction-tuned | 262,144 tokens |

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["CRUSOE_API_KEY"] = ""  # your Crusoe API key
```

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Crusoe Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["CRUSOE_API_KEY"] = ""  # your Crusoe API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Crusoe call
response = completion(
    model="crusoe/meta-llama/Llama-3.3-70B-Instruct",
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Crusoe Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["CRUSOE_API_KEY"] = ""  # your Crusoe API key

messages = [{"content": "Write a short story about AI", "role": "user"}]

# Crusoe call with streaming
response = completion(
    model="crusoe/meta-llama/Llama-3.1-70B-Instruct",
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

### Function Calling

```python showLineNumbers title="Crusoe Function Calling"
import os
import litellm
from litellm import completion

os.environ["CRUSOE_API_KEY"] = ""  # your Crusoe API key

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
    model="crusoe/meta-llama/Llama-3.3-70B-Instruct",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

print(response)
```

## Usage - LiteLLM Proxy Server

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: llama-3.3-70b
    litellm_params:
      model: crusoe/meta-llama/Llama-3.3-70B-Instruct
      api_key: os.environ/CRUSOE_API_KEY
  - model_name: deepseek-r1
    litellm_params:
      model: crusoe/deepseek-ai/DeepSeek-R1-0528
      api_key: os.environ/CRUSOE_API_KEY
  - model_name: deepseek-v3
    litellm_params:
      model: crusoe/deepseek-ai/DeepSeek-V3-0324
      api_key: os.environ/CRUSOE_API_KEY
  - model_name: qwen3-235b
    litellm_params:
      model: crusoe/Qwen/Qwen3-235B-A22B-Instruct-2507
      api_key: os.environ/CRUSOE_API_KEY
  - model_name: kimi-k2
    litellm_params:
      model: crusoe/moonshotai/Kimi-K2-Thinking
      api_key: os.environ/CRUSOE_API_KEY
```

## Custom API Base

**Option 1: Environment variable**

```python showLineNumbers title="Custom API Base via env var"
import os
from litellm import completion

os.environ["CRUSOE_API_BASE"] = "https://custom.crusoecloud.com/v1"
os.environ["CRUSOE_API_KEY"] = ""  # your API key

response = completion(
    model="crusoe/meta-llama/Llama-3.3-70B-Instruct",
    messages=[{"content": "Hello!", "role": "user"}],
)
```

**Option 2: Pass directly**

```python showLineNumbers title="Custom API Base via parameter"
from litellm import completion

response = completion(
    model="crusoe/meta-llama/Llama-3.3-70B-Instruct",
    messages=[{"content": "Hello!", "role": "user"}],
    api_base="https://custom.crusoecloud.com/v1",
    api_key="your-api-key",
)
```

## Supported OpenAI Parameters

- `temperature`
- `max_tokens`
- `max_completion_tokens`
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
- `user`
- `logit_bias`
- `logprobs`
- `top_logprobs`
