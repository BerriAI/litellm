import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# EmpirioLabs AI

## Overview

| Property | Details |
|-------|-------|
| Description | EmpirioLabs AI hosts open, proprietary, and custom models behind one OpenAI-compatible API with pay-as-you-go pricing across text, image, video, audio, search, and 3D endpoints. |
| Provider Route on LiteLLM | `empiriolabs/` |
| Link to Provider Doc | [EmpirioLabs AI Documentation ↗](https://docs.empiriolabs.ai) |
| Base URL | `https://api.empiriolabs.ai/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage), `/responses` |

<br />
<br />

**We support ALL EmpirioLabs chat models, just set `empiriolabs/` as a prefix when sending completion requests**

## Available Models (selection)

The full live catalog with pricing is at [empiriolabs.ai/models](https://empiriolabs.ai/models). Popular chat models:

| Model | Description | Context Window |
|-------|-------------|----------------|
| `empiriolabs/qwen3-7-max` | Qwen3.7 Max flagship text model for coding, agents, and deep thinking | 1M tokens |
| `empiriolabs/qwen3-7-plus` | Cost-effective Qwen3.7 vision-language model (text, image, video input) | 1M tokens |
| `empiriolabs/deepseek-v4-pro` | DeepSeek V4 flagship MoE (1.6T total / 49B active parameters) | 1M tokens |
| `empiriolabs/deepseek-v4-flash` | Lightweight DeepSeek V4 MoE (284B total / 13B active parameters) | 1M tokens |
| `empiriolabs/glm-5-1` | Zhipu AI long-context reasoning model with tool use | 202K tokens |
| `empiriolabs/kimi-k2-6` | Moonshot Kimi K2.6 multimodal reasoning model | 256K tokens |
| `empiriolabs/minimax-m3` | MiniMax M3 multimodal reasoning for coding and agents | 524K tokens |
| `empiriolabs/gemma-4-26b-a4b` | Google Gemma 4 26B A4B open multimodal model | 256K tokens |

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["EMPIRIOLABS_API_KEY"] = ""  # your EmpirioLabs API key
```

Get an API key from the [EmpirioLabs dashboard](https://platform.empiriolabs.ai/dashboard/api-keys).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="EmpirioLabs Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["EMPIRIOLABS_API_KEY"] = ""  # your EmpirioLabs API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# EmpirioLabs call
response = completion(model="empiriolabs/qwen3-7-plus", messages=messages)

print(response)
```

### Streaming

```python showLineNumbers title="EmpirioLabs Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["EMPIRIOLABS_API_KEY"] = ""  # your EmpirioLabs API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# EmpirioLabs call with streaming
response = completion(
    model="empiriolabs/qwen3-7-plus",
    messages=messages,
    stream=True,
)

for chunk in response:
    print(chunk)
```

## Usage - LiteLLM Proxy

Add the following to your LiteLLM Proxy configuration file:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: qwen3-7-plus
    litellm_params:
      model: empiriolabs/qwen3-7-plus
      api_key: os.environ/EMPIRIOLABS_API_KEY

  - model_name: deepseek-v4-flash
    litellm_params:
      model: empiriolabs/deepseek-v4-flash
      api_key: os.environ/EMPIRIOLABS_API_KEY
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="EmpirioLabs via Proxy"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key",      # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="qwen3-7-plus",
    messages=[{"role": "user", "content": "hello from litellm"}],
)

print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="curl" label="cURL">

```bash showLineNumbers title="EmpirioLabs via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "qwen3-7-plus",
    "messages": [{"role": "user", "content": "hello from litellm"}]
  }'
```

</TabItem>
</Tabs>

## Additional Notes

- Thinking-capable models accept `reasoning_effort` (`none`, `low`, `medium`, `high`, `max`); the gateway maps it onto each model's native thinking controls.
- Per-model parameters, limits, and live pricing are listed at [docs.empiriolabs.ai](https://docs.empiriolabs.ai) and on each model page at [empiriolabs.ai/models](https://empiriolabs.ai/models).
