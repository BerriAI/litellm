# SaladCloud AI Gateway

## Overview

| Property | Details |
|-------|-------|
| Description | SaladCloud AI Gateway is an OpenAI-compatible LLM API service powered by distributed GPU infrastructure. |
| Provider Route on LiteLLM | `salad_cloud/` |
| Link to Provider Doc | [SaladCloud AI Gateway ↗](https://docs.salad.com/ai-gateway/explanation/overview) |
| Base URL | `https://ai.salad.cloud/v1` |

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["SALAD_API_KEY"] = ""  # your SaladCloud API key
```

Get your SaladCloud API key by signing up at [portal.salad.com](https://portal.salad.com/)

While the service is in beta, sign up at [salad.com/ai-gateway](https://salad.com/ai-gateway).

## Usage - LiteLLM Python SDK

### Non-streaming

```python
import os
import litellm
from litellm import completion

os.environ["SALAD_API_KEY"] = ""  # your SaladCloud API key

messages = [{"content": "What is the capital of France?", "role": "user"}]

response = completion(
    model="salad_cloud/qwen3.5-35b-a3b",
    messages=messages
)

print(response)
```

### Streaming

```python
import os
import litellm
from litellm import completion

os.environ["SALAD_API_KEY"] = ""  # your SaladCloud API key

messages = [{"content": "Write a short poem about distributed computing.", "role": "user"}]

response = completion(
    model="salad_cloud/qwen3.5-35b-a3b",
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

## Usage - LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export SALAD_API_KEY=""
```

### 2. Start the proxy

```yaml
model_list:
  - model_name: qwen3.5-35b
    litellm_params:
      model: salad_cloud/qwen3.5-35b-a3b
      api_key: os.environ/SALAD_API_KEY
  - model_name: qwen3.5-27b
    litellm_params:
      model: salad_cloud/qwen3.5-27b
      api_key: os.environ/SALAD_API_KEY
  - model_name: qwen3.5-9b
    litellm_params:
      model: salad_cloud/qwen3.5-9b
      api_key: os.environ/SALAD_API_KEY
```

## Supported Models

`salad_cloud/qwen3.5-35b-a3b`
`salad_cloud/qwen3.5-27b`
`salad_cloud/qwen3.5-9b`

## Additional Resources

- [SaladCloud AI Gateway Overview](https://docs.salad.com/ai-gateway/explanation/overview)
- [SaladCloud Sign-up](https://salad.com/ai-gateway)
