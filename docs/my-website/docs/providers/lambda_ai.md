import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Lambda AI

## Overview

| Property | Details |
|-------|-------|
| Description | Lambda AI provides access to a wide range of open-source language models through their cloud GPU infrastructure, optimized for inference at scale. |
| Provider Route on LiteLLM | `lambda_ai/` |
| Link to Provider Doc | [Lambda AI API Documentation ↗](https://docs.lambda.ai/api) |
| Base URL | `https://api.lambda.ai/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />
<br />

https://docs.lambda.ai/api

**We support ALL Lambda AI models, just set `lambda_ai/` as a prefix when sending completion requests**

## Available Models

Lambda AI offers a diverse selection of state-of-the-art open-source models:

### Large Language Models

| Model | Description | Context Window |
|-------|-------------|----------------|
| `lambda_ai/llama3.3-70b-instruct-fp8` | Llama 3.3 70B with FP8 quantization | 8,192 tokens |
| `lambda_ai/llama3.1-405b-instruct-fp8` | Llama 3.1 405B with FP8 quantization | 8,192 tokens |
| `lambda_ai/llama3.1-70b-instruct-fp8` | Llama 3.1 70B with FP8 quantization | 8,192 tokens |
| `lambda_ai/llama3.1-8b-instruct` | Llama 3.1 8B instruction-tuned | 8,192 tokens |
| `lambda_ai/llama3.1-nemotron-70b-instruct-fp8` | Llama 3.1 Nemotron 70B | 8,192 tokens |

### DeepSeek Models

| Model | Description | Context Window |
|-------|-------------|----------------|
| `lambda_ai/deepseek-llama3.3-70b` | DeepSeek Llama 3.3 70B | 8,192 tokens |
| `lambda_ai/deepseek-r1-0528` | DeepSeek R1 0528 | 8,192 tokens |
| `lambda_ai/deepseek-r1-671b` | DeepSeek R1 671B | 8,192 tokens |
| `lambda_ai/deepseek-v3-0324` | DeepSeek V3 0324 | 8,192 tokens |

### Hermes Models

| Model | Description | Context Window |
|-------|-------------|----------------|
| `lambda_ai/hermes3-405b` | Hermes 3 405B | 8,192 tokens |
| `lambda_ai/hermes3-70b` | Hermes 3 70B | 8,192 tokens |
| `lambda_ai/hermes3-8b` | Hermes 3 8B | 8,192 tokens |

### Coding Models

| Model | Description | Context Window |
|-------|-------------|----------------|
| `lambda_ai/qwen25-coder-32b-instruct` | Qwen 2.5 Coder 32B | 8,192 tokens |
| `lambda_ai/qwen3-32b-fp8` | Qwen 3 32B with FP8 | 8,192 tokens |

### Vision Models

| Model | Description | Context Window |
|-------|-------------|----------------|
| `lambda_ai/llama3.2-11b-vision-instruct` | Llama 3.2 11B with vision capabilities | 8,192 tokens |

### Specialized Models

| Model | Description | Context Window |
|-------|-------------|----------------|
| `lambda_ai/llama-4-maverick-17b-128e-instruct-fp8` | Llama 4 Maverick with 128k context | 131,072 tokens |
| `lambda_ai/llama-4-scout-17b-16e-instruct` | Llama 4 Scout with 16k context | 16,384 tokens |
| `lambda_ai/lfm-40b` | LFM 40B model | 8,192 tokens |
| `lambda_ai/lfm-7b` | LFM 7B model | 8,192 tokens |

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["LAMBDA_API_KEY"] = ""  # your Lambda AI API key
```

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Lambda AI Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["LAMBDA_API_KEY"] = ""  # your Lambda AI API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Lambda AI call
response = completion(
    model="lambda_ai/llama3.1-8b-instruct", 
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Lambda AI Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["LAMBDA_API_KEY"] = ""  # your Lambda AI API key

messages = [{"content": "Write a short story about AI", "role": "user"}]

# Lambda AI call with streaming
response = completion(
    model="lambda_ai/llama3.1-70b-instruct-fp8", 
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

### Vision/Multimodal Support

The Llama 3.2 Vision model supports image inputs:

```python showLineNumbers title="Lambda AI Vision/Multimodal"
import os
import litellm
from litellm import completion

os.environ["LAMBDA_API_KEY"] = ""  # your Lambda AI API key

messages = [{
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "What's in this image?"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/image.jpg"
            }
        }
    ]
}]

# Lambda AI vision model call
response = completion(
    model="lambda_ai/llama3.2-11b-vision-instruct",
    messages=messages
)

print(response)
```

### Function Calling

Lambda AI models support function calling:

```python showLineNumbers title="Lambda AI Function Calling"
import os
import litellm
from litellm import completion

os.environ["LAMBDA_API_KEY"] = ""  # your Lambda AI API key

# Define tools
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

# Lambda AI call with function calling
response = completion(
    model="lambda_ai/hermes3-70b",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

print(response)
```

## Usage - LiteLLM Proxy Server

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: llama-8b
    litellm_params:
      model: lambda_ai/llama3.1-8b-instruct
      api_key: os.environ/LAMBDA_API_KEY
  - model_name: deepseek-70b
    litellm_params:
      model: lambda_ai/deepseek-llama3.3-70b
      api_key: os.environ/LAMBDA_API_KEY
  - model_name: hermes-405b
    litellm_params:
      model: lambda_ai/hermes3-405b
      api_key: os.environ/LAMBDA_API_KEY
  - model_name: qwen-coder
    litellm_params:
      model: lambda_ai/qwen25-coder-32b-instruct
      api_key: os.environ/LAMBDA_API_KEY
```

## Custom API Base

If you need to use a custom API base URL:

```python showLineNumbers title="Custom API Base"
import os
import litellm
from litellm import completion

# Using environment variable
os.environ["LAMBDA_API_BASE"] = "https://custom.lambda-api.com/v1"
os.environ["LAMBDA_API_KEY"] = ""  # your API key

# Or pass directly
response = completion(
    model="lambda_ai/llama3.1-8b-instruct",
    messages=[{"content": "Hello!", "role": "user"}],
    api_base="https://custom.lambda-api.com/v1",
    api_key="your-api-key"
)
```

## Supported OpenAI Parameters

Lambda AI supports all standard OpenAI parameters since it's fully OpenAI-compatible:

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
- `user`
- `logit_bias`

Example with parameters:

```python showLineNumbers title="Lambda AI with Parameters"
response = completion(
    model="lambda_ai/hermes3-405b",
    messages=[{"content": "Explain quantum computing", "role": "user"}],
    temperature=0.7,
    max_tokens=500,
    top_p=0.9,
    frequency_penalty=0.2,
    presence_penalty=0.1
)
```