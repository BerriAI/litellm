import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Inference

## Overview

| Property | Details |
|-------|-------|
| Description | Inference.net is a global network of data centers serving fast, scalable, pay-per-token APIs. |
| Provider Route on LiteLLM | `inference/` |
| Link to Provider Doc | [Inference Documentation ↗](https://docs.inference.net) |
| Base URL | `https://api.inference.net/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />
<br />

https://docs.inference.net

**We support the following Inference models. Set `inference/` as a prefix when sending completion requests.**

## Available Models

### Language Models

| Model | Description | Context Window | Pricing per 1M tokens |
|-------|-------------|----------------|----------------------|
| `inference/meta-llama/llama-3.1-8b-instruct/fp-16` | Llama 3.1 8B Instruct (FP16) | 16,384 tokens | $0.03 |
| `inference/meta-llama/llama-3.2-3b-instruct/fp-16` | Llama 3.2 3B Instruct (FP16) | 16,384 tokens | $0.02 |
| `inference/mistralai/mistral-nemo-12b-instruct/fp-8` | Mistral Nemo 12B Instruct (FP8) | 16,384 tokens | $0.04 (input), $0.10 (output) |

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["INFERENCE_API_KEY"] = ""  # your Inference API key
```

Get your API key from [Inference dashboard](https://inference.net/dashboard).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Inference Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["INFERENCE_API_KEY"] = ""  # your Inference API key

messages = [{"content": "What is the capital of France?", "role": "user"}]

# Inference call
response = completion(
    model="inference/meta-llama/llama-3.1-8b-instruct/fp-16", 
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Inference Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["INFERENCE_API_KEY"] = ""  # your Inference API key

messages = [{"content": "Write a short poem about AI", "role": "user"}]

# Inference call with streaming
response = completion(
    model="inference/meta-llama/llama-3.1-8b-instruct/fp-16", 
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

### Function Calling

```python showLineNumbers title="Inference Function Calling"
import os
import litellm
from litellm import completion

os.environ["INFERENCE_API_KEY"] = ""  # your Inference API key

tools = [
    {
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
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = completion(
    model="inference/meta-llama/llama-3.1-8b-instruct/fp-16",
    messages=[{"role": "user", "content": "What's the weather like in New York?"}],
    tools=tools,
    tool_choice="auto"
)

print(response)
```

## Usage - LiteLLM Proxy

Add the following to your LiteLLM Proxy configuration file:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: llama-3.1-8b-instruct
    litellm_params:
      model: inference/meta-llama/llama-3.1-8b-instruct/fp-16
      api_key: os.environ/INFERENCE_API_KEY

  - model_name: llama-3.2-3b-instruct
    litellm_params:
      model: inference/meta-llama/llama-3.2-3b-instruct/fp-16
      api_key: os.environ/INFERENCE_API_KEY

  - model_name: mistral-nemo-12b-instruct
    litellm_params:
      model: inference/mistralai/mistral-nemo-12b-instruct/fp-8
      api_key: os.environ/INFERENCE_API_KEY
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="Inference via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}]
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Inference via Proxy - Streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Streaming response
response = client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Write a Python function to sort a list"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="Inference via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.completion(
    model="litellm_proxy/deepseek-fast",
    messages=[{"role": "user", "content": "What are the benefits of renewable energy?"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key"
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Inference via Proxy - LiteLLM SDK Streaming"
import litellm

# Configure LiteLLM to use your proxy with streaming
response = litellm.completion(
    model="litellm_proxy/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Implement a binary search algorithm"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key",
    stream=True
)

for chunk in response:
    if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="curl" label="cURL">

```bash showLineNumbers title="Inference via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "What is machine learning?"}]
  }'
```

```bash showLineNumbers title="Inference via Proxy - cURL Streaming"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Write a REST API in Python"}],
    "stream": true
  }'
```

</TabItem>
</Tabs>

For more detailed information on using the LiteLLM Proxy, see the [LiteLLM Proxy documentation](../providers/litellm_proxy).

## Supported OpenAI Parameters

Inference supports the following OpenAI-compatible parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | array | **Required**. Array of message objects with 'role' and 'content' |
| `model` | string | **Required**. Model ID (e.g., deepseek-ai/DeepSeek-V3, Qwen/Qwen2.5-72B-Instruct) |
| `stream` | boolean | Optional. Enable streaming responses |
| `temperature` | float | Optional. Sampling temperature (0.0 to 2.0) |
| `top_p` | float | Optional. Nucleus sampling parameter |
| `max_tokens` | integer | Optional. Maximum tokens to generate |
| `frequency_penalty` | float | Optional. Penalize frequent tokens |
| `presence_penalty` | float | Optional. Penalize tokens based on presence |
| `stop` | string/array | Optional. Stop sequences |
| `n` | integer | Optional. Number of completions to generate |
| `tools` | array | Optional. List of available tools/functions |
| `tool_choice` | string/object | Optional. Control tool/function calling |
| `response_format` | object | Optional. Response format specification |
| `seed` | integer | Optional. Random seed for reproducibility |
| `user` | string | Optional. User identifier |

### Rate Limits
- 500 RPM for Language Models

## Pricing

Inference.net provides transparent, pay-as-you-go pricing with no hidden fees or long-term contracts. Refer to the model table above for detailed pricing per million tokens.

### Precision Options
- **BF16**: Best precision and performance, suitable for tasks where accuracy is critical
- **FP8**: Optimized for efficiency and speed, ideal for high-throughput applications at lower cost

## Additional Resources

- [Inference Official Documentation](https://docs.inference.net)
- [Inference Dashboard](https://inference.net/dashboard)
- [Quickstart](https://docs.inference.net/quickstart)