import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Hyperbolic

## Overview

| Property | Details |
|-------|-------|
| Description | Hyperbolic provides access to the latest models at a fraction of legacy cloud costs, with OpenAI-compatible APIs for LLMs, image generation, and more. |
| Provider Route on LiteLLM | `hyperbolic/` |
| Link to Provider Doc | [Hyperbolic Documentation ↗](https://docs.hyperbolic.xyz) |
| Base URL | `https://api.hyperbolic.xyz/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />
<br />

https://docs.hyperbolic.xyz

**We support ALL Hyperbolic models, just set `hyperbolic/` as a prefix when sending completion requests**

## Available Models

### Language Models

| Model | Description | Context Window | Pricing per 1M tokens |
|-------|-------------|----------------|----------------------|
| `hyperbolic/deepseek-ai/DeepSeek-V3` | DeepSeek V3 - Fast and efficient | 131,072 tokens | $0.25 |
| `hyperbolic/deepseek-ai/DeepSeek-V3-0324` | DeepSeek V3 March 2024 version | 131,072 tokens | $0.25 |
| `hyperbolic/deepseek-ai/DeepSeek-R1` | DeepSeek R1 - Reasoning model | 131,072 tokens | $2.00 |
| `hyperbolic/deepseek-ai/DeepSeek-R1-0528` | DeepSeek R1 May 2028 version | 131,072 tokens | $0.25 |
| `hyperbolic/Qwen/Qwen2.5-72B-Instruct` | Qwen 2.5 72B Instruct | 131,072 tokens | $0.40 |
| `hyperbolic/Qwen/Qwen2.5-Coder-32B-Instruct` | Qwen 2.5 Coder 32B for code generation | 131,072 tokens | $0.20 |
| `hyperbolic/Qwen/Qwen3-235B-A22B` | Qwen 3 235B A22B variant | 131,072 tokens | $2.00 |
| `hyperbolic/Qwen/QwQ-32B` | Qwen QwQ 32B | 131,072 tokens | $0.20 |
| `hyperbolic/meta-llama/Llama-3.3-70B-Instruct` | Llama 3.3 70B Instruct | 131,072 tokens | $0.80 |
| `hyperbolic/meta-llama/Meta-Llama-3.1-405B-Instruct` | Llama 3.1 405B Instruct | 131,072 tokens | $5.00 |
| `hyperbolic/moonshotai/Kimi-K2-Instruct` | Kimi K2 Instruct | 131,072 tokens | $2.00 |

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["HYPERBOLIC_API_KEY"] = ""  # your Hyperbolic API key
```

Get your API key from [Hyperbolic dashboard](https://app.hyperbolic.ai).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Hyperbolic Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["HYPERBOLIC_API_KEY"] = ""  # your Hyperbolic API key

messages = [{"content": "What is the capital of France?", "role": "user"}]

# Hyperbolic call
response = completion(
    model="hyperbolic/Qwen/Qwen2.5-72B-Instruct", 
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Hyperbolic Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["HYPERBOLIC_API_KEY"] = ""  # your Hyperbolic API key

messages = [{"content": "Write a short poem about AI", "role": "user"}]

# Hyperbolic call with streaming
response = completion(
    model="hyperbolic/deepseek-ai/DeepSeek-V3", 
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

### Function Calling

```python showLineNumbers title="Hyperbolic Function Calling"
import os
import litellm
from litellm import completion

os.environ["HYPERBOLIC_API_KEY"] = ""  # your Hyperbolic API key

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
    model="hyperbolic/deepseek-ai/DeepSeek-V3",
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
  - model_name: deepseek-fast
    litellm_params:
      model: hyperbolic/deepseek-ai/DeepSeek-V3
      api_key: os.environ/HYPERBOLIC_API_KEY

  - model_name: qwen-coder
    litellm_params:
      model: hyperbolic/Qwen/Qwen2.5-Coder-32B-Instruct
      api_key: os.environ/HYPERBOLIC_API_KEY

  - model_name: deepseek-reasoning
    litellm_params:
      model: hyperbolic/deepseek-ai/DeepSeek-R1
      api_key: os.environ/HYPERBOLIC_API_KEY
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="Hyperbolic via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="deepseek-fast",
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}]
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Hyperbolic via Proxy - Streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Streaming response
response = client.chat.completions.create(
    model="qwen-coder",
    messages=[{"role": "user", "content": "Write a Python function to sort a list"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="Hyperbolic via Proxy - LiteLLM SDK"
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

```python showLineNumbers title="Hyperbolic via Proxy - LiteLLM SDK Streaming"
import litellm

# Configure LiteLLM to use your proxy with streaming
response = litellm.completion(
    model="litellm_proxy/qwen-coder",
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

```bash showLineNumbers title="Hyperbolic via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "deepseek-fast",
    "messages": [{"role": "user", "content": "What is machine learning?"}]
  }'
```

```bash showLineNumbers title="Hyperbolic via Proxy - cURL Streaming"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "qwen-coder",
    "messages": [{"role": "user", "content": "Write a REST API in Python"}],
    "stream": true
  }'
```

</TabItem>
</Tabs>

For more detailed information on using the LiteLLM Proxy, see the [LiteLLM Proxy documentation](../providers/litellm_proxy).

## Supported OpenAI Parameters

Hyperbolic supports the following OpenAI-compatible parameters:

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

## Advanced Usage

### Custom API Base

If you're using a custom Hyperbolic deployment:

```python showLineNumbers title="Custom API Base"
import litellm

response = litellm.completion(
    model="hyperbolic/deepseek-ai/DeepSeek-V3",
    messages=[{"role": "user", "content": "Hello"}],
    api_base="https://your-custom-hyperbolic-endpoint.com/v1",
    api_key="your-api-key"
)
```

### Rate Limits

Hyperbolic offers different tiers:
- **Basic**: 60 requests per minute (RPM)
- **Pro**: 600 RPM
- **Enterprise**: Custom limits

## Pricing

Hyperbolic offers competitive pay-as-you-go pricing with no hidden fees or long-term commitments. See the model table above for specific pricing per million tokens.

### Precision Options
- **BF16**: Best precision and performance, suitable for tasks where accuracy is critical
- **FP8**: Optimized for efficiency and speed, ideal for high-throughput applications at lower cost

## Additional Resources

- [Hyperbolic Official Documentation](https://docs.hyperbolic.xyz)
- [Hyperbolic Dashboard](https://app.hyperbolic.ai)
- [API Reference](https://docs.hyperbolic.xyz/docs/rest-api)