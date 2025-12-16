import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# AI Badgr (Budget/Utility, OpenAI-compatible)

## Overview

| Property | Details |
|-------|-------|
| Description | AI Badgr is a budget-friendly, utility-focused provider offering OpenAI-compatible APIs for LLMs. Ideal for cost-conscious applications and high-volume workloads. |
| Provider Route on LiteLLM | `aibadgr/` |
| Link to Provider Doc | [AI Badgr Documentation ↗](https://aibadgr.com) |
| Base URL | `https://aibadgr.com/api/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage), [`/embeddings`](#embeddings), [`/v1/messages`](#claude-compatible) |

<br />
<br />

**We support ALL AI Badgr models, just set `aibadgr/` as a prefix when sending completion requests**

## Available Models

AI Badgr offers three tiers of models optimized for different use cases and budgets:

| Model | Description | Use Case |
|-------|-------------|----------|
| `aibadgr/basic` | Entry-level model for simple tasks | Basic text generation, simple Q&A |
| `aibadgr/normal` | Balanced performance and cost | General-purpose applications |
| `aibadgr/premium` | Best performance tier | Complex reasoning, production workloads |

:::info
OpenAI model names are accepted and automatically mapped to appropriate AI Badgr models.
:::

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["AIBADGR_API_KEY"] = ""  # your AI Badgr API key
```

Get your API key from [AI Badgr dashboard](https://aibadgr.com).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="AI Badgr Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["AIBADGR_API_KEY"] = ""  # your AI Badgr API key

messages = [{"content": "What is the capital of France?", "role": "user"}]

# AI Badgr call with premium tier
response = completion(
    model="aibadgr/premium", 
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="AI Badgr Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["AIBADGR_API_KEY"] = ""  # your AI Badgr API key

messages = [{"content": "Write a short poem about AI", "role": "user"}]

# AI Badgr call with streaming
response = completion(
    model="aibadgr/premium", 
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

### Using Different Tiers

```python showLineNumbers title="AI Badgr Tier Selection"
import os
import litellm
from litellm import completion

os.environ["AIBADGR_API_KEY"] = ""  # your AI Badgr API key

# Basic tier for simple tasks
basic_response = completion(
    model="aibadgr/basic", 
    messages=[{"role": "user", "content": "Say hello"}]
)

# Normal tier for general use
normal_response = completion(
    model="aibadgr/normal", 
    messages=[{"role": "user", "content": "Explain photosynthesis"}]
)

# Premium tier for complex tasks
premium_response = completion(
    model="aibadgr/premium", 
    messages=[{"role": "user", "content": "Write a detailed technical explanation"}]
)
```

### Function Calling

```python showLineNumbers title="AI Badgr Function Calling"
import os
import litellm
from litellm import completion

os.environ["AIBADGR_API_KEY"] = ""  # your AI Badgr API key

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
    model="aibadgr/premium",
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
  - model_name: budget-basic
    litellm_params:
      model: aibadgr/basic
      api_key: os.environ/AIBADGR_API_KEY

  - model_name: budget-normal
    litellm_params:
      model: aibadgr/normal
      api_key: os.environ/AIBADGR_API_KEY

  - model_name: budget-premium
    litellm_params:
      model: aibadgr/premium
      api_key: os.environ/AIBADGR_API_KEY
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="AI Badgr via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="budget-premium",
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}]
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="AI Badgr via Proxy - Streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Streaming response
response = client.chat.completions.create(
    model="budget-premium",
    messages=[{"role": "user", "content": "Write a Python function to sort a list"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="AI Badgr via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.completion(
    model="litellm_proxy/budget-premium",
    messages=[{"role": "user", "content": "What are the benefits of renewable energy?"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key"
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="AI Badgr via Proxy - LiteLLM SDK Streaming"
import litellm

# Configure LiteLLM to use your proxy with streaming
response = litellm.completion(
    model="litellm_proxy/budget-premium",
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

```bash showLineNumbers title="AI Badgr via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "budget-premium",
    "messages": [{"role": "user", "content": "What is machine learning?"}]
  }'
```

```bash showLineNumbers title="AI Badgr via Proxy - cURL Streaming"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "budget-premium",
    "messages": [{"role": "user", "content": "Write a REST API in Python"}],
    "stream": true
  }'
```

</TabItem>
</Tabs>

For more detailed information on using the LiteLLM Proxy, see the [LiteLLM Proxy documentation](../providers/litellm_proxy).

## Embeddings

AI Badgr supports OpenAI-compatible embeddings for RAG, vector search, and semantic similarity tasks.

```python showLineNumbers title="AI Badgr Embeddings"
import os
import litellm
from litellm import embedding

os.environ["AIBADGR_API_KEY"] = ""  # your AI Badgr API key

# Generate embeddings
response = embedding(
    model="aibadgr/text-embedding-ada-002",  # or your embedding model name
    input=["The quick brown fox", "jumps over the lazy dog"]
)

print(response.data[0].embedding)  # Access embedding vector
```

### Supported Embedding Models

AI Badgr supports OpenAI-compatible embedding models. Use the model name that matches your AI Badgr embedding model:

```python
# Example embedding call
response = litellm.embedding(
    model="aibadgr/text-embedding-ada-002",
    input=["Your text here"]
)
```

## Claude-Compatible Endpoint

AI Badgr supports the Anthropic `/v1/messages` endpoint for Claude-compatible API calls.

```python showLineNumbers title="AI Badgr Claude-Compatible Messages"
import os
import litellm

os.environ["AIBADGR_API_KEY"] = ""  # your AI Badgr API key

# Use Claude-compatible messages endpoint
response = litellm.completion(
    model="aibadgr/premium",
    messages=[{"role": "user", "content": "Hello!"}],
    custom_llm_provider="aibadgr"
)

print(response)
```

:::info
The `/v1/messages` endpoint automatically translates between OpenAI and Anthropic message formats, allowing seamless compatibility with tools that use Claude-style APIs.
:::

## Supported OpenAI Parameters

AI Badgr supports the following OpenAI-compatible parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | array | **Required**. Array of message objects with 'role' and 'content' |
| `model` | string | **Required**. Model ID (e.g., basic, normal, premium) |
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

If you need to override the API base URL:

```python showLineNumbers title="Custom API Base"
import litellm

response = litellm.completion(
    model="aibadgr/premium",
    messages=[{"role": "user", "content": "Hello"}],
    api_base="https://custom-aibadgr-endpoint.com/api/v1",
    api_key="your-api-key"
)
```

Or use the environment variable:

```python showLineNumbers title="Custom API Base via Environment"
import os
import litellm

os.environ["AIBADGR_BASE_URL"] = "https://custom-aibadgr-endpoint.com/api/v1"
os.environ["AIBADGR_API_KEY"] = "your-api-key"

response = litellm.completion(
    model="aibadgr/premium",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Best Practices

### Choose the Right Tier

- **basic**: Use for simple, high-volume tasks where speed and cost matter more than sophistication
- **normal**: Good balance for most applications
- **premium**: Best for complex reasoning, production-critical workloads, or when quality is paramount

### Cost Optimization

AI Badgr is designed for budget-conscious applications:

- Start with `basic` tier and upgrade only if needed
- Use `normal` tier for most production workloads
- Reserve `premium` tier for complex reasoning tasks

## Additional Resources

- [AI Badgr Official Documentation](https://aibadgr.com)
- [AI Badgr Dashboard](https://aibadgr.com)
