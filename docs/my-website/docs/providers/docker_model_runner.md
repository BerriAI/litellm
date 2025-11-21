import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Docker Model Runner

## Overview

| Property | Details |
|-------|-------|
| Description | Docker Model Runner allows you to run large language models locally using Docker Desktop. |
| Provider Route on LiteLLM | `docker_model_runner/` |
| Link to Provider Doc | [Docker Model Runner ↗](https://docs.docker.com/ai/model-runner/) |
| Base URL | `http://localhost:22088` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />
<br />

https://docs.docker.com/ai/model-runner/

**We support ALL Docker Model Runner models, just set `docker_model_runner/` as a prefix when sending completion requests**

## Quick Start

Docker Model Runner is a Docker Desktop feature that lets you run AI models locally. It provides better performance than other local solutions while maintaining OpenAI compatibility.

### Installation

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Enable Docker Model Runner in Docker Desktop settings
3. Download your preferred model through Docker Desktop

## Environment Variables

```python showLineNumbers title="Environment Variables"
os.environ["DOCKER_MODEL_RUNNER_API_BASE"] = "http://localhost:22088"  # Optional - defaults to this
os.environ["DOCKER_MODEL_RUNNER_API_KEY"] = "dummy-key"  # Optional - Docker Model Runner may not require auth for local instances
```

**Note:** Docker Model Runner typically runs locally and may not require authentication. LiteLLM will use a dummy key by default if no key is provided.

## URL Structure

Docker Model Runner uses a unique URL structure:

```
/engines/{engine}/v1/chat/completions
```

Where `{engine}` is typically `llama.cpp` for most models. LiteLLM handles this automatically:

- By default, uses `llama.cpp` as the engine
- You can specify a custom engine in the model name: `docker_model_runner/llama.cpp/model-name`

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Docker Model Runner Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["DOCKER_MODEL_RUNNER_API_BASE"] = "http://localhost:22088"

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Docker Model Runner call
response = completion(
    model="docker_model_runner/llama-3.1", 
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Docker Model Runner Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["DOCKER_MODEL_RUNNER_API_BASE"] = "http://localhost:22088"

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Docker Model Runner call with streaming
response = completion(
    model="docker_model_runner/llama-3.1", 
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

### Custom API Base

```python showLineNumbers title="Custom API Base"
import litellm
from litellm import completion

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Specify custom API base directly
response = completion(
    model="docker_model_runner/llama-3.1",
    messages=messages,
    api_base="http://localhost:22088"
)

print(response)
```

### Specifying Engine Name

```python showLineNumbers title="Custom Engine Name"
import litellm
from litellm import completion

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Specify engine in model name: {engine}/{model}
# This will use /engines/llama.cpp/v1/chat/completions
response = completion(
    model="docker_model_runner/llama.cpp/llama-3.1",
    messages=messages
)

print(response)
```

## Usage - LiteLLM Proxy

Add the following to your LiteLLM Proxy configuration file:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: llama-3.1
    litellm_params:
      model: docker_model_runner/llama-3.1
      api_base: http://localhost:22088

  - model_name: mistral-7b
    litellm_params:
      model: docker_model_runner/mistral-7b
      api_base: http://localhost:22088
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="Docker Model Runner via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="llama-3.1",
    messages=[{"role": "user", "content": "hello from litellm"}]
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Docker Model Runner via Proxy - Streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Streaming response
response = client.chat.completions.create(
    model="llama-3.1",
    messages=[{"role": "user", "content": "hello from litellm"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="Docker Model Runner via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.completion(
    model="litellm_proxy/llama-3.1",
    messages=[{"role": "user", "content": "hello from litellm"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key"
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Docker Model Runner via Proxy - LiteLLM SDK Streaming"
import litellm

# Configure LiteLLM to use your proxy with streaming
response = litellm.completion(
    model="litellm_proxy/llama-3.1",
    messages=[{"role": "user", "content": "hello from litellm"}],
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

```bash showLineNumbers title="Docker Model Runner via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "llama-3.1",
    "messages": [{"role": "user", "content": "hello from litellm"}]
  }'
```

```bash showLineNumbers title="Docker Model Runner via Proxy - cURL Streaming"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "llama-3.1",
    "messages": [{"role": "user", "content": "hello from litellm"}],
    "stream": true
  }'
```

</TabItem>
</Tabs>

For more detailed information on using the LiteLLM Proxy, see the [LiteLLM Proxy documentation](../providers/litellm_proxy).

## Supported OpenAI Parameters

Docker Model Runner is OpenAI-compatible and supports standard parameters:

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
- `logprobs`
- `top_logprobs`

Example with parameters:

```python showLineNumbers title="Docker Model Runner with Parameters"
response = completion(
    model="docker_model_runner/llama-3.1",
    messages=[{"content": "Explain quantum computing", "role": "user"}],
    temperature=0.7,
    max_tokens=500,
    top_p=0.9,
    frequency_penalty=0.2,
    presence_penalty=0.1
)
```

## Benefits of Docker Model Runner

1. **Better Performance**: Docker Model Runner optimizes model inference for better performance compared to other local solutions
2. **Easy Setup**: Integrated with Docker Desktop for simple installation and management
3. **OpenAI Compatible**: Works seamlessly with OpenAI-compatible tools and frameworks
4. **Local Execution**: Run models on your machine without external API calls
5. **Privacy**: Keep your data local without sending it to external services

## API Reference

For detailed API information, see the [Docker Model Runner API Reference](https://docs.docker.com/ai/model-runner/api-reference/).

