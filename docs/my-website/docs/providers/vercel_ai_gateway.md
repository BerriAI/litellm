import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Vercel AI Gateway

## Overview

| Property | Details |
|-------|-------|
| Description | Vercel AI Gateway provides a unified interface to access multiple AI providers through a single endpoint, with built-in caching, rate limiting, and analytics. |
| Provider Route on LiteLLM | `vercel_ai_gateway/` |
| Link to Provider Doc | [Vercel AI Gateway Documentation ↗](https://vercel.com/docs/ai-gateway) |
| Base URL | `https://ai-gateway.vercel.sh/v1` |
| Supported Operations | `/chat/completions`, `/embeddings`, `/models` |

<br />
<br />

https://vercel.com/docs/ai-gateway

**We support ALL models available through Vercel AI Gateway, just set `vercel_ai_gateway/` as a prefix when sending completion requests**

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["VERCEL_AI_GATEWAY_API_KEY"] = ""  # your Vercel AI Gateway API key
# OR
os.environ["VERCEL_OIDC_TOKEN"] = ""  # your Vercel OIDC token for authentication
```

## Optional Variables

```python showLineNumbers title="Environment Variables"
os.environ["VERCEL_SITE_URL"] = ""  # your site url
# OR
os.environ["VERCEL_APP_NAME"] = ""  # your app name
```

Note: see the [Vercel AI Gateway docs](https://vercel.com/docs/ai-gateway#using-the-ai-gateway-with-an-api-key) for instructions on obtaining a key.

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Vercel AI Gateway Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["VERCEL_AI_GATEWAY_API_KEY"] = "your-api-key"

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Vercel AI Gateway call
response = completion(
    model="vercel_ai_gateway/openai/gpt-4o", 
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Vercel AI Gateway Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["VERCEL_AI_GATEWAY_API_KEY"] = "your-api-key"

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Vercel AI Gateway call with streaming
response = completion(
    model="vercel_ai_gateway/openai/gpt-4o",
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

### Embeddings

```python showLineNumbers title="Vercel AI Gateway Embeddings"
import os
from litellm import embedding

os.environ["VERCEL_AI_GATEWAY_API_KEY"] = "your-api-key"

# Vercel AI Gateway embedding call
response = embedding(
    model="vercel_ai_gateway/openai/text-embedding-3-small",
    input="Hello world"
)

print(response.data[0]["embedding"][:5])  # Print first 5 dimensions
```

You can also specify the `dimensions` parameter:

```python showLineNumbers title="Vercel AI Gateway Embeddings with Dimensions"
response = embedding(
    model="vercel_ai_gateway/openai/text-embedding-3-small",
    input=["Hello world", "Goodbye world"],
    dimensions=768
)
```

## Usage - LiteLLM Proxy

Add the following to your LiteLLM Proxy configuration file:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gpt-4o-gateway
    litellm_params:
      model: vercel_ai_gateway/openai/gpt-4o
      api_key: os.environ/VERCEL_AI_GATEWAY_API_KEY

  - model_name: claude-4-sonnet-gateway
    litellm_params:
      model: vercel_ai_gateway/anthropic/claude-4-sonnet
      api_key: os.environ/VERCEL_AI_GATEWAY_API_KEY

  - model_name: text-embedding-3-small-gateway
    litellm_params:
      model: vercel_ai_gateway/openai/text-embedding-3-small
      api_key: os.environ/VERCEL_AI_GATEWAY_API_KEY
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="Vercel AI Gateway via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="gpt-4o-gateway",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Vercel AI Gateway via Proxy - Streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Streaming response
response = client.chat.completions.create(
    model="gpt-4o-gateway",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="Vercel AI Gateway via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.completion(
    model="litellm_proxy/gpt-4o-gateway",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key"
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Vercel AI Gateway via Proxy - LiteLLM SDK Streaming"
import litellm

# Configure LiteLLM to use your proxy with streaming
response = litellm.completion(
    model="litellm_proxy/gpt-4o-gateway",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
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

```bash showLineNumbers title="Vercel AI Gateway via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "gpt-4o-gateway",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
```

```bash showLineNumbers title="Vercel AI Gateway via Proxy - cURL Streaming"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "gpt-4o-gateway",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "stream": true
  }'
```

</TabItem>
</Tabs>

For more detailed information on using the LiteLLM Proxy, see the [LiteLLM Proxy documentation](../providers/litellm_proxy).

## Additional Resources

- [Vercel AI Gateway Documentation](https://vercel.com/docs/ai-gateway)
