import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Public AI

## Overview

| Property | Details |
|-------|-------|
| Description | Public AI provides access to various large language models including swiss-ai apertus and others. |
| Provider Route on LiteLLM | `publicai/` |
| Link to Provider Doc | [Public AI ↗](https://platform.publicai.co/) |
| Base URL | `https://platform.publicai.co/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />
<br />

https://platform.publicai.co/

**We support ALL Public AI models, just set `publicai/` as a prefix when sending completion requests**

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["PUBLICAI_API_KEY"] = ""  # your Public AI API key
```

You can obtain your API key from [Public AI's platform](https://platform.publicai.co/).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Public AI Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["PUBLICAI_API_KEY"] = ""  # your Public AI API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Public AI call
response = completion(
    model="publicai/swiss-ai/Apertus-17B-v1-Instruct-Q6_K", 
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Public AI Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["PUBLICAI_API_KEY"] = ""  # your Public AI API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Public AI call with streaming
response = completion(
    model="publicai/swiss-ai/Apertus-17B-v1-Instruct-Q6_K", 
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

## Usage - LiteLLM Proxy

Add the following to your LiteLLM Proxy configuration file:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: apertus-17b
    litellm_params:
      model: publicai/swiss-ai/Apertus-17B-v1-Instruct-Q6_K
      api_key: os.environ/PUBLICAI_API_KEY

  # Add more models as needed
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="Public AI via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="apertus-17b",
    messages=[{"role": "user", "content": "hello from litellm"}]
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Public AI via Proxy - Streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Streaming response
response = client.chat.completions.create(
    model="apertus-17b",
    messages=[{"role": "user", "content": "hello from litellm"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="Public AI via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.completion(
    model="litellm_proxy/apertus-17b",
    messages=[{"role": "user", "content": "hello from litellm"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key"
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Public AI via Proxy - LiteLLM SDK Streaming"
import litellm

# Configure LiteLLM to use your proxy with streaming
response = litellm.completion(
    model="litellm_proxy/apertus-17b",
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

```bash showLineNumbers title="Public AI via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "apertus-17b",
    "messages": [{"role": "user", "content": "hello from litellm"}]
  }'
```

```bash showLineNumbers title="Public AI via Proxy - cURL Streaming"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "apertus-17b",
    "messages": [{"role": "user", "content": "hello from litellm"}],
    "stream": true
  }'
```

</TabItem>
</Tabs>

For more detailed information on using the LiteLLM Proxy, see the [LiteLLM Proxy documentation](../providers/litellm_proxy).

## Model Configuration

Public AI recommends the following inference parameters for their models (based on swiss-ai recommendations):

- **Context Window:** 65,536 tokens
- **Max Output Tokens:** 8,192 tokens
- **Temperature:** 0.8 (recommended by swiss-ai)
- **Top-p:** 0.9 (recommended by swiss-ai)

## Rate Limits

To ensure fair usage across all users:

- **During Swiss AI Weeks:** 20 requests per minute

## Support

- **Documentation:** [Public AI Platform Docs](https://platform.publicai.co/docs)
- **Community:** [GitHub](https://github.com/forpublicai)
- **Issues:** Report problems via GitHub Issues
