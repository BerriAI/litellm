import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Venice.AI

## Overview

| Property | Details |
|-------|-------|
| Description | Venice.AI provides a privacy-first AI API compatible with OpenAI's API specification. |
| Provider Route on LiteLLM | `veniceai/` |
| Link to Provider Doc | [Venice.AI ↗](https://docs.venice.ai/) |
| Base URL | `https://api.venice.ai/api/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />
<br />

https://docs.venice.ai/

**We support ALL Venice.AI models, just set `veniceai/` as a prefix when sending completion requests**

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["VENICE_AI_API_KEY"] = ""  # your Venice.AI API key
```

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Venice.AI Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["VENICE_AI_API_KEY"] = ""  # your Venice.AI API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Venice.AI call
response = completion(
    model="veniceai/<model-name>", 
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Venice.AI Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["VENICE_AI_API_KEY"] = ""  # your Venice.AI API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Venice.AI call with streaming
response = completion(
    model="veniceai/<model-name>", 
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
  - model_name: venice-model
    litellm_params:
      model: veniceai/<model-name>
      api_key: os.environ/VENICE_AI_API_KEY
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="Venice.AI via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="venice-model",
    messages=[{"role": "user", "content": "hello from litellm"}]
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Venice.AI via Proxy - Streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Streaming response
response = client.chat.completions.create(
    model="venice-model",
    messages=[{"role": "user", "content": "hello from litellm"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="Venice.AI via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.completion(
    model="litellm_proxy/venice-model",
    messages=[{"role": "user", "content": "hello from litellm"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key"
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Venice.AI via Proxy - LiteLLM SDK Streaming"
import litellm

# Configure LiteLLM to use your proxy with streaming
response = litellm.completion(
    model="litellm_proxy/venice-model",
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

```bash showLineNumbers title="Venice.AI via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "venice-model",
    "messages": [{"role": "user", "content": "hello from litellm"}]
  }'
```

```bash showLineNumbers title="Venice.AI via Proxy - cURL Streaming"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "venice-model",
    "messages": [{"role": "user", "content": "hello from litellm"}],
    "stream": true
  }'
```

</TabItem>
</Tabs>

For more detailed information on using the LiteLLM Proxy, see the [LiteLLM Proxy documentation](../providers/litellm_proxy).


