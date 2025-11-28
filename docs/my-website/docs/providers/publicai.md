import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# PublicAI

## Overview

| Property | Details |
|-------|-------|
| Description | PublicAI provides large language models including essential models like the swiss-ai apertus model. |
| Provider Route on LiteLLM | `publicai/` |
| Link to Provider Doc | [PublicAI ↗](https://platform.publicai.co/) |
| Base URL | `https://platform.publicai.co/` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />
<br />

https://platform.publicai.co/

**We support ALL PublicAI models, just set `publicai/` as a prefix when sending completion requests**

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["PUBLICAI_API_KEY"] = ""  # your PublicAI API key
```

You can overwrite the base url with:

```
os.environ["PUBLICAI_API_BASE"] = "https://platform.publicai.co/v1"
```

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="PublicAI Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["PUBLICAI_API_KEY"] = ""  # your PublicAI API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# PublicAI call
response = completion(
    model="publicai/swiss-ai/apertus-8b-instruct", 
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="PublicAI Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["PUBLICAI_API_KEY"] = ""  # your PublicAI API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# PublicAI call with streaming
response = completion(
    model="publicai/swiss-ai/apertus-8b-instruct", 
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
  - model_name: swiss-ai-apertus-8b
    litellm_params:
      model: publicai/swiss-ai/apertus-8b-instruct
      api_key: os.environ/PUBLICAI_API_KEY

  - model_name: swiss-ai-apertus-70b
    litellm_params:
      model: publicai/swiss-ai/apertus-70b-instruct
      api_key: os.environ/PUBLICAI_API_KEY
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="PublicAI via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="swiss-ai-apertus-8b",
    messages=[{"role": "user", "content": "hello from litellm"}]
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="PublicAI via Proxy - Streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Streaming response
response = client.chat.completions.create(
    model="swiss-ai-apertus-8b",
    messages=[{"role": "user", "content": "hello from litellm"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="PublicAI via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.completion(
    model="litellm_proxy/swiss-ai-apertus-8b",
    messages=[{"role": "user", "content": "hello from litellm"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key"
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="PublicAI via Proxy - LiteLLM SDK Streaming"
import litellm

# Configure LiteLLM to use your proxy with streaming
response = litellm.completion(
    model="litellm_proxy/swiss-ai-apertus-8b",
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

```bash showLineNumbers title="PublicAI via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "swiss-ai-apertus-8b",
    "messages": [{"role": "user", "content": "hello from litellm"}]
  }'
```

```bash showLineNumbers title="PublicAI via Proxy - cURL Streaming"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "swiss-ai-apertus-8b",
    "messages": [{"role": "user", "content": "hello from litellm"}],
    "stream": true
  }'
```

</TabItem>
</Tabs>

For more detailed information on using the LiteLLM Proxy, see the [LiteLLM Proxy documentation](../providers/litellm_proxy).
