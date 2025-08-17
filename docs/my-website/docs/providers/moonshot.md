import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Moonshot AI

## Overview

| Property | Details |
|-------|-------|
| Description | Moonshot AI provides large language models including the moonshot-v1 series and kimi models. |
| Provider Route on LiteLLM | `moonshot/` |
| Link to Provider Doc | [Moonshot AI ↗](https://platform.moonshot.ai/) |
| Base URL | `https://api.moonshot.ai/` |
| Supported Operations | [`/chat/completions`](#sample-usage) |

<br />
<br />

https://platform.moonshot.ai/

**We support ALL Moonshot AI models, just set `moonshot/` as a prefix when sending completion requests**

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["MOONSHOT_API_KEY"] = ""  # your Moonshot AI API key
```

**ATTENTION:**

Moonshot AI offers two distinct API endpoints: a global one and a China-specific one.
- Global API Base URL: `https://api.moonshot.ai/v1` (This is the one currently implemented)
- China API Base URL: `https://api.moonshot.cn/v1`

You can overwrite the base url with:

```
os.environ["MOONSHOT_API_BASE"] = "https://api.moonshot.cn/v1"
```

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Moonshot Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["MOONSHOT_API_KEY"] = ""  # your Moonshot AI API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Moonshot call
response = completion(
    model="moonshot/moonshot-v1-8k", 
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Moonshot Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["MOONSHOT_API_KEY"] = ""  # your Moonshot AI API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Moonshot call with streaming
response = completion(
    model="moonshot/moonshot-v1-8k", 
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
  - model_name: moonshot-v1-8k
    litellm_params:
      model: moonshot/moonshot-v1-8k
      api_key: os.environ/MOONSHOT_API_KEY

  - model_name: moonshot-v1-32k
    litellm_params:
      model: moonshot/moonshot-v1-32k
      api_key: os.environ/MOONSHOT_API_KEY

  - model_name: moonshot-v1-128k
    litellm_params:
      model: moonshot/moonshot-v1-128k
      api_key: os.environ/MOONSHOT_API_KEY
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="Moonshot via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[{"role": "user", "content": "hello from litellm"}]
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Moonshot via Proxy - Streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Streaming response
response = client.chat.completions.create(
    model="moonshot-v1-8k",
    messages=[{"role": "user", "content": "hello from litellm"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="Moonshot via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.completion(
    model="litellm_proxy/moonshot-v1-8k",
    messages=[{"role": "user", "content": "hello from litellm"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key"
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Moonshot via Proxy - LiteLLM SDK Streaming"
import litellm

# Configure LiteLLM to use your proxy with streaming
response = litellm.completion(
    model="litellm_proxy/moonshot-v1-8k",
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

```bash showLineNumbers title="Moonshot via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "moonshot-v1-8k",
    "messages": [{"role": "user", "content": "hello from litellm"}]
  }'
```

```bash showLineNumbers title="Moonshot via Proxy - cURL Streaming"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "moonshot-v1-8k",
    "messages": [{"role": "user", "content": "hello from litellm"}],
    "stream": true
  }'
```

</TabItem>
</Tabs>

For more detailed information on using the LiteLLM Proxy, see the [LiteLLM Proxy documentation](../providers/litellm_proxy).

## Moonshot AI Limitations & LiteLLM Handling

LiteLLM automatically handles the following [Moonshot AI limitations](https://platform.moonshot.ai/docs/guide/migrating-from-openai-to-kimi#about-api-compatibility) to provide seamless OpenAI compatibility:

### Temperature Range Limitation
**Limitation**: Moonshot AI only supports temperature range [0, 1] (vs OpenAI's [0, 2])  
**LiteLLM Handling**: Automatically clamps any temperature > 1 to 1

### Temperature + Multiple Outputs Limitation  
**Limitation**: If temperature < 0.3 and n > 1, Moonshot AI raises an exception  
**LiteLLM Handling**: Automatically sets temperature to 0.3 when this condition is detected

### Tool Choice "Required" Not Supported
**Limitation**: Moonshot AI doesn't support `tool_choice="required"`  
**LiteLLM Handling**: Converts this by:
- Adding message: "Please select a tool to handle the current issue."
- Removing the `tool_choice` parameter from the request
