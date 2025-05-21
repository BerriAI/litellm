import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Meta Llama

| Property | Details |
|-------|-------|
| Description | Meta's Llama API provides access to Meta's family of large language models. |
| Provider Route on LiteLLM | `meta_llama/` |
| Supported Endpoints | `/chat/completions`, `/completions`, `/responses` |
| API Reference | [Llama API Reference ↗](https://llama.developer.meta.com?utm_source=partner-litellm&utm_medium=website) |

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["LLAMA_API_KEY"] = ""  # your Meta Llama API key
```

## Supported Models

:::info
All models listed here https://llama.developer.meta.com/docs/models/ are supported. We actively maintain the list of models, token window, etc. [here](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json).

:::


| Model ID | Input context length | Output context length | Input Modalities | Output Modalities |
| --- | --- | --- | --- | --- |
| `Llama-4-Scout-17B-16E-Instruct-FP8` | 128k | 4028 | Text, Image | Text |
| `Llama-4-Maverick-17B-128E-Instruct-FP8` | 128k | 4028 | Text, Image | Text |
| `Llama-3.3-70B-Instruct` | 128k | 4028 | Text | Text |
| `Llama-3.3-8B-Instruct` | 128k | 4028 | Text | Text |

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Meta Llama Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["LLAMA_API_KEY"] = ""  # your Meta Llama API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Meta Llama call
response = completion(model="meta_llama/Llama-3.3-70B-Instruct", messages=messages)
```

### Streaming

```python showLineNumbers title="Meta Llama Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["LLAMA_API_KEY"] = ""  # your Meta Llama API key

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Meta Llama call with streaming
response = completion(
    model="meta_llama/Llama-3.3-70B-Instruct",
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
  - model_name: meta_llama/Llama-3.3-70B-Instruct
    litellm_params:
      model: meta_llama/Llama-3.3-70B-Instruct
      api_key: os.environ/LLAMA_API_KEY

  - model_name: meta_llama/Llama-3.3-8B-Instruct
    litellm_params:
      model: meta_llama/Llama-3.3-8B-Instruct
      api_key: os.environ/LLAMA_API_KEY
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="Meta Llama via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="meta_llama/Llama-3.3-70B-Instruct",
    messages=[{"role": "user", "content": "Write a short poem about AI."}]
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Meta Llama via Proxy - Streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Streaming response
response = client.chat.completions.create(
    model="meta_llama/Llama-3.3-70B-Instruct",
    messages=[{"role": "user", "content": "Write a short poem about AI."}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="Meta Llama via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.completion(
    model="litellm_proxy/meta_llama/Llama-3.3-70B-Instruct",
    messages=[{"role": "user", "content": "Write a short poem about AI."}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key"
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Meta Llama via Proxy - LiteLLM SDK Streaming"
import litellm

# Configure LiteLLM to use your proxy with streaming
response = litellm.completion(
    model="litellm_proxy/meta_llama/Llama-3.3-70B-Instruct",
    messages=[{"role": "user", "content": "Write a short poem about AI."}],
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

```bash showLineNumbers title="Meta Llama via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "meta_llama/Llama-3.3-70B-Instruct",
    "messages": [{"role": "user", "content": "Write a short poem about AI."}]
  }'
```

```bash showLineNumbers title="Meta Llama via Proxy - cURL Streaming"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "meta_llama/Llama-3.3-70B-Instruct",
    "messages": [{"role": "user", "content": "Write a short poem about AI."}],
    "stream": true
  }'
```

</TabItem>
</Tabs>

For more detailed information on using the LiteLLM Proxy, see the [LiteLLM Proxy documentation](../providers/litellm_proxy).
