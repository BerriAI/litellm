import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Venice AI

## Overview

| Property | Details |
|-------|-------|
| Description | Venice AI provides an OpenAI-compatible API for chat completions with advanced features like web search, character customization, and thinking capabilities. |
| Provider Route on LiteLLM | `venice_ai/` |
| Link to Provider Doc | [Venice AI Documentation ↗](https://docs.venice.ai) |
| Base URL | `https://api.venice.ai/api/v1` |
| Supported Operations | `/chat/completions` |

<br />
<br />

Venice AI is an OpenAI-compatible provider that extends the standard chat completions API with Venice-specific parameters. These parameters are automatically nested in a `venice_parameters` object when making requests through LiteLLM.

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["VENICE_AI_API_KEY"] = ""  # your Venice AI API key
```

## Optional Variables

```python showLineNumbers title="Environment Variables"
os.environ["VENICE_AI_API_BASE"] = ""  # custom API base URL (defaults to https://api.venice.ai/api/v1)
```

Note: See the [Venice AI documentation](https://docs.venice.ai/overview/getting-started) for instructions on obtaining an API key.

## Venice-Specific Parameters

Venice AI supports additional parameters that can be passed directly or nested in a `venice_parameters` object:

- `character_slug` - Use a specific character (e.g., "alan-watts")
- `strip_thinking_response` - Remove thinking content from responses
- `disable_thinking` - Disable thinking mode
- `enable_web_search` - Enable web search ("on", "off", or "auto")
- `enable_web_scraping` - Enable web scraping
- `enable_web_citations` - Include web search citations
- `include_search_results_in_stream` - Include search results in streaming responses
- `return_search_results_as_documents` - Return search results as documents
- `include_venice_system_prompt` - Include Venice's default system prompt

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Venice AI Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["VENICE_AI_API_KEY"] = "your-api-key"

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Basic Venice AI call
response = completion(
    model="venice_ai/llama-3.3-70b", 
    messages=messages
)

print(response)
```

### With Venice Parameters

```python showLineNumbers title="Venice AI with Custom Parameters"
import os
import litellm
from litellm import completion

os.environ["VENICE_AI_API_KEY"] = "your-api-key"

messages = [{"content": "What's the latest news about AI?", "role": "user"}]

# Venice AI call with web search enabled
response = completion(
    model="venice_ai/llama-3.3-70b",
    messages=messages,
    enable_web_search="auto",  # Pass Venice params directly
    enable_web_citations=True,
    temperature=0.7,
    max_tokens=1000
)

print(response)
```

### With Nested Venice Parameters

```python showLineNumbers title="Venice AI with Nested Parameters"
import os
import litellm
from litellm import completion

os.environ["VENICE_AI_API_KEY"] = "your-api-key"

messages = [{"content": "Tell me about philosophy", "role": "user"}]

# Venice AI call with nested venice_parameters
response = completion(
    model="venice_ai/default",
    messages=messages,
    venice_parameters={
        "character_slug": "alan-watts",
        "enable_web_search": "on",
        "include_venice_system_prompt": False
    }
)

print(response)
```

### Streaming

```python showLineNumbers title="Venice AI Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["VENICE_AI_API_KEY"] = "your-api-key"

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Venice AI call with streaming
response = completion(
    model="venice_ai/llama-3.3-70b", 
    messages=messages,
    stream=True,
    enable_web_search="auto"
)

for chunk in response:
    print(chunk)
```

## Usage - LiteLLM Proxy

Add the following to your LiteLLM Proxy configuration file:

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: venice-llama-3.3-70b
    litellm_params:
      model: venice_ai/llama-3.3-70b
      api_key: os.environ/VENICE_AI_API_KEY

  - model_name: venice-qwen-235b
    litellm_params:
      model: venice_ai/qwen-3-235b
      api_key: os.environ/VENICE_AI_API_KEY
      extra_body:
        venice_parameters:
          enable_web_search: "auto"
          enable_web_citations: true
```

Start your LiteLLM Proxy server:

```bash showLineNumbers title="Start LiteLLM Proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="Venice AI via Proxy - Non-streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Non-streaming response
response = client.chat.completions.create(
    model="venice-llama-3.3-70b",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Venice AI via Proxy - Streaming"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"       # Your proxy API key
)

# Streaming response
response = client.chat.completions.create(
    model="venice-llama-3.3-70b",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="Venice AI via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.completion(
    model="litellm_proxy/venice-llama-3.3-70b",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    api_base="http://localhost:4000",
    api_key="your-proxy-api-key"
)

print(response.choices[0].message.content)
```

```python showLineNumbers title="Venice AI via Proxy - LiteLLM SDK Streaming"
import litellm

# Configure LiteLLM to use your proxy with streaming
response = litellm.completion(
    model="litellm_proxy/venice-llama-3.3-70b",
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

```bash showLineNumbers title="Venice AI via Proxy - cURL"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "venice-llama-3.3-70b",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
```

```bash showLineNumbers title="Venice AI via Proxy - cURL Streaming"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-proxy-api-key" \
  -d '{
    "model": "venice-llama-3.3-70b",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "stream": true
  }'
```

</TabItem>
</Tabs>

For more detailed information on using the LiteLLM Proxy, see the [LiteLLM Proxy documentation](../providers/litellm_proxy).

## Available Models

Venice AI supports various models. Some popular options include:

- `llama-3.3-70b` - Llama 3.3 70B
- `qwen-3-235b` - Qwen 3 235B (Venice Large)
- `qwen-2.5-vl` - Qwen 2.5 Vision Language
- `venice-uncensored` - Venice Uncensored
- `default` - Default model with character support

For a complete list of available models and pricing, see the [Venice AI documentation](https://docs.venice.ai/overview/pricing).

## Character Support

Venice AI supports character customization using the `character_slug` parameter:

```python showLineNumbers title="Using Characters"
import os
import litellm
from litellm import completion

os.environ["VENICE_AI_API_KEY"] = "your-api-key"

messages = [{"content": "Tell me about mindfulness", "role": "user"}]

# Use a specific character
response = completion(
    model="venice_ai/default",
    messages=messages,
    character_slug="alan-watts"  # Use Alan Watts character
)

print(response)
```

## Web Search Features

Venice AI supports web search capabilities:

```python showLineNumbers title="Web Search Example"
import os
import litellm
from litellm import completion

os.environ["VENICE_AI_API_KEY"] = "your-api-key"

messages = [{"content": "What's the latest news about AI?", "role": "user"}]

# Enable web search with citations
response = completion(
    model="venice_ai/llama-3.3-70b",
    messages=messages,
    enable_web_search="auto",  # "on", "off", or "auto"
    enable_web_citations=True,
    include_search_results_in_stream=False
)

print(response)
```

## Additional Resources

- [Venice AI Documentation](https://docs.venice.ai)
- [Venice AI API Reference](https://docs.venice.ai/api-reference/endpoint/chat/completions)
- [Venice AI Pricing](https://docs.venice.ai/overview/pricing)
