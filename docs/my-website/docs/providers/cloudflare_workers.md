# Cloudflare Workers AI

LiteLLM supports Cloudflare Workers AI for chat completions, embeddings, and image generation.

https://developers.cloudflare.com/workers-ai/models/

## API Key

```python
import os

os.environ["CLOUDFLARE_API_KEY"] = ""
os.environ["CLOUDFLARE_ACCOUNT_ID"] = ""
```

## Chat Completion

```python
from litellm import completion
import os

os.environ["CLOUDFLARE_API_KEY"] = ""
os.environ["CLOUDFLARE_ACCOUNT_ID"] = ""

response = completion(
    model="cloudflare/@cf/meta/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "hello from litellm"}],
    temperature=0.7,
    max_tokens=256,
)
print(response)
```

### Streaming

```python
from litellm import completion
import os

os.environ["CLOUDFLARE_API_KEY"] = ""
os.environ["CLOUDFLARE_ACCOUNT_ID"] = ""

response = completion(
    model="cloudflare/@cf/meta/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "hello from litellm"}],
    stream=True,
)

for chunk in response:
    print(chunk)
```

### Function Calling

```python
from litellm import completion
import os, json

os.environ["CLOUDFLARE_API_KEY"] = ""
os.environ["CLOUDFLARE_ACCOUNT_ID"] = ""

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
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

response = completion(
    model="cloudflare/@cf/meta/llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "What's the weather in San Francisco?"}],
    tools=tools,
    tool_choice="auto",
)
print(response)
```

### Supported Chat Parameters

| Parameter | Type |
|-----------|------|
| `stream` | `bool` |
| `max_tokens` | `int` |
| `max_completion_tokens` | `int` |
| `temperature` | `float` |
| `top_p` | `float` |
| `frequency_penalty` | `float` |
| `presence_penalty` | `float` |
| `tools` | `list` |
| `tool_choice` | `str` |
| `response_format` | `dict` |
| `stop` | `list` |
| `seed` | `int` |

## Embeddings

```python
from litellm import embedding
import os

os.environ["CLOUDFLARE_API_KEY"] = ""
os.environ["CLOUDFLARE_ACCOUNT_ID"] = ""

response = embedding(
    model="cloudflare/@cf/baai/bge-large-en-v1.5",
    input=["hello from litellm"],
)
print(response)
```

## Image Generation

```python
from litellm import image_generation
import os

os.environ["CLOUDFLARE_API_KEY"] = ""
os.environ["CLOUDFLARE_ACCOUNT_ID"] = ""

response = image_generation(
    model="cloudflare/@cf/black-forest-labs/flux-1-schnell",
    prompt="a beautiful sunset over the ocean",
)
print(response)
```

## Supported Models

All models listed at https://developers.cloudflare.com/workers-ai/models/ are supported. Below are some popular ones:

### Chat Models

| Model Name | Function Call |
|---|---|
| @cf/meta/llama-3.1-8b-instruct | `completion(model="cloudflare/@cf/meta/llama-3.1-8b-instruct", messages)` |
| @cf/meta/llama-3.2-3b-instruct | `completion(model="cloudflare/@cf/meta/llama-3.2-3b-instruct", messages)` |
| @cf/meta/llama-3.3-70b-instruct-fp8-fast | `completion(model="cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast", messages)` |
| @cf/mistral/mistral-small-3.1-24b-instruct | `completion(model="cloudflare/@cf/mistral/mistral-small-3.1-24b-instruct", messages)` |
| @cf/google/gemma-3-12b-it | `completion(model="cloudflare/@cf/google/gemma-3-12b-it", messages)` |
| @cf/qwen/qwen2.5-coder-32b-instruct | `completion(model="cloudflare/@cf/qwen/qwen2.5-coder-32b-instruct", messages)` |
| @cf/deepseek/deepseek-r1-distill-qwen-32b | `completion(model="cloudflare/@cf/deepseek/deepseek-r1-distill-qwen-32b", messages)` |

### Embedding Models

| Model Name | Function Call |
|---|---|
| @cf/baai/bge-large-en-v1.5 | `embedding(model="cloudflare/@cf/baai/bge-large-en-v1.5", input)` |
| @cf/baai/bge-m3 | `embedding(model="cloudflare/@cf/baai/bge-m3", input)` |

### Image Generation Models

| Model Name | Function Call |
|---|---|
| @cf/black-forest-labs/flux-1-schnell | `image_generation(model="cloudflare/@cf/black-forest-labs/flux-1-schnell", prompt)` |
