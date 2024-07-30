# Not Diamond

[Not Diamond](https://www.notdiamond.ai/) automatically determines which model is best-suited to respond to any query, for improved quality, and reduced cost and latency. LiteLLM supports automatic model routing to all [models supported by Not Diamond](https://notdiamond.readme.io/docs/llm-models).

## Pre-Requisites

`pip install litellm`

## Required API Keys

Follow this [link](https://app.notdiamond.ai/keys) to create your Not Diamond API key. Additionally, provide API keys for all providers that you want to route between.

```python
os.environ["NOTDIAMOND_API_KEY"] = "NOTDIAMOND_API_KEY"  # NOTDIAMOND_API_KEY
# provide API keys for providers
```

:::info

Not Diamond's API will fail requests if no `llm_providers` are defined. Please provide your desired LLM providers and models to route between.
:::

## Usage

```python
import os
from litellm import completion

os.environ["NOTDIAMOND_API_KEY"] = "NOTDIAMOND_API_KEY"
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["ANTHROPIC_API_KEY"] = "ANTHROPIC_API_KEY"

messages = [{"role": "user", "content": "Hey! How's it going?"}]

llm_providers = [
    {
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307"
    },
    {
        "provider": "openai",
        "model": "gpt-4-turbo"
    }
]

response = completion(
  model="notdiamond/notdiamond",
  messages=messages,
  llm_providers=llm_providers,
  # tradeoff="cost"    # optional parameter to optimize for cost (tradeoff="cost") or latency (tradeoff="latency") without degrading quality
)
print(response)
```

## Usage - Streaming

Set `stream=True` when calling completion to stream responses.

```python
import os
from litellm import completion

os.environ["NOTDIAMOND_API_KEY"] = "NOTDIAMOND_API_KEY"
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["ANTHROPIC_API_KEY"] = "ANTHROPIC_API_KEY"

messages = [{"role": "user", "content": "Hey! How's it going?"}]

llm_providers = [
    {
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307"
    },
    {
        "provider": "openai",
        "model": "gpt-4-turbo"
    }
]

response = completion(
  model="notdiamond/notdiamond",
  messages=messages,
  llm_providers=llm_providers,
  stream=True
)
for chunk in response:
    print(chunk)
```

## Usage - Function Calling

Function calling is also supported through the `tools` parameter for [models that support function calling](https://notdiamond.readme.io/docs/function-calling).

```python
import os
from litellm import completion

os.environ["NOTDIAMOND_API_KEY"] = "NOTDIAMOND_API_KEY"
os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
os.environ["ANTHROPIC_API_KEY"] = "ANTHROPIC_API_KEY"

messages = [{"role": "user", "content": "What is 2 + 5?"}]

llm_providers = [
    {
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307"
    },
    {
        "provider": "openai",
        "model": "gpt-4-turbo"
    }
]

tools = [
    {
        "type": "function",
        "function": {
        "name": "add",
        "description": "Adds a and b.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        },
    },
]

response = completion(
  model="notdiamond/notdiamond",
  messages=messages,
  llm_providers=llm_providers,
  tools=tools
)
print(response)
```
