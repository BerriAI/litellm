# Getting Started

import QuickStart from '../src/components/QuickStart.js'

LiteLLM provides a unified SDK to call 100+ LLM providers using OpenAI-compatible formats.

## Core Functions

| Function | OpenAI Endpoint | Use Case |
|----------|-----------------|----------|
| `completion()` | `/chat/completions` | Chat & text generation |
| `responses()` | `/responses` | Reasoning models (o1, o3) |
| `embedding()` | `/embeddings` | Vector embeddings |
| `image_generation()` | `/images/generations` | Image creation |
| `transcription()` | `/audio/transcriptions` | Speech-to-text |

All functions use the format `provider/model` and return OpenAI-compatible responses. See [Supported Endpoints](./supported_endpoints) for the full list.

## Basic Usage

LiteLLM supports multiple OpenAI endpoints. Below is an example using Chat Completions (`completion()`). See [Responses API](#responses-api) for the `/responses` endpoint.

By default we provide a free $10 community-key to try all providers supported on LiteLLM.

```python
from litellm import completion
import os

## set ENV variables
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

messages = [{"role": "user", "content": "Hello, how are you?"}]

# openai call
response = completion(model="openai/gpt-4o", messages=messages)

# anthropic call
response = completion(model="anthropic/claude-sonnet-4-20250514", messages=messages)

print(response.choices[0].message.content)
```

### Switch Providers with One Line

```python
from litellm import completion
import os

os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

messages = [{"role": "user", "content": "Hello!"}]

# Same code, different providers
response = completion(model="openai/gpt-4o", messages=messages)
response = completion(model="anthropic/claude-sonnet-4-20250514", messages=messages)

# Both return the same OpenAI-compatible format
print(response.choices[0].message.content)
```

**Need a dedicated key?**
Email us @ krrish@berri.ai

Next Steps 👉 [Call all supported models](./providers/)

More details 👉

- [Completion() function details](./completion/)
- [Overview of supported models / providers on LiteLLM](./providers/)
- [Search all models / providers](https://models.litellm.ai/)
- [LiteLLM Proxy Server](./simple_proxy)

## Responses API

For reasoning models (o1, o3) or OpenAI's `/responses` format:

```python
import litellm

response = litellm.responses(
    model="openai/gpt-4o",
    input="Tell me a short story"
)

print(response.output[0].content[0].text)
```

Works with all providers - LiteLLM handles the translation automatically.

More details 👉 [Responses API](./response_api)

## Async

Use `acompletion()` for async operations:

```python
from litellm import acompletion
import asyncio

async def main():
    response = await acompletion(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

## Streaming

Same example from before. Just pass in `stream=True` in the completion args.

```python
from litellm import completion
import os

## set ENV variables
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

messages = [{"role": "user", "content": "Hello, how are you?"}]

# openai call
response = completion(model="openai/gpt-4o", messages=messages, stream=True)

# anthropic call
response = completion(model="anthropic/claude-sonnet-4-20250514", messages=messages, stream=True)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

More details 👉

- [streaming + async](./completion/stream.md)
- [tutorial for streaming Llama2 on TogetherAI](./tutorials/TogetherAI_liteLLM.md)

## Exception Handling

LiteLLM maps exceptions across all supported providers to the OpenAI exceptions. All our exceptions inherit from OpenAI's exception types, so any error-handling you have for that, should work out of the box with LiteLLM.

```python
from openai import OpenAIError
from litellm import completion
import os

os.environ["ANTHROPIC_API_KEY"] = "bad-key"
try:
    # some code
    completion(model="anthropic/claude-sonnet-4-20250514", messages=[{"role": "user", "content": "Hey, how's it going?"}])
except OpenAIError as e:
    print(e)
```

## Logging Observability - Log LLM Input/Output ([Docs](https://docs.litellm.ai/docs/observability/callbacks))

LiteLLM exposes pre defined callbacks to send data to MLflow, Lunary, Langfuse, Helicone, Promptlayer, Traceloop, Slack

```python
from litellm import completion
import litellm
import os

## set env variables for logging tools (API key set up is not required when using MLflow)
os.environ["LUNARY_PUBLIC_KEY"] = "your-lunary-public-key" # get your public key at https://app.lunary.ai/settings
os.environ["HELICONE_API_KEY"] = "your-helicone-key"
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

os.environ["OPENAI_API_KEY"] = "your-api-key"

# set callbacks
litellm.success_callback = ["lunary", "mlflow", "langfuse", "helicone"] # log input/output to MLflow, langfuse, lunary, helicone

#openai call
response = completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```

More details 👉

- [exception mapping](./exception_mapping.md)
- [retries + model fallbacks for completion()](./completion/reliable_completions.md)
- [tutorial for model fallbacks with completion()](./tutorials/fallbacks.md)
