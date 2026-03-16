import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Venice AI

[Venice AI](https://venice.ai) is a privacy-focused AI platform offering uncensored, open-source models with zero data retention (ZDR). Venice provides access to leading LLMs, image generation, and embedding models through an OpenAI-compatible API.

:::tip

**We support ALL Venice AI models, just set `model=veniceai/<model-id>` as a prefix when sending litellm requests**

:::

## Key Features

- **Privacy-First**: Zero data retention - your prompts and outputs are never stored
- **Uncensored Models**: Access to uncensored versions of popular models
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI SDK
- **Trusted Execution Environment (TEE)**: Optional hardware-level encryption via Phala Network

## Table of Contents

- [API Key](#api-key)
- [Sample Usage](#sample-usage)
- [Streaming](#streaming)
- [Async Usage](#async-usage)
- [Vision Models](#vision-models)
- [JSON Mode / Structured Output](#json-mode--structured-output)
- [Function Calling](#function-calling)
- [Supported Models](#supported-models)
- [Embeddings](#embeddings)
- [Image Generation](#image-generation)
- [OpenAI SDK Compatibility](#openai-sdk-compatibility)
- [Why Venice?](#why-venice)

## API Key

Get your API key from [Venice AI Settings](https://venice.ai/settings/api).

```python
import os
os.environ["VENICE_AI_API_KEY"] = "your-api-key"
```

## Sample Usage

<Tabs>
<TabItem value="sdk" label="SDK">

```python
from litellm import completion
import os

os.environ["VENICE_AI_API_KEY"] = "your-api-key"

response = completion(
    model="veniceai/kimi-k2-5",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Add to config.yaml
```yaml
model_list:
  - model_name: kimi-k2-5
    litellm_params:
      model: veniceai/kimi-k2-5
      api_key: os.environ/VENICE_AI_API_KEY
```

2. Start proxy
```bash
litellm --config /path/to/config.yaml
```

3. Test it!
```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
  -H 'Authorization: Bearer sk-1234' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kimi-k2-5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

</TabItem>
</Tabs>

## Streaming

```python
from litellm import completion
import os

os.environ["VENICE_AI_API_KEY"] = "your-api-key"

response = completion(
    model="veniceai/kimi-k2-5",
    messages=[{"role": "user", "content": "Write a short poem"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## Async Usage

```python
import asyncio
from litellm import acompletion
import os

os.environ["VENICE_AI_API_KEY"] = "your-api-key"

async def main():
    response = await acompletion(
        model="veniceai/kimi-k2-5",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

## Vision Models

Venice supports vision-language models for image understanding.

```python
from litellm import completion
import os

os.environ["VENICE_AI_API_KEY"] = "your-api-key"

response = completion(
    model="veniceai/qwen3-vl-235b-a22b",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ]
)
print(response.choices[0].message.content)
```

## JSON Mode / Structured Output

Venice supports JSON mode for structured responses.

```python
from litellm import completion
import os

os.environ["VENICE_AI_API_KEY"] = "your-api-key"

response = completion(
    model="veniceai/kimi-k2-5",
    messages=[{"role": "user", "content": "List 3 colors as JSON"}],
    response_format={"type": "json_object"}
)
print(response.choices[0].message.content)
```

## Function Calling

Venice AI supports function calling on compatible models.

```python
from litellm import completion
import os

os.environ["VENICE_AI_API_KEY"] = "your-api-key"

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
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = completion(
    model="veniceai/kimi-k2-5",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)
print(response.choices[0].message.tool_calls)
```

## Supported Models

### Chat Models

| Model Name | Model ID | Context | Features |
|------------|----------|---------|----------|
| Kimi K2.5 | `veniceai/kimi-k2-5` | 128K | Coding, Function Calling, Reasoning |
| GLM 4.6 | `veniceai/zai-org-glm-4.6` | 198K | Default model, Function Calling |
| GLM 4.7 | `veniceai/zai-org-glm-4.7` | 198K | Enhanced reasoning |
| GLM 4.7 Flash | `veniceai/zai-org-glm-4.7-flash` | 128K | Fast inference, Reasoning |
| GLM 5 | `veniceai/zai-org-glm-5` | 198K | Best reasoning, Code |
| Qwen 3 4B | `veniceai/qwen3-4b` | 32K | Fast, efficient |
| Qwen 3 235B Thinking | `veniceai/qwen3-235b-a22b-thinking-2507` | 128K | Deep reasoning |
| Qwen 3 VL 235B | `veniceai/qwen3-vl-235b-a22b` | 128K | Vision + Language |
| DeepSeek V3.2 | `veniceai/deepseek-v3.2` | 128K | MoE, efficient |
| MiniMax M2.5 | `veniceai/minimax-m25` | 198K | Coding, Agents, Reasoning |
| Llama 3.3 70B | `veniceai/llama-3.3-70b` | 128K | General purpose |
| Llama 3.2 3B | `veniceai/llama-3.2-3b` | 128K | Lightweight |

### Uncensored Models

Venice offers uncensored models for unrestricted AI interactions:

| Model Name | Model ID | Description |
|------------|----------|-------------|
| Venice Uncensored | `veniceai/venice-uncensored` | Maximum creative freedom |
| Venice Uncensored Role Play | `veniceai/venice-uncensored-role-play` | Optimized for roleplay |
| GLM 4.7 Flash Heretic | `veniceai/olafangensan-glm-4.7-flash-heretic` | Uncensored GLM variant |

### TEE (Trusted Execution Environment) Models

For maximum privacy, Venice offers models running in Phala Network's TEE with hardware-level encryption:

| Model Name | Model ID | Description |
|------------|----------|-------------|
| TEE GLM 5 | `veniceai/tee-glm-5` | Hardware-encrypted GLM 5 |
| TEE DeepSeek V3.2 | `veniceai/e2ee-deepseek-v3-2-p` | Hardware-encrypted DeepSeek |
| TEE Kimi K2.5 | `veniceai/e2ee-kimi-k2-5-p` | Hardware-encrypted Kimi |
| TEE Venice Uncensored | `veniceai/e2ee-venice-uncensored-24b-p` | Hardware-encrypted uncensored model |

## Embeddings

Venice supports text embeddings via the BGE-M3 model.

```python
from litellm import embedding
import os

os.environ["VENICE_AI_API_KEY"] = "your-api-key"

response = embedding(
    model="veniceai/text-embedding-bge-m3",
    input=["Hello, world!", "How are you?"]
)
print(response.data[0].embedding[:5])  # First 5 dimensions
```

| Model Name | Model ID | Dimensions |
|------------|----------|------------|
| BGE-M3 | `veniceai/text-embedding-bge-m3` | 1024 |

## Image Generation

Venice supports image generation via the `/images/generations` endpoint.

```python
from litellm import image_generation
import os

os.environ["VENICE_AI_API_KEY"] = "your-api-key"

response = image_generation(
    model="veniceai/flux-2-pro",
    prompt="A beautiful sunset over the ocean",
    n=1,
    size="1024x1024"
)
print(response.data[0].url)
```

### Supported Image Models

| Model Name | Model ID | Description |
|------------|----------|-------------|
| Flux 2 Pro | `veniceai/flux-2-pro` | High-quality image generation |
| Flux 2 Max | `veniceai/flux-2-max` | Maximum quality Flux model |
| Stable Diffusion 3.5 | `veniceai/venice-sd35` | Venice-tuned SD 3.5 |
| Lustify V7 | `veniceai/lustify-v7` | Uncensored image generation |
| Qwen Image | `veniceai/qwen-image` | Alibaba's image model |

## OpenAI SDK Compatibility

Venice is fully compatible with the OpenAI SDK. Just change the base URL:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-venice-api-key",
    base_url="https://api.venice.ai/api/v1"
)

response = client.chat.completions.create(
    model="kimi-k2-5",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Why Venice?

- **Zero Data Retention**: Your prompts and outputs are never stored or logged
- **No Content Filtering**: Access uncensored models for unrestricted AI
- **Privacy by Design**: No accounts required, crypto payments accepted
- **TEE Support**: Optional hardware-level encryption via Phala Network
- **Competitive Pricing**: Pay only for what you use

## Additional Resources

- [Venice AI Documentation](https://docs.venice.ai)
- [Venice AI API Reference](https://docs.venice.ai/api-reference/api-spec)
- [Get API Key](https://venice.ai/settings/api)
- [Venice AI on X/Twitter](https://x.com/AskVenice)
