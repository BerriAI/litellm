# LlamaGate

## Overview

| Property | Details |
|-------|-------|
| Description | LlamaGate is an OpenAI-compatible API gateway for open-source LLMs with credit-based billing. Access 26+ open-source models including Llama, Mistral, DeepSeek, and Qwen at competitive prices. |
| Provider Route on LiteLLM | `llamagate/` |
| Link to Provider Doc | [LlamaGate Documentation ↗](https://llamagate.dev/docs) |
| Base URL | `https://api.llamagate.dev/v1` |
| Supported Operations | [`/chat/completions`](#sample-usage), [`/embeddings`](#embeddings) |

<br />

## What is LlamaGate?

LlamaGate provides access to open-source LLMs through an OpenAI-compatible API:
- **26+ Open-Source Models**: Llama 3.1/3.2, Mistral, Qwen, DeepSeek R1, and more
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI SDK
- **Vision Models**: Qwen VL, LLaVA, olmOCR, UI-TARS for multimodal tasks
- **Reasoning Models**: DeepSeek R1, OpenThinker for complex problem-solving
- **Code Models**: CodeLlama, DeepSeek Coder, Qwen Coder, StarCoder2
- **Embedding Models**: Nomic, Qwen3 Embedding for RAG and search
- **Competitive Pricing**: $0.02-$0.55 per 1M tokens

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["LLAMAGATE_API_KEY"] = ""  # your LlamaGate API key
```

Get your API key from [llamagate.dev](https://llamagate.dev).

## Supported Models

### General Purpose
| Model | Model ID |
|-------|----------|
| Llama 3.1 8B | `llamagate/llama-3.1-8b` |
| Llama 3.2 3B | `llamagate/llama-3.2-3b` |
| Mistral 7B v0.3 | `llamagate/mistral-7b-v0.3` |
| Qwen 3 8B | `llamagate/qwen3-8b` |
| Dolphin 3 8B | `llamagate/dolphin3-8b` |

### Reasoning Models
| Model | Model ID |
|-------|----------|
| DeepSeek R1 8B | `llamagate/deepseek-r1-8b` |
| DeepSeek R1 Distill Qwen 7B | `llamagate/deepseek-r1-7b-qwen` |
| OpenThinker 7B | `llamagate/openthinker-7b` |

### Code Models
| Model | Model ID |
|-------|----------|
| Qwen 2.5 Coder 7B | `llamagate/qwen2.5-coder-7b` |
| DeepSeek Coder 6.7B | `llamagate/deepseek-coder-6.7b` |
| CodeLlama 7B | `llamagate/codellama-7b` |
| CodeGemma 7B | `llamagate/codegemma-7b` |
| StarCoder2 7B | `llamagate/starcoder2-7b` |

### Vision Models
| Model | Model ID |
|-------|----------|
| Qwen 3 VL 8B | `llamagate/qwen3-vl-8b` |
| LLaVA 1.5 7B | `llamagate/llava-7b` |
| Gemma 3 4B | `llamagate/gemma3-4b` |
| olmOCR 7B | `llamagate/olmocr-7b` |
| UI-TARS 1.5 7B | `llamagate/ui-tars-7b` |

### Embedding Models
| Model | Model ID |
|-------|----------|
| Nomic Embed Text | `llamagate/nomic-embed-text` |
| Qwen 3 Embedding 8B | `llamagate/qwen3-embedding-8b` |
| EmbeddingGemma 300M | `llamagate/embeddinggemma-300m` |

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="LlamaGate Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["LLAMAGATE_API_KEY"] = ""  # your LlamaGate API key

messages = [{"content": "What is the capital of France?", "role": "user"}]

# LlamaGate call
response = completion(
    model="llamagate/llama-3.1-8b",
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="LlamaGate Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["LLAMAGATE_API_KEY"] = ""  # your LlamaGate API key

messages = [{"content": "Write a short poem about AI", "role": "user"}]

# LlamaGate call with streaming
response = completion(
    model="llamagate/llama-3.1-8b",
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

### Vision

```python showLineNumbers title="LlamaGate Vision Completion"
import os
import litellm
from litellm import completion

os.environ["LLAMAGATE_API_KEY"] = ""  # your LlamaGate API key

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }
]

# LlamaGate vision call
response = completion(
    model="llamagate/qwen3-vl-8b",
    messages=messages
)

print(response)
```

### Embeddings

```python showLineNumbers title="LlamaGate Embeddings"
import os
import litellm
from litellm import embedding

os.environ["LLAMAGATE_API_KEY"] = ""  # your LlamaGate API key

# LlamaGate embedding call
response = embedding(
    model="llamagate/nomic-embed-text",
    input=["Hello world", "How are you?"]
)

print(response)
```

## Usage - LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export LLAMAGATE_API_KEY=""
```

### 2. Start the proxy

```yaml
model_list:
  - model_name: llama-3.1-8b
    litellm_params:
      model: llamagate/llama-3.1-8b
      api_key: os.environ/LLAMAGATE_API_KEY
  - model_name: deepseek-r1
    litellm_params:
      model: llamagate/deepseek-r1-8b
      api_key: os.environ/LLAMAGATE_API_KEY
  - model_name: qwen-coder
    litellm_params:
      model: llamagate/qwen2.5-coder-7b
      api_key: os.environ/LLAMAGATE_API_KEY
```

## Supported OpenAI Parameters

LlamaGate supports all standard OpenAI-compatible parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | array | **Required**. Array of message objects with 'role' and 'content' |
| `model` | string | **Required**. Model ID |
| `stream` | boolean | Optional. Enable streaming responses |
| `temperature` | float | Optional. Sampling temperature (0-2) |
| `top_p` | float | Optional. Nucleus sampling parameter |
| `max_tokens` | integer | Optional. Maximum tokens to generate |
| `frequency_penalty` | float | Optional. Penalize frequent tokens |
| `presence_penalty` | float | Optional. Penalize tokens based on presence |
| `stop` | string/array | Optional. Stop sequences |
| `tools` | array | Optional. List of available tools/functions |
| `tool_choice` | string/object | Optional. Control tool/function calling |
| `response_format` | object | Optional. JSON mode or JSON schema |

## Pricing

LlamaGate offers competitive per-token pricing:

| Model Category | Input (per 1M) | Output (per 1M) |
|----------------|----------------|-----------------|
| Embeddings | $0.02 | - |
| Small (3-4B) | $0.03-$0.04 | $0.08 |
| Medium (7-8B) | $0.03-$0.15 | $0.05-$0.55 |
| Code Models | $0.06-$0.10 | $0.12-$0.20 |
| Reasoning | $0.08-$0.10 | $0.15-$0.20 |

## Additional Resources

- [LlamaGate Documentation](https://llamagate.dev/docs)
- [LlamaGate Pricing](https://llamagate.dev/pricing)
- [LlamaGate API Reference](https://llamagate.dev/docs/api)
