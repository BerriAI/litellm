# Chutes

## Overview

| Property | Details |
|-------|-------|
| Description | Chutes is a cloud-native AI deployment platform that allows you to deploy, run, and scale LLM applications with OpenAI-compatible APIs using pre-built templates for popular frameworks like vLLM and SGLang. |
| Provider Route on LiteLLM | `chutes/` |
| Link to Provider Doc | [Chutes Website ↗](https://chutes.ai) |
| Base URL | `https://llm.chutes.ai/v1/` |
| Supported Operations | [`/chat/completions`](#sample-usage), Embeddings |

<br />

## What is Chutes?

Chutes is a powerful AI deployment and serving platform that provides:
- **Pre-built Templates**: Ready-to-use configurations for vLLM, SGLang, diffusion models, and embeddings
- **OpenAI-Compatible APIs**: Use standard OpenAI SDKs and clients
- **Multi-GPU Scaling**: Support for large models across multiple GPUs
- **Streaming Responses**: Real-time model outputs
- **Custom Configurations**: Override any parameter for your specific needs
- **Performance Optimization**: Pre-configured optimization settings

## Required Variables

```python showLineNumbers title="Environment Variables"
os.environ["CHUTES_API_KEY"] = ""  # your Chutes API key
```

Get your Chutes API key from [chutes.ai](https://chutes.ai).

## Usage - LiteLLM Python SDK

### Non-streaming

```python showLineNumbers title="Chutes Non-streaming Completion"
import os
import litellm
from litellm import completion

os.environ["CHUTES_API_KEY"] = ""  # your Chutes API key

messages = [{"content": "What is the capital of France?", "role": "user"}]

# Chutes call
response = completion(
    model="chutes/model-name",  # Replace with actual model name
    messages=messages
)

print(response)
```

### Streaming

```python showLineNumbers title="Chutes Streaming Completion"
import os
import litellm
from litellm import completion

os.environ["CHUTES_API_KEY"] = ""  # your Chutes API key

messages = [{"content": "Write a short poem about AI", "role": "user"}]

# Chutes call with streaming
response = completion(
    model="chutes/model-name",  # Replace with actual model name
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk)
```

## Usage - LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export CHUTES_API_KEY=""
```

### 2. Start the proxy

```yaml
model_list:
  - model_name: chutes-model
    litellm_params:
      model: chutes/model-name  # Replace with actual model name
      api_key: os.environ/CHUTES_API_KEY
```

## Supported OpenAI Parameters

Chutes supports all standard OpenAI-compatible parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | array | **Required**. Array of message objects with 'role' and 'content' |
| `model` | string | **Required**. Model ID or HuggingFace model identifier |
| `stream` | boolean | Optional. Enable streaming responses |
| `temperature` | float | Optional. Sampling temperature |
| `top_p` | float | Optional. Nucleus sampling parameter |
| `max_tokens` | integer | Optional. Maximum tokens to generate |
| `frequency_penalty` | float | Optional. Penalize frequent tokens |
| `presence_penalty` | float | Optional. Penalize tokens based on presence |
| `stop` | string/array | Optional. Stop sequences |
| `tools` | array | Optional. List of available tools/functions |
| `tool_choice` | string/object | Optional. Control tool/function calling |
| `response_format` | object | Optional. Response format specification |

## Support Frameworks

Chutes provides optimized templates for popular AI frameworks:

### vLLM (High-Performance LLM Serving)
- OpenAI-compatible endpoints
- Multi-GPU scaling support
- Advanced optimization settings
- Best for production workloads

### SGLang (Advanced LLM Serving)
- Structured generation capabilities
- Advanced features and controls
- Custom configuration options
- Best for complex use cases

### Diffusion Models (Image Generation)
- Pre-configured image generation templates
- Optimized settings for best results
- Support for popular diffusion models

### Embedding Models
- Text embedding templates
- Vector search optimization
- Support for popular embedding models

## Authentication

Chutes supports multiple authentication methods:
- API Key via `X-API-Key` header
- Bearer token via `Authorization` header

Example for LiteLLM (uses environment variable):
```python
os.environ["CHUTES_API_KEY"] = "your-api-key"
```

## Performance Optimization

Chutes offers hardware selection and optimization:
- **Small Models (7B-13B)**: 1 GPU with 24GB VRAM
- **Medium Models (30B-70B)**: 4 GPUs with 80GB VRAM each
- **Large Models (100B+)**: 8 GPUs with 140GB+ VRAM each

Engine optimization parameters available for fine-tuning performance.

## Deployment Options

Chutes provides flexible deployment:
- **Quick Setup**: Use pre-built templates for instant deployment
- **Custom Images**: Deploy with custom Docker images
- **Scaling**: Configure max instances and auto-scaling thresholds
- **Hardware**: Choose specific GPU types and configurations

## Additional Resources

- [Chutes Documentation](https://chutes.ai/docs)
- [Chutes Getting Started](https://chutes.ai/docs/getting-started/running-a-chute)
- [Chutes API Reference](https://chutes.ai/docs/sdk-reference)
