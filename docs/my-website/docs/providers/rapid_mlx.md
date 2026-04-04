import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Rapid-MLX

Rapid-MLX is an OpenAI-compatible inference server optimized for Apple Silicon (MLX). 2-4x faster than Ollama, with full tool calling, reasoning separation, and prompt caching.

| Property | Details |
|---|---|
| Description | Local LLM inference server for Apple Silicon. [Docs](https://github.com/raullenchai/Rapid-MLX) |
| Provider Route on LiteLLM | `rapid_mlx/` |
| Provider Doc | [Rapid-MLX ↗](https://github.com/raullenchai/Rapid-MLX) |
| Supported Endpoints | `/chat/completions` |

## Quick Start

### Install and start Rapid-MLX

```bash
brew tap raullenchai/rapid-mlx
brew install rapid-mlx
rapid-mlx serve qwen3.5-9b
```

Or install via pip:

```bash
pip install vllm-mlx
rapid-mlx serve qwen3.5-9b
```

## Usage - litellm.completion (calling OpenAI compatible endpoint)

<Tabs>

<TabItem value="sdk" label="SDK">

```python
import litellm

response = litellm.completion(
    model="rapid_mlx/default",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

</TabItem>

<TabItem value="proxy" label="PROXY">

1. Add to config.yaml

```yaml
model_list:
  - model_name: my-model
    litellm_params:
      model: rapid_mlx/default
      api_base: http://localhost:8000/v1
```

2. Start the proxy

```bash
$ litellm --config /path/to/config.yaml
```

3. Send a request

```bash
curl --location 'http://0.0.0.0:4000/chat/completions' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
    "model": "my-model",
    "messages": [
        {
        "role": "user",
        "content": "what llm are you"
        }
    ]
}'
```

</TabItem>

</Tabs>

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RAPID_MLX_API_KEY` | API key (optional, Rapid-MLX does not require auth by default) | `not-needed` |
| `RAPID_MLX_API_BASE` | Server URL | `http://localhost:8000/v1` |

## Supported Models

Any MLX model served by Rapid-MLX works. Use the model name as loaded by the server. Common choices:

- `rapid_mlx/default` - Whatever model is currently loaded
- `rapid_mlx/qwen3.5-9b` - Best small model for general use
- `rapid_mlx/qwen3.5-35b` - Smart and fast
- `rapid_mlx/qwen3.5-122b` - Frontier-level MoE model

## Features

- **Streaming** - Full SSE streaming support
- **Tool calling** - 17 parser formats (Qwen, Hermes, MiniMax, GLM, etc.)
- **Reasoning separation** - Native support for thinking models (Qwen3, DeepSeek-R1)
- **Prompt caching** - KV cache reuse and DeltaNet state snapshots for fast TTFT
- **Multi-Token Prediction** - Speculative decoding for supported models
