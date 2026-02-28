# Latitude AI

[Latitude](https://latitude.sh) is a GPU cloud provider offering OpenAI-compatible inference APIs for open-source LLMs.

## API Keys

```python
import os
os.environ["LATITUDE_API_KEY"] = "lat_..."
```

Get your API key from [Latitude AI Dashboard](https://ai.latitude.sh).

## Usage

```python
from litellm import completion

response = completion(
    model="latitude/qwen-2.5-7b",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)
```

## Streaming

```python
from litellm import completion

response = completion(
    model="latitude/llama-3.1-8b",
    messages=[{"role": "user", "content": "Write a poem about AI"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

## Supported Models

| Model | Context | Max Output | Features |
|-------|---------|------------|----------|
| `latitude/qwen-2.5-7b` | 131K | 8K | Tools, JSON mode |
| `latitude/llama-3.1-8b` | 128K | 8K | Tools, JSON mode |
| `latitude/qwen3-32b` | 131K | 8K | Tools, JSON mode |
| `latitude/gemma-2-27b` | 8K | 8K | Tools, JSON mode |
| `latitude/deepseek-r1-distill-14b` | 64K | 8K | Tools, JSON mode, Reasoning |
| `latitude/qwen2.5-coder-32b` | 131K | 8K | Tools, JSON mode |
| `latitude/qwen-2.5-vl-7b` | 32K | 8K | Tools, JSON mode, Vision |

## Tool Calling

```python
from litellm import completion

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]

response = completion(
    model="latitude/qwen-2.5-7b",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)
```

## JSON Mode

```python
from litellm import completion

response = completion(
    model="latitude/qwen-2.5-7b",
    messages=[{"role": "user", "content": "List 3 colors as JSON"}],
    response_format={"type": "json_object"}
)
```

## Vision (qwen-2.5-vl-7b)

```python
from litellm import completion

response = completion(
    model="latitude/qwen-2.5-vl-7b",
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
```

## Supported Parameters

| Parameter | Supported |
|-----------|-----------|
| `temperature` | ✅ |
| `max_tokens` | ✅ |
| `top_p` | ✅ |
| `stop` | ✅ |
| `presence_penalty` | ✅ |
| `frequency_penalty` | ✅ |
| `seed` | ✅ |
| `tools` | ✅ |
| `response_format` | ✅ |
| `stream` | ✅ |
