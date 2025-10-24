import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# TARS (Tetrate Agent Router Service)

https://router.tetrate.ai

TARS is an AI Gateway-as-a-Service from Tetrate that provides intelligent routing for GenAI applications. It's OpenAI-compatible and routes to multiple LLM providers.

## API Key

```python
# env variable
os.environ['TARS_API_KEY']
```

## Quick Start

```python showLineNumbers
import litellm
import os

# Set your TARS API key.
os.environ["TARS_API_KEY"] = "your-tars-api-key"

# Chat Completions.
response = litellm.completion(
    model="tars/claude-haiku-4-5",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response.choices[0].message.content)

# Vision (Image Analysis).
response = litellm.completion(
    model="tars/gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ]
)
print(response.choices[0].message.content)

# Embeddings.
response = litellm.embedding(
    model="tars/text-embedding-3-small",
    input=["Hello world"]
)
print(response.data[0].embedding)
```

## Features

TARS supports:

- ✅ Chat Completions
- ✅ Embeddings
- ✅ Vision (Multi-modal image analysis)
- ✅ Streaming
- ✅ Async calls
- ✅ Function/Tool calling

## API Configuration

### Required Environment Variables

```bash
export TARS_API_KEY="your-tars-api-key"
```

### Optional Configuration

```bash
# Override the default API base URL.
export TARS_API_BASE="https://api.router.tetrate.ai/v1"
```

## Supported Models

TARS provides access to models from multiple providers including:

### Recommended Models

**Claude Haiku 4.5** (`tars/claude-haiku-4-5`) - Fast, cost-effective model with vision support. Great for most use cases.

**Claude Sonnet 4.5** (`tars/claude-sonnet-4-5`) - Balanced performance and cost for complex tasks.

**GPT-4o** (`tars/gpt-4o`) - OpenAI's flagship multimodal model with strong vision capabilities.

### Vision Models

- OpenAI (GPT-4o, GPT-4o-mini, GPT-4.1, GPT-5, etc.)
- Anthropic (Claude 4.5, Claude 4.5 Haiku, Claude 4, Claude 3.7 Sonnet, etc.)
- xAI (Grok 4, Grok 3, etc.)
- Google (Gemini 2.5 Pro, Gemini 2.0 Flash, etc.)
- DeepSeek, Qwen, and many more

### Chat Models

- OpenAI (GPT-4o, GPT-4o-mini, GPT-4.1, GPT-5, O1, O3, etc.)
- Anthropic (Claude 4.5, Claude 4.5 Haiku, Claude 4, Claude 3.7 Sonnet, Claude 3.5 Haiku, etc.)
- xAI (Grok 4, Grok 3, etc.)
- Google (Gemini 2.5 Pro, Gemini 2.0 Flash, etc.)
- DeepSeek, Qwen, and many more

### Embedding Models

- OpenAI (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)
- Custom embedding models from various providers

To see the full list of available models, visit: https://api.router.tetrate.ai/v1/models

## Usage Examples

### Chat Completions

```python showLineNumbers title="LiteLLM python sdk usage - Non-streaming"
import litellm
import os

os.environ["TARS_API_KEY"] = "your-tars-api-key"

response = litellm.completion(
    model="tars/claude-haiku-4-5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ],
    temperature=0.7,
    max_tokens=500
)
print(response.choices[0].message.content)
```

### Streaming

```python showLineNumbers title="LiteLLM python sdk usage - Streaming"
import litellm
import os

os.environ["TARS_API_KEY"] = "your-tars-api-key"

response = litellm.completion(
    model="tars/gpt-4o",
    messages=[{"role": "user", "content": "Write a short poem"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Async Chat Completions

```python showLineNumbers title="LiteLLM python sdk usage - Async"
import litellm
import asyncio
import os

os.environ["TARS_API_KEY"] = "your-tars-api-key"

async def test_async():
    response = await litellm.acompletion(
        model="tars/claude-haiku-4-5",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

asyncio.run(test_async())
```

### Function/Tool Calling

```python showLineNumbers title="LiteLLM python sdk usage - Function calling"
import litellm
import os

os.environ["TARS_API_KEY"] = "your-tars-api-key"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

response = litellm.completion(
    model="tars/gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    tools=tools
)
print(response.choices[0].message.tool_calls)
```

### Vision (Multi-modal)

```python showLineNumbers title="LiteLLM python sdk usage - Vision"
import litellm
import os

os.environ["TARS_API_KEY"] = "your-tars-api-key"

# Vision with image URL
response = litellm.completion(
    model="tars/gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What do you see in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    }
                }
            ]
        }
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)
```

### Vision with Base64 Image

```python showLineNumbers title="LiteLLM python sdk usage - Vision with Base64"
import litellm
import os

os.environ["TARS_API_KEY"] = "your-tars-api-key"

# Vision with base64 encoded image
response = litellm.completion(
    model="tars/claude-haiku-4-5",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkY+g8AAwEB/6P/4AAAAASUVORK5CYII="
                    }
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

### Vision Function Calling

```python showLineNumbers title="LiteLLM python sdk usage - Vision with Function Calling"
import litellm
import os

os.environ["TARS_API_KEY"] = "your-tars-api-key"

tools = [
    {
        "type": "function",
        "function": {
            "name": "analyze_image",
            "description": "Analyze image content",
            "parameters": {
                "type": "object",
                "properties": {
                    "objects_detected": {"type": "array", "items": {"type": "string"}},
                    "scene_type": {"type": "string"}
                }
            }
        }
    }
]

response = litellm.completion(
    model="tars/gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this image and call the analyze_image function."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ],
    tools=tools
)

if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Tool called: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

### Streaming Vision

```python showLineNumbers title="LiteLLM python sdk usage - Streaming Vision"
import litellm
import os

os.environ["TARS_API_KEY"] = "your-tars-api-key"

response = litellm.completion(
    model="tars/gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in detail."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                }
            ]
        }
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Async Vision

```python showLineNumbers title="LiteLLM python sdk usage - Async Vision"
import litellm
import asyncio
import os

os.environ["TARS_API_KEY"] = "your-tars-api-key"

async def test_vision():
    response = await litellm.acompletion(
        model="tars/claude-haiku-4-5",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.jpg"
                        }
                    }
                ]
            }
        ]
    )
    print(response.choices[0].message.content)

asyncio.run(test_vision())
```

### Embeddings

```python showLineNumbers title="LiteLLM python sdk usage - Embeddings"
import litellm
import os

os.environ["TARS_API_KEY"] = "your-tars-api-key"

response = litellm.embedding(
    model="tars/text-embedding-3-large",
    input=["Hello world", "Goodbye world"]
)

for embedding in response.data:
    print(f"Embedding {embedding.index}: {len(embedding.embedding)} dimensions")
```

## Platform Features

TARS provides intelligent routing and platform features that enable:

- Intelligent routing and load balancing
- Automatic fallback to alternative models
- Cost optimization features
- Performance monitoring
- Unified API access

For detailed pricing and platform information, see: https://router.tetrate.ai/models

## Getting Your API Key

1. Sign up at https://router.tetrate.ai
2. Get $5 free credit with a business email
3. Generate your API key from the dashboard
4. Set the `TARS_API_KEY` environment variable

## Usage with LiteLLM Proxy Server

Here's how to call TARS models with the LiteLLM Proxy Server

### 1. Save key in your environment

```bash
export TARS_API_KEY="your-tars-api-key"
```

### 2. Start the proxy

<Tabs>
<TabItem value="config" label="config.yaml">

```yaml
model_list:
  - model_name: claude-haiku
    litellm_params:
      model: tars/claude-haiku-4-5
      api_key: os.environ/TARS_API_KEY

  - model_name: claude-sonnet
    litellm_params:
      model: tars/claude-sonnet-4-5
      api_key: os.environ/TARS_API_KEY

  - model_name: gpt-4o
    litellm_params:
      model: tars/gpt-4o
      api_key: os.environ/TARS_API_KEY

  - model_name: embeddings
    litellm_params:
      model: tars/text-embedding-3-large
      api_key: os.environ/TARS_API_KEY
```

```bash
litellm --config /path/to/config.yaml
```

</TabItem>
<TabItem value="cli" label="CLI">

```bash
$ litellm --model tars/claude-haiku-4-5

# Server running on http://0.0.0.0:4000
```

</TabItem>
</Tabs>

### 3. Test it

<Tabs>
<TabItem value="Curl" label="Curl Request">

```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "gpt-4o",
      "messages": [
        {
          "role": "user",
          "content": "what llm are you"
        }
      ]
    }
'
```

</TabItem>
<TabItem value="openai" label="OpenAI v1.0.0+">

```python
import openai
client = openai.OpenAI(
    api_key="anything",
    base_url="http://0.0.0.0:4000"
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ]
)

print(response)
```

</TabItem>
<TabItem value="langchain" label="Langchain">

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(
    openai_api_base="http://0.0.0.0:4000",
    model="gpt-4o",
    temperature=0.1
)

messages = [
    SystemMessage(
        content="You are a helpful assistant that im using to make a test request to."
    ),
    HumanMessage(
        content="test from litellm. tell me why it's amazing in 1 sentence"
    ),
]
response = chat(messages)

print(response)
```

</TabItem>
</Tabs>

## Advanced Features

### Cost Tracking

LiteLLM automatically tracks costs for TARS models with a 5% margin added to the base model costs. This margin accounts for TARS routing and platform overhead.

**Note:** Cost tracking is only available for models with pricing information in LiteLLM's model catalog. If a model doesn't have pricing information, no cost will be displayed.

```python showLineNumbers
import litellm
import os

os.environ["TARS_API_KEY"] = "your-tars-api-key"

response = litellm.completion(
    model="tars/gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Cost is automatically calculated with 5% margin.
print(f"Response cost: ${response._hidden_params.get('response_cost', 0):.6f}")
```

### Cost Optimization

TARS automatically routes requests to optimize for cost while maintaining performance:

```python showLineNumbers
import litellm
import os

os.environ["TARS_API_KEY"] = "your-tars-api-key"

# TARS can automatically switch to cheaper models.
response = litellm.completion(
    model="tars/gpt-4o-mini",  # Use cost-effective model.
    messages=[{"role": "user", "content": "Simple question"}]
)
```

### Automatic Fallback

TARS provides automatic fallback to alternative models when primary models are unavailable:

```python showLineNumbers
import litellm
import os

os.environ["TARS_API_KEY"] = "your-tars-api-key"

# If the specified model is down, TARS routes to alternatives.
response = litellm.completion(
    model="tars/claude-haiku-4-5",
    messages=[{"role": "user", "content": "Hello"}],
    # TARS handles fallback automatically.
)
```

## Support

- Dashboard: https://router.tetrate.ai
- Documentation: https://docs.tetrate.io
- Support: Contact through the TARS dashboard

## Troubleshooting

### Authentication Errors

If you get authentication errors:

1. Verify your API key is set correctly: `echo $TARS_API_KEY`
2. Check your key hasn't expired in the dashboard
3. Ensure you have sufficient credits

### Rate Limits

TARS respects the rate limits of underlying providers. If you hit rate limits:

1. Check your usage in the dashboard
2. Consider upgrading to Enterprise plan for higher limits
3. Implement exponential backoff in your code

### Model Not Found

If a model isn't available:

1. Check the latest model list: https://router.tetrate.ai/models
2. Verify the model ID is correct (e.g., `tars/claude-sonnet-4-20250514`)
3. Some models may require specific account permissions
