import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Image Generation in Chat Completions, Responses API

This guide covers how to generate images when using the `chat/completions`. Note - if you want this on Responses API please file a Feature Request [here](https://github.com/BerriAI/litellm/issues/new).

:::info

Requires LiteLLM v1.76.1+

:::

Supported Providers:
- Google AI Studio (`gemini`)
- Vertex AI (`vertex_ai/`)

LiteLLM will standardize the `images` response in the assistant message for models that support image generation during chat completions.

```python title="Example response from litellm"
"message": {
    ...
    "content": "Here's the image you requested:",
    "images": [
        {
            "image_url": {
                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
                "detail": "auto"
            },
            "index": 0,
            "type": "image_url"
        }
    ]
}
```

## Quick Start 

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers title="Image generation with chat completion"
from litellm import completion
import os 

os.environ["GEMINI_API_KEY"] = "your-api-key"

response = completion(
    model="gemini/gemini-2.5-flash-image-preview",
    messages=[
        {"role": "user", "content": "Generate an image of a banana wearing a costume that says LiteLLM"}
    ],
)

print(response.choices[0].message.content)  # Text response
print(response.choices[0].message.images)   # List of image objects
```

</TabItem>
<TabItem value="proxy" label="PROXY">

1. Setup config.yaml

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: gemini-image-gen
    litellm_params:
      model: gemini/gemini-2.5-flash-image-preview
      api_key: os.environ/GEMINI_API_KEY
```

2. Run proxy server

```bash showLineNumbers title="Start the proxy"
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

3. Test it!

```bash showLineNumbers title="Make request"
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "gemini-image-gen",
    "messages": [
      {
        "role": "user",
        "content": "Generate an image of a banana wearing a costume that says LiteLLM"
      }
    ]
  }'
```

</TabItem>
</Tabs>

**Expected Response**

```bash
{
    "id": "chatcmpl-3b66124d79a708e10c603496b363574c",
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Here's the image you requested:",
                "role": "assistant",
                "images": [
                    {
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
                            "detail": "auto"
                        },
                        "index": 0,
                        "type": "image_url"
                    }
                ]
            }
        }
    ],
    "created": 1723323084,
    "model": "gemini/gemini-2.5-flash-image-preview",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 12,
        "prompt_tokens": 16,
        "total_tokens": 28
    }
}
```

## Streaming Support

<Tabs>
<TabItem value="sdk" label="SDK">

```python showLineNumbers title="Streaming image generation"
from litellm import completion
import os 

os.environ["GEMINI_API_KEY"] = "your-api-key"

response = completion(
    model="gemini/gemini-2.5-flash-image-preview",
    messages=[
        {"role": "user", "content": "Generate an image of a banana wearing a costume that says LiteLLM"}
    ],
    stream=True,
)

for chunk in response:
    if hasattr(chunk.choices[0].delta, "images") and chunk.choices[0].delta.images is not None:
        print("Generated image:", chunk.choices[0].delta.images[0]["image_url"]["url"])
        break
```

</TabItem>
<TabItem value="proxy" label="PROXY">

```bash showLineNumbers title="Streaming request"
curl http://0.0.0.0:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_KEY" \
  -d '{
    "model": "gemini-image-gen",
    "messages": [
      {
        "role": "user",
        "content": "Generate an image of a banana wearing a costume that says LiteLLM"
      }
    ],
    "stream": true
  }'
```

</TabItem>
</Tabs>

**Expected Streaming Response**

```bash
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1723323084,"model":"gemini/gemini-2.5-flash-image-preview","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1723323084,"model":"gemini/gemini-2.5-flash-image-preview","choices":[{"index":0,"delta":{"content":"Here's the image you requested:"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1723323084,"model":"gemini/gemini-2.5-flash-image-preview","choices":[{"index":0,"delta":{"images":[{"image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...","detail":"auto"},"index":0,"type":"image_url"}]},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1723323084,"model":"gemini/gemini-2.5-flash-image-preview","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Async Support

```python showLineNumbers title="Async image generation"
from litellm import acompletion
import asyncio
import os 

os.environ["GEMINI_API_KEY"] = "your-api-key"

async def generate_image():
    response = await acompletion(
        model="gemini/gemini-2.5-flash-image-preview",
        messages=[
            {"role": "user", "content": "Generate an image of a banana wearing a costume that says LiteLLM"}
        ],
    )
    
    print(response.choices[0].message.content)  # Text response
    print(response.choices[0].message.images)   # List of image objects

    return response

# Run the async function
asyncio.run(generate_image())
```

## Supported Models

| Provider | Model | 
|----------|--------|
| Google AI Studio | `gemini/gemini-2.0-flash-preview-image-generation`, `gemini/gemini-2.5-flash-image-preview` |
| Vertex AI | `vertex_ai/gemini-2.0-flash-preview-image-generation`, `vertex_ai/gemini-2.5-flash-image-preview` |

## Spec

The `images` field in the response follows this structure:

```python
"images": [
    {
        "image_url": {
            "url": "data:image/png;base64,<base64_encoded_image>",
            "detail": "auto"
        },
        "index": 0,
        "type": "image_url"
    }
]
```

- `images` - List[ImageURLListItem]: Array of generated images
  - `image_url` - ImageURLObject: Container for image data
    - `url` - str: Base64 encoded image data in data URI format
    - `detail` - str: Image detail level (always "auto" for generated images)
  - `index` - int: Index of the image in the response
  - `type` - str: Type identifier (always "image_url")

The images are returned as base64-encoded data URIs that can be directly used in HTML `<img>` tags or saved to files.
