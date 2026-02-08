import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# image_generation()

Generate images from text prompts using DALL-E, Stable Diffusion, Imagen, and other models.

## Example

```python
from litellm import image_generation

response = image_generation(
    prompt="A cute baby sea otter",
    model="dall-e-3"
)

print(response.data[0].url)
```

## Signature

```python
def image_generation(
    prompt: str,
    model: Optional[str] = None,
    n: Optional[int] = None,
    quality: Optional[str] = None,
    response_format: Optional[str] = None,
    size: Optional[str] = None,
    style: Optional[str] = None,
    user: Optional[str] = None,
    timeout: int = 600,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    api_version: Optional[str] = None,
    **kwargs
) -> ImageResponse
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | **Required** | Text description of the desired image |
| `model` | `str` | `None` | Model identifier (e.g., `"dall-e-3"`, `"bedrock/stability.stable-diffusion-xl-v0"`) |
| `n` | `int` | `None` | Number of images to generate |
| `quality` | `str` | `None` | Image quality (model-specific: `"hd"`, `"standard"`, `"high"`, `"medium"`, `"low"`) |
| `response_format` | `str` | `None` | `"url"` or `"b64_json"` |
| `size` | `str` | `None` | Image dimensions (e.g., `"1024x1024"`, `"1792x1024"`) |
| `style` | `str` | `None` | Image style (provider-specific) |
| `user` | `str` | `None` | End-user identifier for tracking |
| `timeout` | `int` | `600` | Request timeout in seconds |
| `api_key` | `str` | `None` | API key override |
| `api_base` | `str` | `None` | API base URL override |
| `api_version` | `str` | `None` | API version (required for some Azure models) |

:::info
Any non-OpenAI params will be treated as provider-specific params and sent in the request body.
:::

## Returns

**`ImageResponse`** - OpenAI-compatible response object ([OpenAI Reference](https://platform.openai.com/docs/api-reference/images/object))

```python
ImageResponse(
    created=1703658209,
    data=[
        ImageObject(
            url="https://...",
            b64_json=None,
            revised_prompt="A cute baby sea otter..."
        )
    ],
    usage=ImageUsage(
        prompt_tokens=10,
        completion_tokens=0,
        total_tokens=10
    )
)
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `created` | `int` | Unix timestamp of creation |
| `data` | `List[ImageObject]` | Array of generated images |
| `data[].url` | `str` | URL of the generated image |
| `data[].b64_json` | `str` | Base64-encoded image (if `response_format="b64_json"`) |
| `data[].revised_prompt` | `str` | The prompt actually used (may differ from input) |
| `usage` | `ImageUsage` | Token usage information |

### LiteLLM-Specific Fields

| Field | Type | Description |
|-------|------|-------------|
| `_hidden_params` | `dict` | Contains `model_id`, `api_base`, `response_cost` |
| `_response_headers` | `dict` | Raw HTTP response headers from provider |

## Async Variant

```python
response = await litellm.aimage_generation(
    prompt="A sunset over mountains",
    model="dall-e-3"
)
```

## More Examples

<Tabs>
<TabItem value="multiple" label="Multiple Images">

```python
from litellm import image_generation

response = image_generation(
    prompt="A futuristic city",
    model="dall-e-2",
    n=3,
    size="512x512"
)

for img in response.data:
    print(img.url)
```

</TabItem>
<TabItem value="b64" label="Base64 Response">

```python
from litellm import image_generation

response = image_generation(
    prompt="A mountain landscape",
    model="dall-e-3",
    response_format="b64_json"
)

# Decode and save
import base64
img_data = base64.b64decode(response.data[0].b64_json)
with open("image.png", "wb") as f:
    f.write(img_data)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

```bash
curl -X POST 'http://0.0.0.0:4000/v1/images/generations' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-1234' \
  -d '{
    "model": "dall-e-3",
    "prompt": "A cute baby sea otter",
    "size": "1024x1024"
  }'
```

</TabItem>
</Tabs>

## Supported Providers

See provider-specific documentation for available models and configuration:

- [OpenAI](../providers/openai)
- [Azure OpenAI](../providers/azure/azure)
- [Google AI Studio](../providers/google_ai_studio/image_gen)
- [Vertex AI](../providers/vertex_image)
- [AWS Bedrock](../providers/bedrock)
- [Recraft](../providers/recraft#image-generation)
- [Xinference](../providers/xinference#image-generation)

[**Browse all image models →**](https://models.litellm.ai/)
