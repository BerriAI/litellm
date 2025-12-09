import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# image_generation()

Generate images from text prompts using DALL-E, Stable Diffusion, Imagen, and other models.

## Overview

| Feature | Supported |
|---------|-----------|
| Cost Tracking | ✅ |
| Logging | ✅ |
| Fallbacks | ✅ |
| Loadbalancing | ✅ |
| Guardrails | ✅ |

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
[See Reserved Params](https://github.com/BerriAI/litellm/blob/main/litellm/litellm_core_utils/get_supported_openai_params.py)
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

## Example

<Tabs>
<TabItem value="sdk" label="LiteLLM SDK">

```python
from litellm import image_generation
import os

os.environ["OPENAI_API_KEY"] = "sk-..."

response = image_generation(
    prompt="A cute baby sea otter",
    model="dall-e-3",
    size="1024x1024"
)

print(response.data[0].url)
```

</TabItem>
<TabItem value="proxy" label="LiteLLM Proxy">

**1. Setup config.yaml**

```yaml
model_list:
  - model_name: dall-e-3
    litellm_params:
      model: azure/dall-e-3
      api_base: https://my-endpoint.openai.azure.com/
      api_key: os.environ/AZURE_API_KEY
```

**2. Start proxy**

```bash
litellm --config /path/to/config.yaml
```

**3. Make request**

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
