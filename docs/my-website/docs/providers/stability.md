# Stability AI
https://stability.ai/

## Overview

| Property | Details |
|-------|-------|
| Description | Stability AI creates open AI models for image, video, audio, and 3D generation. Known for Stable Diffusion. |
| Provider Route on LiteLLM | `stability/` |
| Link to Provider Doc | [Stability AI API ↗](https://platform.stability.ai/docs/api-reference) |
| Supported Operations | [`/images/generations`](#image-generation) |

LiteLLM supports Stability AI Image Generation calls via the Stability AI REST API (not via Bedrock).

## API Key

```python
# env variable
os.environ['STABILITY_API_KEY'] = "your-api-key"
```

Get your API key from the [Stability AI Platform](https://platform.stability.ai/).

## Image Generation

### Usage - LiteLLM Python SDK

```python showLineNumbers
from litellm import image_generation
import os

os.environ['STABILITY_API_KEY'] = "your-api-key"

# Stability AI image generation call
response = image_generation(
    model="stability/sd3.5-large",
    prompt="A beautiful sunset over a calm ocean",
)
print(response)
```

### Usage - LiteLLM Proxy Server

#### 1. Setup config.yaml

```yaml showLineNumbers
model_list:
  - model_name: sd3
    litellm_params:
      model: stability/sd3.5-large
      api_key: os.environ/STABILITY_API_KEY
    model_info:
      mode: image_generation

general_settings:
  master_key: sk-1234
```

#### 2. Start the proxy

```bash showLineNumbers
litellm --config config.yaml

# RUNNING on http://0.0.0.0:4000
```

#### 3. Test it

```bash showLineNumbers
curl --location 'http://0.0.0.0:4000/v1/images/generations' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234' \
--data '{
    "model": "sd3",
    "prompt": "A beautiful sunset over a calm ocean"
}'
```

### Advanced Usage - With Additional Parameters

```python showLineNumbers
from litellm import image_generation
import os

os.environ['STABILITY_API_KEY'] = "your-api-key"

response = image_generation(
    model="stability/sd3.5-large",
    prompt="A beautiful sunset over a calm ocean",
    size="1792x1024",  # Maps to aspect_ratio 16:9
    negative_prompt="blurry, low quality",  # Stability-specific
    seed=12345,  # For reproducibility
)
print(response)
```

### Supported Parameters

Stability AI supports the following OpenAI-compatible parameters:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `size` | string | Image dimensions (mapped to aspect_ratio) | `"1024x1024"` |
| `n` | integer | Number of images (note: Stability returns 1 per request) | `1` |
| `response_format` | string | Format of response (`b64_json` only for Stability) | `"b64_json"` |

### Size to Aspect Ratio Mapping

The `size` parameter is automatically mapped to Stability's `aspect_ratio`:

| OpenAI Size | Stability Aspect Ratio |
|-------------|----------------------|
| `1024x1024` | `1:1` |
| `1792x1024` | `16:9` |
| `1024x1792` | `9:16` |
| `512x512` | `1:1` |
| `256x256` | `1:1` |

### Using Stability-Specific Parameters

You can pass parameters that are specific to Stability AI directly in your request:

```python showLineNumbers
from litellm import image_generation
import os

os.environ['STABILITY_API_KEY'] = "your-api-key"

response = image_generation(
    model="stability/sd3.5-large",
    prompt="A beautiful sunset over a calm ocean",
    # Stability-specific parameters
    negative_prompt="blurry, watermark, text",
    aspect_ratio="16:9",  # Use directly instead of size
    seed=42,
    output_format="png",  # png, jpeg, or webp
)
print(response)
```

### Supported Image Generation Models

| Model Name | Function Call | Description |
|------------|---------------|-------------|
| sd3 | `image_generation(model="stability/sd3", ...)` | Stable Diffusion 3 |
| sd3-large | `image_generation(model="stability/sd3-large", ...)` | SD3 Large |
| sd3-large-turbo | `image_generation(model="stability/sd3-large-turbo", ...)` | SD3 Large Turbo (faster) |
| sd3-medium | `image_generation(model="stability/sd3-medium", ...)` | SD3 Medium |
| sd3.5-large | `image_generation(model="stability/sd3.5-large", ...)` | SD 3.5 Large (recommended) |
| sd3.5-large-turbo | `image_generation(model="stability/sd3.5-large-turbo", ...)` | SD 3.5 Large Turbo |
| sd3.5-medium | `image_generation(model="stability/sd3.5-medium", ...)` | SD 3.5 Medium |
| stable-image-ultra | `image_generation(model="stability/stable-image-ultra", ...)` | Stable Image Ultra |
| stable-image-core | `image_generation(model="stability/stable-image-core", ...)` | Stable Image Core |

For more details on available models and features, see: https://platform.stability.ai/docs/api-reference

## Response Format

Stability AI returns images in base64 format. The response is OpenAI-compatible:

```python
{
    "created": 1234567890,
    "data": [
        {
            "b64_json": "iVBORw0KGgo..."  # Base64 encoded image
        }
    ]
}
```

## Comparing with Bedrock

LiteLLM supports Stability AI models via two routes:

| Route | Provider | Use Case |
|-------|----------|----------|
| `stability/` | Stability AI Direct API | Direct access, all latest models |
| `bedrock/stability.*` | AWS Bedrock | AWS integration, enterprise features |

Use `stability/` for direct API access. Use `bedrock/stability.*` if you're already using AWS Bedrock.
