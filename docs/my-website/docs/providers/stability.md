# Stability AI
https://stability.ai/

## Overview

| Property | Details |
|-------|-------|
| Description | Stability AI creates open AI models for image, video, audio, and 3D generation. Known for Stable Diffusion. |
| Provider Route on LiteLLM | `stability/` |
| Link to Provider Doc | [Stability AI API ↗](https://platform.stability.ai/docs/api-reference) |
| Supported Operations | [`/images/generations`](#image-generation), [`/images/edits`](#image-editing) |

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

## Image Editing

Stability AI supports various image editing operations including inpainting, upscaling, outpainting, background removal, and more.

:::info Optional Parameters
**Important:** Different Stability models have different parameter requirements:
- Some models don't require a `prompt` (e.g., upscaling, background removal)
- The `style-transfer` model uses `init_image` and `style_image` instead of `image`
- The `outpaint` model requires numeric parameters (`left`, `right`, `up`, `down`)
LiteLLM automatically handles these differences for you.
:::

### Usage - LiteLLM Python SDK

#### Inpainting (Edit with Mask)

```python showLineNumbers
from litellm import image_edit
import os

os.environ['STABILITY_API_KEY'] = "your-api-key"

# Inpainting - edit specific areas using a mask
response = image_edit(
    model="stability/stable-image-inpaint-v1:0",
    image=open("original_image.png", "rb"),
    mask=open("mask_image.png", "rb"), 
    prompt="Add a beautiful sunset in the masked area",
    size="1024x1024",
)
print(response)
```

#### Image Upscaling

```python showLineNumbers
from litellm import image_edit
import os

os.environ['STABILITY_API_KEY'] = "your-api-key"

# Conservative upscaling - preserves details
response = image_edit(
    model="stability/stable-conservative-upscale-v1:0",
    image=open("low_res_image.png", "rb"),
    prompt="Upscale this image while preserving details",
)

# Creative upscaling - adds creative details
response = image_edit(
    model="stability/stable-creative-upscale-v1:0",
    image=open("low_res_image.png", "rb"),
    prompt="Upscale and enhance with creative details",
    creativity=0.3,  # 0-0.35, higher = more creative
)

# Fast upscaling - quick upscaling (no prompt needed)
response = image_edit(
    model="stability/stable-fast-upscale-v1:0",
    image=open("low_res_image.png", "rb"),
    # No prompt required for fast upscale
)
print(response)
```

#### Image Outpainting

```python showLineNumbers
from litellm import image_edit
import os

os.environ['STABILITY_API_KEY'] = "your-api-key"

# Extend image beyond its borders
response = image_edit(
    model="stability/stable-outpaint-v1:0",
    image=open("original_image.png", "rb"),
    prompt="Extend this landscape with mountains",
    left=100,   # Pixels to extend on the left
    right=100,  # Pixels to extend on the right
    up=50,      # Pixels to extend on top
    down=50,    # Pixels to extend on bottom
)
print(response)
```

#### Background Removal

```python showLineNumbers
from litellm import image_edit
import os

os.environ['STABILITY_API_KEY'] = "your-api-key"

# Remove background from image
response = image_edit(
    model="stability/stable-image-remove-background-v1:0",
    image=open("portrait.png", "rb"),
    # No prompt required for fast upscale
)
print(response)
```

#### Search and Replace

```python showLineNumbers
from litellm import image_edit
import os

os.environ['STABILITY_API_KEY'] = "your-api-key"

# Search and replace objects in image
response = image_edit(
    model="stability/stable-image-search-replace-v1:0",
    image=open("scene.png", "rb"),
    prompt="A red sports car",
    search_prompt="blue sedan",  # What to replace
)

# Search and recolor
response = image_edit(
    model="stability/stable-image-search-recolor-v1:0",
    image=open("scene.png", "rb"),
    prompt="Make it golden yellow",
    select_prompt="the car",  # What to recolor
)
print(response)
```

#### Image Control (Sketch/Structure)

```python showLineNumbers
from litellm import image_edit
import os

os.environ['STABILITY_API_KEY'] = "your-api-key"

# Control with sketch
response = image_edit(
    model="stability/stable-image-control-sketch-v1:0",
    image=open("sketch.png", "rb"),
    prompt="Turn this sketch into a realistic photo",
    control_strength=0.7,  # 0-1, higher = more control
)

# Control with structure
response = image_edit(
    model="stability/stable-image-control-structure-v1:0",
    image=open("structure_reference.png", "rb"),
    prompt="Generate image following this structure",
    control_strength=0.7,
)
print(response)
```

#### Erase Objects

```python showLineNumbers
from litellm import image_edit
import os

os.environ['STABILITY_API_KEY'] = "your-api-key"

# Erase objects from image
response = image_edit(
    model="stability/stable-image-erase-object-v1:0",
    image=open("scene.png", "rb"),
    mask=open("object_mask.png", "rb"),  # Mask the object to erase
    # No prompt needed
)
print(response)
```
#### Style Transfer

```python showLineNumbers
from litellm import image_edit
import os

os.environ['STABILITY_API_KEY'] = "your-api-key"

# Transfer style from one image to another
# Note: Uses init_image (via image param) and style_image
response = image_edit(
    model="stability/stable-style-transfer-v1:0",
    image=open("content_image.png", "rb"),  # Maps to init_image
    style_image=open("style_reference.png", "rb"),  # Style to apply
    fidelity=0.5,  # 0-1, balance between content and style
    # No prompt needed
)

print(response)

### Supported Image Edit Models

| Model Name | Function Call | Description |
|------------|---------------|-------------|
| stable-image-inpaint-v1:0 | `image_edit(model="stability/stable-image-inpaint-v1:0", ...)` | Inpainting with mask |
| stable-conservative-upscale-v1:0 | `image_edit(model="stability/stable-conservative-upscale-v1:0", ...)` | Conservative upscaling |
| stable-creative-upscale-v1:0 | `image_edit(model="stability/stable-creative-upscale-v1:0", ...)` | Creative upscaling |
| stable-fast-upscale-v1:0 | `image_edit(model="stability/stable-fast-upscale-v1:0", ...)` | Fast upscaling |
| stable-outpaint-v1:0 | `image_edit(model="stability/stable-outpaint-v1:0", ...)` | Extend image borders |
| stable-image-remove-background-v1:0 | `image_edit(model="stability/stable-image-remove-background-v1:0", ...)` | Remove background |
| stable-image-search-replace-v1:0 | `image_edit(model="stability/stable-image-search-replace-v1:0", ...)` | Search and replace objects |
| stable-image-search-recolor-v1:0 | `image_edit(model="stability/stable-image-search-recolor-v1:0", ...)` | Search and recolor |
| stable-image-control-sketch-v1:0 | `image_edit(model="stability/stable-image-control-sketch-v1:0", ...)` | Control with sketch |
| stable-image-control-structure-v1:0 | `image_edit(model="stability/stable-image-control-structure-v1:0", ...)` | Control with structure |
| stable-image-erase-object-v1:0 | `image_edit(model="stability/stable-image-erase-object-v1:0", ...)` | Erase objects |
| stable-image-style-guide-v1:0 | `image_edit(model="stability/stable-image-style-guide-v1:0", ...)` | Apply style guide |
| stable-style-transfer-v1:0 | `image_edit(model="stability/stable-style-transfer-v1:0", ...)` | Transfer style |

### Usage - LiteLLM Proxy Server

#### 1. Setup config.yaml

```yaml showLineNumbers
model_list:
  - model_name: stability-inpaint
    litellm_params:
      model: stability/stable-image-inpaint-v1:0
      api_key: os.environ/STABILITY_API_KEY
    model_info:
      mode: image_edit

  - model_name: stability-upscale
    litellm_params:
      model: stability/stable-conservative-upscale-v1:0
      api_key: os.environ/STABILITY_API_KEY
    model_info:
      mode: image_edit

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
curl -X POST "http://0.0.0.0:4000/v1/images/edits" \
  -H "Authorization: Bearer sk-1234" \
  -F "model=stability-inpaint" \
  -F "image=@original_image.png" \
  -F "mask=@mask_image.png" \
  -F "prompt=Add a beautiful garden in the masked area"
```

## AWS Bedrock (Stability)

LiteLLM also supports Stability AI models via AWS Bedrock. This is useful if you're already using AWS infrastructure.

### Usage - Bedrock Stability

```python showLineNumbers
from litellm import image_edit
import os

# Set AWS credentials
os.environ["AWS_ACCESS_KEY_ID"] = "your-access-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret-key"
os.environ["AWS_REGION_NAME"] = "us-east-1"

# Bedrock Stability inpainting
response = image_edit(
    model="bedrock/us.stability.stable-image-inpaint-v1:0",
    image=open("original_image.png", "rb"),
    mask=open("mask_image.png", "rb"),
    prompt="Add flowers in the masked area",
)
print(response)
```
# Fast upscale without prompt
response = image_edit(
    model="bedrock/stability.stable-fast-upscale-v1:0",
    image=open("low_res_image.png", "rb"),
)

# Outpaint with numeric parameters
response = image_edit(
    model="bedrock/stability.stable-outpaint-v1:0",
    image=open("original_image.png", "rb"),
    left=100,   # Automatically converted to int
    right=100,
    up=50,
    down=50,
)

print(response)

### Supported Bedrock Stability Models

All Stability AI image edit models are available via Bedrock with the `bedrock/` prefix:

| Direct API Model | Bedrock Model | Description |
|------------------|---------------|-------------|
| stability/stable-image-inpaint-v1:0 | bedrock/us.stability.stable-image-inpaint-v1:0 | Inpainting |
| stability/stable-conservative-upscale-v1:0 | bedrock/stability.stable-conservative-upscale-v1:0 | Conservative upscaling |
| stability/stable-creative-upscale-v1:0 | bedrock/stability.stable-creative-upscale-v1:0 | Creative upscaling |
| stability/stable-fast-upscale-v1:0 | bedrock/stability.stable-fast-upscale-v1:0 | Fast upscaling |
| stability/stable-outpaint-v1:0 | bedrock/stability.stable-outpaint-v1:0 | Outpainting |
| stability/stable-image-remove-background-v1:0 | bedrock/stability.stable-image-remove-background-v1:0 | Remove background |
| stability/stable-image-search-replace-v1:0 | bedrock/stability.stable-image-search-replace-v1:0 | Search and replace |
| stability/stable-image-search-recolor-v1:0 | bedrock/stability.stable-image-search-recolor-v1:0 | Search and recolor |
| stability/stable-image-control-sketch-v1:0 | bedrock/stability.stable-image-control-sketch-v1:0 | Control with sketch |
| stability/stable-image-control-structure-v1:0 | bedrock/stability.stable-image-control-structure-v1:0 | Control with structure |
| stability/stable-image-erase-object-v1:0 | bedrock/stability.stable-image-erase-object-v1:0 | Erase objects |

**Note:** Bedrock model IDs may use `us.stability.*` or `stability.*` prefix depending on the region and model.

## Comparing Routes

LiteLLM supports Stability AI models via two routes:

| Route | Provider | Use Case | Image Generation | Image Editing |
|-------|----------|----------|------------------|---------------|
| `stability/` | Stability AI Direct API | Direct access, all latest models | ✅ | ✅ |
| `bedrock/stability.*` | AWS Bedrock | AWS integration, enterprise features | ✅ | ✅ |

Use `stability/` for direct API access. Use `bedrock/stability.*` if you're already using AWS Bedrock.
