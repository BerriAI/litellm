import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Black Forest Labs Image Editing

Black Forest Labs provides powerful image editing capabilities using their FLUX models to modify existing images based on text descriptions.

## Overview

| Property | Details |
|----------|---------|
| Description | Black Forest Labs Image Editing uses FLUX Kontext and other models to modify, inpaint, and expand images based on text prompts. |
| Provider Route on LiteLLM | `black_forest_labs/` |
| Provider Doc | [Black Forest Labs API ↗](https://docs.bfl.ai/) |
| Supported Operations | [`/images/edits`](#image-editing) |

## Setup

### API Key

```python showLineNumbers
import os

# Set your Black Forest Labs API key
os.environ["BFL_API_KEY"] = "your-api-key-here"
```

Get your API key from [Black Forest Labs](https://blackforestlabs.ai/).

## Supported Models

| Model Name | Description | Use Case |
|------------|-------------|----------|
| `black_forest_labs/flux-kontext-pro` | FLUX Kontext Pro - General image editing with prompts | General editing, style transfer |
| `black_forest_labs/flux-kontext-max` | FLUX Kontext Max - Premium quality editing | High-quality edits |
| `black_forest_labs/flux-pro-1.0-fill` | FLUX Pro Fill - Inpainting with mask | Remove/replace objects |
| `black_forest_labs/flux-pro-1.0-expand` | FLUX Pro Expand - Outpainting | Expand image borders |

## Image Editing

### Usage - LiteLLM Python SDK

<Tabs>
<TabItem value="basic-edit" label="Basic Usage">

```python showLineNumbers title="Basic Image Editing"
import os
import litellm

# Set your API key
os.environ["BFL_API_KEY"] = "your-api-key-here"

# Edit an image with a prompt
response = litellm.image_edit(
    model="black_forest_labs/flux-kontext-pro",
    image=open("path/to/your/image.png", "rb"),
    prompt="Add a green leaf to the scene",
)

# BFL returns URLs
print(response.data[0].url)
```

</TabItem>

<TabItem value="async-edit" label="Async Usage">

```python showLineNumbers title="Async Image Editing"
import os
import asyncio
import litellm

# Set your API key
os.environ["BFL_API_KEY"] = "your-api-key-here"

async def edit_image():
    response = await litellm.aimage_edit(
        model="black_forest_labs/flux-kontext-pro",
        image=open("path/to/your/image.png", "rb"),
        prompt="Make this image look like a watercolor painting",
    )
    print(response.data[0].url)

# Run the async function
asyncio.run(edit_image())
```

</TabItem>

<TabItem value="inpainting" label="Inpainting (Fill)">

```python showLineNumbers title="Inpainting with Mask"
import os
import litellm

# Set your API key
os.environ["BFL_API_KEY"] = "your-api-key-here"

# Use flux-pro-1.0-fill for inpainting
response = litellm.image_edit(
    model="black_forest_labs/flux-pro-1.0-fill",
    image=open("path/to/your/image.png", "rb"),
    mask=open("path/to/mask.png", "rb"),  # White areas will be edited
    prompt="Replace with a beautiful garden",
    steps=50,  # BFL-specific parameter
    guidance=30,  # BFL-specific parameter
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="outpainting" label="Outpainting (Expand)">

```python showLineNumbers title="Outpainting - Expand Image Borders"
import os
import litellm

# Set your API key
os.environ["BFL_API_KEY"] = "your-api-key-here"

# Use flux-pro-1.0-expand to extend image borders
response = litellm.image_edit(
    model="black_forest_labs/flux-pro-1.0-expand",
    image=open("path/to/your/image.png", "rb"),
    prompt="Continue the scene with a mountain landscape",
    top=256,     # Expand 256 pixels at top
    bottom=256,  # Expand 256 pixels at bottom
    left=128,    # Expand 128 pixels at left
    right=128,   # Expand 128 pixels at right
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="advanced" label="Advanced Parameters">

```python showLineNumbers title="Advanced Image Editing with BFL Parameters"
import os
import litellm

# Set your API key
os.environ["BFL_API_KEY"] = "your-api-key-here"

# Edit image with BFL-specific parameters
response = litellm.image_edit(
    model="black_forest_labs/flux-kontext-pro",
    image=open("path/to/your/image.png", "rb"),
    prompt="Transform into cyberpunk style with neon lights",
    seed=42,                    # For reproducible results
    output_format="png",        # png or jpeg
    safety_tolerance=2,         # 0-6, higher = more permissive
    aspect_ratio="16:9",        # Output aspect ratio
)

print(response.data[0].url)
```

</TabItem>
</Tabs>

### Usage - LiteLLM Proxy Server

#### 1. Configure your config.yaml

```yaml showLineNumbers title="Black Forest Labs Image Editing Configuration"
model_list:
  - model_name: bfl-kontext-pro
    litellm_params:
      model: black_forest_labs/flux-kontext-pro
      api_key: os.environ/BFL_API_KEY
    model_info:
      mode: image_edit

  - model_name: bfl-kontext-max
    litellm_params:
      model: black_forest_labs/flux-kontext-max
      api_key: os.environ/BFL_API_KEY
    model_info:
      mode: image_edit

  - model_name: bfl-fill
    litellm_params:
      model: black_forest_labs/flux-pro-1.0-fill
      api_key: os.environ/BFL_API_KEY
    model_info:
      mode: image_edit

  - model_name: bfl-expand
    litellm_params:
      model: black_forest_labs/flux-pro-1.0-expand
      api_key: os.environ/BFL_API_KEY
    model_info:
      mode: image_edit

general_settings:
  master_key: sk-1234
```

#### 2. Start LiteLLM Proxy Server

```bash showLineNumbers title="Start LiteLLM Proxy Server"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

#### 3. Make image editing requests

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="Black Forest Labs via Proxy - OpenAI SDK"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="sk-1234"
)

# Edit image with FLUX Kontext Pro
response = client.images.edit(
    model="bfl-kontext-pro",
    image=open("path/to/your/image.png", "rb"),
    prompt="Add magical sparkles and fairy dust",
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="curl" label="cURL">

```bash showLineNumbers title="Black Forest Labs via Proxy - cURL"
curl --location 'http://localhost:4000/v1/images/edits' \
--header 'Authorization: Bearer sk-1234' \
--form 'model="bfl-kontext-pro"' \
--form 'prompt="Add a sunset in the background"' \
--form 'image=@"path/to/your/image.png"'
```

</TabItem>
</Tabs>

## Supported Parameters

### OpenAI-Compatible Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `image` | file | The image file to edit | Required |
| `prompt` | string | Text description of the desired changes | Required |
| `model` | string | The FLUX model to use | Required |
| `mask` | file | Mask image for inpainting (flux-pro-1.0-fill) | Optional |
| `n` | integer | Number of images (BFL returns 1 per request) | `1` |
| `size` | string | Maps to aspect_ratio | Optional |
| `response_format` | string | `url` or `b64_json` | `url` |

### Black Forest Labs Specific Parameters

| Parameter | Type | Description | Default | Models |
|-----------|------|-------------|---------|--------|
| `seed` | integer | Seed for reproducible results | Random | All |
| `output_format` | string | Output format: `png` or `jpeg` | `png` | All |
| `safety_tolerance` | integer | Safety filter tolerance (0-6) | 2 | All |
| `aspect_ratio` | string | Output aspect ratio (e.g., `16:9`, `1:1`) | Original | Kontext models |
| `steps` | integer | Number of inference steps | Model default | Fill |
| `guidance` | float | Guidance scale | Model default | Fill |
| `grow_mask` | integer | Pixels to grow mask | 0 | Fill |
| `top` | integer | Pixels to expand at top | 0 | Expand |
| `bottom` | integer | Pixels to expand at bottom | 0 | Expand |
| `left` | integer | Pixels to expand at left | 0 | Expand |
| `right` | integer | Pixels to expand at right | 0 | Expand |

## How It Works

Black Forest Labs uses a polling-based API:

1. **Submit Request**: LiteLLM sends your image and prompt to BFL
2. **Get Task ID**: BFL returns a task ID and polling URL
3. **Poll for Result**: LiteLLM automatically polls until the image is ready
4. **Return Result**: The generated image URL is returned

This polling is handled automatically by LiteLLM - you just call `image_edit()` and get the result.

## Getting Started

1. Create an account at [Black Forest Labs](https://blackforestlabs.ai/)
2. Get your API key from the dashboard
3. Set your `BFL_API_KEY` environment variable
4. Use `litellm.image_edit()` with any supported model

## Additional Resources

- [Black Forest Labs Documentation](https://docs.bfl.ai/)
- [FLUX Model Information](https://blackforestlabs.ai/)
