import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Fal AI

Fal AI provides fast, scalable access to state-of-the-art image generation models including FLUX, Stable Diffusion, Imagen, and more.

## Overview

| Property | Details |
|----------|---------|
| Description | Fal AI offers optimized infrastructure for running image generation models at scale with low latency. |
| Provider Route on LiteLLM | `fal_ai/` |
| Provider Doc | [Fal AI Documentation ↗](https://fal.ai/models) |
| Supported Operations | [`/images/generations`](#image-generation) |

## Setup

### API Key

```python showLineNumbers
import os

# Set your Fal AI API key
os.environ["FAL_AI_API_KEY"] = "your-fal-api-key"
```

Get your API key from [fal.ai](https://fal.ai/).

## Supported Models

| Model Name | Description | Documentation |
|------------|-------------|---------------|
| `fal_ai/fal-ai/flux-pro/v1.1` | FLUX Pro v1.1 - Balanced speed and quality | [Docs ↗](https://fal.ai/models/fal-ai/flux-pro/v1.1) |
| `fal_ai/flux/schnell` | Flux Schnell - Low-latency generation with `image_size` support | [Docs ↗](https://fal.ai/models/fal-ai/flux/schnell) |
| `fal_ai/fal-ai/bytedance/seedream/v3/text-to-image` | ByteDance Seedream v3 - Text-to-image with `image_size` control | [Docs ↗](https://fal.ai/models/fal-ai/bytedance/seedream/v3/text-to-image) |
| `fal_ai/fal-ai/bytedance/dreamina/v3.1/text-to-image` | ByteDance Dreamina v3.1 - Text-to-image with `image_size` control | [Docs ↗](https://fal.ai/models/fal-ai/bytedance/dreamina/v3.1/text-to-image) |
| `fal_ai/fal-ai/flux-pro/v1.1-ultra` | FLUX Pro v1.1 Ultra - High-quality image generation | [Docs ↗](https://fal.ai/models/fal-ai/flux-pro/v1.1-ultra) |
| `fal_ai/fal-ai/imagen4/preview` | Google's Imagen 4 - Highest quality model | [Docs ↗](https://fal.ai/models/fal-ai/imagen4/preview) |
| `fal_ai/fal-ai/recraft/v3/text-to-image` | Recraft v3 - Multiple style options | [Docs ↗](https://fal.ai/models/fal-ai/recraft/v3/text-to-image) |
| `fal_ai/fal-ai/ideogram/v3` | Ideogram v3 - Lettering-first creative model (Balanced: $0.06/image) | [Docs ↗](https://fal.ai/models/fal-ai/ideogram/v3) |
| `fal_ai/fal-ai/stable-diffusion-v35-medium` | Stable Diffusion v3.5 Medium | [Docs ↗](https://fal.ai/models/fal-ai/stable-diffusion-v35-medium) |
| `fal_ai/bria/text-to-image/3.2` | Bria 3.2 - Commercial-grade generation | [Docs ↗](https://fal.ai/models/bria/text-to-image/3.2) |

## Image Generation

### Usage - LiteLLM Python SDK

<Tabs>
<TabItem value="basic" label="Basic Usage">

```python showLineNumbers title="Basic Image Generation"
import litellm
import os

# Set your API key
os.environ["FAL_AI_API_KEY"] = "your-fal-api-key"

# Generate an image
response = litellm.image_generation(
    model="fal_ai/fal-ai/flux-pro/v1.1-ultra",
    prompt="A serene mountain landscape at sunset with vibrant colors"
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="imagen4" label="Imagen 4">

```python showLineNumbers title="Google Imagen 4 Generation"
import litellm
import os

os.environ["FAL_AI_API_KEY"] = "your-fal-api-key"

# Generate with Imagen 4
response = litellm.image_generation(
    model="fal_ai/fal-ai/imagen4/preview",
    prompt="A vintage 1960s kitchen with flour package on countertop",
    aspect_ratio="16:9",
    num_images=1
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="recraft" label="Recraft v3">

```python showLineNumbers title="Recraft v3 with Style"
import litellm
import os

os.environ["FAL_AI_API_KEY"] = "your-fal-api-key"

# Generate with specific style
response = litellm.image_generation(
    model="fal_ai/fal-ai/recraft/v3/text-to-image",
    prompt="A red panda eating bamboo",
    style="realistic_image",
    image_size="landscape_4_3"
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="async" label="Async Usage">

```python showLineNumbers title="Async Image Generation"
import litellm
import asyncio
import os

async def generate_image():
    os.environ["FAL_AI_API_KEY"] = "your-fal-api-key"
    
    response = await litellm.aimage_generation(
        model="fal_ai/fal-ai/stable-diffusion-v35-medium",
        prompt="A cyberpunk cityscape with neon lights",
        guidance_scale=7.5,
        num_inference_steps=50
    )
    
    print(response.data[0].url)
    return response

asyncio.run(generate_image())
```

</TabItem>

<TabItem value="advanced" label="Advanced Parameters">

```python showLineNumbers title="Advanced FLUX Pro Generation"
import litellm
import os

os.environ["FAL_AI_API_KEY"] = "your-fal-api-key"

# Generate with advanced parameters
response = litellm.image_generation(
    model="fal_ai/fal-ai/flux-pro/v1.1-ultra",
    prompt="A majestic dragon soaring over mountains",
    n=2,
    size="1792x1024",  # Maps to aspect_ratio="16:9"
    seed=42,
    safety_tolerance="2",
    enhance_prompt=True
)

for image in response.data:
    print(f"Generated image: {image.url}")
```

</TabItem>
</Tabs>

### Usage - LiteLLM Proxy Server

#### 1. Configure your config.yaml

```yaml showLineNumbers title="Fal AI Image Generation Configuration"
model_list:
  - model_name: flux-ultra
    litellm_params:
      model: fal_ai/fal-ai/flux-pro/v1.1-ultra
      api_key: os.environ/FAL_AI_API_KEY
    model_info:
      mode: image_generation
  
  - model_name: imagen4
    litellm_params:
      model: fal_ai/fal-ai/imagen4/preview
      api_key: os.environ/FAL_AI_API_KEY
    model_info:
      mode: image_generation
  
  - model_name: stable-diffusion
    litellm_params:
      model: fal_ai/fal-ai/stable-diffusion-v35-medium
      api_key: os.environ/FAL_AI_API_KEY
    model_info:
      mode: image_generation

general_settings:
  master_key: sk-1234
```

#### 2. Start LiteLLM Proxy Server

```bash showLineNumbers title="Start Proxy Server"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

#### 3. Make requests

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="Generate via Proxy - OpenAI SDK"
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",
    api_key="sk-1234"
)

response = client.images.generate(
    model="flux-ultra",
    prompt="A beautiful sunset over the ocean",
    n=1,
    size="1024x1024"
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="Generate via Proxy - LiteLLM SDK"
import litellm

response = litellm.image_generation(
    model="litellm_proxy/imagen4",
    prompt="A cozy coffee shop interior",
    api_base="http://localhost:4000",
    api_key="sk-1234"
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="curl" label="cURL">

```bash showLineNumbers title="Generate via Proxy - cURL"
curl --location 'http://localhost:4000/v1/images/generations' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234' \
--data '{
    "model": "stable-diffusion",
    "prompt": "A serene Japanese garden with cherry blossoms",
    "n": 1,
    "size": "1024x1024"
}'
```

</TabItem>
</Tabs>



## Using Model-Specific Parameters

LiteLLM forwards any additional parameters directly to the Fal AI API. You can pass model-specific parameters in your request and they will be sent to Fal AI.

```python showLineNumbers title="Pass Model-Specific Parameters"
import litellm

# Any parameters beyond the standard ones are forwarded to Fal AI
response = litellm.image_generation(
    model="fal_ai/fal-ai/flux-pro/v1.1-ultra",
    prompt="A beautiful sunset",
    # Model-specific Fal AI parameters
    aspect_ratio="16:9",
    safety_tolerance="2",
    enhance_prompt=True,
    seed=42
)
```

For the complete list of parameters supported by each model, see:
- [FLUX Pro v1.1-ultra Parameters ↗](https://fal.ai/models/fal-ai/flux-pro/v1.1-ultra/api)
- [Imagen 4 Parameters ↗](https://fal.ai/models/fal-ai/imagen4/preview/api)
- [Recraft v3 Parameters ↗](https://fal.ai/models/fal-ai/recraft/v3/text-to-image/api)
- [Stable Diffusion v3.5 Parameters ↗](https://fal.ai/models/fal-ai/stable-diffusion-v35-medium/api)
- [Bria 3.2 Parameters ↗](https://fal.ai/models/bria/text-to-image/3.2/api)

## Supported Parameters

Standard OpenAI-compatible parameters that work across all models:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `prompt` | string | Text description of desired image | Required |
| `model` | string | Fal AI model to use | Required |
| `n` | integer | Number of images to generate (1-4) | `1` |
| `size` | string | Image dimensions (maps to model-specific format) | Model default |
| `api_key` | string | Your Fal AI API key | Environment variable |

## Getting Started

1. Sign up at [fal.ai](https://fal.ai/)
2. Get your API key from your account settings
3. Set `FAL_AI_API_KEY` environment variable
4. Choose a model from the [Fal AI model gallery](https://fal.ai/models)
5. Start generating images with LiteLLM

## Additional Resources

- [Fal AI Documentation](https://fal.ai/docs)
- [Model Gallery](https://fal.ai/models)
- [API Reference](https://fal.ai/docs/api-reference)
- [Pricing](https://fal.ai/pricing)

