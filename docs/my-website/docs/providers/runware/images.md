import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Runware - Image Generation

Runware is an AI inference platform offering fast, scalable access to 1000+ image generation models including FLUX, Qwen Image, Grok Imagine, and more.

## Overview

| Property | Details |
|----------|---------|
| Description | Runware provides optimized infrastructure for running image generation models at scale with low latency and competitive pricing. |
| Provider Route on LiteLLM | `runware/` |
| Provider Doc | [Runware Documentation ↗](https://docs.runware.ai/) |
| Supported Operations | [`/images/generations`](#image-generation) |

## Setup

### API Key

```python showLineNumbers
import os

# Set your Runware API key
os.environ["RUNWARE_API_KEY"] = "your-runware-api-key"
```

Get your API key from [runware.ai](https://runware.ai/).

## Supported Models

| Model Name | Description | Cost/Image | Documentation |
|------------|-------------|------------|---------------|
| `runware/runware:400@1` | FLUX.2 [dev] - Fast, high-quality generation | $0.009 | [Docs ↗](https://runware.ai/models/bfl-flux-2-dev) |
| `runware/bfl:7@1` | FLUX.2 [max] - Maximum quality FLUX model | $0.07 | [Docs ↗](https://runware.ai/models/bfl-flux-2-max) |
| `runware/bfl:5@1` | FLUX.2 [pro] - Professional FLUX model | $0.03 | [Docs ↗](https://runware.ai/models/bfl-flux-2-pro) |
| `runware/runware:400@3` | FLUX.2 [klein] 9B Base - Lightweight FLUX | $0.0017 | [Docs ↗](https://runware.ai/models/bfl-flux-2-klein-9b-base) |
| `runware/runware:z-image@turbo` | Z-Image Turbo - Ultra-fast generation | $0.0019 | [Docs ↗](https://runware.ai/models/z-image-turbo) |
| `runware/google:4@3` | Nano Banana 2 | $0.047 | [Docs ↗](https://runware.ai/models/google-nano-banana-2) |
| `runware/google:4@2` | Nano Banana Pro | $0.138 | [Docs ↗](https://runware.ai/models/google-nano-banana-pro) |
| `runware/bytedance:seedream@5.0-lite` | Seedream 5.0 Lite | $0.035 | [Docs ↗](https://runware.ai/models/bytedance-seedream-5-0-lite) |
| `runware/alibaba:qwen-image@2.0` | Qwen Image 2.0 | $0.035 | [Docs ↗](https://runware.ai/models/alibaba-qwen-image-2-0) |
| `runware/klingai:kling-image@o3` | Kling Image o3 | $0.028 | [Docs ↗](https://runware.ai/models/klingai-image-o3) |
| `runware/imagineart:1@5` | ImagineArt 1.5 | $0.03 | [Docs ↗](https://runware.ai/models/imagineart-1-5) |
| `runware/xai:grok-imagine@image` | Grok Imagine Image | $0.02 | [Docs ↗](https://runware.ai/models/xai-grok-imagine-image) |
| `runware/xai:grok-imagine@image-pro` | Grok Imagine Image Pro | $0.07 | [Docs ↗](https://runware.ai/models/xai-grok-imagine-image-pro) |

Runware supports 1000+ additional models via CivitAI. Any model can be used by passing its AIR identifier after the `runware/` prefix.

## Image Generation

### Usage - LiteLLM Python SDK

<Tabs>
<TabItem value="basic" label="Basic Usage">

```python showLineNumbers title="Basic Image Generation"
import litellm
import os

# Set your API key
os.environ["RUNWARE_API_KEY"] = "your-runware-api-key"

# Generate an image
response = litellm.image_generation(
    model="runware/runware:400@1",
    prompt="A serene mountain landscape at sunset with vibrant colors"
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="flux-max" label="FLUX.2 Max">

```python showLineNumbers title="FLUX.2 Max - Highest Quality"
import litellm
import os

os.environ["RUNWARE_API_KEY"] = "your-runware-api-key"

# Generate with FLUX.2 [max]
response = litellm.image_generation(
    model="runware/bfl:7@1",
    prompt="A photorealistic portrait of a cat wearing a tiny crown",
    size="1024x1024",
    n=1
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="advanced" label="Advanced Parameters">

```python showLineNumbers title="Advanced Generation with Runware-Specific Parameters"
import litellm
import os

os.environ["RUNWARE_API_KEY"] = "your-runware-api-key"

# Generate with advanced parameters
response = litellm.image_generation(
    model="runware/runware:400@1",
    prompt="A majestic dragon soaring over mountains",
    size="1024x768",
    n=2,
    # Runware-specific parameters (passed through directly)
    negativePrompt="blurry, low quality, distorted",
    steps=30,
    CFGScale=7.5,
    seed=42,
    scheduler="DPM++ 2M Karras"
)

for image in response.data:
    print(f"Generated image: {image.url}")
```

</TabItem>

<TabItem value="b64" label="Base64 Output">

```python showLineNumbers title="Get Base64 Image Data"
import litellm
import os

os.environ["RUNWARE_API_KEY"] = "your-runware-api-key"

# Get image as base64 instead of URL
response = litellm.image_generation(
    model="runware/runware:z-image@turbo",
    prompt="A minimalist logo of a mountain",
    response_format="b64_json",
    size="512x512"
)

# Access base64 data
b64_data = response.data[0].b64_json
print(f"Base64 length: {len(b64_data)}")
```

</TabItem>

<TabItem value="async" label="Async Usage">

```python showLineNumbers title="Async Image Generation"
import litellm
import asyncio
import os

async def generate_image():
    os.environ["RUNWARE_API_KEY"] = "your-runware-api-key"

    response = await litellm.aimage_generation(
        model="runware/alibaba:qwen-image@2.0",
        prompt="A cyberpunk cityscape with neon lights",
        size="1024x1024"
    )

    print(response.data[0].url)
    return response

asyncio.run(generate_image())
```

</TabItem>
</Tabs>

### Usage - LiteLLM Proxy Server

#### 1. Configure your config.yaml

```yaml showLineNumbers title="Runware Image Generation Configuration"
model_list:
  - model_name: flux-dev
    litellm_params:
      model: runware/runware:400@1
      api_key: os.environ/RUNWARE_API_KEY
    model_info:
      mode: image_generation

  - model_name: flux-max
    litellm_params:
      model: runware/bfl:7@1
      api_key: os.environ/RUNWARE_API_KEY
    model_info:
      mode: image_generation

  - model_name: z-image-turbo
    litellm_params:
      model: runware/runware:z-image@turbo
      api_key: os.environ/RUNWARE_API_KEY
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
    model="flux-dev",
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
    model="litellm_proxy/flux-dev",
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
    "model": "flux-dev",
    "prompt": "A serene Japanese garden with cherry blossoms",
    "n": 1,
    "size": "1024x1024"
}'
```

</TabItem>
</Tabs>

## Using Model-Specific Parameters

LiteLLM forwards any additional parameters directly to the Runware API. You can pass Runware-specific parameters in your request and they will be sent as-is.

```python showLineNumbers title="Pass Runware-Specific Parameters"
import litellm

# Any parameters beyond the standard ones are forwarded to Runware
response = litellm.image_generation(
    model="runware/runware:400@1",
    prompt="A beautiful sunset",
    # Runware-specific parameters
    negativePrompt="blurry, low quality",
    steps=30,
    CFGScale=7.5,
    seed=42,
    scheduler="DPM++ 2M Karras",
    lora=["civitai:12345@67890"],
)
```

### Available Runware-Specific Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `negativePrompt` | string | Text describing what to avoid in the image |
| `steps` | integer | Number of inference steps (higher = more detail) |
| `CFGScale` | float | Classifier-free guidance scale (1.0-30.0) |
| `seed` | integer | Seed for reproducible results |
| `scheduler` | string | Sampling scheduler (e.g., "DPM++ 2M Karras", "Euler a") |
| `lora` | list | LoRA model identifiers to apply |
| `controlNet` | list | ControlNet configurations |
| `refiner` | object | Refiner model configuration |

For the complete list of parameters, see the [Runware API Reference ↗](https://docs.runware.ai/en/image-inference/api-reference).

## Supported OpenAI Parameters

Standard OpenAI-compatible parameters that work across all models:

| Parameter | Type | Description | Runware Mapping |
|-----------|------|-------------|-----------------|
| `prompt` | string | Text description of desired image | `positivePrompt` |
| `model` | string | Runware model AIR identifier | `model` |
| `n` | integer | Number of images to generate | `numberResults` |
| `size` | string | Image dimensions (e.g., "1024x1024") | `width` / `height` |
| `response_format` | string | `"url"` or `"b64_json"` | `outputType` ("URL" or "base64Data") |

## Using Any Runware Model

Runware supports 1000+ models beyond the ones listed above. To use any model, pass its AIR identifier after the `runware/` prefix:

```python showLineNumbers title="Using CivitAI Models"
import litellm

# Use any CivitAI model via its AIR identifier
response = litellm.image_generation(
    model="runware/civitai:36520@76907",
    prompt="A fantasy landscape",
    size="512x512"
)

print(response.data[0].url)
```

Browse available models at [runware.ai](https://runware.ai/).

## Cost Tracking

Runware reports actual generation costs in API responses. LiteLLM automatically captures these for accurate cost tracking, regardless of whether the model is listed in the pricing table.

```python showLineNumbers title="Access Cost Information"
response = litellm.image_generation(
    model="runware/runware:400@1",
    prompt="A sunset"
)

# Actual cost reported by Runware
cost = response._hidden_params.get("response_cost")
print(f"Generation cost: ${cost}")
```

## Getting Started

1. Sign up at [runware.ai](https://runware.ai/)
2. Get your API key from your account dashboard
3. Set `RUNWARE_API_KEY` environment variable
4. Choose a model and start generating images with LiteLLM

## Additional Resources

- [Runware Documentation](https://docs.runware.ai/)
- [API Reference](https://docs.runware.ai/en/image-inference/api-reference)
- [Model Explorer](https://runware.ai/)
