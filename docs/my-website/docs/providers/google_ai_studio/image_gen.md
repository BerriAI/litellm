import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Google AI Studio Image Generation

Google AI Studio provides powerful image generation capabilities using Google's Imagen models to create high-quality images from text descriptions.

## Overview

| Property | Details |
|----------|---------|
| Description | Google AI Studio Image Generation uses Google's Imagen models to generate high-quality images from text descriptions. |
| Provider Route on LiteLLM | `gemini/` |
| Provider Doc | [Google AI Studio Image Generation ↗](https://ai.google.dev/gemini-api/docs/imagen) |
| Supported Operations | [`/images/generations`](#image-generation) |

## Setup

### API Key

```python showLineNumbers
# Set your Google AI Studio API key
import os
os.environ["GEMINI_API_KEY"] = "your-api-key-here"
```

Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## Image Generation

### Usage - LiteLLM Python SDK

<Tabs>
<TabItem value="basic" label="Basic Usage">

```python showLineNumbers title="Basic Image Generation"
import litellm
import os

# Set your API key
os.environ["GEMINI_API_KEY"] = "your-api-key-here"

# Generate a single image
response = litellm.image_generation(
    model="gemini/imagen-4.0-generate-001",
    prompt="A cute baby sea otter swimming in crystal clear water"
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
    # Set your API key
    os.environ["GEMINI_API_KEY"] = "your-api-key-here"
    
    # Generate image asynchronously
    response = await litellm.aimage_generation(
        model="gemini/imagen-4.0-generate-001",
        prompt="A beautiful sunset over mountains with vibrant colors",
        n=1,
    )
    
    print(response.data[0].url)
    return response

# Run the async function
asyncio.run(generate_image())
```

</TabItem>

<TabItem value="advanced" label="Advanced Parameters">

```python showLineNumbers title="Advanced Image Generation with Parameters"
import litellm
import os

# Set your API key
os.environ["GEMINI_API_KEY"] = "your-api-key-here"

# Generate image with additional parameters
response = litellm.image_generation(
    model="gemini/imagen-4.0-generate-001",
    prompt="A futuristic cityscape at night with neon lights",
    n=1,
    size="1024x1024",
    quality="standard",
    response_format="url"
)

for image in response.data:
    print(f"Generated image URL: {image.url}")
```

</TabItem>
</Tabs>

### Usage - LiteLLM Proxy Server

#### 1. Configure your config.yaml

```yaml showLineNumbers title="Google AI Studio Image Generation Configuration"
model_list:
  - model_name: google-imagen
    litellm_params:
      model: gemini/imagen-4.0-generate-001
      api_key: os.environ/GEMINI_API_KEY
  model_info:
    mode: image_generation

general_settings:
  master_key: sk-1234
```

#### 2. Start LiteLLM Proxy Server

```bash showLineNumbers title="Start LiteLLM Proxy Server"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

#### 3. Make requests with OpenAI Python SDK

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="Google AI Studio Image Generation via Proxy - OpenAI SDK"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="sk-1234"                  # Your proxy API key
)

# Generate image
response = client.images.generate(
    model="google-imagen",
    prompt="A majestic eagle soaring over snow-capped mountains",
    n=1,
    size="1024x1024"
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="Google AI Studio Image Generation via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.image_generation(
    model="litellm_proxy/google-imagen",
    prompt="A serene Japanese garden with cherry blossoms",
    api_base="http://localhost:4000",
    api_key="sk-1234"
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="curl" label="cURL">

```bash showLineNumbers title="Google AI Studio Image Generation via Proxy - cURL"
curl --location 'http://localhost:4000/v1/images/generations' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234' \
--data '{
    "model": "google-imagen",
    "prompt": "A cozy coffee shop interior with warm lighting",
    "n": 1,
    "size": "1024x1024"
}'
```

</TabItem>
</Tabs>

## Supported Parameters

Google AI Studio Image Generation supports the following OpenAI-compatible parameters:

| Parameter | Type | Description | Default | Example |
|-----------|------|-------------|---------|---------|
| `prompt` | string | Text description of the image to generate | Required | `"A sunset over the ocean"` |
| `model` | string | The model to use for generation | Required | `"gemini/imagen-4.0-generate-001"` |
| `n` | integer | Number of images to generate (1-4) | `1` | `2` |
| `size` | string | Image dimensions | `"1024x1024"` | `"512x512"`, `"1024x1024"` |

1. Create an account at [Google AI Studio](https://aistudio.google.com/)
2. Generate an API key from [API Keys section](https://aistudio.google.com/app/apikey)
3. Set your `GEMINI_API_KEY` environment variable
4. Start generating images using LiteLLM

## Additional Resources

- [Google AI Studio Documentation](https://ai.google.dev/gemini-api/docs)
- [Imagen Model Overview](https://ai.google.dev/gemini-api/docs/imagen)
- [LiteLLM Image Generation Guide](../../completion/image_generation)
