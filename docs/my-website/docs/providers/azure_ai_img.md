import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Azure AI Image Generation (Black Forest Labs - Flux)

Azure AI provides powerful image generation capabilities using FLUX models from Black Forest Labs to create high-quality images from text descriptions.

## Overview

| Property | Details |
|----------|---------|
| Description | Azure AI Image Generation uses FLUX models to generate high-quality images from text descriptions. |
| Provider Route on LiteLLM | `azure_ai/` |
| Provider Doc | [Azure AI FLUX Models ↗](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/black-forest-labs-flux-1-kontext-pro-and-flux1-1-pro-now-available-in-azure-ai-f/4434659) |
| Supported Operations | [`/images/generations`](#image-generation), [`/images/edits`](#image-editing) |

## Setup

### API Key & Base URL

```python showLineNumbers
# Set your Azure AI API credentials
import os
os.environ["AZURE_AI_API_KEY"] = "your-api-key-here"
os.environ["AZURE_AI_API_BASE"] = "your-azure-ai-endpoint"  # e.g., https://your-endpoint.eastus2.inference.ai.azure.com/
```

Get your API key and endpoint from [Azure AI Studio](https://ai.azure.com/).

## Supported Models

| Model Name | Description | Cost per Image |
|------------|-------------|----------------|
| `azure_ai/FLUX-1.1-pro` | Latest FLUX 1.1 Pro model for high-quality image generation | $0.04 |
| `azure_ai/FLUX.1-Kontext-pro` | FLUX 1 Kontext Pro model with enhanced context understanding | $0.04 |
| `azure_ai/flux.2-pro` | FLUX 2 Pro model for next-generation image generation | $0.04 |

## Image Generation

### Usage - LiteLLM Python SDK

<Tabs>
<TabItem value="basic" label="Basic Usage">

```python showLineNumbers title="Basic Image Generation"
import litellm
import os

# Set your API credentials
os.environ["AZURE_AI_API_KEY"] = "your-api-key-here"
os.environ["AZURE_AI_API_BASE"] = "your-azure-ai-endpoint"

# Generate a single image
response = litellm.image_generation(
    model="azure_ai/FLUX.1-Kontext-pro",
    prompt="A cute baby sea otter swimming in crystal clear water",
    api_base=os.environ["AZURE_AI_API_BASE"],
    api_key=os.environ["AZURE_AI_API_KEY"]
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="flux11" label="FLUX 1.1 Pro">

```python showLineNumbers title="FLUX 1.1 Pro Image Generation"
import litellm
import os

# Set your API credentials
os.environ["AZURE_AI_API_KEY"] = "your-api-key-here"
os.environ["AZURE_AI_API_BASE"] = "your-azure-ai-endpoint"

# Generate image with FLUX 1.1 Pro
response = litellm.image_generation(
    model="azure_ai/FLUX-1.1-pro",
    prompt="A futuristic cityscape at night with neon lights and flying cars",
    api_base=os.environ["AZURE_AI_API_BASE"],
    api_key=os.environ["AZURE_AI_API_KEY"]
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="flux2" label="FLUX 2 Pro">

```python showLineNumbers title="FLUX 2 Pro Image Generation"
import litellm
import os

# Set your API credentials
os.environ["AZURE_AI_API_KEY"] = "your-api-key-here"
os.environ["AZURE_AI_API_BASE"] = "your-azure-ai-endpoint"  # e.g., https://litellm-ci-cd-prod.services.ai.azure.com

# Generate image with FLUX 2 Pro
response = litellm.image_generation(
    model="azure_ai/flux.2-pro",
    prompt="A photograph of a red fox in an autumn forest",
    api_base=os.environ["AZURE_AI_API_BASE"],
    api_key=os.environ["AZURE_AI_API_KEY"],
    api_version="preview",
    size="1024x1024",
    n=1
)

print(response.data[0].b64_json)  # FLUX 2 returns base64 encoded images
```

</TabItem>

<TabItem value="async" label="Async Usage">

```python showLineNumbers title="Async Image Generation"
import litellm
import asyncio
import os

async def generate_image():
    # Set your API credentials
    os.environ["AZURE_AI_API_KEY"] = "your-api-key-here"
    os.environ["AZURE_AI_API_BASE"] = "your-azure-ai-endpoint"
    
    # Generate image asynchronously
    response = await litellm.aimage_generation(
        model="azure_ai/FLUX.1-Kontext-pro",
        prompt="A beautiful sunset over mountains with vibrant colors",
        api_base=os.environ["AZURE_AI_API_BASE"],
        api_key=os.environ["AZURE_AI_API_KEY"],
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

# Set your API credentials
os.environ["AZURE_AI_API_KEY"] = "your-api-key-here"
os.environ["AZURE_AI_API_BASE"] = "your-azure-ai-endpoint"

# Generate image with additional parameters
response = litellm.image_generation(
    model="azure_ai/FLUX-1.1-pro",
    prompt="A majestic dragon soaring over a medieval castle at dawn",
    api_base=os.environ["AZURE_AI_API_BASE"],
    api_key=os.environ["AZURE_AI_API_KEY"],
    n=1,
    size="1024x1024",
    quality="standard"
)

for image in response.data:
    print(f"Generated image URL: {image.url}")
```

</TabItem>
</Tabs>

### Usage - LiteLLM Proxy Server

#### 1. Configure your config.yaml

```yaml showLineNumbers title="Azure AI Image Generation Configuration"
model_list:
  - model_name: azure-flux-kontext
    litellm_params:
      model: azure_ai/FLUX.1-Kontext-pro
      api_key: os.environ/AZURE_AI_API_KEY
      api_base: os.environ/AZURE_AI_API_BASE
    model_info:
      mode: image_generation
  
  - model_name: azure-flux-11-pro
    litellm_params:
      model: azure_ai/FLUX-1.1-pro
      api_key: os.environ/AZURE_AI_API_KEY
      api_base: os.environ/AZURE_AI_API_BASE
    model_info:
      mode: image_generation

  - model_name: azure-flux-2-pro
    litellm_params:
      model: azure_ai/flux.2-pro
      api_key: os.environ/AZURE_AI_API_KEY
      api_base: os.environ/AZURE_AI_API_BASE
      api_version: preview
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

```python showLineNumbers title="Azure AI Image Generation via Proxy - OpenAI SDK"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="sk-1234"                  # Your proxy API key
)

# Generate image with FLUX Kontext Pro
response = client.images.generate(
    model="azure-flux-kontext",
    prompt="A serene Japanese garden with cherry blossoms and a peaceful pond",
    n=1,
    size="1024x1024"
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="litellm-sdk" label="LiteLLM SDK">

```python showLineNumbers title="Azure AI Image Generation via Proxy - LiteLLM SDK"
import litellm

# Configure LiteLLM to use your proxy
response = litellm.image_generation(
    model="litellm_proxy/azure-flux-11-pro",
    prompt="A cyberpunk warrior in a neon-lit alleyway",
    api_base="http://localhost:4000",
    api_key="sk-1234"
)

print(response.data[0].url)
```

</TabItem>

<TabItem value="curl" label="cURL">

```bash showLineNumbers title="Azure AI Image Generation via Proxy - cURL"
curl --location 'http://localhost:4000/v1/images/generations' \
--header 'Content-Type: application/json' \
--header 'Authorization: Bearer sk-1234' \
--data '{
    "model": "azure-flux-kontext",
    "prompt": "A cozy coffee shop interior with warm lighting and rustic wooden furniture",
    "n": 1,
    "size": "1024x1024"
}'
```

</TabItem>
</Tabs>

## Image Editing

FLUX 2 Pro supports image editing by passing an input image along with a prompt describing the desired modifications.

### Usage - LiteLLM Python SDK

<Tabs>
<TabItem value="basic-edit" label="Basic Image Edit">

```python showLineNumbers title="Basic Image Editing with FLUX 2 Pro"
import litellm
import os

# Set your API credentials
os.environ["AZURE_AI_API_KEY"] = "your-api-key-here"
os.environ["AZURE_AI_API_BASE"] = "your-azure-ai-endpoint"  # e.g., https://litellm-ci-cd-prod.services.ai.azure.com

# Edit an existing image
response = litellm.image_edit(
    model="azure_ai/flux.2-pro",
    prompt="Add a red hat to the subject",
    image=open("input_image.png", "rb"),
    api_base=os.environ["AZURE_AI_API_BASE"],
    api_key=os.environ["AZURE_AI_API_KEY"],
    api_version="preview",
)

print(response.data[0].b64_json)  # FLUX 2 returns base64 encoded images
```

</TabItem>

<TabItem value="async-edit" label="Async Image Edit">

```python showLineNumbers title="Async Image Editing"
import litellm
import asyncio
import os

async def edit_image():
    os.environ["AZURE_AI_API_KEY"] = "your-api-key-here"
    os.environ["AZURE_AI_API_BASE"] = "your-azure-ai-endpoint"
    
    response = await litellm.aimage_edit(
        model="azure_ai/flux.2-pro",
        prompt="Change the background to a sunset beach",
        image=open("input_image.png", "rb"),
        api_base=os.environ["AZURE_AI_API_BASE"],
        api_key=os.environ["AZURE_AI_API_KEY"],
        api_version="preview",
    )
    
    return response

asyncio.run(edit_image())
```

</TabItem>
</Tabs>

### Usage - LiteLLM Proxy Server

<Tabs>
<TabItem value="curl-edit" label="cURL">

```bash showLineNumbers title="Image Edit via Proxy - cURL"
curl --location 'http://localhost:4000/v1/images/edits' \
--header 'Authorization: Bearer sk-1234' \
--form 'model="azure-flux-2-pro"' \
--form 'prompt="Add sunglasses to the person"' \
--form 'image=@"input_image.png"'
```

</TabItem>

<TabItem value="openai-sdk-edit" label="OpenAI SDK">

```python showLineNumbers title="Image Edit via Proxy - OpenAI SDK"
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",
    api_key="sk-1234"
)

response = client.images.edit(
    model="azure-flux-2-pro",
    prompt="Make the sky more dramatic with storm clouds",
    image=open("input_image.png", "rb"),
)

print(response.data[0].b64_json)
```

</TabItem>
</Tabs>

## Supported Parameters

Azure AI Image Generation supports the following OpenAI-compatible parameters:

| Parameter | Type | Description | Default | Example |
|-----------|------|-------------|---------|---------|
| `prompt` | string | Text description of the image to generate | Required | `"A sunset over the ocean"` |
| `model` | string | The FLUX model to use for generation | Required | `"azure_ai/FLUX.1-Kontext-pro"` |
| `n` | integer | Number of images to generate (1-4) | `1` | `2` |
| `size` | string | Image dimensions | `"1024x1024"` | `"512x512"`, `"1024x1024"` |
| `api_base` | string | Your Azure AI endpoint URL | Required | `"https://your-endpoint.eastus2.inference.ai.azure.com/"` |
| `api_key` | string | Your Azure AI API key | Required | Environment variable or direct value |

## Getting Started

1. Create an account at [Azure AI Studio](https://ai.azure.com/)
2. Deploy a FLUX model in your Azure AI Studio workspace
3. Get your API key and endpoint from the deployment details
4. Set your `AZURE_AI_API_KEY` and `AZURE_AI_API_BASE` environment variables
5. Start generating images using LiteLLM

## Additional Resources

- [Azure AI Studio Documentation](https://docs.microsoft.com/en-us/azure/ai-services/)
- [FLUX Models Announcement](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/black-forest-labs-flux-1-kontext-pro-and-flux1-1-pro-now-available-in-azure-ai-f/4434659)
