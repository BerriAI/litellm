# Recraft
https://www.recraft.ai/

## Overview

| Property | Details |
|-------|-------|
| Description | Recraft is an AI-powered design tool that generates high-quality images with precise control over style and content. |
| Provider Route on LiteLLM | `recraft/` |
| Link to Provider Doc | [Recraft ↗](https://www.recraft.ai/docs) |
| Supported Operations | [`/images/generations`](#image-generation), [`/images/edits`](#image-edit) |

LiteLLM supports Recraft Image Generation and Image Edit calls.

## API Base, Key
```python
# env variable
os.environ['RECRAFT_API_KEY'] = "your-api-key"
os.environ['RECRAFT_API_BASE'] = "https://external.api.recraft.ai"  # [optional] 
```

## Image Generation

### Usage - LiteLLM Python SDK

```python showLineNumbers
from litellm import image_generation
import os

os.environ['RECRAFT_API_KEY'] = "your-api-key"

# recraft image generation call
response = image_generation(
    model="recraft/recraftv3",
    prompt="A beautiful sunset over a calm ocean",
)
print(response)
```

### Usage - LiteLLM Proxy Server

#### 1. Setup config.yaml

```yaml showLineNumbers
model_list:
  - model_name: recraft-v3
    litellm_params:
      model: recraft/recraftv3
      api_key: os.environ/RECRAFT_API_KEY
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
    "model": "recraft-v3",
    "prompt": "A beautiful sunset over a calm ocean",
}'
```

### Advanced Usage - With Additional Parameters

```python showLineNumbers
from litellm import image_generation
import os

os.environ['RECRAFT_API_KEY'] = "your-api-key"

response = image_generation(
    model="recraft/recraftv3",
    prompt="A beautiful sunset over a calm ocean",
)
print(response)
```

### Supported Parameters

Recraft supports the following OpenAI-compatible parameters:

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `n` | integer | Number of images to generate (1-4) | `1` |
| `response_format` | string | Format of response (`url` or `b64_json`) | `"url"` |
| `size` | string | Image dimensions | `"1024x1024"` |
| `style` | string | Image style/artistic direction | `"realistic"` |

### Using Non-OpenAI Parameters

If you want to pass parameters that are not supported by OpenAI, you can pass them in your request body, LiteLLM will automatically route it to recraft.

In this example we will pass `style_id` parameter to the recraft image generation call.

**Usage with LiteLLM Python SDK**

```python showLineNumbers
from litellm import image_generation
import os

os.environ['RECRAFT_API_KEY'] = "your-api-key"

response = image_generation(
    model="recraft/recraftv3",
    prompt="A beautiful sunset over a calm ocean",
    style_id="your-style-id",
)
```

**Usage with LiteLLM Proxy Server + OpenAI Python SDK**

```python showLineNumbers
from openai import OpenAI
import os

os.environ['RECRAFT_API_KEY'] = "your-api-key"

client = OpenAI(api_key=os.environ['RECRAFT_API_KEY'])

response = client.images.generate(
    model="recraft/recraftv3",
    prompt="A beautiful sunset over a calm ocean",
    extra_body={
        "style_id": "your-style-id",
    },
)
print(response)
```

### Supported Image Generation Models

**Note: All recraft models are supported by LiteLLM** Just pass the model name with `recraft/<model_name>` and litellm will route it to recraft.

| Model Name | Function Call |
|------------|---------------|
| recraftv3 | `image_generation(model="recraft/recraftv3", prompt="...")` |
| recraftv2 | `image_generation(model="recraft/recraftv2", prompt="...")` |

For more details on available models and features, see: https://www.recraft.ai/docs

## Image Edit

### Usage - LiteLLM Python SDK

```python showLineNumbers
from litellm import image_edit
import os

os.environ['RECRAFT_API_KEY'] = "your-api-key"

# Open the image file
with open("reference_image.png", "rb") as image_file:
    # recraft image edit call
    response = image_edit(
        model="recraft/recraftv3",
        prompt="Create a studio ghibli style image that combines all the reference images. Make sure the person looks like a CTO.",
        image=image_file,
    )
print(response)
```

### Usage - LiteLLM Proxy Server

#### 1. Setup config.yaml

```yaml showLineNumbers
model_list:
  - model_name: recraft-v3
    litellm_params:
      model: recraft/recraftv3
      api_key: os.environ/RECRAFT_API_KEY
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
curl --location 'http://0.0.0.0:4000/v1/images/edits' \
--header 'Authorization: Bearer sk-1234' \
--form 'model="recraft-v3"' \
--form 'prompt="Create a studio ghibli style image that combines all the reference images. Make sure the person looks like a CTO."' \
--form 'image=@"reference_image.png"'
```

### Advanced Usage - With Additional Parameters

```python showLineNumbers
from litellm import image_edit
import os

os.environ['RECRAFT_API_KEY'] = "your-api-key"

with open("reference_image.png", "rb") as image_file:
    response = image_edit(
        model="recraft/recraftv3",
        prompt="Create a studio ghibli style image",
        image=image_file,
        n=2,  # Generate 2 variations
        response_format="url",  # Return URLs instead of base64
        style="realistic_image",  # Set artistic style
        strength=0.5  # Control transformation strength (0-1)
    )
print(response)
```

### Supported Image Edit Parameters

Recraft supports the following OpenAI-compatible parameters for image editing:

| Parameter | Type | Description | Default | Example |
|-----------|------|-------------|---------|---------|
| `n` | integer | Number of images to generate (1-4) | `1` | `2` |
| `response_format` | string | Format of response (`url` or `b64_json`) | `"url"` | `"b64_json"` |
| `style` | string | Image style/artistic direction | - | `"realistic_image"` |
| `strength` | float | Controls how much to transform the image (0.0-1.0) | `0.2` | `0.5` |

### Using Non-OpenAI Parameters

You can pass Recraft-specific parameters that are not part of the OpenAI API by including them in your request:

**Usage with LiteLLM Python SDK**

```python showLineNumbers
from litellm import image_edit
import os

os.environ['RECRAFT_API_KEY'] = "your-api-key"

with open("reference_image.png", "rb") as image_file:
    response = image_edit(
        model="recraft/recraftv3",
        prompt="Create a studio ghibli style image",
        image=image_file,
        style_id="your-style-id",  # Recraft-specific parameter
        strength=0.7
    )
```

**Usage with LiteLLM Proxy Server + OpenAI Python SDK**

```python showLineNumbers
from openai import OpenAI
import os

client = OpenAI(
    api_key="sk-1234",  # your LiteLLM proxy master key
    base_url="http://0.0.0.0:4000"  # your LiteLLM proxy URL
)

with open("reference_image.png", "rb") as image_file:
    response = client.images.edit(
        model="recraft-v3",
        prompt="Create a studio ghibli style image",
        image=image_file,
        extra_body={
            "style_id": "your-style-id",
            "strength": 0.7
        }
    )
print(response)
```

### Supported Image Edit Models

**Note: All recraft models are supported by LiteLLM** Just pass the model name with `recraft/<model_name>` and litellm will route it to recraft.

| Model Name | Function Call |
|------------|---------------|
| recraftv3 | `image_edit(model="recraft/recraftv3", ...)` |

## API Key Setup

Get your API key from [Recraft's website](https://www.recraft.ai/) and set it as an environment variable:

```bash
export RECRAFT_API_KEY="your-api-key"
```
