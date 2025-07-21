# Recraft
https://www.recraft.ai/

## Overview

| Property | Details |
|-------|-------|
| Description | Recraft is an AI-powered design tool that generates high-quality images with precise control over style and content. |
| Provider Route on LiteLLM | `recraft/` |
| Link to Provider Doc | [Recraft ↗](https://www.recraft.ai/docs) |
| Supported Operations | [`/images/generations`](#image-generation) |

LiteLLM supports Recraft Image Generation calls.

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

### Supported Image Generation Models

**Note: All recraft models are supported by LiteLLM** Just pass the model name with `recraft/<model_name>` and litellm will route it to recraft.

| Model Name | Function Call |
|------------|---------------|
| recraftv3 | `image_generation(model="recraft/recraftv3", prompt="...")` |
| recraftv2 | `image_generation(model="recraft/recraftv2", prompt="...")` |

For more details on available models and features, see: https://www.recraft.ai/docs

## API Key Setup

Get your API key from [Recraft's website](https://www.recraft.ai/) and set it as an environment variable:

```bash
export RECRAFT_API_KEY="your-api-key"
```
