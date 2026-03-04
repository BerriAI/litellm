# Xinference [Xorbits Inference]
https://inference.readthedocs.io/en/latest/index.html

## Overview

| Property | Details |
|-------|-------|
| Description | Xinference is an open-source platform to run inference with any open-source LLMs, image generation models, and more. |
| Provider Route on LiteLLM | `xinference/` |
| Link to Provider Doc | [Xinference ↗](https://inference.readthedocs.io/en/latest/index.html) |
| Supported Operations | [`/embeddings`](#sample-usage---embedding), [`/images/generations`](#image-generation), [`/images/edits`](#image-editing) |

LiteLLM supports Xinference Embedding, Image Generation, and Image Editing calls.

## API Base, Key
```python
# env variable
os.environ['XINFERENCE_API_BASE'] = "http://127.0.0.1:9997/v1"
os.environ['XINFERENCE_API_KEY'] = "anything" #[optional] no api key required
```

## Sample Usage - Embedding
```python showLineNumbers
from litellm import embedding
import os

os.environ['XINFERENCE_API_BASE'] = "http://127.0.0.1:9997/v1"
response = embedding(
    model="xinference/bge-base-en",
    input=["good morning from litellm"],
)
print(response)
```

## Sample Usage `api_base` param
```python showLineNumbers
from litellm import embedding
import os

response = embedding(
    model="xinference/bge-base-en",
    api_base="http://127.0.0.1:9997/v1",
    input=["good morning from litellm"],
)
print(response)
```

## Image Generation

### Usage - LiteLLM Python SDK

```python showLineNumbers
from litellm import image_generation
import os

# xinference image generation call
response = image_generation(
    model="xinference/stabilityai/stable-diffusion-3.5-large",
    prompt="A beautiful sunset over a calm ocean",
    api_base="http://127.0.0.1:9997/v1",
)
print(response)
```

### Usage - LiteLLM Proxy Server

#### 1. Setup config.yaml

```yaml showLineNumbers
model_list:
  - model_name: xinference-sd
    litellm_params:
      model: xinference/stabilityai/stable-diffusion-3.5-large
      api_base: http://127.0.0.1:9997/v1
      api_key: anything
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
    "model": "xinference-sd",
    "prompt": "A beautiful sunset over a calm ocean",
    "n": 1,
    "size": "1024x1024",
    "response_format": "url"
}'
```

### Advanced Usage - With Additional Parameters

```python showLineNumbers
from litellm import image_generation
import os

os.environ['XINFERENCE_API_BASE'] = "http://127.0.0.1:9997/v1"

response = image_generation(
    model="xinference/stabilityai/stable-diffusion-3.5-large",
    prompt="A beautiful sunset over a calm ocean",
    n=1,                           # number of images
    size="1024x1024",             # image size
    response_format="b64_json",   # return format
)
print(response)
```

### Supported Image Generation Models

Xinference supports various stable diffusion models. Here are some examples:

| Model Name                                              | Function Call                                                                                      |
|---------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| stabilityai/stable-diffusion-3.5-large                 | `image_generation(model="xinference/stabilityai/stable-diffusion-3.5-large", prompt="...")`      |
| stabilityai/stable-diffusion-xl-base-1.0               | `image_generation(model="xinference/stabilityai/stable-diffusion-xl-base-1.0", prompt="...")`    |
| runwayml/stable-diffusion-v1-5                         | `image_generation(model="xinference/runwayml/stable-diffusion-v1-5", prompt="...")`              |

For a complete list of supported image generation models, see: https://inference.readthedocs.io/en/latest/models/builtin/image/index.html

## Image Editing

Xinference supports image editing (image-to-image transformation) through the `/v1/images/edits` endpoint. This allows you to modify existing images based on text prompts.

### Usage - LiteLLM Python SDK

```python showLineNumbers
from litellm import image_edit
import os

# xinference image edit call
response = image_edit(
    model="xinference/stabilityai/stable-diffusion-3.5-large",
    image=open("input_image.png", "rb"),
    prompt="Make the sky more vibrant and colorful",
    api_base="http://127.0.0.1:9997/v1",
)
print(response)
```

### Usage - LiteLLM Proxy Server

#### 1. Setup config.yaml

```yaml showLineNumbers
model_list:
  - model_name: xinference-sd-edit
    litellm_params:
      model: xinference/stabilityai/stable-diffusion-3.5-large
      api_base: http://127.0.0.1:9997/v1
      api_key: anything
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
curl --location 'http://0.0.0.0:4000/v1/images/edits' \
--header 'Authorization: Bearer sk-1234' \
--form 'model="stabilityai/stable-diffusion-3.5-large"' \
--form 'image=@"/path/to/input_image.png"' \
--form 'prompt="Make the sky more vibrant and colorful"' \
--form 'n="1"' \
--form 'size="1024x1024"'
```

### Supported Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `model` | string | The model to use for image editing | Yes |
| `image` | file | The image to edit, can be passed multiple times | Yes |
| `prompt` | string | A text description of the desired edit | Yes |
| `mask` | file | An image whose transparent areas indicate where to edit | No |
| `n` | integer | Number of images to generate (1-10) | No |
| `size` | string | Size of the generated images | No |
| `response_format` | string | Format of the response: `url` or `b64_json` | No |

## Supported Models
All models listed here https://inference.readthedocs.io/en/latest/models/builtin/embedding/index.html are supported

| Model Name                  | Function Call                                                      |
|-----------------------------|--------------------------------------------------------------------|
| bge-base-en                 | `embedding(model="xinference/bge-base-en", input)`                 |
| bge-base-en-v1.5            | `embedding(model="xinference/bge-base-en-v1.5", input)`            |
| bge-base-zh                 | `embedding(model="xinference/bge-base-zh", input)`                 |
| bge-base-zh-v1.5            | `embedding(model="xinference/bge-base-zh-v1.5", input)`            |
| bge-large-en                | `embedding(model="xinference/bge-large-en", input)`                |
| bge-large-en-v1.5           | `embedding(model="xinference/bge-large-en-v1.5", input)`           |
| bge-large-zh                | `embedding(model="xinference/bge-large-zh", input)`                |
| bge-large-zh-noinstruct     | `embedding(model="xinference/bge-large-zh-noinstruct", input)`     |
| bge-large-zh-v1.5           | `embedding(model="xinference/bge-large-zh-v1.5", input)`           |
| bge-small-en-v1.5           | `embedding(model="xinference/bge-small-en-v1.5", input)`           |
| bge-small-zh                | `embedding(model="xinference/bge-small-zh", input)`                |
| bge-small-zh-v1.5           | `embedding(model="xinference/bge-small-zh-v1.5", input)`           |
| e5-large-v2                 | `embedding(model="xinference/e5-large-v2", input)`                 |
| gte-base                    | `embedding(model="xinference/gte-base", input)`                    |
| gte-large                   | `embedding(model="xinference/gte-large", input)`                   |
| jina-embeddings-v2-base-en  | `embedding(model="xinference/jina-embeddings-v2-base-en", input)`  |
| jina-embeddings-v2-small-en | `embedding(model="xinference/jina-embeddings-v2-small-en", input)` |
| multilingual-e5-large       | `embedding(model="xinference/multilingual-e5-large", input)`       |



