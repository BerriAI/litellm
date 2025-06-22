# Vertex AI Image Generation

Vertex AI Image Generation uses Google's Imagen models to generate high-quality images from text descriptions.

| Property | Details |
|----------|---------|
| Description | Vertex AI Image Generation uses Google's Imagen models to generate high-quality images from text descriptions. |
| Provider Route on LiteLLM | `vertex_ai/` |
| Provider Doc | [Google Cloud Vertex AI Image Generation ↗](https://cloud.google.com/vertex-ai/docs/generative-ai/image/generate-images) |

## Quick Start

### LiteLLM Python SDK

```python showLineNumbers title="Basic Image Generation"
import litellm

# Generate a single image
response = await litellm.aimage_generation(
    prompt="An olympic size swimming pool with crystal clear water and modern architecture",
    model="vertex_ai/imagen-4.0-generate-preview-06-06",
    vertex_ai_project="your-project-id",
    vertex_ai_location="us-central1",
)

print(response.data[0].url)
```

### LiteLLM Proxy

#### 1. Configure your config.yaml

```yaml showLineNumbers title="Vertex AI Image Generation Configuration"
model_list:
  - model_name: vertex-imagen
    litellm_params:
      model: vertex_ai/imagen-4.0-generate-preview-06-06
      vertex_ai_project: "your-project-id"
      vertex_ai_location: "us-central1"
      vertex_ai_credentials: "path/to/service-account.json"  # Optional if using environment auth
```

#### 2. Start LiteLLM Proxy Server

```bash title="Start LiteLLM Proxy Server"
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

#### 3. Make requests with OpenAI Python SDK

```python showLineNumbers title="Basic Image Generation via Proxy"
from openai import OpenAI

# Initialize client with your proxy URL
client = OpenAI(
    base_url="http://localhost:4000",  # Your proxy URL
    api_key="your-proxy-api-key"      # Your proxy API key
)

# Generate image
response = client.images.generate(
    model="vertex-imagen",
    prompt="An olympic size swimming pool with crystal clear water and modern architecture",
)

print(response.data[0].url)
```

## Supported Models


:::tip

**We support ALL Vertex AI Image Generation models, just set `model=vertex_ai/<any-model-on-vertex_ai>` as a prefix when sending litellm requests**

:::

LiteLLM supports all Vertex AI Imagen models available through Google Cloud.

For the complete and up-to-date list of supported models, visit: [https://models.litellm.ai/](https://models.litellm.ai/)

