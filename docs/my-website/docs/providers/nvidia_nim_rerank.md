import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Nvidia NIM - Rerank

Use Nvidia NIM Rerank models through LiteLLM.

| Property | Details |
|----------|---------|
| Description | Nvidia NIM provides high-performance reranking models for semantic search and retrieval-augmented generation (RAG) |
| Provider Doc | [Nvidia NIM Rerank API ↗](https://docs.api.nvidia.com/nim/reference/nvidia-llama-3_2-nv-rerankqa-1b-v2-infer) |
| Supported Endpoint | `/rerank` |

## Overview

Nvidia NIM rerank models help you:
- Reorder search results by relevance to a query
- Improve RAG (Retrieval-Augmented Generation) accuracy
- Filter and rank large document sets efficiently

**Supported Models:**
- All Nvidia NIM rerank models on their platform

:::tip

See the full list of LiteLLM supported Nvidia NIM rerank models on [Nvidia NIM](https://models.litellm.ai)

:::

## Usage

### LiteLLM Python SDK

<Tabs>
<TabItem value="llama-1b" label="LLaMa 1B Model">

```python
import litellm
import os

os.environ['NVIDIA_NIM_API_KEY'] = "nvapi-..."

response = litellm.rerank(
    model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
    query="What is the GPU memory bandwidth of H100 SXM?",
    documents=[
        "The Hopper GPU is paired with the Grace CPU using NVIDIA's ultra-fast chip-to-chip interconnect, delivering 900GB/s of bandwidth.",
        "A100 provides up to 20X higher performance over the prior generation.",
        "Accelerated servers with H100 deliver 3 terabytes per second (TB/s) of memory bandwidth per GPU."
    ],
    top_n=3,
)

print(response)
```

</TabItem>
<TabItem value="mistral-4b" label="Mistral 4B Model">

```python
import litellm
import os

os.environ['NVIDIA_NIM_API_KEY'] = "nvapi-..."

response = litellm.rerank(
    model="nvidia_nim/nvidia/nv-rerankqa-mistral-4b-v3",
    query="What is the GPU memory bandwidth of H100 SXM?",
    documents=[
        "The Hopper GPU is paired with the Grace CPU using NVIDIA's ultra-fast chip-to-chip interconnect, delivering 900GB/s of bandwidth.",
        "A100 provides up to 20X higher performance over the prior generation.",
        "Accelerated servers with H100 deliver 3 terabytes per second (TB/s) of memory bandwidth per GPU."
    ],
    top_n=3,
)

print(response)
```

</TabItem>
</Tabs>

**Response:**
```json
{
    "results": [
        {
            "index": 2,
            "relevance_score": 6.828125,
            "document": {
                "text": "Accelerated servers with H100 deliver 3 terabytes per second (TB/s) of memory bandwidth per GPU."
            }
        },
        {
            "index": 0,
            "relevance_score": -1.564453125,
            "document": {
                "text": "The Hopper GPU is paired with the Grace CPU using NVIDIA's ultra-fast chip-to-chip interconnect, delivering 900GB/s of bandwidth."
            }
        }
    ]
}
```


## Usage with LiteLLM Proxy

### 1. Setup Config

Add Nvidia NIM rerank models to your proxy configuration:

```yaml
model_list:
  - model_name: nvidia-rerank
    litellm_params:
      model: nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2
      api_key: os.environ/NVIDIA_NIM_API_KEY
```

### 2. Start Proxy

```bash
litellm --config /path/to/config.yaml
```

### 3. Make Rerank Requests

```bash
curl -X POST http://0.0.0.0:4000/rerank \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia-rerank",
    "query": "What is the GPU memory bandwidth of H100?",
    "documents": [
      "H100 delivers 3TB/s memory bandwidth",
      "A100 has 2TB/s memory bandwidth",
      "V100 offers 900GB/s memory bandwidth"
    ],
    "top_n": 2
  }'
```

## `/v1/ranking` Models (llama-3.2-nv-rerankqa-1b-v2)

Some Nvidia NIM rerank models use the `/v1/ranking` endpoint instead of the default `/v1/retrieval/{model}/reranking` endpoint.

Use the `ranking/` prefix to force requests to the `/v1/ranking` endpoint:

### LiteLLM Python SDK

```python showLineNumbers title="Force /v1/ranking endpoint with ranking/ prefix"
import litellm
import os

os.environ['NVIDIA_NIM_API_KEY'] = "nvapi-..."

# Use "ranking/" prefix to force /v1/ranking endpoint
response = litellm.rerank(
    model="nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2",
    query="which way did the traveler go?",
    documents=[
        "two roads diverged in a yellow wood...",
        "then took the other, as just as fair...",
        "i shall be telling this with a sigh somewhere ages and ages hence..."
    ],
    top_n=3,
    truncate="END",  # Optional: truncate long text from the end
)

print(response)
```

### LiteLLM Proxy

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: nvidia-ranking
    litellm_params:
      model: nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2
      api_key: os.environ/NVIDIA_NIM_API_KEY
```

```bash title="Request to LiteLLM Proxy"
curl -X POST http://0.0.0.0:4000/rerank \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia-ranking",
    "query": "which way did the traveler go?",
    "documents": [
      "two roads diverged in a yellow wood...",
      "then took the other, as just as fair..."
    ],
    "top_n": 2
  }'
```

### Understanding Model Resolution

**Ranking Endpoint (`/v1/ranking`):**

```
model: nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2
       └────┬────┘ └──┬──┘ └─────────────┬──────────────────┘
            │        │                   │
            │        │                   └────▶ Model name sent to provider
            │        │
            │        └────────────────────────▶ Tells LiteLLM the request/response and url should be sent to Nvidia NIM /v1/ranking endpoint
            │
            └─────────────────────────────────▶ Provider prefix

API URL: https://ai.api.nvidia.com/v1/ranking
```

**Visual Flow:**

```
Client Request                LiteLLM                              Provider API
──────────────              ────────────                         ─────────────

# Default reranking endpoint
model: "nvidia_nim/nvidia/model-name"
                            1. Extracts model: nvidia/model-name
                            2. Routes to default endpoint ──────▶ POST /v1/retrieval/nvidia/model-name/reranking


# Forced ranking endpoint  
model: "nvidia_nim/ranking/nvidia/model-name"
                            1. Detects "ranking/" prefix
                            2. Extracts model: nvidia/model-name
                            3. Routes to ranking endpoint ──────▶ POST /v1/ranking
                                                                  Body: {"model": "nvidia/model-name", ...}
```

**When to use each endpoint:**

| Endpoint | Model Prefix | Use Case |
|----------|--------------|----------|
| `/v1/retrieval/{model}/reranking` | `nvidia_nim/<model>` | Default for most rerank models |
| `/v1/ranking` | `nvidia_nim/ranking/<model>` | For models like `nvidia/llama-3.2-nv-rerankqa-1b-v2` that require this endpoint |

:::tip

Check the [Nvidia NIM model deployment page](https://build.nvidia.com/nvidia/llama-3_2-nv-rerankqa-1b-v2/deploy) to see which endpoint your model requires.

:::

## API Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | The Nvidia NIM rerank model name with `nvidia_nim/` prefix |
| `query` | string | The search query to rank documents against |
| `documents` | array | List of documents to rank (1-1000 documents) |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_n` | integer | All documents | Number of top-ranked documents to return |

### Nvidia-Specific Parameters

**`truncate`**: Controls how text is truncated if it exceeds the model's context window
- `"NONE"`: No truncation (request may fail if too long)
- `"END"`: Truncate from the end of the text

```python
response = litellm.rerank(
    model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
    query="GPU performance",
    documents=["High performance computing", "Fast GPU processing"],
    top_n=2,
    truncate="END",  # Nvidia-specific parameter
)
```

## Authentication

Set your Nvidia NIM API key:

<Tabs>
<TabItem value="env" label="Environment Variable">

```bash
export NVIDIA_NIM_API_KEY="nvapi-..."
```

</TabItem>
<TabItem value="python" label="Python">

```python
import os
os.environ['NVIDIA_NIM_API_KEY'] = "nvapi-..."

# Or pass directly
response = litellm.rerank(
    model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
    query="test",
    documents=["doc1"],
    api_key="nvapi-...",
)
```

</TabItem>
</Tabs>

## Custom API Base URL

You can override the default base URL in several ways:

**Option 1: Environment Variable**

```bash
export NVIDIA_NIM_API_BASE="https://your-custom-endpoint.com"
```

**Option 2: Pass as parameter**

```python
response = litellm.rerank(
    model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
    query="test",
    documents=["doc1"],
    api_base="https://your-custom-endpoint.com",
)
```

**Option 3: Full URL (including model path)**

If you have the complete endpoint URL, you can pass it directly:

```python
response = litellm.rerank(
    model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
    query="test",
    documents=["doc1"],
    api_base="https://your-custom-endpoint.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
)
```

LiteLLM will detect the full URL (by checking for `/retrieval/` in the path) and use it as-is.

### How do I get an API key?

Get your Nvidia NIM API key from [Nvidia's website](https://developer.nvidia.com/nim/).

## Related Documentation

- [Nvidia NIM - Main Documentation](./nvidia_nim)
- [Nvidia NIM Chat Completions](./nvidia_nim#sample-usage)
- [LiteLLM Rerank Endpoint](../rerank)
- [Nvidia NIM Official Docs ↗](https://docs.api.nvidia.com/nim/reference/)
