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
- `nvidia/llama-3_2-nv-rerankqa-1b-v2`
- All Nvidia NIM rerank models on their platform

## Quick Start

### Basic Usage

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
| `truncate` | string | `"END"` | How to truncate long inputs: `"NONE"` or `"END"` |

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

## Model Name Format

**Important:** Use underscores (`_`) instead of periods (`.`) in version numbers for the model name.

✅ **Correct:**
```python
model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2"
```

❌ **Incorrect:**
```python
model="nvidia_nim/nvidia/llama-3.2-nv-rerankqa-1b-v2"  # Don't use periods
```

LiteLLM automatically converts underscores to periods when making API requests to Nvidia.

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

<Tabs>
<TabItem value="python" label="Python">

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",  # Your proxy key
)

# Note: OpenAI SDK doesn't have native rerank support
# Use requests or httpx instead
import requests

response = requests.post(
    "http://0.0.0.0:4000/rerank",
    headers={"Authorization": "Bearer sk-1234"},
    json={
        "model": "nvidia-rerank",
        "query": "What is the GPU memory bandwidth of H100?",
        "documents": [
            "H100 delivers 3TB/s memory bandwidth",
            "A100 has 2TB/s memory bandwidth",
            "V100 offers 900GB/s memory bandwidth"
        ],
        "top_n": 2
    }
)

print(response.json())
```

</TabItem>
<TabItem value="curl" label="Curl">

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

</TabItem>
</Tabs>

## Advanced Usage

### Document Objects

You can pass documents as objects with a `text` field:

```python
response = litellm.rerank(
    model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
    query="Best GPU for AI",
    documents=[
        {"text": "H100 is optimized for AI workloads", "id": "doc1"},
        {"text": "A100 provides excellent performance", "id": "doc2"},
    ],
    top_n=1,
)
```

### Handling Long Documents

Use the `truncate` parameter to handle documents that exceed the model's context window:

```python
response = litellm.rerank(
    model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
    query="Technical specifications",
    documents=[very_long_document_1, very_long_document_2],
    truncate="END",  # Automatically truncate from the end
)
```

## Cost Tracking

LiteLLM automatically tracks rerank API costs for Nvidia NIM models when configured with pricing information.

```python
response = litellm.rerank(
    model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
    query="AI performance",
    documents=["doc1", "doc2", "doc3"],
)

# Access cost information
print(f"Cost: ${response._hidden_params.get('response_cost', 0)}")
```

## Error Handling

```python
try:
    response = litellm.rerank(
        model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
        query="test query",
        documents=["doc1", "doc2"],
    )
except litellm.exceptions.APIConnectionError as e:
    print(f"Connection error: {e}")
except litellm.exceptions.RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except Exception as e:
    print(f"Error: {e}")
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

## API Endpoint

The rerank endpoint uses a different base URL than chat/embeddings:

- **Chat/Embeddings:** `https://integrate.api.nvidia.com/v1/`
- **Rerank:** `https://ai.api.nvidia.com/v1/`

LiteLLM automatically uses the correct endpoint for rerank requests.

### How do I get an API key?

Get your Nvidia NIM API key from [Nvidia's website](https://developer.nvidia.com/nim/).

## Related Documentation

- [Nvidia NIM - Main Documentation](./nvidia_nim)
- [Nvidia NIM Chat Completions](./nvidia_nim#sample-usage)
- [LiteLLM Rerank Endpoint](../rerank)
- [Nvidia NIM Official Docs ↗](https://docs.api.nvidia.com/nim/reference/)

