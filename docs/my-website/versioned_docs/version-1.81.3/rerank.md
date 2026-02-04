# /rerank

:::tip

LiteLLM Follows the [cohere api request / response for the rerank api](https://cohere.com/rerank)

:::

## Overview

| Feature | Supported | Notes |
|---------|-----------|-------|
| Cost Tracking | ✅ | Works with all supported models |
| Logging | ✅ | Works across all integrations |
| End-user Tracking | ✅ | |
| Fallbacks | ✅ | Works between supported models |
| Loadbalancing | ✅ | Works between supported models |
| Guardrails | ✅ | Applies to input query only (not documents) |
| Supported Providers | Cohere, Together AI, Azure AI, DeepInfra, Nvidia NIM, Infinity, Fireworks AI, Voyage AI | |

## **LiteLLM Python SDK Usage**
### Quick Start 

```python
from litellm import rerank
import os

os.environ["COHERE_API_KEY"] = "sk-.."

query = "What is the capital of the United States?"
documents = [
    "Carson City is the capital city of the American state of Nevada.",
    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
    "Washington, D.C. is the capital of the United States.",
    "Capital punishment has existed in the United States since before it was a country.",
]

response = rerank(
    model="cohere/rerank-english-v3.0",
    query=query,
    documents=documents,
    top_n=3,
)
print(response)
```

### Async Usage 

```python
from litellm import arerank
import os, asyncio

os.environ["COHERE_API_KEY"] = "sk-.."

async def test_async_rerank(): 
    query = "What is the capital of the United States?"
    documents = [
        "Carson City is the capital city of the American state of Nevada.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
        "Washington, D.C. is the capital of the United States.",
        "Capital punishment has existed in the United States since before it was a country.",
    ]

    response = await arerank(
        model="cohere/rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=3,
    )
    print(response)

asyncio.run(test_async_rerank())
```

## **LiteLLM Proxy Usage**

LiteLLM provides an cohere api compatible `/rerank` endpoint for Rerank calls.

**Setup**

Add this to your litellm proxy config.yaml

```yaml
model_list:
  - model_name: Salesforce/Llama-Rank-V1
    litellm_params:
      model: together_ai/Salesforce/Llama-Rank-V1
      api_key: os.environ/TOGETHERAI_API_KEY
  - model_name: rerank-english-v3.0
    litellm_params:
      model: cohere/rerank-english-v3.0
      api_key: os.environ/COHERE_API_KEY
```

Start litellm

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

Test request

```bash
curl http://0.0.0.0:4000/rerank \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rerank-english-v3.0",
    "query": "What is the capital of the United States?",
    "documents": [
        "Carson City is the capital city of the American state of Nevada.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
        "Washington, D.C. is the capital of the United States.",
        "Capital punishment has existed in the United States since before it was a country."
    ],
    "top_n": 3
  }'
```

## **Supported Providers**

#### ⚡️See all supported models and providers at [models.litellm.ai](https://models.litellm.ai/)

| Provider    | Link to Usage      |
|-------------|--------------------|
| Cohere (v1 + v2 clients)      |   [Usage](#quick-start)                 |
| Together AI|   [Usage](../docs/providers/togetherai)                 |  
| Azure AI|   [Usage](../docs/providers/azure_ai#rerank-endpoint)                 |  
| Jina AI|   [Usage](../docs/providers/jina_ai)                 |  
| AWS Bedrock|   [Usage](../docs/providers/bedrock#rerank-api)                 |  
| HuggingFace|   [Usage](../docs/providers/huggingface_rerank)                 |  
| Infinity|   [Usage](../docs/providers/infinity)                 |  
| vLLM|   [Usage](../docs/providers/vllm#rerank-endpoint)                 |  
| DeepInfra|   [Usage](../docs/providers/deepinfra#rerank-endpoint)                 |
| Vertex AI|   [Usage](../docs/providers/vertex#rerank-api)                 |
| Fireworks AI|   [Usage](../docs/providers/fireworks_ai#rerank-endpoint)                 |
| Voyage AI|   [Usage](../docs/providers/voyage#rerank)                 |  