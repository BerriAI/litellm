import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Azure AI Search - Vector Store (Unified API)

Use this to **search** Azure AI Search Vector Stores, with LiteLLM's unified `/chat/completions` API.

## Quick Start

You need three things:
1. An Azure AI Search service
2. An embedding model (to convert your queries to vectors)
3. A search index with vector fields

## Usage

<Tabs>
<TabItem value="sdk" label="SDK">

### Basic Search

```python
from litellm import vector_stores
import os

# Set your credentials
os.environ["AZURE_SEARCH_API_KEY"] = "your-search-api-key"
os.environ["AZURE_AI_SEARCH_EMBEDDING_API_BASE"] = "your-embedding-endpoint"
os.environ["AZURE_AI_SEARCH_EMBEDDING_API_KEY"] = "your-embedding-api-key"

# Search the vector store
response = vector_stores.search(
    vector_store_id="my-vector-index",  # Your Azure AI Search index name
    query="What is the capital of France?",
    custom_llm_provider="azure_ai",
    azure_search_service_name="your-search-service",
    litellm_embedding_model="azure/text-embedding-3-large",
    litellm_embedding_config={
        "api_base": os.getenv("AZURE_AI_SEARCH_EMBEDDING_API_BASE"),
        "api_key": os.getenv("AZURE_AI_SEARCH_EMBEDDING_API_KEY"),
    },
    api_key=os.getenv("AZURE_SEARCH_API_KEY"),
)

print(response)
```

### Async Search

```python
from litellm import vector_stores

response = await vector_stores.asearch(
    vector_store_id="my-vector-index",
    query="What is the capital of France?",
    custom_llm_provider="azure_ai",
    azure_search_service_name="your-search-service",
    litellm_embedding_model="azure/text-embedding-3-large",
    litellm_embedding_config={
        "api_base": os.getenv("AZURE_AI_SEARCH_EMBEDDING_API_BASE"),
        "api_key": os.getenv("AZURE_AI_SEARCH_EMBEDDING_API_KEY"),
    },
    api_key=os.getenv("AZURE_SEARCH_API_KEY"),
)

print(response)
```

### Advanced Options

```python
from litellm import vector_stores

response = vector_stores.search(
    vector_store_id="my-vector-index",
    query="What is the capital of France?",
    custom_llm_provider="azure_ai",
    azure_search_service_name="your-search-service",
    litellm_embedding_model="azure/text-embedding-3-large",
    litellm_embedding_config={
        "api_base": os.getenv("AZURE_AI_SEARCH_EMBEDDING_API_BASE"),
        "api_key": os.getenv("AZURE_AI_SEARCH_EMBEDDING_API_KEY"),
    },
    api_key=os.getenv("AZURE_SEARCH_API_KEY"),
    top_k=10,  # Number of results to return
    azure_search_vector_field="contentVector",  # Custom vector field name
)

print(response)
```

</TabItem>

<TabItem value="proxy" label="PROXY">

### Setup Config

Add this to your config.yaml:

```yaml
vector_store_registry:
  - vector_store_name: "azure-ai-search-litellm-website-knowledgebase"
    litellm_params:
        vector_store_id: "test-litellm-app_1761094730750"
        custom_llm_provider: "azure_ai"
        api_key: os.environ/AZURE_SEARCH_API_KEY
        litellm_embedding_model: "azure/text-embedding-3-large"
        litellm_embedding_config:
            api_base: https://krris-mh44uf7y-eastus2.cognitiveservices.azure.com/
            api_key: os.environ/AZURE_API_KEY
            api_version: "2025-09-01"
```

### Start Proxy

```bash
litellm --config /path/to/config.yaml
```

### Search via API

```bash
curl -X POST 'http://0.0.0.0:4000/v1/vector_stores/my-vector-index/search' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "query": "What is the capital of France?",
}'
```

</TabItem>
</Tabs>

## Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `vector_store_id` | string | Your Azure AI Search index name |
| `custom_llm_provider` | string | Set to `"azure_ai"` |
| `azure_search_service_name` | string | Name of your Azure AI Search service |
| `litellm_embedding_model` | string | Model to generate query embeddings (e.g., `"azure/text-embedding-3-large"`) |
| `litellm_embedding_config` | dict | Config for the embedding model (api_base, api_key, api_version) |
| `api_key` | string | Your Azure AI Search API key |

## Supported Features

| Feature | Status | Notes |
|---------|--------|-------|
| Logging | ✅ Supported | Full logging support available |
| Guardrails | ❌ Not Yet Supported | Guardrails are not currently supported for vector stores |
| Cost Tracking | ✅ Supported | Cost is $0 according to Azure |
| Unified API | ✅ Supported | Call via OpenAI compatible `/v1/vector_stores/search` endpoint |
| Passthrough | ❌ Not yet supported |  |

## Response Format

The response follows the standard LiteLLM vector store format:

```json
{
  "object": "vector_store.search_results.page",
  "search_query": "What is the capital of France?",
  "data": [
    {
      "score": 0.95,
      "content": [
        {
          "text": "Paris is the capital of France...",
          "type": "text"
        }
      ],
      "file_id": "doc_123",
      "filename": "Document doc_123",
      "attributes": {
        "document_id": "doc_123"
      }
    }
  ]
}
```

## How It Works

When you search:

1. LiteLLM converts your query to a vector using the embedding model you specified
2. It sends the vector to Azure AI Search
3. Azure AI Search finds the most similar documents in your index
4. Results come back with similarity scores

The embedding model can be any model supported by LiteLLM - Azure OpenAI, OpenAI, Bedrock, etc.

## Setting Up Your Azure AI Search Index

Your index needs a vector field. Here's what that looks like:

```json
{
  "name": "my-vector-index",
  "fields": [
    {
      "name": "id",
      "type": "Edm.String",
      "key": true
    },
    {
      "name": "content",
      "type": "Edm.String"
    },
    {
      "name": "contentVector",
      "type": "Collection(Edm.Single)",
      "searchable": true,
      "dimensions": 1536,
      "vectorSearchProfile": "myVectorProfile"
    }
  ]
}
```

The vector dimensions must match your embedding model. For example:
- `text-embedding-3-large`: 1536 dimensions
- `text-embedding-3-small`: 1536 dimensions
- `text-embedding-ada-002`: 1536 dimensions


## Common Issues

**"Failed to generate embedding for query"**

Your embedding model config is wrong. Check:
- `litellm_embedding_config` has the right api_base and api_key
- The embedding model name is correct
- Your credentials work

**"Index not found"**

The `vector_store_id` doesn't match any index in your search service. Check:
- The index name is correct
- You're using the right search service name

**"Field 'contentVector' not found"**

Your index uses a different vector field name. Pass it via `azure_search_vector_field`.

