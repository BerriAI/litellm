import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Milvus - Vector Store

Use Milvus as a vector store for RAG.

## Quick Start

You need three things:
1. A Milvus instance (cloud or self-hosted)
2. An embedding model (to convert your queries to vectors)
3. A Milvus collection with vector fields

## Usage

<Tabs>
<TabItem value="sdk" label="SDK">

### Basic Search

```python
from litellm import vector_stores
import os

# Set your credentials
os.environ["MILVUS_API_KEY"] = "your-milvus-api-key"
os.environ["MILVUS_API_BASE"] = "https://your-milvus-instance.milvus.io"

# Search the vector store
response = vector_stores.search(
    vector_store_id="my-collection-name",  # Your Milvus collection name
    query="What is the capital of France?",
    custom_llm_provider="milvus",
    litellm_embedding_model="azure/text-embedding-3-large",
    litellm_embedding_config={
        "api_base": "your-embedding-endpoint",
        "api_key": "your-embedding-api-key",
        "api_version": "2025-09-01"
    },
    milvus_text_field="book_intro",  # Field name that contains text content
    api_key=os.getenv("MILVUS_API_KEY"),
)

print(response)
```

### Async Search

```python
from litellm import vector_stores

response = await vector_stores.asearch(
    vector_store_id="my-collection-name",
    query="What is the capital of France?",
    custom_llm_provider="milvus",
    litellm_embedding_model="azure/text-embedding-3-large",
    litellm_embedding_config={
        "api_base": "your-embedding-endpoint",
        "api_key": "your-embedding-api-key",
        "api_version": "2025-09-01"
    },
    milvus_text_field="book_intro",
    api_key=os.getenv("MILVUS_API_KEY"),
)

print(response)
```

### Advanced Options

```python
from litellm import vector_stores

response = vector_stores.search(
    vector_store_id="my-collection-name",
    query="What is the capital of France?",
    custom_llm_provider="milvus",
    litellm_embedding_model="azure/text-embedding-3-large",
    litellm_embedding_config={
        "api_base": "your-embedding-endpoint",
        "api_key": "your-embedding-api-key",
    },
    milvus_text_field="book_intro",
    api_key=os.getenv("MILVUS_API_KEY"),
    # Milvus-specific parameters
    limit=10,  # Number of results to return
    offset=0,  # Pagination offset
    dbName="default",  # Database name
    annsField="book_intro_vector",  # Vector field name
    outputFields=["id", "book_intro", "title"],  # Fields to return
    filter='book_id > 0',  # Metadata filter expression
    searchParams={"metric_type": "L2", "params": {"nprobe": 10}},  # Search parameters
)

print(response)
```

</TabItem>

<TabItem value="proxy" label="PROXY">

### Setup Config

Add this to your config.yaml:

```yaml
vector_store_registry:
  - vector_store_name: "milvus-knowledgebase"
    litellm_params:
        vector_store_id: "my-collection-name"
        custom_llm_provider: "milvus"
        api_key: os.environ/MILVUS_API_KEY
        api_base: https://your-milvus-instance.milvus.io
        litellm_embedding_model: "azure/text-embedding-3-large"
        litellm_embedding_config:
            api_base: https://your-endpoint.cognitiveservices.azure.com/
            api_key: os.environ/AZURE_API_KEY
            api_version: "2025-09-01"
        milvus_text_field: "book_intro"
        # Optional Milvus parameters
        annsField: "book_intro_vector"
        limit: 10
```

### Start Proxy

```bash
litellm --config /path/to/config.yaml
```

### Search via API

```bash
curl -X POST 'http://0.0.0.0:4000/v1/vector_stores/my-collection-name/search' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "query": "What is the capital of France?"
}'
```

</TabItem>
</Tabs>

## Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `vector_store_id` | string | Your Milvus collection name |
| `custom_llm_provider` | string | Set to `"milvus"` |
| `litellm_embedding_model` | string | Model to generate query embeddings (e.g., `"azure/text-embedding-3-large"`) |
| `litellm_embedding_config` | dict | Config for the embedding model (api_base, api_key, api_version) |
| `milvus_text_field` | string | Field name in your collection that contains text content |
| `api_key` | string | Your Milvus API key (or set `MILVUS_API_KEY` env var) |
| `api_base` | string | Your Milvus API base URL (or set `MILVUS_API_BASE` env var) |

## Optional Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dbName` | string | Database name (default: "default") |
| `annsField` | string | Vector field name to search (default: "book_intro_vector") |
| `limit` | integer | Maximum number of results to return |
| `offset` | integer | Pagination offset |
| `filter` | string | Filter expression for metadata filtering |
| `groupingField` | string | Field to group results by |
| `outputFields` | list | List of fields to return in results |
| `searchParams` | dict | Search parameters like metric type and search parameters |
| `partitionNames` | list | List of partition names to search |
| `consistencyLevel` | string | Consistency level for the search |

## Supported Features

| Feature | Status | Notes |
|---------|--------|-------|
| Logging | ✅ Supported | Full logging support available |
| Guardrails | ❌ Not Yet Supported | Guardrails are not currently supported for vector stores |
| Cost Tracking | ✅ Supported | Cost is $0 for Milvus searches |
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
      "file_id": null,
      "filename": null,
      "attributes": {
        "id": "123",
        "title": "France Geography"
      }
    }
  ]
}
```

## How It Works

When you search:

1. LiteLLM converts your query to a vector using the embedding model you specified
2. It sends the vector to your Milvus instance via the `/v2/vectordb/entities/search` endpoint
3. Milvus finds the most similar documents in your collection using vector similarity search
4. Results come back with distance scores

The embedding model can be any model supported by LiteLLM - Azure OpenAI, OpenAI, Bedrock, etc.

