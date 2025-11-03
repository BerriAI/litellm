import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /vector_stores - Create Vector Store

Create a vector store which can be used to store and search document chunks for retrieval-augmented generation (RAG) use cases.

## Overview

| Feature | Supported | Notes |
|---------|-----------|-------|
| Cost Tracking | ✅ | Tracked per vector store operation |
| Logging | ✅ | Works across all integrations |
| End-user Tracking | ✅ | |
| Support LLM Providers (OpenAI `/vector_stores` API) | **OpenAI** | Full vector stores API support across providers |
| Support LLM Providers (Passthrough API) | [**Azure AI**](/docs/providers/azure_ai/azure_ai_vector_stores_passthrough) | Full vector stores API support across providers |

## Usage

### LiteLLM Python SDK

<Tabs>
<TabItem value="basic" label="Basic Usage">

#### Async example
```python showLineNumbers title="Create Vector Store - Basic"
import litellm

response = await litellm.vector_stores.acreate(
    name="My Document Store",
    file_ids=["file-abc123", "file-def456"]
)
print(response)
```

#### Sync example
```python showLineNumbers title="Create Vector Store - Sync"
import litellm

response = litellm.vector_stores.create(
    name="My Document Store", 
    file_ids=["file-abc123", "file-def456"]
)
print(response)
```

</TabItem>

<TabItem value="advanced" label="Advanced Configuration">

#### With expiration and chunking strategy
```python showLineNumbers title="Create Vector Store - Advanced"
import litellm

response = await litellm.vector_stores.acreate(
    name="My Document Store",
    file_ids=["file-abc123", "file-def456"],
    expires_after={
        "anchor": "last_active_at",
        "days": 7
    },
    chunking_strategy={
        "type": "static",
        "static": {
            "max_chunk_size_tokens": 800,
            "chunk_overlap_tokens": 400
        }
    },
    metadata={
        "project": "rag-system",
        "environment": "production"
    }
)
print(response)
```

</TabItem>

<TabItem value="openai-provider" label="OpenAI Provider">

#### Using OpenAI provider explicitly
```python showLineNumbers title="Create Vector Store - OpenAI Provider"
import litellm
import os

# Set API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

response = await litellm.vector_stores.acreate(
    name="My Document Store",
    file_ids=["file-abc123", "file-def456"],
    custom_llm_provider="openai"
)
print(response)
```

</TabItem>
</Tabs>

### LiteLLM Proxy Server

<Tabs>
<TabItem value="proxy-setup" label="Setup & Usage">

1. Setup config.yaml

```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

general_settings:
  # Vector store settings can be added here if needed
```

2. Start proxy 

```bash
litellm --config /path/to/config.yaml
```

3. Test it with OpenAI SDK!

```python showLineNumbers title="OpenAI SDK via LiteLLM Proxy"
from openai import OpenAI

# Point OpenAI SDK to LiteLLM proxy
client = OpenAI(
    base_url="http://0.0.0.0:4000",
    api_key="sk-1234",  # Your LiteLLM API key
)

vector_store = client.beta.vector_stores.create(
    name="My Document Store",
    file_ids=["file-abc123", "file-def456"]
)
print(vector_store)
```

</TabItem>

<TabItem value="curl-proxy" label="curl">

```bash showLineNumbers title="Create Vector Store via curl"
curl -L -X POST 'http://0.0.0.0:4000/v1/vector_stores' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "name": "My Document Store",
  "file_ids": ["file-abc123", "file-def456"],
  "expires_after": {
    "anchor": "last_active_at", 
    "days": 7
  },
  "chunking_strategy": {
    "type": "static",
    "static": {
      "max_chunk_size_tokens": 800,
      "chunk_overlap_tokens": 400
    }
  },
  "metadata": {
    "project": "rag-system",
    "environment": "production"
  }
}'
```

</TabItem>
</Tabs>

### OpenAI SDK (Standalone)

<Tabs>
<TabItem value="openai-direct" label="Direct OpenAI Usage">

```python showLineNumbers title="OpenAI SDK Direct"
from openai import OpenAI

client = OpenAI(api_key="your-openai-api-key")

vector_store = client.beta.vector_stores.create(
    name="My Document Store",
    file_ids=["file-abc123", "file-def456"]
)
print(vector_store)
```

</TabItem>
</Tabs>

## Request Format

The request body follows OpenAI's vector stores API format.

#### Example request body

```json
{
  "name": "My Document Store",
  "file_ids": ["file-abc123", "file-def456"],
  "expires_after": {
    "anchor": "last_active_at",
    "days": 7
  },
  "chunking_strategy": {
    "type": "static",
    "static": {
      "max_chunk_size_tokens": 800,
      "chunk_overlap_tokens": 400
    }
  },
  "metadata": {
    "project": "rag-system",
    "environment": "production"
  }
}
```

#### Optional Fields
- **name** (string): The name of the vector store.
- **file_ids** (array of strings): A list of File IDs that the vector store should use. Useful for tools like `file_search` that can access files.
- **expires_after** (object): The expiration policy for the vector store.
  - **anchor** (string): Anchor timestamp after which the expiration policy applies. Supported anchors: `last_active_at`.
  - **days** (integer): The number of days after the anchor time that the vector store will expire.
- **chunking_strategy** (object): The chunking strategy used to chunk the file(s). If not set, will use the `auto` strategy.
  - **type** (string): Always `static`.
  - **static** (object): The static chunking strategy.
    - **max_chunk_size_tokens** (integer): The maximum number of tokens in each chunk. The default value is `800`. The minimum value is `100` and the maximum value is `4096`.
    - **chunk_overlap_tokens** (integer): The number of tokens that overlap between chunks. The default value is `400`.
- **metadata** (object): Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values can be a maximum of 512 characters long.

## Response Format

#### Example Response

```json
{
  "id": "vs_abc123",
  "object": "vector_store",
  "created_at": 1699061776,
  "name": "My Document Store",
  "bytes": 139920,
  "file_counts": {
    "in_progress": 0,
    "completed": 2,
    "failed": 0,
    "cancelled": 0,
    "total": 2
  },
  "status": "completed",
  "expires_after": {
    "anchor": "last_active_at",
    "days": 7
  },
  "expires_at": null,
  "last_active_at": 1699061776,
  "metadata": {
    "project": "rag-system",
    "environment": "production"
  }
}
```

#### Response Fields

- **id** (string): The identifier, which can be referenced in API endpoints.
- **object** (string): The object type, which is always `vector_store`.
- **created_at** (integer): The Unix timestamp (in seconds) for when the vector store was created.
- **name** (string): The name of the vector store.
- **bytes** (integer): The total number of bytes used by the files in the vector store.
- **file_counts** (object): The file counts for the vector store.
  - **in_progress** (integer): The number of files that are currently being processed.
  - **completed** (integer): The number of files that have been successfully processed.
  - **failed** (integer): The number of files that failed to process.
  - **cancelled** (integer): The number of files that were cancelled.
  - **total** (integer): The total number of files.
- **status** (string): The status of the vector store, which can be either `expired`, `in_progress`, or `completed`. A status of `completed` indicates that the vector store is ready for use.
- **expires_after** (object or null): The expiration policy for the vector store.
- **expires_at** (integer or null): The Unix timestamp (in seconds) for when the vector store will expire.
- **last_active_at** (integer or null): The Unix timestamp (in seconds) for when the vector store was last active.
- **metadata** (object or null): Set of 16 key-value pairs that can be attached to an object.

## Mock Response Testing

For testing purposes, you can use mock responses:

```python showLineNumbers title="Mock Response Example"
import litellm

# Mock response for testing
mock_response = {
    "id": "vs_mock123",
    "object": "vector_store", 
    "created_at": 1699061776,
    "name": "Mock Vector Store",
    "bytes": 0,
    "file_counts": {
        "in_progress": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
        "total": 0
    },
    "status": "completed"
}

response = await litellm.vector_stores.acreate(
    name="Test Store",
    mock_response=mock_response
)
print(response)
``` 