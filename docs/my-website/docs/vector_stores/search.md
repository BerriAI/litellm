import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /vector_stores/search - Search Vector Store

Search a vector store for relevant chunks based on a query and file attributes filter. This is useful for retrieval-augmented generation (RAG) use cases.

## Overview

| Feature | Supported | Notes |
|---------|-----------|-------|
| Cost Tracking | ✅ | Tracked per search operation |
| Logging | ✅ | Works across all integrations |
| End-user Tracking | ✅ | |
| Support LLM Providers | **OpenAI, Azure OpenAI, Bedrock, Vertex RAG Engine, Azure AI, Milvus** | Full vector stores API support across providers |

## Usage

### LiteLLM Python SDK

<Tabs>
<TabItem value="basic" label="Basic Usage">

#### Non-streaming example
```python showLineNumbers title="Search Vector Store - Basic"
import litellm

response = await litellm.vector_stores.asearch(
    vector_store_id="vs_abc123",
    query="What is the capital of France?"
)
print(response)
```

#### Synchronous example
```python showLineNumbers title="Search Vector Store - Sync"
import litellm

response = litellm.vector_stores.search(
    vector_store_id="vs_abc123",
    query="What is the capital of France?"
)
print(response)
```

</TabItem>

<TabItem value="advanced" label="Advanced Configuration">

#### With filters and ranking options
```python showLineNumbers title="Search Vector Store - Advanced"
import litellm

response = await litellm.vector_stores.asearch(
    vector_store_id="vs_abc123",
    query="What is the capital of France?",
    filters={
        "file_ids": ["file-abc123", "file-def456"]
    },
    max_num_results=5,
    ranking_options={
        "score_threshold": 0.7
    },
    rewrite_query=True
)
print(response)
```

</TabItem>

<TabItem value="multiple-queries" label="Multiple Queries">

#### Searching with multiple queries
```python showLineNumbers title="Search Vector Store - Multiple Queries"
import litellm

response = await litellm.vector_stores.asearch(
    vector_store_id="vs_abc123",
    query=[
        "What is the capital of France?",
        "What is the population of Paris?"
    ],
    max_num_results=10
)
print(response)
```

</TabItem>

<TabItem value="openai-provider" label="OpenAI Provider">

#### Using OpenAI provider explicitly
```python showLineNumbers title="Search Vector Store - OpenAI Provider"
import litellm
import os

# Set API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

response = await litellm.vector_stores.asearch(
    vector_store_id="vs_abc123",
    query="What is the capital of France?",
    custom_llm_provider="openai"
)
print(response)
```

</TabItem>

<TabItem value="azure-ai-provider" label="Azure AI Provider">

#### Using Azure AI Search
```python showLineNumbers title="Search Vector Store - Azure AI Provider"
import litellm
import os

# Set credentials
os.environ["AZURE_SEARCH_API_KEY"] = "your-search-api-key"

response = await litellm.vector_stores.asearch(
    vector_store_id="my-vector-index",
    query="What is the capital of France?",
    custom_llm_provider="azure_ai",
    azure_search_service_name="your-search-service",
    litellm_embedding_model="azure/text-embedding-3-large",
    litellm_embedding_config={
        "api_base": "your-embedding-endpoint",
        "api_key": "your-embedding-api-key",
    },
    api_key=os.getenv("AZURE_SEARCH_API_KEY"),
)
print(response)
```

[See full Azure AI vector store documentation](../providers/azure_ai_vector_stores.md)

</TabItem>

<TabItem value="milvus-provider" label="Milvus Provider">

#### Using Milvus
```python showLineNumbers title="Search Vector Store - Milvus Provider"
import litellm
import os

# Set credentials
os.environ["MILVUS_API_KEY"] = "your-milvus-api-key"
os.environ["MILVUS_API_BASE"] = "https://your-milvus-instance.milvus.io"

response = await litellm.vector_stores.asearch(
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
)
print(response)
```

[See full Milvus vector store documentation](../providers/milvus_vector_stores.md)

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

search_results = client.beta.vector_stores.search(
    vector_store_id="vs_abc123",
    query="What is the capital of France?",
    max_num_results=5
)
print(search_results)
```

</TabItem>

<TabItem value="curl-proxy" label="curl">

```bash showLineNumbers title="Search Vector Store via curl"
curl -L -X POST 'http://0.0.0.0:4000/v1/vector_stores/vs_abc123/search' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "query": "What is the capital of France?",
  "filters": {
    "file_ids": ["file-abc123", "file-def456"]
  },
  "max_num_results": 5,
  "ranking_options": {
    "score_threshold": 0.7
  },
  "rewrite_query": true
}'
```

</TabItem>
</Tabs>

## Setting Up Vector Stores

To use vector store search, configure your vector stores in the `vector_store_registry`. See the [Vector Store Configuration Guide](../completion/knowledgebase.md) for:

- Provider-specific configuration (Bedrock, OpenAI, Azure, Vertex AI, PG Vector)
- Python SDK and Proxy setup examples  
- Authentication and credential management

## Using Vector Stores with Chat Completions

Pass `vector_store_ids` in chat completion requests to automatically retrieve relevant context. See [Using Vector Stores with Chat Completions](../completion/knowledgebase.md#2-make-a-request-with-vector_store_ids-parameter) for implementation details.