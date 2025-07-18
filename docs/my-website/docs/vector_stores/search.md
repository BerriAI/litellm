import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /vector_stores/\{vector_store_id\}/search - Search Vector Store

Search a vector store for relevant chunks based on a query and file attributes filter. This is useful for retrieval-augmented generation (RAG) use cases.

## Overview

| Feature | Supported | Notes |
|---------|-----------|-------|
| Cost Tracking | ✅ | Tracked per search operation |
| Logging | ✅ | Works across all integrations |
| End-user Tracking | ✅ | |
| Support LLM Providers | **OpenAI, Azure OpenAI, Bedrock, Vertex RAG Engine** | Full vector stores API support across providers |

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

### OpenAI SDK (Standalone)

<Tabs>
<TabItem value="openai-direct" label="Direct OpenAI Usage">

```python showLineNumbers title="OpenAI SDK Direct"
from openai import OpenAI

client = OpenAI(api_key="your-openai-api-key")

search_results = client.beta.vector_stores.search(
    vector_store_id="vs_abc123",
    query="What is the capital of France?",
    max_num_results=5
)
print(search_results)
```

</TabItem>
</Tabs>

## Request Format

The request body follows OpenAI's vector stores search API format.

#### Example request body

```json
{
  "query": "What is the capital of France?",
  "filters": {
    "file_ids": ["file-abc123", "file-def456"]
  },
  "max_num_results": 5,
  "ranking_options": {
    "score_threshold": 0.7
  },
  "rewrite_query": true
}
```

#### Required Fields
- **query** (string or array of strings): A query string or array for the search. The query is used to find relevant chunks in the vector store.

#### Optional Fields
- **filters** (object): Optional filter to apply based on file attributes.
  - **file_ids** (array of strings): Filter chunks based on specific file IDs.
- **max_num_results** (integer): Maximum number of results to return. Must be between 1 and 50. Default is 10.
- **ranking_options** (object): Optional ranking options for search.
  - **score_threshold** (number): Minimum similarity score threshold for results.
- **rewrite_query** (boolean): Whether to rewrite the natural language query for vector search optimization. Default is true.

## Response Format

#### Example Response

```json
{
  "object": "vector_store.search_results.page",
  "search_query": "What is the capital of France?",
  "data": [
    {
      "score": 0.95,
      "content": [
        {
          "type": "text",
          "text": "Paris is the capital and most populous city of France. With an official estimated population of 2,102,650 residents as of 1 January 2023 in an area of more than 105 km², Paris is the fourth-most populated city in the European Union and the 30th most densely populated city in the world in 2022."
        }
      ]
    },
    {
      "score": 0.87,
      "content": [
        {
          "type": "text", 
          "text": "France, officially the French Republic, is a country located primarily in Western Europe. Its capital is Paris, one of the most important cultural and economic centers in Europe."
        }
      ]
    }
  ]
}
```

#### Response Fields

- **object** (string): The object type, which is always `vector_store.search_results.page`.
- **search_query** (string): The query that was used for the search.
- **data** (array): An array of search result objects.
  - **score** (number): The similarity score of the search result, typically between 0 and 1, where 1 is the most similar.
  - **content** (array): Array of content objects containing the retrieved text.
    - **type** (string): The type of content, typically `text`.
    - **text** (string): The actual text content that was retrieved from the vector store.

## Mock Response Testing

For testing purposes, you can use mock responses:

```python showLineNumbers title="Mock Response Example"
import litellm

# Mock response for testing
mock_results = [
    {
        "score": 0.95,
        "content": [
            {
                "text": "Paris is the capital of France.",
                "type": "text"
            }
        ]
    },
    {
        "score": 0.87,
        "content": [
            {
                "text": "France is a country in Western Europe.",
                "type": "text"
            }
        ]
    }
]

response = await litellm.vector_stores.asearch(
    vector_store_id="vs_abc123",
    query="What is the capital of France?",
    mock_response=mock_results
)
print(response)
```

## Error Handling

Common errors you might encounter:

```python showLineNumbers title="Error Handling Example"
import litellm

try:
    response = await litellm.vector_stores.asearch(
        vector_store_id="vs_invalid",
        query="What is the capital of France?"
    )
except litellm.NotFoundError as e:
    print(f"Vector store not found: {e}")
except litellm.RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Best Practices

1. **Query Optimization**: Use clear, specific queries for better search results.
2. **Result Filtering**: Use file_ids filter to limit search scope when needed.
3. **Score Thresholds**: Set appropriate score thresholds to filter out irrelevant results.
4. **Batch Queries**: Use array queries when searching for multiple related topics.
5. **Error Handling**: Always implement proper error handling for production use.

```python showLineNumbers title="Best Practices Example"
import litellm

async def search_documents(vector_store_id: str, user_query: str):
    """
    Search documents with best practices applied
    """
    try:
        response = await litellm.vector_stores.asearch(
            vector_store_id=vector_store_id,
            query=user_query,
            max_num_results=5,
            ranking_options={
                "score_threshold": 0.7  # Filter out low-relevance results
            },
            rewrite_query=True  # Optimize query for vector search
        )
        
        # Filter results by score for additional quality control
        high_quality_results = [
            result for result in response.data 
            if result.score >= 0.8
        ]
        
        return high_quality_results
        
    except Exception as e:
        print(f"Search failed: {e}")
        return []

# Usage
results = await search_documents("vs_abc123", "What is the capital of France?")
``` 