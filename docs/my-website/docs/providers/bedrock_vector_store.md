import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# Bedrock Knowledge Bases

AWS Bedrock Knowledge Bases allows you to connect your LLM's to your organization's data, letting your models retrieve and reference information specific to your business.

| Property | Details |
|----------|---------|
| Description | Bedrock Knowledge Bases connects your data to LLM's, enabling them to retrieve and reference your organization's information in their responses. |
| Provider Route on LiteLLM | `bedrock` in the litellm vector_store_registry |
| Provider Doc | [AWS Bedrock Knowledge Bases ↗](https://aws.amazon.com/bedrock/knowledge-bases/) |

## Quick Start

### LiteLLM Python SDK

```python showLineNumbers title="Example using LiteLLM Python SDK"
import os
import litellm

from litellm.vector_stores.vector_store_registry import VectorStoreRegistry, LiteLLM_ManagedVectorStore

# Init vector store registry with your Bedrock Knowledge Base
litellm.vector_store_registry = VectorStoreRegistry(
    vector_stores=[
        LiteLLM_ManagedVectorStore(
            vector_store_id="YOUR_KNOWLEDGE_BASE_ID",  # KB ID from AWS Bedrock
            custom_llm_provider="bedrock"
        )
    ]
)

# Make a completion request using your Knowledge Base
response = await litellm.acompletion(
    model="anthropic/claude-3-5-sonnet", 
    messages=[{"role": "user", "content": "What does our company policy say about remote work?"}],
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": ["YOUR_KNOWLEDGE_BASE_ID"]
        }
    ],
)

print(response.choices[0].message.content)
```

### LiteLLM Proxy

#### 1. Configure your vector_store_registry

<Tabs>
<TabItem value="config-yaml" label="config.yaml">

```yaml
model_list:
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet
      api_key: os.environ/ANTHROPIC_API_KEY

vector_store_registry:
  - vector_store_name: "bedrock-company-docs"
    litellm_params:
      vector_store_id: "YOUR_KNOWLEDGE_BASE_ID"
      custom_llm_provider: "bedrock"
      vector_store_description: "Bedrock Knowledge Base for company documents"
      vector_store_metadata:
        source: "Company internal documentation"
```

</TabItem>

<TabItem value="litellm-ui" label="LiteLLM UI">

On the LiteLLM UI, Navigate to Experimental > Vector Stores > Create Vector Store. On this page you can create a vector store with a name, vector store id and credentials.

<Image 
  img={require('../../img/kb_2.png')}
  style={{width: '50%'}}
/>

</TabItem>
</Tabs>

#### 2. Make a request with vector_store_ids parameter

<Tabs>
<TabItem value="curl" label="Curl">

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "claude-3-5-sonnet",
    "messages": [{"role": "user", "content": "What does our company policy say about remote work?"}],
    "tools": [
        {
            "type": "file_search",
            "vector_store_ids": ["YOUR_KNOWLEDGE_BASE_ID"]
        }
    ]
  }'
```

</TabItem>

<TabItem value="openai-sdk" label="OpenAI Python SDK">

```python
from openai import OpenAI

# Initialize client with your LiteLLM proxy URL
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-litellm-api-key"
)

# Make a completion request with vector_store_ids parameter
response = client.chat.completions.create(
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "What does our company policy say about remote work?"}],
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": ["YOUR_KNOWLEDGE_BASE_ID"]
        }
    ]
)

print(response.choices[0].message.content)
```

</TabItem>
</Tabs>


## Filter Results

Filter by metadata attributes.

**Operators** (OpenAI-style, auto-translated):
- `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `nin`

**AWS operators** (use directly):
- `equals`, `notEquals`, `greaterThan`, `greaterThanOrEquals`, `lessThan`, `lessThanOrEquals`, `in`, `notIn`, `startsWith`, `listContains`, `stringContains`

<Tabs>
<TabItem value="single-filter" label="Single Filter">

```python
response = await litellm.acompletion(
    model="anthropic/claude-3-5-sonnet",
    messages=[{"role": "user", "content": "What are the latest updates?"}],
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["YOUR_KNOWLEDGE_BASE_ID"],
        "filters": {
            "key": "category",
            "value": "updates",
            "operator": "eq"
        }
    }]
)
```

</TabItem>

<TabItem value="and-filters" label="AND">

```python
response = await litellm.acompletion(
    model="anthropic/claude-3-5-sonnet",
    messages=[{"role": "user", "content": "What are the policies?"}],
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["YOUR_KNOWLEDGE_BASE_ID"],
        "filters": {
            "and": [
                {"key": "category", "value": "policy", "operator": "eq"},
                {"key": "year", "value": 2024, "operator": "gte"}
            ]
        }
    }]
)
```

</TabItem>

<TabItem value="or-filters" label="OR">

```python
response = await litellm.acompletion(
    model="anthropic/claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Show me technical docs"}],
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["YOUR_KNOWLEDGE_BASE_ID"],
        "filters": {
            "or": [
                {"key": "category", "value": "api", "operator": "eq"},
                {"key": "category", "value": "sdk", "operator": "eq"}
            ]
        }
    }]
)
```

</TabItem>

<TabItem value="advanced-filters" label="AWS Operators">

```python
response = await litellm.acompletion(
    model="anthropic/claude-3-5-sonnet",
    messages=[{"role": "user", "content": "Find docs"}],
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["YOUR_KNOWLEDGE_BASE_ID"],
        "filters": {
            "and": [
                {"key": "title", "value": "Guide", "operator": "stringContains"},
                {"key": "tags", "value": "important", "operator": "listContains"}
            ]
        }
    }]
)
```

</TabItem>

<TabItem value="proxy-filters" label="Proxy">

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "claude-3-5-sonnet",
    "messages": [{"role": "user", "content": "What are our policies?"}],
    "tools": [{
        "type": "file_search",
        "vector_store_ids": ["YOUR_KNOWLEDGE_BASE_ID"],
        "filters": {
            "and": [
                {"key": "department", "value": "engineering", "operator": "eq"},
                {"key": "type", "value": "policy", "operator": "eq"}
            ]
        }
    }]
  }'
```

</TabItem>
</Tabs>

## Accessing Search Results

See how to access vector store search results in your response:
- [Accessing Search Results (Non-Streaming & Streaming)](../completion/knowledgebase#accessing-search-results-citations)

## Further Reading

Vector Stores:
- [Always on Vector Stores](https://docs.litellm.ai/docs/completion/knowledgebase#always-on-for-a-model)
- [Listing available vector stores on litellm proxy](https://docs.litellm.ai/docs/completion/knowledgebase#listing-available-vector-stores)
- [How LiteLLM Vector Stores Work](https://docs.litellm.ai/docs/completion/knowledgebase#how-it-works)