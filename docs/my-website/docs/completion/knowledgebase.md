import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import Image from '@theme/IdealImage';

# Using Vector Stores (Knowledge Bases)

<Image 
  img={require('../../img/kb.png')}
  style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>
<p style={{textAlign: 'left', color: '#666'}}>
  Use Vector Stores with any LiteLLM supported model
</p>


LiteLLM integrates with vector stores, allowing your models to access your organization's data for more accurate and contextually relevant responses.

## Supported Vector Stores
- [Bedrock Knowledge Bases](https://aws.amazon.com/bedrock/knowledge-bases/)

## Quick Start

In order to use a vector store with LiteLLM, you need to 

- Initialize litellm.vector_store_registry
- Pass tools with vector_store_ids to the completion request. Where `vector_store_ids` is a list of vector store ids you initialized in litellm.vector_store_registry

### LiteLLM Python SDK

LiteLLM's allows you to use vector stores in the [OpenAI API spec](https://platform.openai.com/docs/api-reference/chat/create) by passing a tool with vector_store_ids you want to use

```python showLineNumbers title="Basic Bedrock Knowledge Base Usage"
import os
import litellm

from litellm.vector_stores.vector_store_registry import VectorStoreRegistry, LiteLLM_ManagedVectorStore

# Init vector store registry
litellm.vector_store_registry = VectorStoreRegistry(
    vector_stores=[
        LiteLLM_ManagedVectorStore(
            vector_store_id="T37J8R4WTM",
            custom_llm_provider="bedrock"
        )
    ]
)


# Make a completion request with vector_store_ids parameter
response = await litellm.acompletion(
    model="anthropic/claude-3-5-sonnet", 
    messages=[{"role": "user", "content": "What is litellm?"}],
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": ["T37J8R4WTM"]
        }
    ],
)

print(response.choices[0].message.content)
```

### LiteLLM Proxy

#### 1. Configure your vector_store_registry

In order to use a vector store with LiteLLM, you need to configure your vector_store_registry. This tells litellm which vector stores to use and api provider to use for the vector store.

<Tabs>
<TabItem value="config-yaml" label="config.yaml">

```yaml showLineNumbers title="config.yaml"
model_list:
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet
      api_key: os.environ/ANTHROPIC_API_KEY

vector_store_registry:
  - vector_store_name: "bedrock-litellm-website-knowledgebase"
    litellm_params:
      vector_store_id: "T37J8R4WTM"
      custom_llm_provider: "bedrock"
      vector_store_description: "Bedrock vector store for the Litellm website knowledgebase"
      vector_store_metadata:
        source: "https://www.litellm.com/docs"

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

```bash showLineNumbers title="Curl Request to LiteLLM Proxy"
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -d '{
    "model": "claude-3-5-sonnet",
    "messages": [{"role": "user", "content": "What is litellm?"}],
    "tools": [
        {
            "type": "file_search",
            "vector_store_ids": ["T37J8R4WTM"]
        }
    ]
  }'
```

</TabItem>

<TabItem value="openai-sdk" label="OpenAI Python SDK">

```python showLineNumbers title="OpenAI Python SDK Request"
from openai import OpenAI

# Initialize client with your LiteLLM proxy URL
client = OpenAI(
    base_url="http://localhost:4000",
    api_key="your-litellm-api-key"
)

# Make a completion request with vector_store_ids parameter
response = client.chat.completions.create(
    model="claude-3-5-sonnet",
    messages=[{"role": "user", "content": "What is litellm?"}],
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": ["T37J8R4WTM"]
        }
    ]
)

print(response.choices[0].message.content)
```

</TabItem>
</Tabs>




## Advanced

### Logging Vector Store Usage

LiteLLM allows you to view your vector store usage in the LiteLLM UI on the `Logs` page.

After completing a request with a vector store, navigate to the `Logs` page on LiteLLM. Here you should be able to see the query sent to the vector store and corresponding response with scores.

<Image 
  img={require('../../img/kb_4.png')}
  style={{width: '80%'}}
/>
<p style={{textAlign: 'left', color: '#666'}}>
  LiteLLM Logs Page: Vector Store Usage
</p>


### Listing available vector stores

You can list all available vector stores using the /vector_store/list endpoint

**Request:**
```bash showLineNumbers title="List all available vector stores"
curl -X GET "http://localhost:4000/vector_store/list" \
  -H "Authorization: Bearer $LITELLM_API_KEY"
```

**Response:**

The response will be a list of all vector stores that are available to use with LiteLLM.

```json
{
  "object": "list",
  "data": [
    {
      "vector_store_id": "T37J8R4WTM",
      "custom_llm_provider": "bedrock",
      "vector_store_name": "bedrock-litellm-website-knowledgebase",
      "vector_store_description": "Bedrock vector store for the Litellm website knowledgebase",
      "vector_store_metadata": {
        "source": "https://www.litellm.com/docs"
      },
      "created_at": "2023-05-03T18:21:36.462Z",
      "updated_at": "2023-05-03T18:21:36.462Z",
      "litellm_credential_name": "bedrock_credentials"
    }
  ],
  "total_count": 1,
  "current_page": 1,
  "total_pages": 1
}
```


### Always on for a model

**Use this if you want vector stores to be used by default for a specific model.**

In this config, we add `vector_store_ids` to the claude-3-5-sonnet-with-vector-store model. This means that any request to the claude-3-5-sonnet-with-vector-store model will always use the vector store with the id `T37J8R4WTM` defined in the `vector_store_registry`.

```yaml showLineNumbers title="Always on for a model"
model_list:
  - model_name: claude-3-5-sonnet-with-vector-store
    litellm_params:
      model: anthropic/claude-3-5-sonnet
      vector_store_ids: ["T37J8R4WTM"]

vector_store_registry:
  - vector_store_name: "bedrock-litellm-website-knowledgebase"
    litellm_params:
      vector_store_id: "T37J8R4WTM"
      custom_llm_provider: "bedrock"
      vector_store_description: "Bedrock vector store for the Litellm website knowledgebase"
      vector_store_metadata:
        source: "https://www.litellm.com/docs"
```

## How It Works

LiteLLM implements a `BedrockKnowledgeBaseHook` that intercepts your completion requests for handling the integration with Bedrock Knowledge Bases.

1. You make a completion request with the `vector_store_ids` parameter
2. LiteLLM automatically:
   - Uses your last message as the query to retrieve relevant information from the Knowledge Base
   - Adds the retrieved context to your conversation
   - Sends the augmented messages to the model

#### Example Transformation

When you pass `vector_store_ids=["YOUR_KNOWLEDGE_BASE_ID"]`, your request flows through these steps:

**1. Original Request to LiteLLM:**
```json
{
    "model": "anthropic/claude-3-5-sonnet",
    "messages": [
        {"role": "user", "content": "What is litellm?"}
    ],
    "vector_store_ids": ["YOUR_KNOWLEDGE_BASE_ID"]
}
```

**2. Request to AWS Bedrock Knowledge Base:**
```json
{
    "retrievalQuery": {
        "text": "What is litellm?"
    }
}
```
This is sent to: `https://bedrock-agent-runtime.{aws_region}.amazonaws.com/knowledgebases/YOUR_KNOWLEDGE_BASE_ID/retrieve`

**3. Final Request to LiteLLM:**
```json
{
    "model": "anthropic/claude-3-5-sonnet",
    "messages": [
        {"role": "user", "content": "What is litellm?"},
        {"role": "user", "content": "Context: \n\nLiteLLM is an open-source SDK to simplify LLM API calls across providers (OpenAI, Claude, etc). It provides a standardized interface with robust error handling, streaming, and observability tools."}
    ]
}
```

This process happens automatically whenever you include the `vector_store_ids` parameter in your request.

## API Reference

### LiteLLM Completion Knowledge Base Parameters

When using the Knowledge Base integration with LiteLLM, you can include the following parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `vector_store_ids` | List[str] | List of Knowledge Base IDs to query |
