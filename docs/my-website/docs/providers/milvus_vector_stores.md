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
| Passthrough | ✅ Supported | Use native Milvus API format |

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

## Passthrough API (Native Milvus Format)

Use this to allow developers to **create** and **search** vector stores using the native Milvus API format, without giving them the Milvus credentials.

This is for the proxy only.

### Admin Flow

#### 1. Add the vector store to LiteLLM

```yaml
model_list:  
  - model_name: embedding-model
    litellm_params:
      model: azure/text-embedding-3-large
      api_base: https://your-endpoint.cognitiveservices.azure.com/
      api_key: os.environ/AZURE_API_KEY
      api_version: "2025-09-01"

vector_store_registry:
  - vector_store_name: "milvus-store"
    litellm_params:
      vector_store_id: "can-be-anything" # vector store id can be anything for the purpose of passthrough api
      custom_llm_provider: "milvus"
      api_key: os.environ/MILVUS_API_KEY
      api_base: https://your-milvus-instance.milvus.io

general_settings:
    database_url: "postgresql://user:password@host:port/database"
    master_key: "sk-1234"
```

Add your vector store credentials to LiteLLM.

#### 2. Start the proxy

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

#### 3. Create a virtual index

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/indexes' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{ 
    "index_name": "dall-e-6",
    "litellm_params": {
        "vector_store_index": "real-collection-name",
        "vector_store_name": "milvus-store"
    }
}'
```

This is a virtual index, which the developer can use to create and search vector stores.

#### 4. Create a key with the vector store permissions

```bash
curl -L -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "allowed_vector_store_indexes": [{"index_name": "dall-e-6", "index_permissions": ["write", "read"]}],
    "models": ["embedding-model"]
}'
```

Give the key access to the virtual index and the embedding model.

**Expected response**

```json
{
    "key": "sk-my-virtual-key"
}
```

### Developer Flow

#### 1. Create a collection with schema

Note: Use the `/milvus` endpoint for the passthrough api that uses the `milvus` provider in your config.

```python
from milvus_rest_client import MilvusRESTClient, DataType
import random
import time

# Configuration
uri = "http://0.0.0.0:4000/milvus"  # IMPORTANT: Use the '/milvus' endpoint for passthrough
token = "sk-my-virtual-key"
collection_name = "dall-e-6"  # Virtual index name

# Initialize client
milvus_client = MilvusRESTClient(uri=uri, token=token)
print(f"Connected to DB: {uri} successfully")

# Check if the collection exists and drop if it does
check_collection = milvus_client.has_collection(collection_name)
if check_collection:
    milvus_client.drop_collection(collection_name)
    print(f"Dropped the existing collection {collection_name} successfully")

# Define schema
dim = 64  # Vector dimension

print("Start to create the collection schema")
schema = milvus_client.create_schema()
schema.add_field(
    "book_id", DataType.INT64, is_primary=True, description="customized primary id"
)
schema.add_field("word_count", DataType.INT64, description="word count")
schema.add_field(
    "book_intro", DataType.FLOAT_VECTOR, dim=dim, description="book introduction"
)

# Prepare index parameters
print("Start to prepare index parameters with default AUTOINDEX")
index_params = milvus_client.prepare_index_params()
index_params.add_index("book_intro", metric_type="L2")

# Create collection
print(f"Start to create example collection: {collection_name}")
milvus_client.create_collection(
    collection_name, schema=schema, index_params=index_params
)
collection_property = milvus_client.describe_collection(collection_name)
print("Collection details: %s" % collection_property)
```

#### 2. Insert data into the collection

```python
# Insert data with customized ids
nb = 1000
insert_rounds = 2
start = 0  # first primary key id
total_rt = 0  # total response time for insert

print(
    f"Start to insert {nb*insert_rounds} entities into example collection: {collection_name}"
)
for i in range(insert_rounds):
    vector = [random.random() for _ in range(dim)]
    rows = [
        {"book_id": i, "word_count": random.randint(1, 100), "book_intro": vector}
        for i in range(start, start + nb)
    ]
    t0 = time.time()
    milvus_client.insert(collection_name, rows)
    ins_rt = time.time() - t0
    start += nb
    total_rt += ins_rt
print(f"Insert completed in {round(total_rt, 4)} seconds")

# Flush the collection
print("Start to flush")
start_flush = time.time()
milvus_client.flush(collection_name)
end_flush = time.time()
print(f"Flush completed in {round(end_flush - start_flush, 4)} seconds")
```

#### 3. Search the collection

```python
# Search configuration
nq = 3  # Number of query vectors
search_params = {"metric_type": "L2", "params": {"level": 2}}
limit = 2  # Number of results to return

# Perform searches
for i in range(5):
    search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
    t0 = time.time()
    results = milvus_client.search(
        collection_name,
        data=search_vectors,
        limit=limit,
        search_params=search_params,
        anns_field="book_intro",
    )
    t1 = time.time()
    print(f"Search {i} results: {results}")
    print(f"Search {i} latency: {round(t1-t0, 4)} seconds")
```

#### Complete Example

Here's a full working example:

```python
from milvus_rest_client import MilvusRESTClient, DataType
import random
import time

# ----------------------------
# 🔐 CONFIGURATION
# ----------------------------
uri = "http://0.0.0.0:4000/milvus"  # IMPORTANT: Use the '/milvus' endpoint
token = "sk-my-virtual-key"
collection_name = "dall-e-6"  # Your virtual index name

# ----------------------------
# 📋 STEP 1 — Initialize Client
# ----------------------------
milvus_client = MilvusRESTClient(uri=uri, token=token)
print(f"✅ Connected to DB: {uri} successfully")

# ----------------------------
# 🗑️  STEP 2 — Drop Existing Collection (if needed)
# ----------------------------
check_collection = milvus_client.has_collection(collection_name)
if check_collection:
    milvus_client.drop_collection(collection_name)
    print(f"🗑️  Dropped the existing collection {collection_name} successfully")

# ----------------------------
# 📐 STEP 3 — Create Collection Schema
# ----------------------------
dim = 64  # Vector dimension

print("📐 Creating the collection schema")
schema = milvus_client.create_schema()
schema.add_field(
    "book_id", DataType.INT64, is_primary=True, description="customized primary id"
)
schema.add_field("word_count", DataType.INT64, description="word count")
schema.add_field(
    "book_intro", DataType.FLOAT_VECTOR, dim=dim, description="book introduction"
)

# ----------------------------
# 🔍 STEP 4 — Create Index
# ----------------------------
print("🔍 Preparing index parameters with default AUTOINDEX")
index_params = milvus_client.prepare_index_params()
index_params.add_index("book_intro", metric_type="L2")

# ----------------------------
# 🏗️  STEP 5 — Create Collection
# ----------------------------
print(f"🏗️  Creating collection: {collection_name}")
milvus_client.create_collection(
    collection_name, schema=schema, index_params=index_params
)
collection_property = milvus_client.describe_collection(collection_name)
print(f"✅ Collection created: {collection_property}")

# ----------------------------
# 📤 STEP 6 — Insert Data
# ----------------------------
nb = 1000
insert_rounds = 2
start = 0
total_rt = 0

print(f"📤 Inserting {nb*insert_rounds} entities into collection")
for i in range(insert_rounds):
    vector = [random.random() for _ in range(dim)]
    rows = [
        {"book_id": i, "word_count": random.randint(1, 100), "book_intro": vector}
        for i in range(start, start + nb)
    ]
    t0 = time.time()
    milvus_client.insert(collection_name, rows)
    ins_rt = time.time() - t0
    start += nb
    total_rt += ins_rt
print(f"✅ Insert completed in {round(total_rt, 4)} seconds")

# ----------------------------
# 💾 STEP 7 — Flush Collection
# ----------------------------
print("💾 Flushing collection")
start_flush = time.time()
milvus_client.flush(collection_name)
end_flush = time.time()
print(f"✅ Flush completed in {round(end_flush - start_flush, 4)} seconds")

# ----------------------------
# 🔍 STEP 8 — Search
# ----------------------------
nq = 3
search_params = {"metric_type": "L2", "params": {"level": 2}}
limit = 2

print(f"🔍 Performing {5} search operations")
for i in range(5):
    search_vectors = [[random.random() for _ in range(dim)] for _ in range(nq)]
    t0 = time.time()
    results = milvus_client.search(
        collection_name,
        data=search_vectors,
        limit=limit,
        search_params=search_params,
        anns_field="book_intro",
    )
    t1 = time.time()
    print(f"✅ Search {i} results: {results}")
    print(f"   Search {i} latency: {round(t1-t0, 4)} seconds")
```

## How It Works

When you search:

1. LiteLLM converts your query to a vector using the embedding model you specified
2. It sends the vector to your Milvus instance via the `/v2/vectordb/entities/search` endpoint
3. Milvus finds the most similar documents in your collection using vector similarity search
4. Results come back with distance scores

The embedding model can be any model supported by LiteLLM - Azure OpenAI, OpenAI, Bedrock, etc.

