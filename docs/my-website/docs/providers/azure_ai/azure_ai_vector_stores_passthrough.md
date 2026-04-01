# Azure AI Search - Vector Store (Passthrough API)

Use this to allow developers to **create** and **search** vector stores using the Azure AI Search API in the **native** Azure AI Search API format, without giving them the Azure AI credentials.

This is for the proxy only. 

## Admin Flow

### 1. Add the vector store to LiteLLM 

```yaml
model_list:  
  - model_name: embedding-model
    litellm_params:
      model: openai/text-embedding-3-large


vector_store_registry:
  - vector_store_name: "azure-ai-search"
    litellm_params:
      vector_store_id: "can-be-anything" # vector store id can be anything for the purpose of passthrough api
      custom_llm_provider: "azure_ai"
      api_key: os.environ/AZURE_SEARCH_API_KEY
      api_base: https://azure-kb-search.search.windows.net
      litellm_embedding_model: "azure/text-embedding-3-large"
      litellm_embedding_config:
          api_base: https://krris-mh44uf7y-eastus2.cognitiveservices.azure.com/
          api_key: os.environ/AZURE_API_KEY
          api_version: "2025-09-01"

general_settings:
    database_url: "postgresql://user:password@host:port/database"
    master_key: "sk-1234"
```

Add your vector store credentials to LiteLLM. 

### 2. Start the proxy. 

```bash
litellm --config /path/to/config.yaml

# RUNNING on http://0.0.0.0:4000
```

### 3. Create a virtual index. 

```bash
curl -L -X POST 'http://0.0.0.0:4000/v1/indexes' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{ 
    "index_name": "dall-e-4",
    "litellm_params": {
        "vector_store_index": "real-index-name-2",
        "vector_store_name": "azure-ai-search"
    }

}'
```

This is a virtual index, which the developer can use to create and search vector stores.

### 4. Create a key with the vector store permissions. 

```bash
curl -L -X POST 'http://0.0.0.0:4000/key/generate' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
    "allowed_vector_store_indexes": [{"index_name": "dall-e-4", "index_permissions": ["write", "read"]}],
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

## Developer Flow

### 1. Create a vector store with some documents. 

Note: Use the '/azure_ai' endpoint for the passthrough api that uses the `azure_ai` provider in your `_new_secret_config.yaml` file. 

```python
import requests
import json

# ----------------------------
# 🔐 CONFIGURATION
# ----------------------------
# Azure OpenAI (for embeddings)
AZURE_OPENAI_ENDPOINT = "http://0.0.0.0:4000"
AZURE_OPENAI_KEY = "sk-my-virtual-key"
EMBEDDING_DEPLOYMENT_NAME = "embedding-model"

# Azure AI Search
AZURE_AI_SEARCH_ENDPOINT = "http://0.0.0.0:4000/azure_ai" # IMPORTANT: Use the '/azure_ai' endpoint for the passthrough api to Azure 
SEARCH_API_KEY = "sk-my-virtual-key"
INDEX_NAME = "dall-e-4"



# Vector dimensions (text-embedding-3-large uses 3072 dimensions)
VECTOR_DIMENSIONS = 3072

# Example docs (replace with your own)
documents = [
    {"id": "1", "content": "Refunds must be requested within 30 days."},
    {"id": "2", "content": "We offer 24/7 support for all enterprise customers."},
]


# ----------------------------
# 📋 STEP 0 — Create Index Schema
# ----------------------------
def delete_index_if_exists():
    """Delete the index if it exists"""
    index_url = f"{AZURE_AI_SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version=2024-07-01"
    headers = {"api-key": SEARCH_API_KEY}

    response = requests.delete(index_url, headers=headers)

    if response.status_code == 204:
        print(f"🗑️  Deleted existing index '{INDEX_NAME}'")
        return True
    elif response.status_code == 404:
        print(f"ℹ️  Index '{INDEX_NAME}' does not exist yet")
        return False
    else:
        print(f"⚠️  Delete response: {response.status_code}")
        print(f"    Message: {response.text}")
        return False


def create_index():
    """Create the Azure AI Search index with proper schema"""
    index_url = f"{AZURE_AI_SEARCH_ENDPOINT}/indexes/{INDEX_NAME}?api-version=2024-07-01"
    headers = {"Content-Type": "application/json", "api-key": SEARCH_API_KEY}

    index_schema = {
        "name": INDEX_NAME,
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
            {
                "name": "content",
                "type": "Edm.String",
                "searchable": True,
                "filterable": False,
            },
            {
                "name": "contentVector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": VECTOR_DIMENSIONS,
                "vectorSearchProfile": "my-vector-profile",
            },
        ],
        "vectorSearch": {
            "algorithms": [
                {
                    "name": "my-hnsw-algorithm",
                    "kind": "hnsw",
                    "hnswParameters": {
                        "metric": "cosine",
                        "m": 4,
                        "efConstruction": 400,
                        "efSearch": 500,
                    },
                }
            ],
            "profiles": [
                {"name": "my-vector-profile", "algorithm": "my-hnsw-algorithm"}
            ],
        },
    }

    # Create the index
    response = requests.put(index_url, headers=headers, json=index_schema)

    if response.status_code == 201:
        print(f"✅ Index '{INDEX_NAME}' created successfully.")
        return True
    elif response.status_code == 204:
        print(f"✅ Index '{INDEX_NAME}' updated successfully.")
        return True
    else:
        print(f"❌ Failed to create index: {response.status_code}")
        print(f"    Message: {response.text}")
        return False


# Delete and recreate the index with correct schema
print("🔄 Setting up Azure AI Search index...")
delete_index_if_exists()
if not create_index():
    print("❌ Could not create index. Exiting.")
    exit(1)


# ----------------------------
# 🧠 STEP 1 — Generate Embeddings
# ----------------------------
def get_embedding(text: str):
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{EMBEDDING_DEPLOYMENT_NAME}/embeddings?api-version=2024-10-21"
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}
    payload = {"input": text}
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Embedding failed: {response.status_code}\n{response.text}")
    return response.json()["data"][0]["embedding"]


# Generate embeddings for each document
for doc in documents:
    doc["contentVector"] = get_embedding(doc["content"])
    print(f"✅ Embedded doc {doc['id']} (vector length: {len(doc['contentVector'])})")

# ----------------------------
# 📤 STEP 2 — Upload to Azure AI Search
# ----------------------------
upload_url = f"{AZURE_AI_SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/index?api-version=2024-07-01"
headers = {"Content-Type": "application/json", "api-key": SEARCH_API_KEY}

payload = {
    "value": [
        {
            "@search.action": "upload",
            "id": doc["id"],
            "content": doc["content"],
            "contentVector": doc["contentVector"],
        }
        for doc in documents
    ]
}

response = requests.post(upload_url, headers=headers, data=json.dumps(payload))

# ----------------------------
# 🧾 RESULT
# ----------------------------
if response.status_code == 200:
    print("✅ Documents uploaded successfully.")
else:
    print(f"❌ Upload failed: {response.status_code}")
    print(response.text)

```


### 2. Search the vector store. 


```python
import requests
import json

# ----------------------------
# 🔐 CONFIGURATION
# ----------------------------
# Azure OpenAI (for embeddings)
AZURE_OPENAI_ENDPOINT = "http://0.0.0.0:4000"
AZURE_OPENAI_KEY = "sk-my-virtual-key"
EMBEDDING_DEPLOYMENT_NAME = "embedding-model"

# Azure AI Search
AZURE_AI_SEARCH_ENDPOINT = "http://0.0.0.0:4000/azure_ai"
SEARCH_API_KEY = "sk-my-virtual-key"
INDEX_NAME = "dall-e-4"


# ----------------------------
# 🧠 Generate Query Embedding
# ----------------------------
def get_embedding(text: str):
    """Generate embedding for the query text"""
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{EMBEDDING_DEPLOYMENT_NAME}/embeddings?api-version=2024-10-21"
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}
    payload = {"input": text}
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Embedding failed: {response.status_code}\n{response.text}")
    return response.json()["data"][0]["embedding"]


# ----------------------------
# 🔍 Vector Search Function
# ----------------------------
def search_knowledge_base(query: str, top_k: int = 3):
    """
    Search the knowledge base using vector similarity

    Args:
        query: The search query string
        top_k: Number of top results to return (default: 3)

    Returns:
        List of search results with content and scores
    """
    print(f"🔍 Searching for: '{query}'")

    # Step 1: Generate embedding for the query
    print("   Generating query embedding...")
    query_vector = get_embedding(query)

    # Step 2: Perform vector search
    search_url = f"{AZURE_AI_SEARCH_ENDPOINT}/indexes/{INDEX_NAME}/docs/search?api-version=2024-07-01"
    headers = {"Content-Type": "application/json", "api-key": SEARCH_API_KEY}

    # Build the search request with vector search
    search_payload = {
        "search": "*",  # Get all documents
        "vectorQueries": [
            {
                "vector": query_vector,
                "fields": "contentVector",
                "kind": "vector",
                "k": top_k,  # Number of nearest neighbors to return
            }
        ],
        "select": "id,content",  # Fields to return
        "top": top_k,
    }

    # Execute the search
    response = requests.post(search_url, headers=headers, json=search_payload)

    if response.status_code != 200:
        raise Exception(f"Search failed: {response.status_code}\n{response.text}")

    # Parse and return results
    results = response.json()
    return results.get("value", [])


# ----------------------------
# 📊 Display Results
# ----------------------------
def display_results(results):
    """Pretty print the search results"""
    if not results:
        print("\n❌ No results found.")
        return

    print(f"\n✅ Found {len(results)} results:\n")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n📄 Result #{i}")
        print(f"   ID: {result.get('id', 'N/A')}")
        print(f"   Score: {result.get('@search.score', 'N/A')}")
        print(f"   Content: {result.get('content', 'N/A')}")
        print("-" * 80)


# ----------------------------
# 🎯 MAIN - Example Queries
# ----------------------------
if __name__ == "__main__":
    # Example 1: Search for refund policy
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Refund Policy Query")
    print("=" * 80)
    results = search_knowledge_base("How do I get a refund?", top_k=2)
    display_results(results)

    # Example 2: Search for customer support
    print("\n\n" + "=" * 80)
    print("EXAMPLE 2: Customer Support Query")
    print("=" * 80)
    results = search_knowledge_base("When can I contact support?", top_k=2)
    display_results(results)

    # Example 3: Custom query - uncomment to use
    # print("\n\n" + "=" * 80)
    # print("CUSTOM QUERY")
    # print("=" * 80)
    # custom_query = input("Enter your query: ")
    # results = search_knowledge_base(custom_query, top_k=3)
    # display_results(results)

```