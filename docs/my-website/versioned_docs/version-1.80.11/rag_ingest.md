# /rag/ingest

All-in-one document ingestion pipeline: **Upload → Chunk → Embed → Vector Store**

| Feature | Supported |
|---------|-----------|
| Logging | ✅ |
| Supported Providers | `openai`, `bedrock`, `vertex_ai`, `gemini` |

## Quick Start

### OpenAI

```bash showLineNumbers title="Ingest to OpenAI vector store"
curl -X POST "http://localhost:4000/v1/rag/ingest" \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d "{
        \"file\": {
            \"filename\": \"document.txt\",
            \"content\": \"$(base64 -i document.txt)\",
            \"content_type\": \"text/plain\"
        },
        \"ingest_options\": {
            \"vector_store\": {
                \"custom_llm_provider\": \"openai\"
            }
        }
    }"
```

### Bedrock

```bash showLineNumbers title="Ingest to Bedrock Knowledge Base"
curl -X POST "http://localhost:4000/v1/rag/ingest" \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d "{
        \"file\": {
            \"filename\": \"document.txt\",
            \"content\": \"$(base64 -i document.txt)\",
            \"content_type\": \"text/plain\"
        },
        \"ingest_options\": {
            \"vector_store\": {
                \"custom_llm_provider\": \"bedrock\"
            }
        }
    }"
```

### Vertex AI RAG Engine

```bash showLineNumbers title="Ingest to Vertex AI RAG Corpus"
curl -X POST "http://localhost:4000/v1/rag/ingest" \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d "{
        \"file\": {
            \"filename\": \"document.txt\",
            \"content\": \"$(base64 -i document.txt)\",
            \"content_type\": \"text/plain\"
        },
        \"ingest_options\": {
            \"vector_store\": {
                \"custom_llm_provider\": \"vertex_ai\",
                \"vector_store_id\": \"your-corpus-id\",
                \"gcs_bucket\": \"your-gcs-bucket\"
            }
        }
    }"
```

## Response

```json
{
  "id": "ingest_abc123",
  "status": "completed",
  "vector_store_id": "vs_xyz789",
  "file_id": "file_123"
}
```

## Query the Vector Store

After ingestion, query with `/vector_stores/{vector_store_id}/search`:

```bash showLineNumbers title="Search the vector store"
curl -X POST "http://localhost:4000/v1/vector_stores/vs_xyz789/search" \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "What is the main topic?",
        "max_num_results": 5
    }'
```

## End-to-End Example

### OpenAI

#### 1. Ingest Document

```bash showLineNumbers title="Step 1: Ingest"
curl -X POST "http://localhost:4000/v1/rag/ingest" \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d "{
        \"file\": {
            \"filename\": \"test_document.txt\",
            \"content\": \"$(base64 -i test_document.txt)\",
            \"content_type\": \"text/plain\"
        },
        \"ingest_options\": {
            \"name\": \"test-basic-ingest\",
            \"vector_store\": {
                \"custom_llm_provider\": \"openai\"
            }
        }
    }"
```

Response:
```json
{
  "id": "ingest_d834f544-fc5e-4751-902d-fb0bcc183b85",
  "status": "completed",
  "vector_store_id": "vs_692658d337c4819183f2ad8488d12fc9",
  "file_id": "file-M2pJJiWH56cfUP4Fe7rJay"
}
```

#### 2. Query

```bash showLineNumbers title="Step 2: Query"
curl -X POST "http://localhost:4000/v1/vector_stores/vs_692658d337c4819183f2ad8488d12fc9/search" \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "What is LiteLLM?",
        "custom_llm_provider": "openai"
    }'
```

Response:
```json
{
  "object": "vector_store.search_results.page",
  "search_query": ["What is LiteLLM?"],
  "data": [
    {
      "file_id": "file-M2pJJiWH56cfUP4Fe7rJay",
      "filename": "test_document.txt",
      "score": 0.4004629778869299,
      "attributes": {},
      "content": [
        {
          "type": "text",
          "text": "Test document abc123 for RAG ingestion.\nThis is a sample document to test the RAG ingest API.\nLiteLLM provides a unified interface for vector stores."
        }
      ]
    }
  ],
  "has_more": false,
  "next_page": null
}
```

## Request Parameters

### Top-Level

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | object | One of file/file_url/file_id required | Base64-encoded file |
| `file.filename` | string | Yes | Filename with extension |
| `file.content` | string | Yes | Base64-encoded content |
| `file.content_type` | string | Yes | MIME type (e.g., `text/plain`) |
| `file_url` | string | One of file/file_url/file_id required | URL to fetch file from |
| `file_id` | string | One of file/file_url/file_id required | Existing file ID |
| `ingest_options` | object | Yes | Pipeline configuration |

### ingest_options

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `vector_store` | object | Yes | Vector store configuration |
| `name` | string | No | Pipeline name for logging |

### vector_store (OpenAI)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `custom_llm_provider` | string | - | `"openai"` |
| `vector_store_id` | string | auto-create | Existing vector store ID |

### vector_store (Bedrock)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `custom_llm_provider` | string | - | `"bedrock"` |
| `vector_store_id` | string | auto-create | Existing Knowledge Base ID |
| `wait_for_ingestion` | boolean | `false` | Wait for indexing to complete |
| `ingestion_timeout` | integer | `300` | Timeout in seconds (if waiting) |
| `s3_bucket` | string | auto-create | S3 bucket for documents |
| `s3_prefix` | string | `"data/"` | S3 key prefix |
| `embedding_model` | string | `amazon.titan-embed-text-v2:0` | Bedrock embedding model |
| `aws_region_name` | string | `us-west-2` | AWS region |

:::info Bedrock Auto-Creation
When `vector_store_id` is omitted, LiteLLM automatically creates:
- S3 bucket for document storage
- OpenSearch Serverless collection
- IAM role with required permissions
- Bedrock Knowledge Base
- Data Source
:::

### vector_store (Vertex AI)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `custom_llm_provider` | string | - | `"vertex_ai"` |
| `vector_store_id` | string | **required** | RAG corpus ID |
| `gcs_bucket` | string | **required** | GCS bucket for file uploads |
| `vertex_project` | string | env `VERTEXAI_PROJECT` | GCP project ID |
| `vertex_location` | string | `us-central1` | GCP region |
| `vertex_credentials` | string | ADC | Path to credentials JSON |
| `wait_for_import` | boolean | `true` | Wait for import to complete |
| `import_timeout` | integer | `600` | Timeout in seconds (if waiting) |

:::info Vertex AI Prerequisites
1. Create a RAG corpus in Vertex AI console or via API
2. Create a GCS bucket for file uploads
3. Authenticate via `gcloud auth application-default login`
4. Install: `pip install 'google-cloud-aiplatform>=1.60.0'`
:::

## Input Examples

### File (Base64)

```json title="Request body"
{
  "file": {
    "filename": "document.txt",
    "content": "<base64-encoded-content>",
    "content_type": "text/plain"
  },
  "ingest_options": {
    "vector_store": {"custom_llm_provider": "openai"}
  }
}
```

### File URL

```bash showLineNumbers title="Ingest from URL"
curl -X POST "http://localhost:4000/v1/rag/ingest" \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d '{
        "file_url": "https://example.com/document.pdf",
        "ingest_options": {"vector_store": {"custom_llm_provider": "openai"}}
    }'
```

## Chunking Strategy

Control how documents are split into chunks before embedding. Specify `chunking_strategy` in `ingest_options`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | integer | `1000` | Maximum size of each chunk |
| `chunk_overlap` | integer | `200` | Overlap between consecutive chunks |

### Vertex AI RAG Engine

Vertex AI RAG Engine supports custom chunking via the `chunking_strategy` parameter. Chunks are processed server-side during import.

```bash showLineNumbers title="Vertex AI with custom chunking"
curl -X POST "http://localhost:4000/v1/rag/ingest" \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d "{
        \"file\": {
            \"filename\": \"document.txt\",
            \"content\": \"$(base64 -i document.txt)\",
            \"content_type\": \"text/plain\"
        },
        \"ingest_options\": {
            \"chunking_strategy\": {
                \"chunk_size\": 500,
                \"chunk_overlap\": 100
            },
            \"vector_store\": {
                \"custom_llm_provider\": \"vertex_ai\",
                \"vector_store_id\": \"your-corpus-id\",
                \"gcs_bucket\": \"your-gcs-bucket\"
            }
        }
    }"
```

