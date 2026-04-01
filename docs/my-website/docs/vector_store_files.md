# /vector_stores/\{vector_store_id\}/files

Vector store files represent the individual files that live inside a vector store.

| Feature | Supported |
|---------|-----------|
| Logging | ✅ (full request/response logging) |
| Supported Providers | `openai` |


## Supported operations

| Operation | Description | OpenAI Python Client | LiteLLM Proxy |
|-----------|-------------|----------------------|---------------|
| Create vector store file | Attach a file to a vector store with optional chunking overrides | ✅ | ✅ |
| List vector store files | Paginated listing with filters | ✅ | ✅ |
| Retrieve vector store file | Fetch metadata for a single file | ✅ | ✅ |
| Delete vector store file | Remove a file from a store (file object persists) | ✅ | ✅ |
| Retrieve vector store file content | Stream processed chunks | ❌ | ✅ |
| Update vector store file attributes | Patch custom attributes | ❌ | ✅ |

:::note
Vector store support currently works **only with OpenAI vector stores and OpenAI-uploaded file IDs**.
:::


## Create vector store file

<code>POST http://localhost:4000/v1/vector_stores/&#123;vector_store_id&#125;/files</code>

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000",  # LiteLLM proxy or OpenAI base
    api_key="sk-1234"
)

vector_store_file = client.vector_stores.files.create(
    vector_store_id="vs_69172088a18c8191ab3e2621aa87d1ee",
    file_id="file-NDbEDJTfqVh7S4Ugi3CGYw",
    chunking_strategy={
        "type": "static",
        "static": {
            "max_chunk_size_tokens": 800,
            "chunk_overlap_tokens": 400,
        },
    },
)

print(vector_store_file)
```

## List vector store files

<code>GET http://localhost:4000/v1/vector_stores/&#123;vector_store_id&#125;/files</code>

Parameters:

- `vector_store_id` (path, required)
- `after` / `before` (query, optional) – pagination cursors
- `filter` (query, optional) – `in_progress`, `completed`, `failed`, `cancelled`
- `limit` (query, optional, default `20`, range `1-100`)
- `order` (query, optional, default `desc`)

```python
vector_store_files = client.vector_stores.files.list(
    vector_store_id="vs_abc123"
)
print(vector_store_files)
```

## Retrieve vector store file

<code>GET http://localhost:4000/v1/vector_stores/&#123;vector_store_id&#125;/files/&#123;file_id&#125;</code>

```python
vector_store_file = client.vector_stores.files.retrieve(
    vector_store_id="vs_abc123",
    file_id="file-abc123"
)
print(vector_store_file)
```

## Delete vector store file

<code>DELETE http://localhost:4000/v1/vector_stores/&#123;vector_store_id&#125;/files/&#123;file_id&#125;</code>

```python
deleted_vector_store_file = client.vector_stores.files.delete(
    vector_store_id="vs_abc123",
    file_id="file-abc123"
)
print(deleted_vector_store_file)
```

## Proxy-only endpoints

When you need raw content chunks or attribute updates, call the LiteLLM Proxy directly.

### Retrieve file content

```bash
curl -X GET "http://localhost:4000/v1/vector_stores/\{vector_store_id\}/files/\{file_id\}/content" \
  -H "Authorization: Bearer sk-1234"
```

### Update file attributes

```bash
curl -X POST "http://localhost:4000/v1/vector_stores/\{vector_store_id\}/files/\{file_id\}" \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{
        "attributes": {
          "category": "support-faq",
          "language": "en"
        }
      }'
```
