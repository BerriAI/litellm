---
id: container_files
title: /containers/files
---

# Container Files API

Manage files within Code Interpreter containers. Files are created automatically when code interpreter generates outputs (charts, CSVs, images, etc.).

:::tip
Looking for how to use Code Interpreter? See the [Code Interpreter Guide](/docs/guides/code_interpreter).
:::

| Feature | Supported |
|---------|-----------|
| Cost Tracking | ✅ |
| Logging | ✅ |
| Supported Providers | `openai` |

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/containers/{container_id}/files` | GET | List files in container |
| `/v1/containers/{container_id}/files/{file_id}` | GET | Get file metadata |
| `/v1/containers/{container_id}/files/{file_id}/content` | GET | Download file content |
| `/v1/containers/{container_id}/files/{file_id}` | DELETE | Delete file |

## LiteLLM Python SDK

### List Container Files

```python showLineNumbers title="list_container_files.py"
from litellm import list_container_files

files = list_container_files(
    container_id="cntr_123...",
    custom_llm_provider="openai"
)

for file in files.data:
    print(f"  - {file.id}: {file.filename}")
```

**Async:**

```python showLineNumbers title="alist_container_files.py"
from litellm import alist_container_files

files = await alist_container_files(
    container_id="cntr_123...",
    custom_llm_provider="openai"
)
```

### Retrieve Container File

```python showLineNumbers title="retrieve_container_file.py"
from litellm import retrieve_container_file

file = retrieve_container_file(
    container_id="cntr_123...",
    file_id="cfile_456...",
    custom_llm_provider="openai"
)

print(f"File: {file.filename}")
print(f"Size: {file.bytes} bytes")
```

### Download File Content

```python showLineNumbers title="retrieve_container_file_content.py"
from litellm import retrieve_container_file_content

content = retrieve_container_file_content(
    container_id="cntr_123...",
    file_id="cfile_456...",
    custom_llm_provider="openai"
)

# content is raw bytes
with open("output.png", "wb") as f:
    f.write(content)
```

### Delete Container File

```python showLineNumbers title="delete_container_file.py"
from litellm import delete_container_file

result = delete_container_file(
    container_id="cntr_123...",
    file_id="cfile_456...",
    custom_llm_provider="openai"
)

print(f"Deleted: {result.deleted}")
```

## LiteLLM AI Gateway (Proxy)

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

### List Files

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="list_files.py"
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

files = client.containers.files.list(
    container_id="cntr_123..."
)

for file in files.data:
    print(f"  - {file.id}: {file.filename}")
```

</TabItem>
<TabItem value="curl" label="curl">

```bash showLineNumbers title="list_files.sh"
curl "http://localhost:4000/v1/containers/cntr_123.../files" \
    -H "Authorization: Bearer sk-1234"
```

</TabItem>
</Tabs>

### Retrieve File Metadata

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="retrieve_file.py"
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

file = client.containers.files.retrieve(
    container_id="cntr_123...",
    file_id="cfile_456..."
)

print(f"File: {file.filename}")
print(f"Size: {file.bytes} bytes")
```

</TabItem>
<TabItem value="curl" label="curl">

```bash showLineNumbers title="retrieve_file.sh"
curl "http://localhost:4000/v1/containers/cntr_123.../files/cfile_456..." \
    -H "Authorization: Bearer sk-1234"
```

</TabItem>
</Tabs>

### Download File Content

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="download_content.py"
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

content = client.containers.files.content(
    container_id="cntr_123...",
    file_id="cfile_456..."
)

with open("output.png", "wb") as f:
    f.write(content.read())
```

</TabItem>
<TabItem value="curl" label="curl">

```bash showLineNumbers title="download_content.sh"
curl "http://localhost:4000/v1/containers/cntr_123.../files/cfile_456.../content" \
    -H "Authorization: Bearer sk-1234" \
    --output downloaded_file.png
```

</TabItem>
</Tabs>

### Delete File

<Tabs>
<TabItem value="openai-sdk" label="OpenAI SDK">

```python showLineNumbers title="delete_file.py"
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",
    base_url="http://localhost:4000"
)

result = client.containers.files.delete(
    container_id="cntr_123...",
    file_id="cfile_456..."
)

print(f"Deleted: {result.deleted}")
```

</TabItem>
<TabItem value="curl" label="curl">

```bash showLineNumbers title="delete_file.sh"
curl -X DELETE "http://localhost:4000/v1/containers/cntr_123.../files/cfile_456..." \
    -H "Authorization: Bearer sk-1234"
```

</TabItem>
</Tabs>

## Parameters

### List Files

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `container_id` | string | Yes | Container ID |
| `after` | string | No | Pagination cursor |
| `limit` | integer | No | Items to return (1-100, default: 20) |
| `order` | string | No | Sort order: `asc` or `desc` |

### Retrieve/Delete File

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `container_id` | string | Yes | Container ID |
| `file_id` | string | Yes | File ID |

## Response Objects

### ContainerFileObject

```json showLineNumbers title="ContainerFileObject"
{
  "id": "cfile_456...",
  "object": "container.file",
  "container_id": "cntr_123...",
  "bytes": 12345,
  "created_at": 1234567890,
  "filename": "chart.png",
  "path": "/mnt/data/chart.png",
  "source": "code_interpreter"
}
```

### ContainerFileListResponse

```json showLineNumbers title="ContainerFileListResponse"
{
  "object": "list",
  "data": [...],
  "first_id": "cfile_456...",
  "last_id": "cfile_789...",
  "has_more": false
}
```

### DeleteContainerFileResponse

```json showLineNumbers title="DeleteContainerFileResponse"
{
  "id": "cfile_456...",
  "object": "container.file.deleted",
  "deleted": true
}
```

## Supported Providers

| Provider | Status |
|----------|--------|
| OpenAI | ✅ Supported |

## Related

- [Containers API](/docs/containers) - Manage containers
- [Code Interpreter Guide](/docs/guides/code_interpreter) - Using Code Interpreter with LiteLLM
