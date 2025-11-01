import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# /containers

Manage OpenAI code interpreter containers (sessions) for executing code in isolated environments.

| Feature | Supported | 
|---------|-----------|
| Cost Tracking | ✅ |
| Logging | ✅ (Full request/response logging) |
| Load Balancing | ✅ |
| Proxy Server Support | ✅ Full proxy integration with virtual keys |
| Spend Management | ✅ Budget tracking and rate limiting |
| Supported Providers | `openai`|

## **Supported Providers**:
- [OpenAI](#quick-start)

## Quick Start 

Containers provide isolated execution environments for code interpreter sessions. You can create, list, retrieve, and delete containers.

### SDK, PROXY, and OpenAI Client

<Tabs>
<TabItem value="sdk" label="SDK">

**Create a Container**

```python
import litellm
import os 

# setup env
os.environ["OPENAI_API_KEY"] = "sk-.."

container = litellm.create_container(
    name="My Code Interpreter Container",
    custom_llm_provider="openai",
    expires_after={
        "anchor": "last_active_at",
        "minutes": 20
    }
)

print(f"Container ID: {container.id}")
print(f"Container Name: {container.name}")

### ASYNC USAGE ### 
# container = await litellm.acreate_container(
#     name="My Code Interpreter Container",
#     custom_llm_provider="openai",
#     expires_after={
#         "anchor": "last_active_at",
#         "minutes": 20
#     }
# )
```

**List Containers**

```python
from litellm import list_containers, alist_containers
import os 

os.environ["OPENAI_API_KEY"] = "sk-.."

containers = list_containers(
    custom_llm_provider="openai",
    limit=20,
    order="desc"
)

print(f"Found {len(containers.data)} containers")
for container in containers.data:
    print(f"  - {container.id}: {container.name}")

### ASYNC USAGE ### 
# containers = await alist_containers(
#     custom_llm_provider="openai",
#     limit=20,
#     order="desc"
# )
```

**Retrieve a Container**

```python
from litellm import retrieve_container, aretrieve_container
import os 

os.environ["OPENAI_API_KEY"] = "sk-.."

container = retrieve_container(
    container_id="cntr_123...",
    custom_llm_provider="openai"
)

print(f"Container: {container.name}")
print(f"Status: {container.status}")
print(f"Created: {container.created_at}")

### ASYNC USAGE ### 
# container = await aretrieve_container(
#     container_id="cntr_123...",
#     custom_llm_provider="openai"
# )
```

**Delete a Container**

```python
from litellm import delete_container, adelete_container
import os 

os.environ["OPENAI_API_KEY"] = "sk-.."

result = delete_container(
    container_id="cntr_123...",
    custom_llm_provider="openai"
)

print(f"Deleted: {result.deleted}")
print(f"Container ID: {result.id}")

### ASYNC USAGE ### 
# result = await adelete_container(
#     container_id="cntr_123...",
#     custom_llm_provider="openai"
# )
```

</TabItem>
<TabItem value="proxy" label="LiteLLM PROXY Server">

```bash
$ export OPENAI_API_KEY="sk-..."

$ litellm

# RUNNING on http://0.0.0.0:4000
```

**Custom Provider Specification**

You can specify the custom LLM provider in multiple ways (priority order):
1. Header: `-H "custom-llm-provider: openai"`
2. Query param: `?custom_llm_provider=openai`
3. Request body: `{"custom_llm_provider": "openai", ...}`
4. Defaults to "openai" if not specified

**Create a Container**

```bash
# Default provider (openai)
curl -X POST "http://localhost:4000/v1/containers" \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "My Container",
        "expires_after": {
            "anchor": "last_active_at",
            "minutes": 20
        }
    }'
```

```bash
# Via header
curl -X POST "http://localhost:4000/v1/containers" \
    -H "Authorization: Bearer sk-1234" \
    -H "custom-llm-provider: openai" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "My Container"
    }'
```

```bash
# Via query parameter
curl -X POST "http://localhost:4000/v1/containers?custom_llm_provider=openai" \
    -H "Authorization: Bearer sk-1234" \
    -H "Content-Type: application/json" \
    -d '{
        "name": "My Container"
    }'
```

**List Containers**

```bash
curl "http://localhost:4000/v1/containers?limit=20&order=desc" \
    -H "Authorization: Bearer sk-1234"
```

**Retrieve a Container**

```bash
curl "http://localhost:4000/v1/containers/cntr_123..." \
    -H "Authorization: Bearer sk-1234"
```

**Delete a Container**

```bash
curl -X DELETE "http://localhost:4000/v1/containers/cntr_123..." \
    -H "Authorization: Bearer sk-1234"
```

</TabItem>
<TabItem value="openai" label="OpenAI Python Client">

**Setup**

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-1234",  # Your LiteLLM proxy key
    base_url="http://localhost:4000"  # LiteLLM proxy URL
)
```

**Create a Container**

```python
container = client.containers.create(
    name="test-container",
    expires_after={
        "anchor": "last_active_at",
        "minutes": 20
    },
    extra_body={"custom_llm_provider": "openai"}
)

print(f"Container ID: {container.id}")
print(f"Container Name: {container.name}")
print(f"Created at: {container.created_at}")
```

**List Containers**

```python
containers = client.containers.list(
    limit=20,
    extra_body={"custom_llm_provider": "openai"}
)

print(f"Found {len(containers.data)} containers")
for container in containers.data:
    print(f"  - {container.id}: {container.name}")
```

**Retrieve a Container**

```python
container = client.containers.retrieve(
    container_id="cntr_6901d28b3c8881908b702815828a5bde0380b3408aeae8c7",
    extra_body={"custom_llm_provider": "openai"}
)

print(f"Container: {container.name}")
print(f"Status: {container.status}")
print(f"Last active: {container.last_active_at}")
```

**Delete a Container**

```python
result = client.containers.delete(
    container_id="cntr_6901d28b3c8881908b702815828a5bde0380b3408aeae8c7",
    extra_body={"custom_llm_provider": "openai"}
)

print(f"Deleted: {result.deleted}")
print(f"Container ID: {result.id}")
```

</TabItem>
</Tabs>

## Container Parameters

### Create Container Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Name of the container |
| `expires_after` | object | No | Container expiration settings |
| `expires_after.anchor` | string | No | Anchor point for expiration (e.g., "last_active_at") |
| `expires_after.minutes` | integer | No | Minutes until expiration from anchor |
| `file_ids` | array | No | List of file IDs to include in the container |
| `custom_llm_provider` | string | No | LLM provider to use (default: "openai") |

### List Container Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `after` | string | No | Cursor for pagination |
| `limit` | integer | No | Number of items to return (1-100, default: 20) |
| `order` | string | No | Sort order: "asc" or "desc" (default: "desc") |
| `custom_llm_provider` | string | No | LLM provider to use (default: "openai") |

### Retrieve/Delete Container Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `container_id` | string | Yes | ID of the container to retrieve/delete |
| `custom_llm_provider` | string | No | LLM provider to use (default: "openai") |

## Response Objects

### ContainerObject

```json
{
  "id": "cntr_123...",
  "object": "container",
  "created_at": 1234567890,
  "name": "My Container",
  "status": "active",
  "last_active_at": 1234567890,
  "expires_at": 1234569090,
  "file_ids": []
}
```

### ContainerListResponse

```json
{
  "object": "list",
  "data": [
    {
      "id": "cntr_123...",
      "object": "container",
      "created_at": 1234567890,
      "name": "My Container",
      "status": "active"
    }
  ],
  "first_id": "cntr_123...",
  "last_id": "cntr_456...",
  "has_more": false
}
```

### DeleteContainerResult

```json
{
  "id": "cntr_123...",
  "object": "container.deleted",
  "deleted": true
}
```

