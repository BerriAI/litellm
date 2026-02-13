# Container Files API

This module provides a unified interface for container file operations across multiple LLM providers (OpenAI, Azure OpenAI, etc.).

## Architecture

```
endpoints.json                    # Declarative endpoint definitions
       ↓
endpoint_factory.py              # Auto-generates SDK functions
       ↓
container_handler.py             # Generic HTTP handler
       ↓
BaseContainerConfig              # Provider-specific transformations
├── OpenAIContainerConfig
└── AzureContainerConfig (example)
```

## Files Overview

| File | Purpose |
|------|---------|
| `endpoints.json` | **Single source of truth** - Defines all container file endpoints |
| `endpoint_factory.py` | Auto-generates SDK functions (`list_container_files`, etc.) |
| `main.py` | Core container operations (create, list, retrieve, delete containers) |
| `utils.py` | Request parameter utilities |

## Adding a New Endpoint

To add a new container file endpoint (e.g., `get_container_file_content`):

### Step 1: Add to `endpoints.json`

```json
{
  "name": "get_container_file_content",
  "async_name": "aget_container_file_content",
  "path": "/containers/{container_id}/files/{file_id}/content",
  "method": "GET",
  "path_params": ["container_id", "file_id"],
  "query_params": [],
  "response_type": "ContainerFileContentResponse"
}
```

### Step 2: Add Response Type (if new)

In `litellm/types/containers/main.py`:

```python
class ContainerFileContentResponse(BaseModel):
    """Response for file content download."""
    content: bytes
    # ... other fields
```

### Step 3: Register Response Type

In `litellm/llms/custom_httpx/container_handler.py`, add to `RESPONSE_TYPES`:

```python
RESPONSE_TYPES = {
    # ... existing types
    "ContainerFileContentResponse": ContainerFileContentResponse,
}
```

### Step 4: Update Router (one-time setup)

In `litellm/router.py`, add the call_type to the factory_function Literal and `_init_containers_api_endpoints` condition.

In `litellm/proxy/route_llm_request.py`, add to the route mappings and skip-model-routing lists.

### Step 5: Update Proxy Handler Factory (if new path params)

If your endpoint has a new combination of path parameters, add a handler in `litellm/proxy/container_endpoints/handler_factory.py`:

```python
elif path_params == ["container_id", "file_id", "new_param"]:
    async def handler(...):
        # handler implementation
```

---

## Adding a New Provider (e.g., Azure OpenAI)

### Step 1: Create Provider Config

Create `litellm/llms/azure/containers/transformation.py`:

```python
from typing import Dict, Optional, Tuple, Any
import httpx

from litellm.llms.base_llm.containers.transformation import BaseContainerConfig
from litellm.types.containers.main import (
    ContainerFileListResponse,
    ContainerFileObject,
    DeleteContainerFileResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.secret_managers.main import get_secret_str


class AzureContainerConfig(BaseContainerConfig):
    """Configuration class for Azure OpenAI container API."""

    def get_supported_openai_params(self) -> list:
        return ["name", "expires_after", "file_ids", "extra_headers"]

    def map_openai_params(
        self,
        container_create_optional_params,
        drop_params: bool,
    ) -> Dict:
        return dict(container_create_optional_params)

    def validate_environment(
        self,
        headers: dict,
        api_key: Optional[str] = None,
    ) -> dict:
        """Azure uses api-key header instead of Bearer token."""
        import litellm
        
        api_key = (
            api_key
            or litellm.azure_key
            or get_secret_str("AZURE_API_KEY")
        )
        headers["api-key"] = api_key
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Azure format: 
        https://{resource}.openai.azure.com/openai/containers?api-version=2024-xx
        """
        if api_base is None:
            raise ValueError("api_base is required for Azure")
        
        api_version = litellm_params.get("api_version", "2024-02-15-preview")
        return f"{api_base.rstrip('/')}/openai/containers?api-version={api_version}"

    # Implement remaining abstract methods from BaseContainerConfig:
    # - transform_container_create_request
    # - transform_container_create_response
    # - transform_container_list_request
    # - transform_container_list_response
    # - transform_container_retrieve_request
    # - transform_container_retrieve_response
    # - transform_container_delete_request
    # - transform_container_delete_response
    # - transform_container_file_list_request
    # - transform_container_file_list_response
```

### Step 2: Register Provider Config

In `litellm/utils.py`, find `ProviderConfigManager.get_provider_container_config()` and add:

```python
@staticmethod
def get_provider_container_config(
    provider: LlmProviders,
) -> Optional[BaseContainerConfig]:
    if provider == LlmProviders.OPENAI:
        from litellm.llms.openai.containers.transformation import OpenAIContainerConfig
        return OpenAIContainerConfig()
    elif provider == LlmProviders.AZURE:
        from litellm.llms.azure.containers.transformation import AzureContainerConfig
        return AzureContainerConfig()
    return None
```

### Step 3: Test the New Provider

```bash
# Create container via Azure
curl -X POST "http://localhost:4000/v1/containers" \
    -H "Authorization: Bearer sk-1234" \
    -H "custom-llm-provider: azure" \
    -H "Content-Type: application/json" \
    -d '{"name": "My Azure Container"}'

# List container files via Azure
curl -X GET "http://localhost:4000/v1/containers/cntr_123/files" \
    -H "Authorization: Bearer sk-1234" \
    -H "custom-llm-provider: azure"
```

---

## How Provider Selection Works

1. **Proxy receives request** with `custom-llm-provider` header/query/body
2. **Router calls** `ProviderConfigManager.get_provider_container_config(provider)`
3. **Generic handler** uses the provider config for:
   - URL construction (`get_complete_url`)
   - Authentication (`validate_environment`)
   - Request/response transformation

---

## Testing

Run the container API tests:

```bash
cd /Users/ishaanjaffer/github/litellm
python -m pytest tests/test_litellm/containers/ -v
```

Test via proxy:

```bash
# Start proxy
cd litellm/proxy && python proxy_cli.py --config proxy_config.yaml --port 4000

# Test endpoints
curl -X GET "http://localhost:4000/v1/containers/cntr_123/files" \
    -H "Authorization: Bearer sk-1234"
```

---

## Endpoint Reference

| Endpoint | Method | Path |
|----------|--------|------|
| List container files | GET | `/v1/containers/{container_id}/files` |
| Retrieve container file | GET | `/v1/containers/{container_id}/files/{file_id}` |
| Delete container file | DELETE | `/v1/containers/{container_id}/files/{file_id}` |

See `endpoints.json` for the complete list.

