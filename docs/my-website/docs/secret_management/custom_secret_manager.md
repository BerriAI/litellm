# Custom Secret Manager

LiteLLM now supports custom secret manager implementations, allowing you to integrate any secret management system with LiteLLM.

## Overview

Similar to [custom guardrails](/docs/proxy/guardrails#custom-guardrails), you can now create custom secret managers by extending the `CustomSecretManager` base class. This enables you to:

- Integrate proprietary secret management systems
- Use custom authentication methods
- Implement organization-specific security policies
- Support any secret storage backend

## Quick Start

### 1. Create Your Custom Secret Manager

```python
from typing import Optional, Union
import httpx
from litellm.integrations.custom_secret_manager import CustomSecretManager

class MySecretManager(CustomSecretManager):
    def __init__(self, vault_url: str, token: str):
        super().__init__(secret_manager_name="my_vault")
        self.vault_url = vault_url
        self.token = token
    
    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.vault_url}/v1/secret/{secret_name}",
                headers={"X-Vault-Token": self.token},
                timeout=timeout or 30.0
            )
            response.raise_for_status()
            return response.json()["data"]["value"]
    
    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        with httpx.Client() as client:
            response = client.get(
                f"{self.vault_url}/v1/secret/{secret_name}",
                headers={"X-Vault-Token": self.token},
                timeout=timeout or 30.0
            )
            response.raise_for_status()
            return response.json()["data"]["value"]
```

### 2. Initialize and Use

```python
import litellm
from litellm.types.secret_managers.main import KeyManagementSystem, KeyManagementSettings

# Set up your custom secret manager
litellm.secret_manager_client = MySecretManager(
    vault_url="https://vault.example.com",
    token="your-vault-token"
)
litellm._key_management_system = KeyManagementSystem.CUSTOM
litellm._key_management_settings = KeyManagementSettings(
    access_mode="read_only"
)

# Now use LiteLLM normally - it will use your secret manager
from litellm.secret_managers.main import get_secret

api_key = get_secret("OPENAI_API_KEY")
```

### 3. Use with LiteLLM Proxy

```python
# In your proxy startup script
import litellm
from litellm.types.secret_managers.main import KeyManagementSystem, KeyManagementSettings

# Initialize your custom secret manager
litellm.secret_manager_client = MySecretManager(
    vault_url=os.getenv("VAULT_URL"),
    token=os.getenv("VAULT_TOKEN")
)
litellm._key_management_system = KeyManagementSystem.CUSTOM
litellm._key_management_settings = KeyManagementSettings(
    access_mode="read_only"
)

# Start the proxy - it will use your secret manager for all API keys
```

## Required Methods

When implementing a custom secret manager, you **must** implement these two methods:

### `async_read_secret()`

```python
async def async_read_secret(
    self,
    secret_name: str,
    optional_params: Optional[dict] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> Optional[str]:
    """
    Read a secret asynchronously.
    
    Args:
        secret_name: Name/path of the secret to read
        optional_params: Additional parameters for your secret manager
        timeout: Request timeout
        
    Returns:
        The secret value if found, None otherwise
    """
    pass
```

### `sync_read_secret()`

```python
def sync_read_secret(
    self,
    secret_name: str,
    optional_params: Optional[dict] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> Optional[str]:
    """
    Read a secret synchronously.
    
    Args:
        secret_name: Name/path of the secret to read
        optional_params: Additional parameters for your secret manager
        timeout: Request timeout
        
    Returns:
        The secret value if found, None otherwise
    """
    pass
```

## Optional Methods

You can optionally implement these methods for additional functionality:

### `async_write_secret()`

```python
async def async_write_secret(
    self,
    secret_name: str,
    secret_value: str,
    description: Optional[str] = None,
    optional_params: Optional[dict] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    tags: Optional[Union[dict, list]] = None,
) -> dict:
    """Write a secret to your secret manager."""
    pass
```

### `async_delete_secret()`

```python
async def async_delete_secret(
    self,
    secret_name: str,
    recovery_window_in_days: Optional[int] = 7,
    optional_params: Optional[dict] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
) -> dict:
    """Delete a secret from your secret manager."""
    pass
```

### `validate_environment()`

```python
def validate_environment(self) -> bool:
    """
    Validate that all required configuration is present.
    
    Raises:
        ValueError: If required configuration is missing
    """
    pass
```

### `async_health_check()`

```python
async def async_health_check(
    self, 
    timeout: Optional[Union[float, httpx.Timeout]] = None
) -> bool:
    """
    Perform a health check on your secret manager.
    
    Returns:
        True if healthy, False otherwise
    """
    pass
```

## Complete Example

Here's a complete example with all optional methods implemented:

```python
from typing import Optional, Union, Dict, Any
import httpx
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm._logging import verbose_logger

class ProductionSecretManager(CustomSecretManager):
    """
    Production-ready secret manager with full CRUD operations.
    """
    
    def __init__(self, api_url: str, api_key: str):
        super().__init__(secret_manager_name="production_vault")
        self.api_url = api_url
        self.api_key = api_key
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration on initialization."""
        if not self.api_url:
            raise ValueError("api_url is required")
        if not self.api_key:
            raise ValueError("api_key is required")
    
    def _get_headers(self) -> dict:
        """Get common headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """Read a secret asynchronously."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_url}/secrets/{secret_name}",
                    headers=self._get_headers(),
                    timeout=timeout or 30.0
                )
                response.raise_for_status()
                return response.json()["value"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            verbose_logger.error(f"Error reading secret: {e}")
            raise
        except Exception as e:
            verbose_logger.error(f"Unexpected error reading secret: {e}")
            raise
    
    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """Read a secret synchronously."""
        try:
            with httpx.Client() as client:
                response = client.get(
                    f"{self.api_url}/secrets/{secret_name}",
                    headers=self._get_headers(),
                    timeout=timeout or 30.0
                )
                response.raise_for_status()
                return response.json()["value"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            verbose_logger.error(f"Error reading secret: {e}")
            raise
        except Exception as e:
            verbose_logger.error(f"Unexpected error reading secret: {e}")
            raise
    
    async def async_write_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: Optional[str] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        tags: Optional[Union[dict, list]] = None,
    ) -> Dict[str, Any]:
        """Write a secret."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/secrets",
                headers=self._get_headers(),
                json={
                    "name": secret_name,
                    "value": secret_value,
                    "description": description,
                    "tags": tags
                },
                timeout=timeout or 30.0
            )
            response.raise_for_status()
            return response.json()
    
    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: Optional[int] = 7,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> dict:
        """Delete a secret."""
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{self.api_url}/secrets/{secret_name}",
                headers=self._get_headers(),
                params={"recovery_window": recovery_window_in_days},
                timeout=timeout or 30.0
            )
            response.raise_for_status()
            return response.json()
    
    def validate_environment(self) -> bool:
        """Validate environment configuration."""
        self._validate_config()
        return True
    
    async def async_health_check(
        self, 
        timeout: Optional[Union[float, httpx.Timeout]] = None
    ) -> bool:
        """Check if the secret manager is accessible."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_url}/health",
                    headers=self._get_headers(),
                    timeout=timeout or 10.0
                )
                return response.status_code == 200
        except Exception as e:
            verbose_logger.error(f"Health check failed: {e}")
            return False
```

## Configuration Options

Use `KeyManagementSettings` to configure how LiteLLM uses your secret manager:

```python
from litellm.types.secret_managers.main import KeyManagementSettings

litellm._key_management_settings = KeyManagementSettings(
    # Access mode: "read_only", "write_only", or "read_and_write"
    access_mode="read_only",
    
    # Only check these specific keys in the secret manager
    hosted_keys=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
    
    # Store virtual keys created by LiteLLM
    store_virtual_keys=True,
    
    # Prefix for stored virtual keys
    prefix_for_stored_virtual_keys="litellm/",
    
    # Optional description for created secrets
    description="Managed by LiteLLM",
    
    # Optional tags for created secrets
    tags={"Environment": "Production", "ManagedBy": "LiteLLM"}
)
```

## Best Practices

1. **Error Handling**: Always handle network errors and invalid responses gracefully
2. **Logging**: Use `verbose_logger` for debugging and error logging
3. **Security**: Never log secret values
4. **Timeouts**: Always configure appropriate timeouts for network requests
5. **Validation**: Implement `validate_environment()` to catch configuration issues early
6. **Health Checks**: Implement `async_health_check()` for monitoring
7. **Thread Safety**: Ensure your implementation is thread-safe if used in concurrent environments

## Testing

See `tests/test_litellm/secret_managers/test_custom_secret_manager.py` for example tests.

## Examples

See `cookbook/secret_managers/custom_secret_manager_example.py` for complete working examples.

## Comparison with Built-in Secret Managers

| Feature | Custom Secret Manager | AWS Secrets Manager | HashiCorp Vault |
|---------|----------------------|---------------------|-----------------|
| Flexibility | ✅ Unlimited | ❌ AWS-specific | ❌ Vault-specific |
| Setup Complexity | ⚠️ You implement | ✅ Pre-built | ✅ Pre-built |
| Cost | 💰 Depends on provider | 💰 AWS pricing | 💰 Self-hosted or Cloud |
| Integration Time | 🕐 1-2 hours | 🕐 5 minutes | 🕐 10 minutes |

## Related Documentation

- [Secret Management Overview](/docs/secret_management)
- [AWS Secrets Manager](/docs/secret_management/aws)
- [HashiCorp Vault](/docs/secret_management/hashicorp)
- [Custom Guardrails](/docs/proxy/guardrails#custom-guardrails)
