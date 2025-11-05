# Secret Managers

LiteLLM supports various secret management systems to securely store and retrieve API keys and other sensitive information.

## Supported Secret Managers

- **AWS Secrets Manager** (`aws_secret_manager`)
- **AWS KMS** (`aws_kms`)
- **Azure Key Vault** (`azure_key_vault`)
- **Google Secret Manager** (`google_secret_manager`)
- **Google KMS** (`google_kms`)
- **HashiCorp Vault** (`hashicorp_vault`)
- **CyberArk** (`cyberark`)
- **Custom Secret Manager** (`custom`) - Implement your own!

## Using Built-in Secret Managers

### AWS Secrets Manager

```python
import litellm
from litellm.types.secret_managers.main import KeyManagementSystem, KeyManagementSettings

# Initialize AWS Secrets Manager
from litellm.secret_managers.aws_secret_manager_v2 import AWSSecretsManagerV2

litellm.secret_manager_client = AWSSecretsManagerV2()
litellm._key_management_system = KeyManagementSystem.AWS_SECRET_MANAGER
litellm._key_management_settings = KeyManagementSettings(
    access_mode="read_only"
)

# Now get_secret will use AWS Secrets Manager
from litellm.secret_managers.main import get_secret

api_key = get_secret("OPENAI_API_KEY")
```

### HashiCorp Vault

```python
import litellm
from litellm.types.secret_managers.main import KeyManagementSystem, KeyManagementSettings

# Initialize HashiCorp Vault
from litellm.secret_managers.hashicorp_secret_manager import HashicorpVaultSecretManager

litellm.secret_manager_client = HashicorpVaultSecretManager(
    vault_url="https://vault.example.com",
    token="your-vault-token"
)
litellm._key_management_system = KeyManagementSystem.HASHICORP_VAULT
litellm._key_management_settings = KeyManagementSettings(
    access_mode="read_only"
)
```

## Creating a Custom Secret Manager

You can create your own custom secret manager by extending the `CustomSecretManager` base class:

```python
from typing import Optional, Union
import httpx
from litellm.integrations.custom_secret_manager import CustomSecretManager

class MyVaultSecretManager(CustomSecretManager):
    """
    Custom secret manager for my proprietary vault system.
    """
    
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
        """
        Read a secret asynchronously from your vault.
        """
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
        """
        Read a secret synchronously from your vault.
        """
        with httpx.Client() as client:
            response = client.get(
                f"{self.vault_url}/v1/secret/{secret_name}",
                headers={"X-Vault-Token": self.token},
                timeout=timeout or 30.0
            )
            response.raise_for_status()
            return response.json()["data"]["value"]
    
    # Optional: Override write and delete if supported
    async def async_write_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: Optional[str] = None,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        tags: Optional[Union[dict, list]] = None,
    ) -> dict:
        """
        Write a secret to your vault (optional).
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.vault_url}/v1/secret/{secret_name}",
                headers={"X-Vault-Token": self.token},
                json={"value": secret_value, "description": description},
                timeout=timeout or 30.0
            )
            response.raise_for_status()
            return {"status": "success", "secret_name": secret_name}


# Initialize your custom secret manager
import litellm
from litellm.types.secret_managers.main import KeyManagementSystem, KeyManagementSettings

litellm.secret_manager_client = MyVaultSecretManager(
    vault_url="https://my-vault.example.com",
    token="my-vault-token"
)
litellm._key_management_system = KeyManagementSystem.CUSTOM
litellm._key_management_settings = KeyManagementSettings(
    access_mode="read_only"
)

# Use it with LiteLLM
from litellm.secret_managers.main import get_secret

api_key = get_secret("OPENAI_API_KEY")
```

### Custom Secret Manager Methods

When implementing a custom secret manager, you **must** implement these methods:

- `async_read_secret()` - Read a secret asynchronously
- `sync_read_secret()` - Read a secret synchronously

Optional methods you can implement:

- `async_write_secret()` - Write a secret asynchronously
- `async_delete_secret()` - Delete a secret asynchronously
- `async_rotate_secret()` - Rotate a secret (inherited from base class)
- `validate_environment()` - Validate environment configuration
- `async_health_check()` - Check if the secret manager is accessible

### Minimal Example

Here's a minimal example that only implements the required methods:

```python
from typing import Optional, Union
import httpx
from litellm.integrations.custom_secret_manager import CustomSecretManager

class SimpleSecretManager(CustomSecretManager):
    def __init__(self, secrets: dict):
        super().__init__(secret_manager_name="simple")
        self.secrets = secrets
    
    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        return self.secrets.get(secret_name)
    
    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        return self.secrets.get(secret_name)


# Use it
import litellm
from litellm.types.secret_managers.main import KeyManagementSystem, KeyManagementSettings

litellm.secret_manager_client = SimpleSecretManager({
    "OPENAI_API_KEY": "sk-...",
    "ANTHROPIC_API_KEY": "sk-ant-...",
})
litellm._key_management_system = KeyManagementSystem.CUSTOM
litellm._key_management_settings = KeyManagementSettings(access_mode="read_only")
```

## Configuration Options

### KeyManagementSettings

- `access_mode`: `"read_only"`, `"write_only"`, or `"read_and_write"`
- `hosted_keys`: List of specific keys to check in the secret manager
- `store_virtual_keys`: Whether to store virtual keys created by LiteLLM
- `prefix_for_stored_virtual_keys`: Prefix for stored virtual keys (default: `"litellm/"`)
- `primary_secret_name`: For services that support multiple secrets in one store
- `description`: Optional description for created secrets
- `tags`: Optional tags to attach to created secrets

## Best Practices

1. **Security**: Never log or print secret values in production
2. **Error Handling**: Always handle cases where secrets might not exist
3. **Timeouts**: Configure appropriate timeouts for network requests
4. **Caching**: Consider implementing caching for frequently accessed secrets
5. **Rotation**: Implement secret rotation if your system supports it
6. **Validation**: Use `validate_environment()` to check configuration on startup

## Testing

See `tests/test_litellm/secret_managers/test_custom_secret_manager.py` for example tests.

```bash
pytest tests/test_litellm/secret_managers/test_custom_secret_manager.py -v
```
