# Custom Secret Manager - Quick Start Guide

## What is it?

Custom Secret Manager allows you to integrate **any** secret management system with LiteLLM, just like you can create custom guardrails. 

## Why use it?

✅ Integrate proprietary vault systems  
✅ Use custom authentication methods  
✅ Implement organization-specific security policies  
✅ Support any secret storage backend  

## 5-Minute Implementation

### Step 1: Create Your Secret Manager

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
        # Your async implementation
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.vault_url}/secrets/{secret_name}",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=timeout or 30.0
            )
            return response.json()["value"]
    
    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        # Your sync implementation
        with httpx.Client() as client:
            response = client.get(
                f"{self.vault_url}/secrets/{secret_name}",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=timeout or 30.0
            )
            return response.json()["value"]
```

### Step 2: Configure LiteLLM

```python
import litellm
from litellm.types.secret_managers.main import KeyManagementSystem, KeyManagementSettings

# Initialize your secret manager
litellm.secret_manager_client = MySecretManager(
    vault_url="https://vault.example.com",
    token="your-token"
)

# Set to CUSTOM mode
litellm._key_management_system = KeyManagementSystem.CUSTOM

# Configure access mode
litellm._key_management_settings = KeyManagementSettings(
    access_mode="read_only"
)
```

### Step 3: Use It!

```python
from litellm.secret_managers.main import get_secret

# Now all secret reads go through your custom manager
api_key = get_secret("OPENAI_API_KEY")

# Or use with LiteLLM proxy - it automatically uses the secret manager
import litellm
response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Minimal Example (In-Memory Store)

Here's the absolute minimal implementation for testing:

```python
from typing import Optional, Union
import httpx
from litellm.integrations.custom_secret_manager import CustomSecretManager

class SimpleSecretManager(CustomSecretManager):
    def __init__(self, secrets: dict):
        super().__init__(secret_manager_name="simple")
        self.secrets = secrets
    
    async def async_read_secret(self, secret_name, optional_params=None, timeout=None):
        return self.secrets.get(secret_name)
    
    def sync_read_secret(self, secret_name, optional_params=None, timeout=None):
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

## With LiteLLM Proxy

Add to your proxy startup script:

```python
# proxy_config.py
import litellm
from litellm.types.secret_managers.main import KeyManagementSystem, KeyManagementSettings
from my_module import MySecretManager

# Initialize before starting proxy
litellm.secret_manager_client = MySecretManager(
    vault_url=os.getenv("VAULT_URL"),
    token=os.getenv("VAULT_TOKEN")
)
litellm._key_management_system = KeyManagementSystem.CUSTOM
litellm._key_management_settings = KeyManagementSettings(access_mode="read_only")

# Now start proxy - it will use your secret manager
# python -m litellm --config proxy_config.yaml
```

## Optional: Add Write/Delete Support

```python
class MySecretManager(CustomSecretManager):
    # ... include async_read_secret and sync_read_secret from above ...
    
    async def async_write_secret(
        self, secret_name, secret_value, description=None, 
        optional_params=None, timeout=None, tags=None
    ):
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.vault_url}/secrets",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"name": secret_name, "value": secret_value},
                timeout=timeout or 30.0
            )
        return {"status": "success"}
    
    async def async_delete_secret(
        self, secret_name, recovery_window_in_days=7, 
        optional_params=None, timeout=None
    ):
        async with httpx.AsyncClient() as client:
            await client.delete(
                f"{self.vault_url}/secrets/{secret_name}",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=timeout or 30.0
            )
        return {"status": "deleted"}
```

## Comparison with Custom Guardrails

| Feature | Custom Guardrails | Custom Secret Managers |
|---------|------------------|----------------------|
| **Purpose** | Filter/validate requests & responses | Securely store/retrieve credentials |
| **Base Class** | `CustomLogger` | `BaseSecretManager` |
| **Import** | `litellm.integrations.custom_guardrail` | `litellm.integrations.custom_secret_manager` |
| **Required Methods** | Event hooks (pre/post call) | `async_read_secret()`, `sync_read_secret()` |
| **Configuration** | Event hooks, modes | Access mode, hosted keys |
| **Use Case** | Content filtering, PII masking | API key management |

## Files Reference

### Implementation
- **Base Class**: `litellm/integrations/custom_secret_manager.py`
- **Handler**: `litellm/secret_managers/secret_manager_handler.py`
- **Types**: `litellm/types/secret_managers/main.py`

### Examples & Tests
- **Tests**: `tests/test_litellm/secret_managers/test_custom_secret_manager.py`
- **Example**: `cookbook/secret_managers/custom_secret_manager_example.py`

### Documentation
- **Guide**: `litellm/secret_managers/Readme.md`
- **Docs**: `docs/my-website/docs/secret_management/custom_secret_manager.md`

## Need Help?

1. Check the example: `cookbook/secret_managers/custom_secret_manager_example.py`
2. Look at tests: `tests/test_litellm/secret_managers/test_custom_secret_manager.py`
3. Review built-in implementations in `litellm/secret_managers/`
4. See documentation: `docs/my-website/docs/secret_management/custom_secret_manager.md`

## Testing Your Implementation

```python
import pytest
from your_module import YourSecretManager

def test_read_secret():
    manager = YourSecretManager(vault_url="...", token="...")
    
    # Test sync read
    secret = manager.sync_read_secret("TEST_KEY")
    assert secret is not None
    
    # Test missing key
    missing = manager.sync_read_secret("NON_EXISTENT")
    assert missing is None

@pytest.mark.asyncio
async def test_async_read_secret():
    manager = YourSecretManager(vault_url="...", token="...")
    
    # Test async read
    secret = await manager.async_read_secret("TEST_KEY")
    assert secret is not None
```

---

**That's it!** You now have a custom secret manager integrated with LiteLLM. 🎉
