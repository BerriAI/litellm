# 🔐 Custom Secret Manager for LiteLLM

## 🎯 Overview

You can now create custom secret managers in LiteLLM, just like custom guardrails! Integrate **any** secret management system by extending the `CustomSecretManager` base class.

## ⚡ Quick Start (60 seconds)

```python
from typing import Optional, Union
import httpx
from litellm.integrations.custom_secret_manager import CustomSecretManager
import litellm
from litellm.types.secret_managers.main import KeyManagementSystem, KeyManagementSettings

# 1. Create your secret manager
class MyVault(CustomSecretManager):
    def __init__(self, secrets: dict):
        super().__init__(secret_manager_name="my_vault")
        self.secrets = secrets
    
    async def async_read_secret(self, secret_name, optional_params=None, timeout=None):
        return self.secrets.get(secret_name)
    
    def sync_read_secret(self, secret_name, optional_params=None, timeout=None):
        return self.secrets.get(secret_name)

# 2. Configure LiteLLM
litellm.secret_manager_client = MyVault({"OPENAI_API_KEY": "sk-..."})
litellm._key_management_system = KeyManagementSystem.CUSTOM
litellm._key_management_settings = KeyManagementSettings(access_mode="read_only")

# 3. Use it!
from litellm.secret_managers.main import get_secret
api_key = get_secret("OPENAI_API_KEY")  # Uses your custom manager!
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[Quick Start](CUSTOM_SECRET_MANAGER_QUICKSTART.md)** | 5-minute guide with examples |
| **[Implementation Details](IMPLEMENTATION_SUMMARY.md)** | Technical overview and design |
| **[Complete Guide](docs/my-website/docs/secret_management/custom_secret_manager.md)** | Full API reference |
| **[Working Example](cookbook/secret_managers/custom_secret_manager_example.py)** | Runnable code |
| **[Tests](tests/test_litellm/secret_managers/test_custom_secret_manager.py)** | Test examples |

## 🚀 Features

✅ Integrate any secret management system  
✅ Only 2 required methods (`async_read_secret`, `sync_read_secret`)  
✅ Optional write/delete support  
✅ Works with LiteLLM Proxy  
✅ Follows custom guardrail pattern  
✅ Production-ready with error handling  
✅ Comprehensive tests included  
✅ Zero breaking changes  

## 📦 Files Created

### Core Implementation
- **`litellm/integrations/custom_secret_manager.py`** - Base class for custom secret managers

### Tests & Examples
- **`tests/test_litellm/secret_managers/test_custom_secret_manager.py`** - Test suite
- **`cookbook/secret_managers/custom_secret_manager_example.py`** - Working example

### Documentation
- **`CUSTOM_SECRET_MANAGER_QUICKSTART.md`** - Quick start guide
- **`IMPLEMENTATION_SUMMARY.md`** - Technical details
- **`CUSTOM_SECRET_MANAGER_IMPLEMENTATION.md`** - Complete overview
- **`docs/my-website/docs/secret_management/custom_secret_manager.md`** - Website docs
- **`litellm/secret_managers/Readme.md`** - Secret manager guide

### Modified Files
- **`litellm/types/secret_managers/main.py`** - Added `CUSTOM` enum
- **`litellm/secret_managers/secret_manager_handler.py`** - Added custom handler

## 🔨 Implementation Pattern

### Minimal (In-Memory)
```python
class SimpleManager(CustomSecretManager):
    def __init__(self, secrets):
        super().__init__(secret_manager_name="simple")
        self.secrets = secrets
    
    async def async_read_secret(self, secret_name, **kwargs):
        return self.secrets.get(secret_name)
    
    def sync_read_secret(self, secret_name, **kwargs):
        return self.secrets.get(secret_name)
```

### Production (HTTP API)
```python
class VaultManager(CustomSecretManager):
    def __init__(self, vault_url, token):
        super().__init__(secret_manager_name="vault")
        self.vault_url = vault_url
        self.token = token
    
    async def async_read_secret(self, secret_name, optional_params=None, timeout=None):
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.vault_url}/secrets/{secret_name}",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=timeout or 30.0
            )
            response.raise_for_status()
            return response.json()["value"]
    
    def sync_read_secret(self, secret_name, optional_params=None, timeout=None):
        with httpx.Client() as client:
            response = client.get(
                f"{self.vault_url}/secrets/{secret_name}",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=timeout or 30.0
            )
            response.raise_for_status()
            return response.json()["value"]
    
    # Optional: Add write/delete support
    async def async_write_secret(self, secret_name, secret_value, **kwargs):
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.vault_url}/secrets",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"name": secret_name, "value": secret_value}
            )
        return {"status": "success"}
```

## 🎓 Usage Examples

### With LiteLLM Library
```python
import litellm
from litellm.types.secret_managers.main import KeyManagementSystem, KeyManagementSettings

litellm.secret_manager_client = MyVault(...)
litellm._key_management_system = KeyManagementSystem.CUSTOM
litellm._key_management_settings = KeyManagementSettings(access_mode="read_only")

# Use normally - secrets come from your manager
response = litellm.completion(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### With LiteLLM Proxy
```python
# proxy_startup.py
import litellm
from litellm.types.secret_managers.main import KeyManagementSystem, KeyManagementSettings

litellm.secret_manager_client = MyVault(...)
litellm._key_management_system = KeyManagementSystem.CUSTOM
litellm._key_management_settings = KeyManagementSettings(access_mode="read_only")

# Start proxy - it will use your secret manager for API keys
```

## 🧪 Testing

### Run the Example
```bash
python cookbook/secret_managers/custom_secret_manager_example.py
```

### Run Tests
```bash
pytest tests/test_litellm/secret_managers/test_custom_secret_manager.py -v
```

### Write Your Own Tests
```python
import pytest
from your_module import YourSecretManager

def test_read_secret():
    manager = YourSecretManager(...)
    secret = manager.sync_read_secret("TEST_KEY")
    assert secret is not None

@pytest.mark.asyncio
async def test_async_read():
    manager = YourSecretManager(...)
    secret = await manager.async_read_secret("TEST_KEY")
    assert secret is not None
```

## 📋 Required Methods

You **must** implement these two methods:

1. **`async_read_secret(secret_name, optional_params, timeout) -> Optional[str]`**
   - Read a secret asynchronously
   - Return `None` if not found

2. **`sync_read_secret(secret_name, optional_params, timeout) -> Optional[str]`**
   - Read a secret synchronously
   - Return `None` if not found

## 🎁 Optional Methods

You **can** implement these for additional functionality:

- `async_write_secret()` - Write secrets
- `async_delete_secret()` - Delete secrets
- `async_rotate_secret()` - Rotate secrets (inherited)
- `validate_environment()` - Validate configuration
- `async_health_check()` - Health check

## 🆚 Comparison

| Feature | Custom Guardrails | Custom Secret Managers |
|---------|------------------|----------------------|
| **Purpose** | Filter requests/responses | Manage secrets |
| **Base Class** | `CustomLogger` | `BaseSecretManager` |
| **Location** | `litellm/integrations/` | `litellm/integrations/` |
| **Pattern** | Event hooks | Read/write operations |
| **Config** | `GuardrailEventHooks` | `KeyManagementSystem` |

## 🔍 Implementation Details

### Architecture
- Base class: `BaseSecretManager` → `CustomSecretManager` (your class)
- Handler: `secret_manager_handler.py` routes to custom manager
- Type: `KeyManagementSystem.CUSTOM` enum value
- Integration: Works with `get_secret()` function

### Configuration Options
```python
KeyManagementSettings(
    access_mode="read_only",              # or "write_only", "read_and_write"
    hosted_keys=["KEY1", "KEY2"],         # Only check these keys
    store_virtual_keys=True,              # Store virtual keys
    prefix_for_stored_virtual_keys="litellm/",
    description="Managed by LiteLLM",
    tags={"Environment": "Production"}
)
```

## 💡 Best Practices

1. **Error Handling**: Handle network errors gracefully
2. **Logging**: Use `verbose_logger` for debugging
3. **Security**: Never log secret values
4. **Timeouts**: Always configure timeouts
5. **Validation**: Implement `validate_environment()`
6. **Health Checks**: Implement `async_health_check()`
7. **Thread Safety**: Ensure concurrent access is safe

## 🎯 Use Cases

✅ Proprietary vault systems  
✅ Custom authentication (mTLS, OAuth, etc.)  
✅ Organization-specific security policies  
✅ Legacy secret storage systems  
✅ Multi-region secret replication  
✅ Secret versioning and rotation  
✅ Audit logging requirements  
✅ Compliance requirements (HIPAA, SOC2, etc.)  

## 📖 Further Reading

- [Quick Start Guide](CUSTOM_SECRET_MANAGER_QUICKSTART.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)
- [Complete Documentation](docs/my-website/docs/secret_management/custom_secret_manager.md)
- [Working Example](cookbook/secret_managers/custom_secret_manager_example.py)
- [Test Suite](tests/test_litellm/secret_managers/test_custom_secret_manager.py)

## 🤝 Contributing

See `IMPLEMENTATION_SUMMARY.md` for technical details on the implementation.

## ✅ Status

**Implementation**: Complete  
**Tests**: Passing  
**Documentation**: Complete  
**Examples**: Ready  
**Backwards Compatible**: Yes  
**Breaking Changes**: None  

---

**Ready to use!** 🎉

See [CUSTOM_SECRET_MANAGER_QUICKSTART.md](CUSTOM_SECRET_MANAGER_QUICKSTART.md) to get started in 5 minutes.
