"""
Example: Custom Secret Manager Implementation

This example shows how to create and use a custom secret manager with LiteLLM.
We'll implement a simple file-based secret manager that reads secrets from JSON files.
"""

import json
import os
from pathlib import Path
from typing import Optional, Union

import httpx

from litellm.integrations.custom_secret_manager import CustomSecretManager


class FileBasedSecretManager(CustomSecretManager):
    """
    A simple file-based secret manager that stores secrets in JSON files.
    
    This is useful for development and testing, but should NOT be used in production.
    
    Secrets are stored in a JSON file with the following structure:
    {
        "OPENAI_API_KEY": "sk-...",
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "CUSTOM_SECRET": "my-secret-value"
    }
    """

    def __init__(self, secrets_file_path: str):
        """
        Initialize the file-based secret manager.
        
        Args:
            secrets_file_path: Path to the JSON file containing secrets
        """
        super().__init__(secret_manager_name="file_based_secret_manager")
        self.secrets_file_path = Path(secrets_file_path)
        self._validate_file()

    def _validate_file(self):
        """
        Validate that the secrets file exists and is readable.
        """
        if not self.secrets_file_path.exists():
            raise FileNotFoundError(
                f"Secrets file not found: {self.secrets_file_path}"
            )
        if not self.secrets_file_path.is_file():
            raise ValueError(
                f"Secrets path must be a file: {self.secrets_file_path}"
            )

    def _load_secrets(self) -> dict:
        """
        Load secrets from the JSON file.
        """
        with open(self.secrets_file_path, "r") as f:
            return json.load(f)

    def _save_secrets(self, secrets: dict):
        """
        Save secrets to the JSON file.
        """
        with open(self.secrets_file_path, "w") as f:
            json.dump(secrets, f, indent=2)

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Read a secret from the JSON file asynchronously.
        """
        secrets = self._load_secrets()
        return secrets.get(secret_name)

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Read a secret from the JSON file synchronously.
        """
        secrets = self._load_secrets()
        return secrets.get(secret_name)

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
        Write a secret to the JSON file.
        """
        secrets = self._load_secrets()
        secrets[secret_name] = secret_value
        self._save_secrets(secrets)
        return {
            "status": "success",
            "secret_name": secret_name,
            "description": description,
        }

    async def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: Optional[int] = 7,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> dict:
        """
        Delete a secret from the JSON file.
        """
        secrets = self._load_secrets()
        if secret_name in secrets:
            del secrets[secret_name]
            self._save_secrets(secrets)
            return {"status": "deleted", "secret_name": secret_name}
        return {"status": "not_found", "secret_name": secret_name}

    def validate_environment(self) -> bool:
        """
        Validate that the secrets file exists and is readable.
        """
        self._validate_file()
        return True


def example_1_basic_usage():
    """
    Example 1: Basic usage with file-based secret manager
    """
    print("\n=== Example 1: Basic Usage ===\n")
    
    # Create a temporary secrets file
    secrets_file = "/tmp/litellm_secrets.json"
    with open(secrets_file, "w") as f:
        json.dump({
            "OPENAI_API_KEY": "sk-test-key-12345",
            "ANTHROPIC_API_KEY": "sk-ant-test-key-67890",
        }, f)
    
    # Initialize the secret manager
    secret_manager = FileBasedSecretManager(secrets_file)
    
    # Read secrets
    openai_key = secret_manager.sync_read_secret("OPENAI_API_KEY")
    print(f"OpenAI API Key: {openai_key}")
    
    anthropic_key = secret_manager.sync_read_secret("ANTHROPIC_API_KEY")
    print(f"Anthropic API Key: {anthropic_key}")
    
    # Try to read a non-existent secret
    missing_key = secret_manager.sync_read_secret("NON_EXISTENT_KEY")
    print(f"Non-existent key: {missing_key}")
    
    # Clean up
    os.remove(secrets_file)


def example_2_integration_with_litellm():
    """
    Example 2: Integration with LiteLLM's secret management system
    """
    print("\n=== Example 2: Integration with LiteLLM ===\n")
    
    import litellm
    from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem
    from litellm.secret_managers.main import get_secret
    
    # Create a temporary secrets file
    secrets_file = "/tmp/litellm_secrets.json"
    with open(secrets_file, "w") as f:
        json.dump({
            "OPENAI_API_KEY": "sk-test-openai-integration",
            "COHERE_API_KEY": "test-cohere-key",
        }, f)
    
    # Set up the custom secret manager
    litellm.secret_manager_client = FileBasedSecretManager(secrets_file)
    litellm._key_management_system = KeyManagementSystem.CUSTOM
    litellm._key_management_settings = KeyManagementSettings(
        access_mode="read_only"
    )
    
    # Get secrets using LiteLLM's get_secret function
    openai_key = get_secret("OPENAI_API_KEY")
    print(f"OpenAI API Key via get_secret: {openai_key}")
    
    cohere_key = get_secret("COHERE_API_KEY")
    print(f"Cohere API Key via get_secret: {cohere_key}")
    
    # Test with default value
    missing_key = get_secret("MISSING_KEY", default_value="default-value")
    print(f"Missing key with default: {missing_key}")
    
    # Clean up
    litellm.secret_manager_client = None
    litellm._key_management_system = None
    litellm._key_management_settings = None
    os.remove(secrets_file)


async def example_3_async_operations():
    """
    Example 3: Asynchronous secret operations
    """
    print("\n=== Example 3: Async Operations ===\n")
    
    # Create a temporary secrets file
    secrets_file = "/tmp/litellm_secrets.json"
    with open(secrets_file, "w") as f:
        json.dump({
            "API_KEY_1": "key-1-value",
            "API_KEY_2": "key-2-value",
        }, f)
    
    secret_manager = FileBasedSecretManager(secrets_file)
    
    # Async read
    key1 = await secret_manager.async_read_secret("API_KEY_1")
    print(f"API Key 1 (async): {key1}")
    
    # Async write
    write_result = await secret_manager.async_write_secret(
        secret_name="NEW_API_KEY",
        secret_value="new-key-value",
        description="A newly created API key",
    )
    print(f"Write result: {write_result}")
    
    # Verify write
    new_key = await secret_manager.async_read_secret("NEW_API_KEY")
    print(f"New API Key (async): {new_key}")
    
    # Async delete
    delete_result = await secret_manager.async_delete_secret("API_KEY_2")
    print(f"Delete result: {delete_result}")
    
    # Verify delete
    deleted_key = await secret_manager.async_read_secret("API_KEY_2")
    print(f"Deleted key (should be None): {deleted_key}")
    
    # Clean up
    os.remove(secrets_file)


def example_4_using_with_litellm_proxy():
    """
    Example 4: Using custom secret manager with LiteLLM Proxy
    
    This example shows how to configure a custom secret manager for use with
    the LiteLLM Proxy server.
    """
    print("\n=== Example 4: LiteLLM Proxy Configuration ===\n")
    
    # Create a secrets file
    secrets_file = "/tmp/proxy_secrets.json"
    with open(secrets_file, "w") as f:
        json.dump({
            "OPENAI_API_KEY": "sk-proxy-openai-key",
            "ANTHROPIC_API_KEY": "sk-ant-proxy-key",
            "AZURE_API_KEY": "azure-proxy-key",
        }, f)
    
    print(f"Created secrets file: {secrets_file}")
    print("\nTo use with LiteLLM Proxy, add this to your proxy startup code:\n")
    
    code = f"""
import litellm
from litellm.types.secret_managers.main import KeyManagementSystem, KeyManagementSettings
from your_module import FileBasedSecretManager

# Initialize custom secret manager
litellm.secret_manager_client = FileBasedSecretManager("{secrets_file}")
litellm._key_management_system = KeyManagementSystem.CUSTOM
litellm._key_management_settings = KeyManagementSettings(
    access_mode="read_only"
)

# Now start the proxy
# The proxy will use the custom secret manager to retrieve API keys
"""
    
    print(code)
    
    # Clean up
    os.remove(secrets_file)


if __name__ == "__main__":
    # Run the examples
    example_1_basic_usage()
    example_2_integration_with_litellm()
    
    # Run async example
    import asyncio
    asyncio.run(example_3_async_operations())
    
    example_4_using_with_litellm_proxy()
    
    print("\n=== All Examples Completed ===\n")
