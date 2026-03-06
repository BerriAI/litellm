"""
Test custom secret manager implementation
"""

import os
import sys
from typing import Optional, Union

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm.types.secret_managers.main import KeyManagementSystem


class TestCustomSecretManager(CustomSecretManager):
    """
    Test implementation of a custom secret manager.

    This simulates a simple in-memory secret store for testing purposes.
    """

    def __init__(self, secrets: Optional[dict] = None):
        super().__init__(secret_manager_name="test_custom_secret_manager")
        self.secrets = secrets or {
            "TEST_API_KEY": "test-api-key-12345",
            "TEST_SECRET": "super-secret-value",
            "OPENAI_API_KEY": "sk-test-openai-key",
        }

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Read a secret from the in-memory store asynchronously.
        """
        return self.secrets.get(secret_name)

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Read a secret from the in-memory store synchronously.
        """
        return self.secrets.get(secret_name)

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
        Write a secret to the in-memory store.
        """
        self.secrets[secret_name] = secret_value
        return {
            "secret_name": secret_name,
            "status": "success",
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
        Delete a secret from the in-memory store.
        """
        if secret_name in self.secrets:
            del self.secrets[secret_name]
            return {"secret_name": secret_name, "status": "deleted"}
        return {"secret_name": secret_name, "status": "not_found"}


def test_custom_secret_manager_initialization():
    """
    Test that a custom secret manager can be initialized correctly.
    """
    secret_manager = TestCustomSecretManager()
    assert secret_manager.secret_manager_name == "test_custom_secret_manager"
    assert "TEST_API_KEY" in secret_manager.secrets


def test_custom_secret_manager_sync_read():
    """
    Test synchronous secret reading.
    """
    secret_manager = TestCustomSecretManager()

    # Test reading an existing secret
    api_key = secret_manager.sync_read_secret("TEST_API_KEY")
    assert api_key == "test-api-key-12345"

    # Test reading a non-existent secret
    missing_key = secret_manager.sync_read_secret("NON_EXISTENT_KEY")
    assert missing_key is None


@pytest.mark.asyncio
async def test_custom_secret_manager_async_read():
    """
    Test asynchronous secret reading.
    """
    secret_manager = TestCustomSecretManager()

    # Test reading an existing secret
    api_key = await secret_manager.async_read_secret("TEST_SECRET")
    assert api_key == "super-secret-value"

    # Test reading a non-existent secret
    missing_key = await secret_manager.async_read_secret("NON_EXISTENT_KEY")
    assert missing_key is None


@pytest.mark.asyncio
async def test_custom_secret_manager_async_write():
    """
    Test asynchronous secret writing.
    """
    secret_manager = TestCustomSecretManager()

    # Write a new secret
    result = await secret_manager.async_write_secret(
        secret_name="NEW_SECRET",
        secret_value="new-secret-value",
        description="A new test secret",
    )

    assert result["status"] == "success"
    assert result["secret_name"] == "NEW_SECRET"

    # Verify the secret was written
    secret_value = secret_manager.sync_read_secret("NEW_SECRET")
    assert secret_value == "new-secret-value"


@pytest.mark.asyncio
async def test_custom_secret_manager_async_delete():
    """
    Test asynchronous secret deletion.
    """
    secret_manager = TestCustomSecretManager()

    # Verify secret exists
    assert secret_manager.sync_read_secret("TEST_API_KEY") is not None

    # Delete the secret
    result = await secret_manager.async_delete_secret("TEST_API_KEY")
    assert result["status"] == "deleted"

    # Verify secret is deleted
    assert secret_manager.sync_read_secret("TEST_API_KEY") is None


def test_custom_secret_manager_integration_with_litellm():
    """
    Test that the custom secret manager integrates with LiteLLM's secret management system.
    """
    # Create and set the custom secret manager
    secret_manager = TestCustomSecretManager()
    litellm.secret_manager_client = secret_manager
    litellm._key_management_system = KeyManagementSystem.CUSTOM

    # Set access mode to enable secret reading
    from litellm.types.secret_managers.main import KeyManagementSettings
    litellm._key_management_settings = KeyManagementSettings(
        access_mode="read_only"
    )

    try:
        # Test getting a secret through LiteLLM's get_secret function
        from litellm.secret_managers.main import get_secret

        api_key = get_secret("TEST_API_KEY")
        assert api_key == "test-api-key-12345"

        openai_key = get_secret("OPENAI_API_KEY")
        assert openai_key == "sk-test-openai-key"

    finally:
        # Clean up
        litellm.secret_manager_client = None
        litellm._key_management_system = None
        litellm._key_management_settings = None



class MinimalCustomSecretManager(CustomSecretManager):
    """
    Minimal implementation that only implements required methods.
    """

    def __init__(self):
        super().__init__(secret_manager_name="minimal_custom")

    async def async_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Minimal async read implementation.
        """
        return f"async-{secret_name}-value"

    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Optional[dict] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Optional[str]:
        """
        Minimal sync read implementation.
        """
        return f"sync-{secret_name}-value"


def test_minimal_custom_secret_manager():
    """
    Test that a minimal implementation without write/delete methods works.
    """
    secret_manager = MinimalCustomSecretManager()

    # Read should work
    value = secret_manager.sync_read_secret("TEST_KEY")
    assert value == "sync-TEST_KEY-value"

    # Write should raise NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        import asyncio
        asyncio.run(secret_manager.async_write_secret("KEY", "value"))

    assert "Write operations are not implemented" in str(exc_info.value)

    # Delete should raise NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        import asyncio
        asyncio.run(secret_manager.async_delete_secret("KEY"))

    assert "Delete operations are not implemented" in str(exc_info.value)
