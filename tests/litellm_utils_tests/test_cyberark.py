"""
Integration test for CyberArk Conjur Secret Manager.
"""
import os
import sys
import pytest
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath("../.."))
from unittest.mock import AsyncMock, MagicMock, patch
from litellm._uuid import uuid

# Set up environment variables for testing
os.environ["CYBERARK_API_KEY"] = "test-cyberark-api-key-909"
os.environ["CYBERARK_API_BASE"] = "http://0.0.0.0:8080"
os.environ["CYBERARK_ACCOUNT"] = "default"
os.environ["CYBERARK_USERNAME"] = "admin"

from litellm.secret_managers.cyberark_secret_manager import CyberArkSecretManager


def create_mock_response(status_code: int, text: str = ""):
    """
    Helper function to create a mock HTTP response.
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.text = text
    mock_response.raise_for_status = MagicMock()
    
    if status_code >= 400:
        import httpx
        error = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=mock_response
        )
        mock_response.raise_for_status.side_effect = error
    
    return mock_response


@pytest.mark.asyncio
async def test_cyberark_write_and_read_secret():
    """
    Test writing a secret to CyberArk Conjur and reading it back using mocked HTTP requests.
    """
    with patch("litellm.proxy.proxy_server.premium_user", True):
        # Generate unique secret name and value
        secret_name = f"test-secret-{uuid.uuid4()}"
        secret_value = f"test-value-{uuid.uuid4()}"

        # Mock sync httpx client (for auth, ensure variable exists, sync read)
        # The _get_httpx_client returns an HTTPHandler with a .client property
        mock_sync_client = MagicMock()
        # Auth response - note: the actual client is accessed via .client property
        mock_sync_client.client.post.return_value = create_mock_response(
            status_code=200, text="mock-token"
        )
        # Sync read response
        mock_sync_client.client.get.return_value = create_mock_response(
            status_code=200, text=secret_value
        )

        # Mock async httpx client (for async write)
        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = create_mock_response(
            status_code=201, text=""
        )

        with patch(
            "litellm.secret_managers.cyberark_secret_manager._get_httpx_client",
            return_value=mock_sync_client,
        ), patch(
            "litellm.secret_managers.cyberark_secret_manager.get_async_httpx_client",
            return_value=mock_async_client,
        ):
            # Create CyberArk secret manager instance
            cyberark_manager = CyberArkSecretManager()

            # Write the secret
            write_response = await cyberark_manager.async_write_secret(
                secret_name=secret_name,
                secret_value=secret_value,
            )
            print("write_response=", write_response)

            # Validate write was successful
            assert write_response["status"] == "success"

            # Read the secret back
            read_value = cyberark_manager.sync_read_secret(secret_name=secret_name)
            print("READ VALUE=", read_value)

            # Validate the secret exists and has the correct value
            assert read_value is not None
            assert read_value == secret_value


@pytest.mark.asyncio
async def test_cyberark_rotate_secret():
    """
    Test key rotation in CyberArk Conjur using mocked HTTP requests.
    
    This test simulates what happens when a virtual key is rotated:
    1. Write initial secret with alias (like sk-1234)
    2. Rotate to new value (like sk-12359)
    3. Verify reading the secret returns the NEW value
    """
    with patch("litellm.proxy.proxy_server.premium_user", True):
        # Simulate initial virtual key creation
        secret_alias = f"test-rotation-key-{uuid.uuid4()}"
        initial_key_value = f"sk-initial-{uuid.uuid4()}"
        rotated_key_value = f"sk-rotated-{uuid.uuid4()}"

        print(f"\n=== Testing Key Rotation ===")
        print(f"Alias: {secret_alias}")
        print(f"Initial value: {initial_key_value}")
        print(f"Rotated value: {rotated_key_value}")

        # Store the current value to simulate actual storage behavior
        current_value = {"value": initial_key_value}

        # Mock sync httpx client (for auth, ensure variable exists, sync reads)
        # The _get_httpx_client returns an HTTPHandler with a .client property
        mock_sync_client = MagicMock()
        # Auth response - note: the actual client is accessed via .client property
        mock_sync_client.client.post.return_value = create_mock_response(
            status_code=200, text="mock-token"
        )
        
        # Sync reads return the current value from our simulated storage
        def get_mock_sync_read_response(*args, **kwargs):
            return create_mock_response(status_code=200, text=current_value["value"])
        
        mock_sync_client.client.get.side_effect = get_mock_sync_read_response

        # Mock async httpx client (for async writes and reads)
        mock_async_client = AsyncMock()
        
        # Async writes update the current value
        async def mock_async_post(*args, **kwargs):
            content = kwargs.get("content", "")
            if content:
                current_value["value"] = content
            return create_mock_response(status_code=201, text="")
        
        mock_async_client.post.side_effect = mock_async_post
        
        # Async reads also return the current value
        async def get_mock_async_read_response(*args, **kwargs):
            return create_mock_response(status_code=200, text=current_value["value"])
        
        mock_async_client.get.side_effect = get_mock_async_read_response

        with patch(
            "litellm.secret_managers.cyberark_secret_manager._get_httpx_client",
            return_value=mock_sync_client,
        ), patch(
            "litellm.secret_managers.cyberark_secret_manager.get_async_httpx_client",
            return_value=mock_async_client,
        ):
            # Create CyberArk secret manager instance
            cyberark_manager = CyberArkSecretManager()

            # Step 1: Write initial secret (simulates key creation)
            write_response = await cyberark_manager.async_write_secret(
                secret_name=secret_alias,
                secret_value=initial_key_value,
            )
            print(f"\n1. Initial write response: {write_response}")
            assert write_response["status"] == "success"

            # Verify initial value was written
            initial_read = cyberark_manager.sync_read_secret(secret_name=secret_alias)
            print(f"2. Initial read value: {initial_read}")
            assert initial_read == initial_key_value

            # Step 2: Rotate the secret (simulates key rotation)
            # In key rotation, we keep the same secret_name but update the value
            rotation_response = await cyberark_manager.async_rotate_secret(
                current_secret_name=secret_alias,
                new_secret_name=secret_alias,
                new_secret_value=rotated_key_value,
            )
            print(f"3. Rotation response: {rotation_response}")
            assert rotation_response["status"] == "success"

            # Clear cache to force a fresh read
            cyberark_manager.cache.flush_cache()

            # Step 3: Verify the secret now returns the NEW value
            rotated_read = cyberark_manager.sync_read_secret(secret_name=secret_alias)
            print(f"4. After rotation, read value: {rotated_read}")
            
            # This is the key assertion: after rotation, reading should return the NEW value
            assert rotated_read is not None
            assert rotated_read == rotated_key_value
            assert rotated_read != initial_key_value

            print(f"\n✅ Rotation successful: {initial_key_value} → {rotated_key_value}")


@pytest.mark.asyncio
async def test_cyberark_rotate_secret_with_new_alias():
    """
    Test key rotation with a new alias using mocked HTTP requests.
    
    This simulates rotating a key and changing its alias at the same time:
    1. Write secret with alias-v1
    2. Rotate to alias-v2 with new value
    3. Verify alias-v2 has the new value
    4. Verify alias-v1 still exists with old value (CyberArk doesn't delete)
    """
    with patch("litellm.proxy.proxy_server.premium_user", True):
        # Simulate key rotation with alias change
        base_alias = f"test-alias-change-{uuid.uuid4()}"
        old_alias = f"{base_alias}-v1"
        new_alias = f"{base_alias}-v2"
        old_value = f"sk-old-{uuid.uuid4()}"
        new_value = f"sk-new-{uuid.uuid4()}"

        print(f"\n=== Testing Key Rotation with Alias Change ===")
        print(f"Old alias: {old_alias} = {old_value}")
        print(f"New alias: {new_alias} = {new_value}")

        # Store secrets in a dict to simulate actual storage
        secrets_store = {}

        # Mock sync httpx client (for auth, ensure variable exists, sync reads)
        # The _get_httpx_client returns an HTTPHandler with a .client property
        mock_sync_client = MagicMock()
        # Auth response - note: the actual client is accessed via .client property
        mock_sync_client.client.post.return_value = create_mock_response(
            status_code=200, text="mock-token"
        )
        
        # Mock sync reads to return from our store
        def get_mock_sync_read(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            # Extract secret name from URL
            for secret_name, secret_val in secrets_store.items():
                if secret_name in url:
                    return create_mock_response(status_code=200, text=secret_val)
            return create_mock_response(status_code=404, text="Not found")
        
        mock_sync_client.client.get.side_effect = get_mock_sync_read

        # Mock async httpx client (for async writes and reads)
        mock_async_client = AsyncMock()
        
        # Mock async write to update our store
        async def mock_async_post(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            content = kwargs.get("content", "")
            
            # Extract secret name from URL and store the value
            if old_alias in url:
                secrets_store[old_alias] = content
            elif new_alias in url:
                secrets_store[new_alias] = content
            
            return create_mock_response(status_code=201, text="")
        
        mock_async_client.post.side_effect = mock_async_post
        
        # Mock async reads to return from our store
        async def get_mock_async_read(*args, **kwargs):
            url = args[0] if args else kwargs.get("url", "")
            # Extract secret name from URL
            for secret_name, secret_val in secrets_store.items():
                if secret_name in url:
                    return create_mock_response(status_code=200, text=secret_val)
            return create_mock_response(status_code=404, text="Not found")
        
        mock_async_client.get.side_effect = get_mock_async_read

        with patch(
            "litellm.secret_managers.cyberark_secret_manager._get_httpx_client",
            return_value=mock_sync_client,
        ), patch(
            "litellm.secret_managers.cyberark_secret_manager.get_async_httpx_client",
            return_value=mock_async_client,
        ):
            # Create CyberArk secret manager instance
            cyberark_manager = CyberArkSecretManager()

            # Step 1: Create initial secret with old alias
            write_response = await cyberark_manager.async_write_secret(
                secret_name=old_alias,
                secret_value=old_value,
            )
            print(f"\n1. Initial write: {write_response}")
            assert write_response["status"] == "success"

            # Step 2: Rotate to new alias with new value
            rotation_response = await cyberark_manager.async_rotate_secret(
                current_secret_name=old_alias,
                new_secret_name=new_alias,
                new_secret_value=new_value,
            )
            print(f"2. Rotation response: {rotation_response}")
            assert rotation_response["status"] == "success"

            # Clear cache to force fresh reads
            cyberark_manager.cache.flush_cache()

            # Step 3: Verify new alias has new value
            new_read = cyberark_manager.sync_read_secret(secret_name=new_alias)
            print(f"3. Read new alias: {new_read}")
            assert new_read == new_value

            # Step 4: Verify old alias still exists (CyberArk doesn't delete via API)
            old_read = cyberark_manager.sync_read_secret(secret_name=old_alias)
            print(f"4. Read old alias (should still exist): {old_read}")
            assert old_read == old_value

            print(f"\n✅ Alias rotation successful: {old_alias} → {new_alias}")
            print(f"   Note: Old alias still exists in CyberArk (expected behavior)")

