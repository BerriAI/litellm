"""
End-to-End tests for Infisical Secret Manager integration.

These tests connect to an actual Infisical instance and perform real operations.
They require the following environment variables to be set:
- INFISICAL_CLIENT_ID: Machine Identity Client ID
- INFISICAL_CLIENT_SECRET: Machine Identity Client Secret
- INFISICAL_PROJECT_ID: Project/workspace ID
- INFISICAL_ENVIRONMENT: Environment slug (e.g., 'dev', 'staging', 'prod')

Optional:
- INFISICAL_URL: Base URL (defaults to https://app.infisical.com)
- INFISICAL_SECRET_PATH: Secret path (defaults to '/')

To run these tests:
    pytest tests/test_litellm/secret_managers/test_infisical_e2e.py -v
"""

import json
import os
import sys
from typing import Optional
from unittest.mock import MagicMock

import pytest

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

# Mock the premium_user check at module level before imports
sys.modules["litellm.proxy.proxy_server"] = MagicMock()
sys.modules["litellm.proxy.proxy_server"].premium_user = True
sys.modules["litellm.proxy.proxy_server"].CommonProxyErrors = MagicMock()


def check_infisical_credentials() -> Optional[str]:
    """
    Check if Infisical credentials are configured.
    
    Returns:
        None if all credentials are present, or a skip message if missing.
    """
    required_vars = [
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
        "INFISICAL_PROJECT_ID",
        "INFISICAL_ENVIRONMENT",
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        return f"Missing required Infisical credentials: {', '.join(missing_vars)}"
    return None


def skip_if_no_credentials():
    """Skip test if Infisical credentials are not configured."""
    skip_reason = check_infisical_credentials()
    if skip_reason:
        pytest.skip(skip_reason)


def get_unique_secret_name(prefix: str = "litellm_e2e_test") -> str:
    """Generate a unique secret name for testing."""
    from litellm._uuid import uuid
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.mark.asyncio
async def test_infisical_e2e_write_and_read_secret():
    """
    E2E test: Write a secret to Infisical and read it back.
    
    This test:
    1. Creates a new secret in Infisical
    2. Reads the secret back
    3. Verifies the value matches
    4. Cleans up by deleting the secret
    """
    skip_if_no_credentials()
    
    from litellm.secret_managers.infisical_secret_manager import (
        InfisicalSecretManager,
    )

    manager = InfisicalSecretManager()
    secret_name = get_unique_secret_name()
    secret_value = "e2e-test-value-12345"
    
    try:
        # Write the secret
        print(f"\n[E2E] Writing secret: {secret_name}")
        write_result = await manager.async_write_secret(
            secret_name=secret_name,
            secret_value=secret_value,
            description="LiteLLM E2E test secret",
        )
        
        print(f"[E2E] Write result: {write_result}")
        assert write_result["status"] == "success", f"Failed to write secret: {write_result}"
        assert write_result["secret_name"] == secret_name
        
        # Clear cache to ensure we read from Infisical
        manager.cache.delete_cache(f"infisical_secret_{secret_name}")
        
        # Read the secret back
        print(f"[E2E] Reading secret: {secret_name}")
        read_value = await manager.async_read_secret(secret_name=secret_name)
        
        print(f"[E2E] Read value: {read_value}")
        assert read_value == secret_value, f"Expected '{secret_value}', got '{read_value}'"
        
        print(f"[E2E] SUCCESS: Secret write and read verified")
        
    finally:
        # Cleanup: Delete the secret
        print(f"[E2E] Cleaning up: Deleting secret {secret_name}")
        delete_result = await manager.async_delete_secret(secret_name=secret_name)
        print(f"[E2E] Delete result: {delete_result}")


@pytest.mark.asyncio
async def test_infisical_e2e_write_json_secret():
    """
    E2E test: Write a JSON-structured secret to Infisical and read it back.
    
    This test verifies that complex JSON values can be stored and retrieved.
    """
    skip_if_no_credentials()
    
    from litellm.secret_managers.infisical_secret_manager import (
        InfisicalSecretManager,
    )

    manager = InfisicalSecretManager()
    secret_name = get_unique_secret_name(prefix="litellm_e2e_json")
    secret_data = {
        "api_key": "sk-test-12345",
        "model": "gpt-4",
        "temperature": 0.7,
        "metadata": {
            "team": "ml-engineering",
            "project": "litellm-integration",
        },
    }
    secret_value = json.dumps(secret_data)
    
    try:
        # Write the JSON secret
        print(f"\n[E2E] Writing JSON secret: {secret_name}")
        write_result = await manager.async_write_secret(
            secret_name=secret_name,
            secret_value=secret_value,
            description="LiteLLM E2E JSON test secret",
        )
        
        print(f"[E2E] Write result: {write_result}")
        assert write_result["status"] == "success", f"Failed to write secret: {write_result}"
        
        # Clear cache
        manager.cache.delete_cache(f"infisical_secret_{secret_name}")
        
        # Read and parse the JSON secret
        print(f"[E2E] Reading JSON secret: {secret_name}")
        read_value = await manager.async_read_secret(secret_name=secret_name)
        
        print(f"[E2E] Read value: {read_value}")
        parsed_value = json.loads(read_value)
        
        assert parsed_value == secret_data, f"JSON mismatch: expected {secret_data}, got {parsed_value}"
        assert parsed_value["api_key"] == "sk-test-12345"
        assert parsed_value["metadata"]["team"] == "ml-engineering"
        
        print(f"[E2E] SUCCESS: JSON secret verified")
        
    finally:
        # Cleanup
        print(f"[E2E] Cleaning up: Deleting secret {secret_name}")
        delete_result = await manager.async_delete_secret(secret_name=secret_name)
        print(f"[E2E] Delete result: {delete_result}")


@pytest.mark.asyncio
async def test_infisical_e2e_update_secret():
    """
    E2E test: Create a secret, update it, and verify the new value.
    
    This test verifies that the update functionality works correctly.
    """
    skip_if_no_credentials()
    
    from litellm.secret_managers.infisical_secret_manager import (
        InfisicalSecretManager,
    )

    manager = InfisicalSecretManager()
    secret_name = get_unique_secret_name(prefix="litellm_e2e_update")
    initial_value = "initial-value-v1"
    updated_value = "updated-value-v2"
    
    try:
        # Create the secret
        print(f"\n[E2E] Creating secret: {secret_name}")
        create_result = await manager.async_write_secret(
            secret_name=secret_name,
            secret_value=initial_value,
        )
        
        print(f"[E2E] Create result: {create_result}")
        assert create_result["status"] == "success"
        assert create_result["operation"] == "create"
        
        # Clear cache
        manager.cache.delete_cache(f"infisical_secret_{secret_name}")
        
        # Verify initial value
        read_initial = await manager.async_read_secret(secret_name=secret_name)
        assert read_initial == initial_value, f"Initial value mismatch: expected '{initial_value}', got '{read_initial}'"
        
        # Clear cache again
        manager.cache.delete_cache(f"infisical_secret_{secret_name}")
        
        # Update the secret
        print(f"[E2E] Updating secret: {secret_name}")
        update_result = await manager.async_write_secret(
            secret_name=secret_name,
            secret_value=updated_value,
        )
        
        print(f"[E2E] Update result: {update_result}")
        assert update_result["status"] == "success"
        assert update_result["operation"] == "update"
        
        # Clear cache
        manager.cache.delete_cache(f"infisical_secret_{secret_name}")
        
        # Verify updated value
        read_updated = await manager.async_read_secret(secret_name=secret_name)
        assert read_updated == updated_value, f"Updated value mismatch: expected '{updated_value}', got '{read_updated}'"
        
        print(f"[E2E] SUCCESS: Secret update verified")
        
    finally:
        # Cleanup
        print(f"[E2E] Cleaning up: Deleting secret {secret_name}")
        delete_result = await manager.async_delete_secret(secret_name=secret_name)
        print(f"[E2E] Delete result: {delete_result}")


@pytest.mark.asyncio
async def test_infisical_e2e_read_nonexistent_secret():
    """
    E2E test: Attempt to read a secret that doesn't exist.
    
    This test verifies that reading a non-existent secret returns None.
    """
    skip_if_no_credentials()
    
    from litellm.secret_managers.infisical_secret_manager import (
        InfisicalSecretManager,
    )

    manager = InfisicalSecretManager()
    # Use a very unique name that definitely doesn't exist
    secret_name = get_unique_secret_name(prefix="litellm_nonexistent_xyz")
    
    print(f"\n[E2E] Reading non-existent secret: {secret_name}")
    read_value = await manager.async_read_secret(secret_name=secret_name)
    
    print(f"[E2E] Read value: {read_value}")
    assert read_value is None, f"Expected None for non-existent secret, got '{read_value}'"
    
    print(f"[E2E] SUCCESS: Non-existent secret correctly returned None")


@pytest.mark.asyncio
async def test_infisical_e2e_delete_secret():
    """
    E2E test: Create a secret and then delete it.
    
    This test verifies that deletion works correctly.
    """
    skip_if_no_credentials()
    
    from litellm.secret_managers.infisical_secret_manager import (
        InfisicalSecretManager,
    )

    manager = InfisicalSecretManager()
    secret_name = get_unique_secret_name(prefix="litellm_e2e_delete")
    secret_value = "to-be-deleted"
    
    # Create the secret
    print(f"\n[E2E] Creating secret for deletion test: {secret_name}")
    create_result = await manager.async_write_secret(
        secret_name=secret_name,
        secret_value=secret_value,
    )
    
    print(f"[E2E] Create result: {create_result}")
    assert create_result["status"] == "success"
    
    # Clear cache
    manager.cache.delete_cache(f"infisical_secret_{secret_name}")
    
    # Verify it exists
    read_before = await manager.async_read_secret(secret_name=secret_name)
    assert read_before == secret_value, "Secret should exist before deletion"
    
    # Clear cache
    manager.cache.delete_cache(f"infisical_secret_{secret_name}")
    
    # Delete the secret
    print(f"[E2E] Deleting secret: {secret_name}")
    delete_result = await manager.async_delete_secret(secret_name=secret_name)
    
    print(f"[E2E] Delete result: {delete_result}")
    assert delete_result["status"] == "success", f"Delete failed: {delete_result}"
    
    # Clear cache
    manager.cache.delete_cache(f"infisical_secret_{secret_name}")
    
    # Verify it no longer exists
    read_after = await manager.async_read_secret(secret_name=secret_name)
    assert read_after is None, f"Secret should not exist after deletion, got '{read_after}'"
    
    print(f"[E2E] SUCCESS: Secret deletion verified")


@pytest.mark.asyncio
async def test_infisical_e2e_sync_operations():
    """
    E2E test: Test synchronous read operations.
    
    This test verifies that sync_read_secret works correctly.
    """
    skip_if_no_credentials()
    
    from litellm.secret_managers.infisical_secret_manager import (
        InfisicalSecretManager,
    )

    manager = InfisicalSecretManager()
    secret_name = get_unique_secret_name(prefix="litellm_e2e_sync")
    secret_value = "sync-test-value"
    
    try:
        # Create secret using async
        print(f"\n[E2E] Creating secret for sync test: {secret_name}")
        create_result = await manager.async_write_secret(
            secret_name=secret_name,
            secret_value=secret_value,
        )
        assert create_result["status"] == "success"
        
        # Clear cache
        manager.cache.delete_cache(f"infisical_secret_{secret_name}")
        
        # Read using sync method
        print(f"[E2E] Reading secret synchronously: {secret_name}")
        sync_value = manager.sync_read_secret(secret_name=secret_name)
        
        print(f"[E2E] Sync read value: {sync_value}")
        assert sync_value == secret_value, f"Sync read mismatch: expected '{secret_value}', got '{sync_value}'"
        
        print(f"[E2E] SUCCESS: Sync read verified")
        
    finally:
        # Cleanup
        print(f"[E2E] Cleaning up: Deleting secret {secret_name}")
        delete_result = await manager.async_delete_secret(secret_name=secret_name)
        print(f"[E2E] Delete result: {delete_result}")


@pytest.mark.asyncio
async def test_infisical_e2e_list_secrets():
    """
    E2E test: List secrets in a path.
    
    This test creates multiple secrets and verifies they appear in the list.
    """
    skip_if_no_credentials()
    
    from litellm.secret_managers.infisical_secret_manager import (
        InfisicalSecretManager,
    )

    manager = InfisicalSecretManager()
    
    # Create two test secrets
    secret_names = [
        get_unique_secret_name(prefix="litellm_e2e_list_a"),
        get_unique_secret_name(prefix="litellm_e2e_list_b"),
    ]
    
    try:
        for name in secret_names:
            print(f"\n[E2E] Creating secret: {name}")
            result = await manager.async_write_secret(
                secret_name=name,
                secret_value=f"value-for-{name}",
            )
            assert result["status"] == "success", f"Failed to create {name}: {result}"
        
        # List secrets
        print(f"[E2E] Listing secrets")
        secrets_list = await manager.async_list_secrets()
        
        print(f"[E2E] Found {len(secrets_list)} secrets")
        
        # Verify our test secrets are in the list
        secret_keys = [s.get("secretKey") for s in secrets_list]
        for name in secret_names:
            assert name in secret_keys, f"Secret {name} not found in list"
        
        print(f"[E2E] SUCCESS: List secrets verified")
        
    finally:
        # Cleanup
        for name in secret_names:
            print(f"[E2E] Cleaning up: Deleting secret {name}")
            await manager.async_delete_secret(secret_name=name)


@pytest.mark.asyncio
async def test_infisical_e2e_authentication():
    """
    E2E test: Verify authentication works and tokens are cached.
    
    This test verifies that authentication succeeds and caching works.
    """
    skip_if_no_credentials()
    
    from litellm.secret_managers.infisical_secret_manager import (
        InfisicalSecretManager,
    )

    manager = InfisicalSecretManager()
    
    # Clear any cached token
    manager.cache.delete_cache("infisical_access_token")
    
    print(f"\n[E2E] Testing authentication")
    
    # First authentication should hit the API
    token1 = await manager._async_authenticate()
    assert token1 is not None and len(token1) > 0, "Token should not be empty"
    print(f"[E2E] First auth successful, token length: {len(token1)}")
    
    # Second authentication should use cache
    token2 = await manager._async_authenticate()
    assert token2 == token1, "Cached token should be the same"
    print(f"[E2E] Second auth used cache successfully")
    
    print(f"[E2E] SUCCESS: Authentication and caching verified")


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    import asyncio
    
    print("Running Infisical E2E Tests...")
    print("=" * 60)
    
    skip_reason = check_infisical_credentials()
    if skip_reason:
        print(f"SKIPPED: {skip_reason}")
        print("\nTo run these tests, set the following environment variables:")
        print("  - INFISICAL_CLIENT_ID")
        print("  - INFISICAL_CLIENT_SECRET")
        print("  - INFISICAL_PROJECT_ID")
        print("  - INFISICAL_ENVIRONMENT")
        exit(1)
    
    async def run_all_tests():
        tests = [
            ("Authentication", test_infisical_e2e_authentication),
            ("Write and Read", test_infisical_e2e_write_and_read_secret),
            ("JSON Secret", test_infisical_e2e_write_json_secret),
            ("Update Secret", test_infisical_e2e_update_secret),
            ("Read Non-existent", test_infisical_e2e_read_nonexistent_secret),
            ("Delete Secret", test_infisical_e2e_delete_secret),
            ("Sync Operations", test_infisical_e2e_sync_operations),
            ("List Secrets", test_infisical_e2e_list_secrets),
        ]
        
        passed = 0
        failed = 0
        
        for name, test_func in tests:
            print(f"\n{'=' * 60}")
            print(f"Running: {name}")
            print("-" * 60)
            try:
                await test_func()
                print(f"PASSED: {name}")
                passed += 1
            except Exception as e:
                print(f"FAILED: {name}")
                print(f"Error: {e}")
                failed += 1
        
        print(f"\n{'=' * 60}")
        print(f"Results: {passed} passed, {failed} failed")
        return failed == 0
    
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
