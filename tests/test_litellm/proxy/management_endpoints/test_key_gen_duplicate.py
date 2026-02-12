import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from litellm.proxy._types import GenerateKeyRequest, LitellmUserRoles
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.management_endpoints.key_management_endpoints import generate_key_fn

@pytest.mark.asyncio
async def test_generate_key_with_duplicate_raises_error(monkeypatch):
    """
    Test that generating a key with an existing secret key returns HTTP 400.
    
    This verifies the fix for issue #20494 and the related race condition protection.
    """
    mock_prisma_client = AsyncMock()
    
    # Mock hash_token
    mock_prisma_client.hash_token = lambda token: f"hashed-{token}"
    
    # Mock insert_data to fail with Unique constraint violation (simulating atomic failure)
    mock_prisma_client.insert_data = AsyncMock(side_effect=Exception("Unique constraint failed on the fields: (`token`)"))
    
    # Mock other necessary things
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken = MagicMock()
    # Mock find_first for unique key alias check
    mock_prisma_client.db.litellm_verificationtoken.find_first = AsyncMock(return_value=None)
    
    # Use monkeypatch to set the prisma_client
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin", user_id="admin")
    
    # Generate a key with a specific secret that 'already exists' in our mock
    req = GenerateKeyRequest(key="sk-existing-key", key_alias="duplicate test")
    
    # Import ProxyException to check if it's raised (management_endpoint_wrapper might re-raise it)
    from litellm.proxy._types import ProxyException
    
    # We expect an exception because of the duplicate key
    with pytest.raises((HTTPException, ProxyException)) as exc_info:
        await generate_key_fn(data=req, user_api_key_dict=auth)
    
    # Verify status code and message
    if isinstance(exc_info.value, HTTPException):
        assert exc_info.value.status_code == 400
        assert "Key already exists" in str(exc_info.value.detail)
    else:
        # If it's a ProxyException, code is a string
        assert str(exc_info.value.code) == "400"
        assert "Key already exists" in str(exc_info.value.message)
