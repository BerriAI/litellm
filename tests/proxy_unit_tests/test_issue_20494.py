import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from litellm.proxy._types import GenerateKeyRequest, LitellmUserRoles
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.management_endpoints.key_management_endpoints import generate_key_fn

@pytest.mark.asyncio
async def test_repro_issue_20494(monkeypatch):
    """
    Reproduce issue #20494: generating a key with an existing secret key fails silently (returns 200 but doesn't create/update properly).
    """
    mock_prisma_client = AsyncMock()
    
    # Simple database mock
    database = {}
    
    async def mock_insert_data(data, table_name, ignore_duplicates=True):
        if table_name == "key":
            token = data["token"]
            hashed_token = f"hashed-{token}"
            if hashed_token in database:
                if ignore_duplicates is False:
                    raise Exception("Unique constraint failed on the fields: (`token`)")
                return database[hashed_token]
            else:
                record = MagicMock()
                record.token = hashed_token
                record.key_alias = data.get("key_alias")
                database[hashed_token] = record
                return record
        return MagicMock()

    mock_prisma_client.insert_data = AsyncMock(side_effect=mock_insert_data)
    async def mock_find_unique(where):
        token = where.get("token")
        return database.get(token)

    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(side_effect=mock_find_unique)
    # Mock find_first for unique key alias check
    mock_prisma_client.db.litellm_verificationtoken.find_first = AsyncMock(return_value=None)
    
    # Mock hash_token as well to be consistent
    mock_prisma_client.hash_token = lambda token: f"hashed-{token}"

    # Use monkeypatch to set the prisma_client
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin", user_id="admin")
    
    # 1. Generate first key
    req1 = GenerateKeyRequest(key_alias="test duplicate")
    resp1 = await generate_key_fn(data=req1, user_api_key_dict=auth)
    key1 = resp1.key
    
    # 2. Generate second key with same secret key
    # It should ideally throw an error, but currently it returns 200 OK.
    req2 = GenerateKeyRequest(key_alias="test duplicate 2", key=key1)
    
    from litellm.proxy._types import ProxyException
    # THE ISSUE: This call should raise an Exception (e.g. 400 Bad Request)
    with pytest.raises(ProxyException) as excinfo:
        await generate_key_fn(data=req2, user_api_key_dict=auth)
    
    assert str(excinfo.value.code) == "400"
    assert "Key already exists" in str(excinfo.value.message)

@pytest.mark.asyncio
async def test_repro_issue_20494_race_condition(monkeypatch):
    """
    Simulate race condition: find_unique returns None, but insert_data fails with Unique constraint violation.
    """
    mock_prisma_client = AsyncMock()
    
    # 1. Mock find_unique to return None (simulating key not found yet)
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_verificationtoken.find_first = AsyncMock(return_value=None)
    
    # 2. Mock insert_data to fail with Unique constraint violation (simulating another request won the race)
    mock_prisma_client.insert_data = AsyncMock(side_effect=Exception("Unique constraint failed on the fields: (`token`)"))
    
    # Mock hash_token
    mock_prisma_client.hash_token = lambda token: f"hashed-{token}"

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    
    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin", user_id="admin")
    req = GenerateKeyRequest(key_alias="test race", key="sk-already-exists")
    
    from litellm.proxy._types import ProxyException
    with pytest.raises(ProxyException) as excinfo:
        await generate_key_fn(data=req, user_api_key_dict=auth)
    
    assert str(excinfo.value.code) == "400"
    assert "Key already exists" in str(excinfo.value.message)
