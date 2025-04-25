import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import AsyncMock, MagicMock

from litellm.proxy.management_endpoints.key_management_endpoints import _list_key_helper
from litellm.proxy.proxy_server import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_list_keys():
    mock_prisma_client = AsyncMock()
    mock_find_many = AsyncMock(return_value=[])
    mock_prisma_client.db.litellm_verificationtoken.find_many = mock_find_many
    args = {
        "prisma_client": mock_prisma_client,
        "page": 1,
        "size": 50,
        "user_id": "cda88cb4-cc2c-4e8c-b871-dc71ca111b00",
        "team_id": None,
        "organization_id": None,
        "key_alias": None,
        "exclude_team_id": None,
        "return_full_object": True,
        "admin_team_ids": ["28bd3181-02c5-48f2-b408-ce790fb3d5ba"],
    }
    try:
        result = await _list_key_helper(**args)
    except Exception as e:
        print(f"error: {e}")

    mock_find_many.assert_called_once()

    where_condition = mock_find_many.call_args.kwargs["where"]
    print(f"where_condition: {where_condition}")
    assert json.dumps({"team_id": {"not": "litellm-dashboard"}}) in json.dumps(
        where_condition
    )


@pytest.mark.asyncio
async def test_key_token_handling(monkeypatch):
    """
    Test that token handling in key generation follows the expected behavior:
    1. token field should not equal key field
    2. if token_id exists, it should equal token field
    """
    mock_prisma_client = AsyncMock()
    mock_insert_data = AsyncMock(
        return_value=MagicMock(token="hashed_token_123", litellm_budget_table=None)
    )
    mock_prisma_client.insert_data = mock_insert_data
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=None
    )
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[]
    )
    mock_prisma_client.db.litellm_verificationtoken.count = AsyncMock(return_value=0)
    mock_prisma_client.db.litellm_verificationtoken.update = AsyncMock(
        return_value=MagicMock(token="hashed_token_123", litellm_budget_table=None)
    )

    from litellm.proxy._types import GenerateKeyRequest, LitellmUserRoles
    from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_key_fn,
    )
    from litellm.proxy.proxy_server import prisma_client

    # Use monkeypatch to set the prisma_client
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Test key generation
    response = await generate_key_fn(
        data=GenerateKeyRequest(),
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-1234", user_id="1234"
        ),
    )

    # Verify token handling
    assert response.key != response.token, "Token should not equal key"
    if hasattr(response, "token_id"):
        assert (
            response.token == response.token_id
        ), "Token should equal token_id if token_id exists"
