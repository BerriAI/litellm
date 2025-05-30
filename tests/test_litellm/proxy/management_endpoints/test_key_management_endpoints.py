import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import AsyncMock, MagicMock

import pytest

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
        "key_hash": None,
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
        return_value=MagicMock(
            token="hashed_token_123", litellm_budget_table=None, object_permission=None
        )
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
        return_value=MagicMock(
            token="hashed_token_123", litellm_budget_table=None, object_permission=None
        )
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


@pytest.mark.asyncio
async def test_budget_reset_at_first_of_month(monkeypatch):
    """
    Test that when budget_duration is "1mo", budget_reset_at is set to first of next month
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

    from datetime import datetime, timezone

    import pytest

    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_key_helper_fn,
    )
    from litellm.proxy.proxy_server import prisma_client

    # Use monkeypatch to set the prisma_client
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Test key generation with budget_duration="1mo"
    response = await generate_key_helper_fn(
        request_type="user",
        budget_duration="1mo",
        user_id="test_user",
    )

    print(f"response: {response}\n")
    # Get the current date
    now = datetime.now(timezone.utc)

    # Calculate expected reset date (first of next month)
    if now.month == 12:
        expected_month = 1
        expected_year = now.year + 1
    else:
        expected_month = now.month + 1
        expected_year = now.year

    # Parse the response date
    response_date = response["budget_reset_at"]

    # Verify budget_reset_at is set to first of next month
    assert (
        response_date.year == expected_year
    ), f"Expected year {expected_year}, got {response_date.year}"
    assert (
        response_date.month == expected_month
    ), f"Expected month {expected_month}, got {response_date.month}"
    assert response_date.day == 1, f"Expected day 1, got {response_date.day}"


@pytest.mark.asyncio
async def test_key_generation_with_object_permission(monkeypatch):
    """Ensure /key/generate correctly handles `object_permission` input by
    1. Creating a record in litellm_objectpermissiontable
    2. Passing the returned `object_permission_id` into the key insert payload
    """
    # --- Setup mocked prisma client ---
    mock_prisma_client = AsyncMock()

    # identity helper for jsonify_object (used inside generate_key_helper_fn)
    mock_prisma_client.jsonify_object = lambda data: data  # type: ignore

    # Mock the prisma_client.db.litellm_objectpermissiontable.create call
    mock_object_permission_create = AsyncMock(
        return_value=MagicMock(object_permission_id="objperm123")
    )
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.litellm_objectpermissiontable = MagicMock()
    mock_prisma_client.db.litellm_objectpermissiontable.create = (
        mock_object_permission_create
    )

    # Mock prisma_client.insert_data for both user and key tables
    async def _insert_data_side_effect(*args, **kwargs):  # type: ignore
        table_name = kwargs.get("table_name")
        if table_name == "user":
            # minimal attributes accessed later in generate_key_helper_fn
            return MagicMock(models=[], spend=0)
        elif table_name == "key":
            return MagicMock(
                token="hashed_token_456",
                litellm_budget_table=None,
                object_permission=None,
            )
        return MagicMock()

    mock_prisma_client.insert_data = AsyncMock(side_effect=_insert_data_side_effect)

    # Attach the mocked prisma client to the proxy_server module
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # --- Import objects after monkeypatching ---
    from litellm.proxy._types import (
        GenerateKeyRequest,
        LiteLLM_ObjectPermissionBase,
        LitellmUserRoles,
    )
    from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_key_fn,
    )

    # --- Call generate_key_fn with object_permission ---
    request_data = GenerateKeyRequest(
        object_permission=LiteLLM_ObjectPermissionBase(vector_stores=["my-vector"])
    )

    await generate_key_fn(
        data=request_data,
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="user-1",
        ),
    )

    # --- Assertions ---
    # 1. Object permission creation was triggered
    mock_object_permission_create.assert_called_once()

    # 2. Key insert received the generated object_permission_id
    key_insert_calls = [
        call.kwargs
        for call in mock_prisma_client.insert_data.call_args_list
        if call.kwargs.get("table_name") == "key"
    ]
    assert len(key_insert_calls) == 1
    assert key_insert_calls[0]["data"].get("object_permission_id") == "objperm123"
