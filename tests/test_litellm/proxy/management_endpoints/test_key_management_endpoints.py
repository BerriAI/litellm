import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException

from litellm.proxy._types import (
    GenerateKeyRequest,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_VerificationToken,
    LitellmUserRoles,
    ProxyException,
    UpdateKeyRequest,
)
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _check_team_key_limits,
    _common_key_generation_helper,
    _list_key_helper,
    check_team_key_model_specific_limits,
    generate_key_helper_fn,
    prepare_key_update_data,
    validate_key_team_change,
)
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
async def test_list_keys_include_created_by_keys():
    """
    Test that include_created_by_keys parameter correctly includes keys created by the user
    and applies specific filtering to both user's own keys and created_by keys.
    """
    mock_prisma_client = AsyncMock()
    mock_find_many = AsyncMock(return_value=[])
    mock_count = AsyncMock(return_value=0)
    mock_prisma_client.db.litellm_verificationtoken.find_many = mock_find_many
    mock_prisma_client.db.litellm_verificationtoken.count = mock_count

    test_user_id = "user-123"
    test_org_id = "org-456"
    test_key_alias = "test-alias"
    test_key_hash = "hashed-token-789"

    # Test Case 1: include_created_by_keys=True with specific filters
    args = {
        "prisma_client": mock_prisma_client,
        "page": 1,
        "size": 50,
        "user_id": test_user_id,
        "team_id": None,
        "organization_id": test_org_id,
        "key_alias": test_key_alias,
        "key_hash": test_key_hash,
        "exclude_team_id": None,
        "return_full_object": True,
        "admin_team_ids": None,
        "include_created_by_keys": True,
    }

    try:
        result = await _list_key_helper(**args)
    except Exception as e:
        print(f"error: {e}")

    mock_find_many.assert_called_once()
    mock_count.assert_called_once()

    where_condition = mock_find_many.call_args.kwargs["where"]
    print(f"where_condition with include_created_by_keys=True: {where_condition}")

    # Verify the structure contains AND with OR conditions
    assert "AND" in where_condition
    assert "OR" in where_condition["AND"][1]

    or_conditions = where_condition["AND"][1]["OR"]

    # Should have 2 OR conditions: user's own keys and created_by keys
    assert len(or_conditions) == 2

    # First condition should be user's own keys with all filters applied
    user_condition = None
    created_by_condition = None

    for condition in or_conditions:
        if "user_id" in condition:
            user_condition = condition
        elif "created_by" in condition:
            created_by_condition = condition

    assert user_condition is not None, "User condition should be present"
    assert created_by_condition is not None, "Created by condition should be present"

    # Verify user condition has all the filters
    assert user_condition["user_id"] == test_user_id
    assert user_condition["organization_id"] == test_org_id
    assert user_condition["key_alias"] == test_key_alias
    assert user_condition["token"] == test_key_hash

    # Verify created_by condition only has the created_by filter (no other filters applied)
    # This is the current behavior - created_by keys don't inherit other filters
    assert created_by_condition["created_by"] == test_user_id
    assert (
        len(created_by_condition) == 1
    ), "Created by condition should only have created_by field"

    # Reset mocks for Test Case 2
    mock_find_many.reset_mock()
    mock_count.reset_mock()

    # Test Case 2: include_created_by_keys=False should not include created_by condition
    args["include_created_by_keys"] = False

    try:
        result = await _list_key_helper(**args)
    except Exception as e:
        print(f"error: {e}")

    where_condition_no_created_by = mock_find_many.call_args.kwargs["where"]
    print(
        f"where_condition with include_created_by_keys=False: {where_condition_no_created_by}"
    )

    # Should not have OR conditions when include_created_by_keys=False and no admin_team_ids
    # The user condition should be merged directly into the where clause
    assert "created_by" not in json.dumps(where_condition_no_created_by)

    # Reset mocks for Test Case 3
    mock_find_many.reset_mock()
    mock_count.reset_mock()

    # Test Case 3: include_created_by_keys=True with exclude_team_id
    args.update(
        {
            "include_created_by_keys": True,
            "exclude_team_id": "excluded-team-123",
            "team_id": None,  # Make sure no specific team is set
        }
    )

    try:
        result = await _list_key_helper(**args)
    except Exception as e:
        print(f"error: {e}")

    where_condition_with_exclude = mock_find_many.call_args.kwargs["where"]
    print(f"where_condition with exclude_team_id: {where_condition_with_exclude}")

    or_conditions_with_exclude = where_condition_with_exclude["AND"][1]["OR"]

    # Find the user condition and created_by condition
    user_condition_with_exclude = None
    created_by_condition_with_exclude = None

    for condition in or_conditions_with_exclude:
        if "user_id" in condition:
            user_condition_with_exclude = condition
        elif "created_by" in condition:
            created_by_condition_with_exclude = condition

    # Verify exclude_team_id is applied to user condition
    assert (
        user_condition_with_exclude is not None
    ), "User condition with exclude should be present"
    assert user_condition_with_exclude["team_id"] == {"not": "excluded-team-123"}

    # Verify created_by condition still only has created_by filter
    assert (
        created_by_condition_with_exclude is not None
    ), "Created by condition with exclude should be present"
    assert created_by_condition_with_exclude["created_by"] == test_user_id
    assert len(created_by_condition_with_exclude) == 1


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
async def test_budget_reset_and_expires_at_first_of_month(monkeypatch):
    """
    Test that when budget_duration, duration, and key_budget_duration are "1mo", budget_reset_at and expires are set to first of next month
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

    # Test key generation with budget_duration="1mo", duration="1mo", key_budget_duration="1mo"
    response = await generate_key_helper_fn(
        request_type="user",
        budget_duration="1mo",
        duration="1mo",
        key_budget_duration="1mo",
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

    # Verify budget_reset_at, expires is set to first of next month
    for key in ["budget_reset_at", "expires"]:
        response_date = response.get(key)
        assert response_date is not None, f"{key} not found in response"
        assert (
            response_date.year == expected_year
        ), f"Expected year {expected_year}, got {response_date.year} for {key}"
        assert (
            response_date.month == expected_month
        ), f"Expected month {expected_month}, got {response_date.month} for {key}"
        assert (
            response_date.day == 1
        ), f"Expected day 1, got {response_date.day} for {key}"


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


@pytest.mark.asyncio
async def test_key_generation_with_mcp_tool_permissions(monkeypatch):
    """
    Test that /key/generate correctly handles mcp_tool_permissions in object_permission.

    This test verifies that:
    1. mcp_tool_permissions is accepted in the object_permission field
    2. The field is properly stored in the LiteLLM_ObjectPermissionTable
    3. The key is correctly linked to the object_permission record
    """
    mock_prisma_client = AsyncMock()
    mock_prisma_client.jsonify_object = lambda data: data

    # Track what data is passed to create
    created_permission_data = {}

    async def mock_create(**kwargs):
        created_permission_data.update(kwargs.get("data", {}))
        return MagicMock(object_permission_id="objperm_mcp_123")

    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.litellm_objectpermissiontable = MagicMock()
    mock_prisma_client.db.litellm_objectpermissiontable.create = mock_create

    async def _insert_data_side_effect(*args, **kwargs):
        table_name = kwargs.get("table_name")
        if table_name == "user":
            return MagicMock(models=[], spend=0)
        elif table_name == "key":
            return MagicMock(
                token="hashed_token_789",
                litellm_budget_table=None,
                object_permission=None,
            )
        return MagicMock()

    mock_prisma_client.insert_data = AsyncMock(side_effect=_insert_data_side_effect)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    from litellm.proxy._types import (
        GenerateKeyRequest,
        LiteLLM_ObjectPermissionBase,
        LitellmUserRoles,
    )
    from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_key_fn,
    )

    # Create request with mcp_tool_permissions
    request_data = GenerateKeyRequest(
        object_permission=LiteLLM_ObjectPermissionBase(
            mcp_servers=["server_1"],
            mcp_tool_permissions={"server_1": ["tool1", "tool2", "tool3"]},
        )
    )

    await generate_key_fn(
        data=request_data,
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="user-mcp-1",
        ),
    )

    # Verify mcp_tool_permissions was stored (serialized to JSON string for GraphQL compatibility)
    assert "mcp_tool_permissions" in created_permission_data
    import json

    assert json.loads(created_permission_data["mcp_tool_permissions"]) == {
        "server_1": ["tool1", "tool2", "tool3"]
    }
    assert created_permission_data["mcp_servers"] == ["server_1"]


@pytest.mark.asyncio
async def test_key_update_object_permissions_existing_permission(monkeypatch):
    """
    Test updating object permissions when a key already has an existing object_permission_id.

    This test verifies that when updating vector stores for a key that already has an
    object_permission_id, the existing LiteLLM_ObjectPermissionTable record is updated
    with the new permissions and the object_permission_id remains the same.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionBase,
        LiteLLM_VerificationToken,
    )
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _handle_update_object_permission,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Mock existing key with object_permission_id
    existing_key_row = LiteLLM_VerificationToken(
        token="test_token_hash",
        object_permission_id="existing_perm_id_123",
        user_id="user123",
        team_id=None,
    )

    # Mock existing object permission record
    existing_object_permission = MagicMock()
    existing_object_permission.model_dump.return_value = {
        "object_permission_id": "existing_perm_id_123",
        "vector_stores": ["old_store_1", "old_store_2"],
    }

    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=existing_object_permission
    )

    # Mock upsert operation
    updated_permission = MagicMock()
    updated_permission.object_permission_id = "existing_perm_id_123"
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock(
        return_value=updated_permission
    )

    # Test data with new object permission
    data_json = {
        "object_permission": LiteLLM_ObjectPermissionBase(
            vector_stores=["new_store_1", "new_store_2", "new_store_3"]
        ).model_dump(exclude_unset=True, exclude_none=True),
        "user_id": "user123",
    }

    # Call the function
    result = await _handle_update_object_permission(
        data_json=data_json,
        existing_key_row=existing_key_row,
    )

    # Verify the object_permission was removed from data_json and object_permission_id was set
    assert "object_permission" not in result
    assert result["object_permission_id"] == "existing_perm_id_123"

    # Verify database operations were called correctly
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_called_once_with(
        where={"object_permission_id": "existing_perm_id_123"}
    )
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_key_update_object_permissions_no_existing_permission(monkeypatch):
    """
    Test creating object permissions when a key has no existing object_permission_id.

    This test verifies that when updating object permissions for a key that has
    object_permission_id set to None, a new entry is created in the
    LiteLLM_ObjectPermissionTable and the key is updated with the new object_permission_id.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionBase,
        LiteLLM_VerificationToken,
    )
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _handle_update_object_permission,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    existing_key_row_no_perm = LiteLLM_VerificationToken(
        token="test_token_hash_2",
        object_permission_id=None,
        user_id="user456",
        team_id=None,
    )

    # Mock find_unique to return None (no existing permission)
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=None
    )

    # Mock upsert to create new record
    new_permission = MagicMock()
    new_permission.object_permission_id = "new_perm_id_456"
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock(
        return_value=new_permission
    )

    data_json = {
        "object_permission": LiteLLM_ObjectPermissionBase(
            vector_stores=["brand_new_store"]
        ).model_dump(exclude_unset=True, exclude_none=True),
        "user_id": "user456",
    }

    result = await _handle_update_object_permission(
        data_json=data_json,
        existing_key_row=existing_key_row_no_perm,
    )

    # Verify new object_permission_id was set
    assert "object_permission" not in result
    assert result["object_permission_id"] == "new_perm_id_456"
    # Verify upsert was called to create new record
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_key_update_object_permissions_missing_permission_record(monkeypatch):
    """
    Test creating object permissions when existing object_permission_id record is not found.

    This test verifies that when updating object permissions for a key that has an
    object_permission_id but the corresponding record cannot be found in the database,
    a new entry is created in the LiteLLM_ObjectPermissionTable with the new permissions.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import (
        LiteLLM_ObjectPermissionBase,
        LiteLLM_VerificationToken,
    )
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _handle_update_object_permission,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    existing_key_row_missing_perm = LiteLLM_VerificationToken(
        token="test_token_hash_3",
        object_permission_id="missing_perm_id_789",
        user_id="user789",
        team_id=None,
    )

    # Mock find_unique to return None (permission record not found)
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=None
    )

    # Mock upsert to create new record
    new_permission = MagicMock()
    new_permission.object_permission_id = "recreated_perm_id_789"
    mock_prisma_client.db.litellm_objectpermissiontable.upsert = AsyncMock(
        return_value=new_permission
    )

    data_json = {
        "object_permission": LiteLLM_ObjectPermissionBase(
            vector_stores=["recreated_store"]
        ).model_dump(exclude_unset=True, exclude_none=True),
        "user_id": "user789",
    }

    result = await _handle_update_object_permission(
        data_json=data_json,
        existing_key_row=existing_key_row_missing_perm,
    )

    # Verify new object_permission_id was set
    assert "object_permission" not in result
    assert result["object_permission_id"] == "recreated_perm_id_789"

    # Verify find_unique was called with the missing permission ID
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_called_once_with(
        where={"object_permission_id": "missing_perm_id_789"}
    )

    # Verify upsert was called to create new record
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_called_once()


def test_get_new_token_with_valid_key():
    """Test get_new_token function when provided with a valid key that starts with 'sk-'"""
    from litellm.proxy._types import RegenerateKeyRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        get_new_token,
    )

    # Test with valid new_key
    data = RegenerateKeyRequest(new_key="sk-test123456789")
    result = get_new_token(data)

    assert result == "sk-test123456789"


def test_get_new_token_with_invalid_key():
    """Test get_new_token function when provided with an invalid key that doesn't start with 'sk-'"""
    from fastapi import HTTPException

    from litellm.proxy._types import RegenerateKeyRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        get_new_token,
    )

    # Test with invalid new_key (doesn't start with 'sk-')
    data = RegenerateKeyRequest(new_key="invalid-key-123")

    with pytest.raises(HTTPException) as exc_info:
        get_new_token(data)

    assert exc_info.value.status_code == 400
    assert "New key must start with 'sk-'" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_generate_service_account_requires_team_id():
    with pytest.raises(HTTPException):
        await _common_key_generation_helper(
            data=GenerateKeyRequest(
                metadata={"service_account_id": "sa"},
                team_id=None,
            ),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-1"
            ),
            litellm_changed_by=None,
            team_table=None,
        )


@pytest.mark.asyncio
async def test_generate_service_account_works_with_team_id():
    from unittest.mock import patch

    # Mock the database and router dependencies from proxy_server
    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.llm_router"
    ) as mock_router, patch("litellm.proxy.proxy_server.premium_user", False), patch(
        "litellm.proxy.management_endpoints.key_management_endpoints.generate_key_helper_fn"
    ) as mock_generate_key:

        # Configure mocks
        mock_prisma.return_value = AsyncMock()
        mock_router.return_value = None
        # Mock the response from generate_key_helper_fn
        mock_generate_key.return_value = {
            "key": "sk-test-key",
            "expires": None,
            "user_id": "test-user",
            "team_id": "IJ",
        }

        # This should not raise an exception since team_id is provided
        await _common_key_generation_helper(
            data=GenerateKeyRequest(
                metadata={"service_account_id": "sa"},
                team_id="IJ",
            ),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-1"
            ),
            litellm_changed_by=None,
            team_table=None,
        )


@pytest.mark.asyncio
async def test_update_service_account_requires_team_id():
    data = UpdateKeyRequest(key="sk-1", metadata={"service_account_id": "sa"})
    existing_key = LiteLLM_VerificationToken(token="hashed", team_id=None)

    with pytest.raises(HTTPException):
        await prepare_key_update_data(data=data, existing_key_row=existing_key)


@pytest.mark.asyncio
async def test_update_service_account_works_with_team_id():
    data = UpdateKeyRequest(
        key="sk-1", metadata={"service_account_id": "sa"}, team_id="IJ"
    )
    existing_key = LiteLLM_VerificationToken(token="hashed")

    await prepare_key_update_data(data=data, existing_key_row=existing_key)


@pytest.mark.asyncio
async def test_validate_team_id_used_in_service_account_request_requires_team_id():
    """
    Test that validate_team_id_used_in_service_account_request raises HTTPException
    when team_id is None for service account key generation.
    """
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        validate_team_id_used_in_service_account_request,
    )

    mock_prisma_client = AsyncMock()

    # Test that HTTPException is raised when team_id is None
    with pytest.raises(HTTPException) as exc_info:
        await validate_team_id_used_in_service_account_request(
            team_id=None,
            prisma_client=mock_prisma_client,
        )

    assert exc_info.value.status_code == 400
    assert "team_id is required for service account keys" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_validate_team_id_used_in_service_account_request_requires_prisma_client():
    """
    Test that validate_team_id_used_in_service_account_request raises HTTPException
    when prisma_client is None for service account key generation.
    """
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        validate_team_id_used_in_service_account_request,
    )

    # Test that HTTPException is raised when prisma_client is None
    with pytest.raises(HTTPException) as exc_info:
        await validate_team_id_used_in_service_account_request(
            team_id="test-team-id",
            prisma_client=None,
        )

    assert exc_info.value.status_code == 400
    assert "prisma_client is required for service account keys" in str(
        exc_info.value.detail
    )


@pytest.mark.asyncio
async def test_validate_team_id_used_in_service_account_request_checks_team_exists():
    """
    Test that validate_team_id_used_in_service_account_request validates that
    the team_id exists in the database for service account key generation.
    """
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        validate_team_id_used_in_service_account_request,
    )

    mock_prisma_client = AsyncMock()

    # Mock the database query to return None (team doesn't exist)
    mock_find_unique = AsyncMock(return_value=None)
    mock_prisma_client.db.litellm_teamtable.find_unique = mock_find_unique

    # Test that HTTPException is raised when team doesn't exist in DB
    with pytest.raises(HTTPException) as exc_info:
        await validate_team_id_used_in_service_account_request(
            team_id="non-existent-team-id",
            prisma_client=mock_prisma_client,
        )

    assert exc_info.value.status_code == 400
    assert "team_id does not exist in the database" in str(exc_info.value.detail)

    # Verify the database was queried with the correct parameters
    mock_find_unique.assert_called_once_with(where={"team_id": "non-existent-team-id"})


@pytest.mark.asyncio
async def test_validate_team_id_used_in_service_account_request_success():
    """
    Test that validate_team_id_used_in_service_account_request returns True
    when team_id exists in the database for service account key generation.
    """
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        validate_team_id_used_in_service_account_request,
    )

    mock_prisma_client = AsyncMock()

    # Mock the database query to return a team object (team exists)
    mock_team = {"team_id": "existing-team-id", "team_name": "Test Team"}
    mock_find_unique = AsyncMock(return_value=mock_team)
    mock_prisma_client.db.litellm_teamtable.find_unique = mock_find_unique

    # Test that function returns True when team exists
    result = await validate_team_id_used_in_service_account_request(
        team_id="existing-team-id",
        prisma_client=mock_prisma_client,
    )

    assert result is True

    # Verify the database was queried with the correct parameters
    mock_find_unique.assert_called_once_with(where={"team_id": "existing-team-id"})


@pytest.mark.asyncio
async def test_generate_service_account_key_endpoint_validation():
    """
    Test that the /key/service-account/generate endpoint properly validates
    team_id requirement and team existence in database.
    """
    from unittest.mock import patch

    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_service_account_key_fn,
    )

    # Test case 1: Missing team_id
    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
        # Mock prisma_client to be not None so we can reach team_id validation
        mock_prisma_instance = AsyncMock()
        mock_prisma.return_value = mock_prisma_instance

        with pytest.raises(HTTPException) as exc_info:
            await generate_service_account_key_fn(
                data=GenerateKeyRequest(team_id=None),
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-1"
                ),
                litellm_changed_by=None,
            )

        assert exc_info.value.status_code == 400
        assert "team_id is required for service account keys" in str(
            exc_info.value.detail
        )

    # Test case 2: Team doesn't exist in database
    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
        # Mock team not found
        mock_find_unique = AsyncMock(return_value=None)
        mock_prisma.db.litellm_teamtable.find_unique = mock_find_unique

        with pytest.raises(HTTPException) as exc_info:
            await generate_service_account_key_fn(
                data=GenerateKeyRequest(team_id="non-existent-team"),
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-1"
                ),
                litellm_changed_by=None,
            )

        assert exc_info.value.status_code == 400
        assert "team_id does not exist in the database" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_unblock_key_supports_both_sk_and_hashed_tokens(monkeypatch):
    """
    Test that the unblock_key endpoint correctly handles both sk- prefixed tokens
    and hashed tokens by properly converting sk- tokens to hashed format before
    database operations.
    """
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._types import BlockKeyRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import unblock_key

    # Mock dependencies
    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()

    # Use a proper 64-character hex hash for testing
    test_hashed_token = (
        "a1b2c3d4e5f6789012345678901234567890123456789012345678901234abcd"
    )

    # Mock the key record that will be returned from database
    mock_key_record = MagicMock()
    mock_key_record.token = test_hashed_token
    mock_key_record.blocked = False
    mock_key_record.model_dump_json.return_value = (
        f'{{"token": "{test_hashed_token}", "blocked": false}}'
    )

    # Mock database operations
    mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=mock_key_record
    )
    mock_prisma_client.db.litellm_verificationtoken.update = AsyncMock(
        return_value=mock_key_record
    )

    # Mock get_key_object and _cache_key_object functions
    mock_key_object = MagicMock()
    mock_key_object.blocked = True  # Initially blocked

    # Mock hash_token function
    def mock_hash_token(token):
        if token == "sk-test123456789":
            return test_hashed_token
        return token

    # Apply monkeypatch
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_user_api_key_cache
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging_obj
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.hash_token", mock_hash_token)
    monkeypatch.setattr(
        "litellm.store_audit_logs", False
    )  # Disable audit logs for simpler test

    # Mock get_key_object and _cache_key_object
    async def mock_get_key_object(**kwargs):
        return mock_key_object

    async def mock_cache_key_object(**kwargs):
        pass

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.get_key_object",
        mock_get_key_object,
    )
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints._cache_key_object",
        mock_cache_key_object,
    )

    # Create mock request and user auth
    mock_request = MagicMock()
    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin", user_id="admin_user"
    )

    # Test Case 1: Using sk- prefixed token
    sk_token_request = BlockKeyRequest(key="sk-test123456789")

    result = await unblock_key(
        data=sk_token_request,
        http_request=mock_request,
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by=None,
    )

    # Verify that the database update was called with hashed token
    mock_prisma_client.db.litellm_verificationtoken.update.assert_called_with(
        where={"token": test_hashed_token}, data={"blocked": False}
    )

    assert result == mock_key_record
    assert mock_key_object.blocked == False  # Should be updated to unblocked

    # Reset mocks for second test
    mock_prisma_client.db.litellm_verificationtoken.update.reset_mock()
    mock_key_object.blocked = True  # Reset to blocked state

    # Test Case 2: Using already hashed token
    hashed_token_request = BlockKeyRequest(key=test_hashed_token)

    result = await unblock_key(
        data=hashed_token_request,
        http_request=mock_request,
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by=None,
    )

    # Verify that the database update was called with the same hashed token
    mock_prisma_client.db.litellm_verificationtoken.update.assert_called_with(
        where={"token": test_hashed_token}, data={"blocked": False}
    )

    assert result == mock_key_record
    assert mock_key_object.blocked == False  # Should be updated to unblocked


@pytest.mark.asyncio
async def test_unblock_key_invalid_key_format(monkeypatch):
    """
    Test that unblock_key properly validates key format and raises appropriate errors
    for invalid keys.
    """
    from litellm.proxy._types import BlockKeyRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import unblock_key
    from litellm.proxy.utils import ProxyException

    # Mock prisma_client to avoid DB connection error
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Mock request and user auth
    mock_request = MagicMock()
    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin", user_id="admin_user"
    )

    # Test with invalid key format
    invalid_key_request = BlockKeyRequest(key="invalid-key-format")

    with pytest.raises(ProxyException) as exc_info:
        await unblock_key(
            data=invalid_key_request,
            http_request=mock_request,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=None,
        )

    assert exc_info.value.code == "400"
    assert "Invalid key format" in str(exc_info.value.message)


def test_validate_key_team_change_with_member_permissions():
    """
    Test validate_key_team_change function with team member permissions.

    This test covers the new logic that allows team members with specific
    permissions to update keys, not just team admins.
    """
    from unittest.mock import MagicMock, patch

    from litellm.proxy._types import KeyManagementRoutes

    # Create mock objects
    mock_key = MagicMock()
    mock_key.user_id = "test-user-123"
    mock_key.models = ["gpt-4"]
    mock_key.tpm_limit = None
    mock_key.rpm_limit = None

    mock_team = MagicMock()
    mock_team.team_id = "test-team-456"
    mock_team.members_with_roles = []
    mock_team.tpm_limit = None
    mock_team.rpm_limit = None

    mock_change_initiator = MagicMock()
    mock_change_initiator.user_id = "test-user-123"

    mock_router = MagicMock()

    # Mock the member object returned by _get_user_in_team
    mock_member_object = MagicMock()

    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints.can_team_access_model"
    ):
        with patch(
            "litellm.proxy.management_endpoints.key_management_endpoints._get_user_in_team"
        ) as mock_get_user:
            with patch(
                "litellm.proxy.management_endpoints.key_management_endpoints._is_user_team_admin"
            ) as mock_is_admin:
                with patch(
                    "litellm.proxy.management_endpoints.key_management_endpoints.TeamMemberPermissionChecks.does_team_member_have_permissions_for_endpoint"
                ) as mock_has_perms:

                    mock_get_user.return_value = mock_member_object
                    mock_is_admin.return_value = False
                    mock_has_perms.return_value = True

                    # This should not raise an exception due to member permissions
                    validate_key_team_change(
                        key=mock_key,
                        team=mock_team,
                        change_initiated_by=mock_change_initiator,
                        llm_router=mock_router,
                    )

                    # Verify the permission check was called with correct parameters
                    mock_has_perms.assert_called_once_with(
                        team_member_object=mock_member_object,
                        team_table=mock_team,
                        route=KeyManagementRoutes.KEY_UPDATE.value,
                    )


def test_key_rotation_fields_helper():
    """
    Test the key data update logic for rotation fields.

    This test focuses on the core logic that adds rotation fields to key_data
    when auto_rotate is enabled, without the complexity of full key generation.
    """
    # Test Case 1: With rotation enabled
    key_data = {"models": ["gpt-3.5-turbo"], "user_id": "test-user"}

    auto_rotate = True
    rotation_interval = "30d"

    # Simulate the rotation logic from generate_key_helper_fn
    if auto_rotate and rotation_interval:
        key_data.update(
            {"auto_rotate": auto_rotate, "rotation_interval": rotation_interval}
        )

    # Verify rotation fields are added
    assert key_data["auto_rotate"] == True
    assert key_data["rotation_interval"] == "30d"
    assert key_data["models"] == ["gpt-3.5-turbo"]  # Original fields preserved

    # Test Case 2: Without rotation enabled
    key_data2 = {"models": ["gpt-4"], "user_id": "test-user"}

    auto_rotate2 = False
    rotation_interval2 = None

    # Simulate the rotation logic
    if auto_rotate2 and rotation_interval2:
        key_data2.update(
            {"auto_rotate": auto_rotate2, "rotation_interval": rotation_interval2}
        )

    # Verify rotation fields are NOT added
    assert "auto_rotate" not in key_data2
    assert "rotation_interval" not in key_data2
    assert key_data2["models"] == ["gpt-4"]  # Original fields preserved

    # Test Case 3: auto_rotate=True but no interval
    key_data3 = {"models": ["claude-3"], "user_id": "test-user"}

    auto_rotate3 = True
    rotation_interval3 = None

    # Simulate the rotation logic
    if auto_rotate3 and rotation_interval3:
        key_data3.update(
            {"auto_rotate": auto_rotate3, "rotation_interval": rotation_interval3}
        )

    # Verify rotation fields are NOT added (missing interval)
    assert "auto_rotate" not in key_data3
    assert "rotation_interval" not in key_data3


@pytest.mark.asyncio
async def test_update_key_fn_auto_rotate_enable():
    """Test that update_key_fn properly handles enabling auto rotation."""
    from litellm.proxy._types import LiteLLM_VerificationToken, UpdateKeyRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        prepare_key_update_data,
    )

    # Mock existing key
    existing_key = LiteLLM_VerificationToken(
        token="test-token",
        key_alias="test-key",
        models=["gpt-3.5-turbo"],
        user_id="test-user",
        team_id=None,
        auto_rotate=False,
        rotation_interval=None,
        metadata={},
    )

    # Test enabling auto rotation
    update_request = UpdateKeyRequest(
        key="test-token", auto_rotate=True, rotation_interval="30d"
    )

    result = await prepare_key_update_data(
        data=update_request, existing_key_row=existing_key
    )

    # Verify rotation fields are included
    assert result["auto_rotate"] is True
    assert result["rotation_interval"] == "30d"


@pytest.mark.asyncio
async def test_update_key_fn_auto_rotate_disable():
    """Test that update_key_fn properly handles disabling auto rotation."""
    from litellm.proxy._types import LiteLLM_VerificationToken, UpdateKeyRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        prepare_key_update_data,
    )

    # Mock existing key with rotation enabled
    existing_key = LiteLLM_VerificationToken(
        token="test-token",
        key_alias="test-key",
        models=["gpt-3.5-turbo"],
        user_id="test-user",
        team_id=None,
        auto_rotate=True,
        rotation_interval="30d",
        metadata={},
    )

    # Test disabling auto rotation
    update_request = UpdateKeyRequest(key="test-token", auto_rotate=False)

    result = await prepare_key_update_data(
        data=update_request, existing_key_row=existing_key
    )

    # Verify auto_rotate is set to False
    assert result["auto_rotate"] is False


@pytest.mark.asyncio
async def test_check_team_key_limits_no_existing_keys():
    """
    Test _check_team_key_limits when team has no existing keys.
    Should allow any TPM/RPM limits within team bounds.
    """
    # Mock prisma client
    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[]
    )

    # Create team table with limits
    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-123",
        team_alias="test-team",
        tpm_limit=10000,
        rpm_limit=1000,
        max_budget=100.0,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[],
    )

    # Create request with limits within team bounds
    data = GenerateKeyRequest(
        tpm_limit=5000,
        rpm_limit=500,
        tpm_limit_type="guaranteed_throughput",
        rpm_limit_type="guaranteed_throughput",
    )

    # Should not raise any exception
    await _check_team_key_limits(
        team_table=team_table,
        data=data,
        prisma_client=mock_prisma_client,
    )

    # Verify database was queried
    mock_prisma_client.db.litellm_verificationtoken.find_many.assert_called_once_with(
        where={"team_id": "test-team-123"}
    )


@pytest.mark.asyncio
async def test_check_team_key_limits_with_existing_keys_within_bounds():
    """
    Test _check_team_key_limits when team has existing keys but total allocation
    is still within team limits.
    """
    # Create mock existing keys
    existing_key1 = MagicMock()
    existing_key1.tpm_limit = 3000
    existing_key1.rpm_limit = 200

    existing_key2 = MagicMock()
    existing_key2.tpm_limit = 2000
    existing_key2.rpm_limit = 300

    existing_key3 = MagicMock()
    existing_key3.tpm_limit = None  # Should be ignored in calculation
    existing_key3.rpm_limit = None  # Should be ignored in calculation

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[existing_key1, existing_key2, existing_key3]
    )

    # Create team table with limits
    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-456",
        team_alias="test-team",
        tpm_limit=10000,  # Total: 3000 + 2000 + 4000 (new) = 9000 < 10000 ✓
        rpm_limit=1000,  # Total: 200 + 300 + 400 (new) = 900 < 1000 ✓
        max_budget=100.0,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[],
    )

    # Create request that would still be within bounds
    data = GenerateKeyRequest(
        tpm_limit=4000,
        rpm_limit=400,
    )

    # Should not raise any exception
    await _check_team_key_limits(
        team_table=team_table,
        data=data,
        prisma_client=mock_prisma_client,
    )


@pytest.mark.asyncio
async def test_check_team_key_limits_tpm_overallocation():
    """
    Test _check_team_key_limits when new key would cause TPM overallocation.
    Should raise HTTPException with appropriate error message.
    """
    # Create mock existing keys with high TPM usage
    existing_key1 = MagicMock()
    existing_key1.tpm_limit = 6000
    existing_key1.rpm_limit = 100
    existing_key1.metadata = {}

    existing_key2 = MagicMock()
    existing_key2.tpm_limit = 3000
    existing_key2.rpm_limit = 200
    existing_key2.metadata = {}

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[existing_key1, existing_key2]
    )

    # Create team table with limits
    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-789",
        team_alias="test-team",
        tpm_limit=10000,  # Allocated: 6000 + 3000 = 9000, New: 2000, Total: 11000 > 10000 ✗
        rpm_limit=1000,
        max_budget=100.0,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[],
    )

    # Create request that would exceed TPM limits
    data = GenerateKeyRequest(
        tpm_limit=2000,
        rpm_limit=100,
        tpm_limit_type="guaranteed_throughput",
    )

    # Should raise HTTPException for TPM overallocation
    with pytest.raises(HTTPException) as exc_info:
        await _check_team_key_limits(
            team_table=team_table,
            data=data,
            prisma_client=mock_prisma_client,
        )

    assert exc_info.value.status_code == 400
    assert (
        "Allocated TPM limit=9000 + Key TPM limit=2000 is greater than team TPM limit=10000"
        in str(exc_info.value.detail)
    )


@pytest.mark.asyncio
async def test_check_team_key_limits_rpm_overallocation():
    """
    Test _check_team_key_limits when new key would cause RPM overallocation.
    Should raise HTTPException with appropriate error message.
    """
    # Create mock existing keys with high RPM usage
    existing_key1 = MagicMock()
    existing_key1.tpm_limit = 1000
    existing_key1.rpm_limit = 600
    existing_key1.metadata = {}

    existing_key2 = MagicMock()
    existing_key2.tpm_limit = 2000
    existing_key2.rpm_limit = 300
    existing_key2.metadata = {}

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[existing_key1, existing_key2]
    )

    # Create team table with limits
    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-101",
        team_alias="test-team",
        tpm_limit=10000,
        rpm_limit=1000,  # Allocated: 600 + 300 = 900, New: 200, Total: 1100 > 1000 ✗
        max_budget=100.0,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[],
    )

    # Create request that would exceed RPM limits
    data = GenerateKeyRequest(
        tpm_limit=1000,
        rpm_limit=200,
        rpm_limit_type="guaranteed_throughput",
    )

    # Should raise HTTPException for RPM overallocation
    with pytest.raises(HTTPException) as exc_info:
        await _check_team_key_limits(
            team_table=team_table,
            data=data,
            prisma_client=mock_prisma_client,
        )

    assert exc_info.value.status_code == 400
    assert (
        "Allocated RPM limit=900 + Key RPM limit=200 is greater than team RPM limit=1000"
        in str(exc_info.value.detail)
    )


@pytest.mark.asyncio
async def test_check_team_key_limits_no_team_limits():
    """
    Test _check_team_key_limits when team has no TPM/RPM limits set.
    Should allow any key limits since there are no team constraints.
    """
    # Create mock existing keys
    existing_key = MagicMock()
    existing_key.tpm_limit = 5000
    existing_key.rpm_limit = 500

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[existing_key]
    )

    # Create team table with no limits
    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-202",
        team_alias="test-team",
        tpm_limit=None,  # No team limit
        rpm_limit=None,  # No team limit
        max_budget=100.0,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[],
    )

    # Create request with any limits
    data = GenerateKeyRequest(
        tpm_limit=10000,  # High limit should be allowed
        rpm_limit=2000,  # High limit should be allowed
    )

    # Should not raise any exception
    await _check_team_key_limits(
        team_table=team_table,
        data=data,
        prisma_client=mock_prisma_client,
    )


@pytest.mark.asyncio
async def test_check_team_key_limits_no_key_limits():
    """
    Test _check_team_key_limits when new key has no TPM/RPM limits.
    Should not raise any exceptions since no limits are being allocated.
    """
    # Create mock existing keys
    existing_key = MagicMock()
    existing_key.tpm_limit = 8000
    existing_key.rpm_limit = 800

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[existing_key]
    )

    # Create team table with limits
    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-303",
        team_alias="test-team",
        tpm_limit=10000,
        rpm_limit=1000,
        max_budget=100.0,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[],
    )

    # Create request with no limits
    data = GenerateKeyRequest(
        tpm_limit=None,  # No limit being set
        rpm_limit=None,  # No limit being set
    )

    # Should not raise any exception
    await _check_team_key_limits(
        team_table=team_table,
        data=data,
        prisma_client=mock_prisma_client,
    )


@pytest.mark.asyncio
async def test_check_team_key_limits_mixed_scenarios():
    """
    Test _check_team_key_limits with mixed scenarios:
    - Some existing keys have limits, others don't
    - New key has only one type of limit
    - Team has only one type of limit
    """
    # Create mock existing keys with mixed limits
    existing_key1 = MagicMock()
    existing_key1.tpm_limit = 3000
    existing_key1.rpm_limit = None  # No RPM limit

    existing_key2 = MagicMock()
    existing_key2.tpm_limit = None  # No TPM limit
    existing_key2.rpm_limit = 400

    existing_key3 = MagicMock()
    existing_key3.tpm_limit = 2000
    existing_key3.rpm_limit = 300

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[existing_key1, existing_key2, existing_key3]
    )

    # Create team table with only TPM limit
    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-404",
        team_alias="test-team",
        tpm_limit=10000,  # Allocated: 3000 + 0 + 2000 = 5000, New: 4000, Total: 9000 < 10000 ✓
        rpm_limit=None,  # No team RPM limit
        max_budget=100.0,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[],
    )

    # Create request with only TPM limit
    data = GenerateKeyRequest(
        tpm_limit=4000,
        rpm_limit=None,  # No RPM limit being set
    )

    # Should not raise any exception
    await _check_team_key_limits(
        team_table=team_table,
        data=data,
        prisma_client=mock_prisma_client,
    )


@pytest.mark.asyncio
async def test_check_team_key_limits_exact_boundary():
    """
    Test _check_team_key_limits when allocation exactly matches team limits.
    Should allow the allocation (boundary case).
    """
    # Create mock existing keys
    existing_key = MagicMock()
    existing_key.tpm_limit = 7000
    existing_key.rpm_limit = 700

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[existing_key]
    )

    # Create team table with limits
    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-505",
        team_alias="test-team",
        tpm_limit=10000,  # Allocated: 7000, New: 3000, Total: 10000 = 10000 ✓
        rpm_limit=1000,  # Allocated: 700, New: 300, Total: 1000 = 1000 ✓
        max_budget=100.0,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[],
    )

    # Create request that exactly matches remaining capacity
    data = GenerateKeyRequest(
        tpm_limit=3000,
        rpm_limit=300,
    )

    # Should not raise any exception (exact boundary should be allowed)
    await _check_team_key_limits(
        team_table=team_table,
        data=data,
        prisma_client=mock_prisma_client,
    )


def test_check_team_key_model_specific_limits_no_limits():
    """
    Test check_team_key_model_specific_limits when no model-specific limits are set.
    Should return without raising any exceptions.
    """
    # Create existing key with no model-specific limits
    existing_key = LiteLLM_VerificationToken(
        token="test-token-1",
        user_id="test-user",
        team_id="test-team-123",
        metadata={},
    )

    keys = [existing_key]

    # Create team table
    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-123",
        team_alias="test-team",
        tpm_limit=10000,
        rpm_limit=1000,
        max_budget=100.0,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[],
        metadata={},
    )

    # Create request with no model-specific limits
    data = GenerateKeyRequest(
        model_rpm_limit=None,
        model_tpm_limit=None,
    )

    # Should not raise any exception
    check_team_key_model_specific_limits(
        keys=keys,
        team_table=team_table,
        data=data,
    )


def test_check_team_key_model_specific_limits_rpm_overallocation():
    """
    Test check_team_key_model_specific_limits when model-specific RPM would cause overallocation.
    Should raise HTTPException with appropriate error message.
    """
    # Create existing keys with model-specific RPM limits
    existing_key1 = LiteLLM_VerificationToken(
        token="test-token-1",
        user_id="test-user-1",
        team_id="test-team-456",
        metadata={
            "model_rpm_limit": {
                "gpt-4": 500,
                "gpt-3.5-turbo": 300,
            }
        },
    )

    existing_key2 = LiteLLM_VerificationToken(
        token="test-token-2",
        user_id="test-user-2",
        team_id="test-team-456",
        metadata={
            "model_rpm_limit": {
                "gpt-4": 300,
            }
        },
    )

    keys = [existing_key1, existing_key2]

    # Create team table with RPM limit
    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-456",
        team_alias="test-team",
        tpm_limit=10000,
        rpm_limit=1000,  # Total team RPM limit
        max_budget=100.0,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[],
        metadata={},
    )

    # Create request that would exceed model-specific RPM limits
    # Existing gpt-4: 500 + 300 = 800, New: 300, Total: 1100 > 1000 (team limit)
    data = GenerateKeyRequest(
        model_rpm_limit={
            "gpt-4": 300,  # This would cause overallocation
        },
        model_tpm_limit=None,
    )

    # Should raise HTTPException for model-specific RPM overallocation
    with pytest.raises(HTTPException) as exc_info:
        check_team_key_model_specific_limits(
            keys=keys,
            team_table=team_table,
            data=data,
        )

    assert exc_info.value.status_code == 400
    assert (
        "Allocated RPM limit=800 + Key RPM limit=300 is greater than team RPM limit=1000"
        in str(exc_info.value.detail)
    )


def test_check_team_key_model_specific_limits_team_model_rpm_overallocation():
    """
    Test check_team_key_model_specific_limits when team has model-specific RPM limits
    in metadata and key allocation would exceed those limits.

    This tests the scenario where team_table.metadata["model_rpm_limit"] is set
    with per-model limits, not just a global team RPM limit.
    """
    # Create existing keys with model-specific RPM limits
    existing_key1 = LiteLLM_VerificationToken(
        token="test-token-1",
        user_id="test-user-1",
        team_id="test-team-789",
        metadata={
            "model_rpm_limit": {
                "gpt-4": 300,
                "gpt-3.5-turbo": 200,
            }
        },
    )

    existing_key2 = LiteLLM_VerificationToken(
        token="test-token-2",
        user_id="test-user-2",
        team_id="test-team-789",
        metadata={
            "model_rpm_limit": {
                "gpt-4": 250,
            }
        },
    )

    keys = [existing_key1, existing_key2]

    # Create team table with model-specific RPM limits in metadata
    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-789",
        team_alias="test-team",
        tpm_limit=None,
        rpm_limit=None,
        max_budget=100.0,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[],
        metadata={
            "model_rpm_limit": {
                "gpt-4": 700,  # Team-level model-specific limit for gpt-4
                "gpt-3.5-turbo": 500,
            }
        },
    )

    # Create request that would exceed team's model-specific RPM limits
    # Existing gpt-4: 300 + 250 = 550, New: 200, Total: 750 > 700 (team model-specific limit)
    data = GenerateKeyRequest(
        model_rpm_limit={
            "gpt-4": 200,  # This would cause overallocation against team model-specific limit
        },
        model_tpm_limit=None,
    )

    # Should raise HTTPException for team model-specific RPM overallocation
    with pytest.raises(HTTPException) as exc_info:
        check_team_key_model_specific_limits(
            keys=keys,
            team_table=team_table,
            data=data,
        )

    assert exc_info.value.status_code == 400
    assert (
        "Allocated RPM limit=550 + Key RPM limit=200 is greater than team RPM limit=700"
        in str(exc_info.value.detail)
    )


def test_check_team_key_model_specific_limits_team_model_tpm_overallocation():
    """
    Test check_team_key_model_specific_limits when team has model-specific TPM limits
    in metadata and key allocation would exceed those limits.

    This tests the scenario where team_table.metadata["model_tpm_limit"] is set
    with per-model limits, not just a global team TPM limit.
    """
    # Create existing keys with model-specific TPM limits
    existing_key1 = LiteLLM_VerificationToken(
        token="test-token-1",
        user_id="test-user-1",
        team_id="test-team-101",
        metadata={
            "model_tpm_limit": {
                "gpt-4": 5000,
                "claude-3": 3000,
            }
        },
    )

    existing_key2 = LiteLLM_VerificationToken(
        token="test-token-2",
        user_id="test-user-2",
        team_id="test-team-101",
        metadata={
            "model_tpm_limit": {
                "gpt-4": 3500,
            }
        },
    )

    keys = [existing_key1, existing_key2]

    # Create team table with model-specific TPM limits in metadata
    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-101",
        team_alias="test-team",
        tpm_limit=None,
        rpm_limit=None,
        max_budget=100.0,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[],
        metadata={
            "model_tpm_limit": {
                "gpt-4": 10000,  # Team-level model-specific limit for gpt-4
                "claude-3": 8000,
            }
        },
    )

    # Create request that would exceed team's model-specific TPM limits
    # Existing gpt-4: 5000 + 3500 = 8500, New: 2000, Total: 10500 > 10000 (team model-specific limit)
    data = GenerateKeyRequest(
        model_rpm_limit=None,
        model_tpm_limit={
            "gpt-4": 2000,  # This would cause overallocation against team model-specific limit
        },
    )

    # Should raise HTTPException for team model-specific TPM overallocation
    with pytest.raises(HTTPException) as exc_info:
        check_team_key_model_specific_limits(
            keys=keys,
            team_table=team_table,
            data=data,
        )

    assert exc_info.value.status_code == 400
    assert (
        "Allocated TPM limit=8500 + Key TPM limit=2000 is greater than team TPM limit=10000"
        in str(exc_info.value.detail)
    )


@pytest.mark.asyncio
async def test_generate_key_with_object_permission():
    """
    Test that /key/generate correctly handles object_permission by:
    1. Creating a record in litellm_objectpermissiontable
    2. Passing the returned object_permission_id into the key insert payload
    3. NOT passing the object_permission dict to the key table
    """
    from unittest.mock import patch

    from litellm.proxy._types import (
        GenerateKeyRequest,
        LiteLLM_ObjectPermissionBase,
        LitellmUserRoles,
    )
    from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _common_key_generation_helper,
    )

    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_prisma_client.jsonify_object = lambda x: x

    # Mock object permission creation
    mock_object_perm_create = AsyncMock(
        return_value=MagicMock(object_permission_id="objperm_key_456")
    )
    mock_prisma_client.db.litellm_objectpermissiontable = MagicMock()
    mock_prisma_client.db.litellm_objectpermissiontable.create = mock_object_perm_create

    # Mock key insertion
    mock_key_insert = AsyncMock(
        return_value=MagicMock(
            token="hashed_token_123",
            litellm_budget_table=None,
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )
    )
    mock_prisma_client.insert_data = mock_key_insert

    # Create request with object_permission
    key_request = GenerateKeyRequest(
        models=["gpt-4"],
        object_permission=LiteLLM_ObjectPermissionBase(
            vector_stores=["vector_store_1"],
            mcp_servers=["mcp_server_1"],
        ),
    )

    mock_admin_auth = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user",
    )

    # Patch the prisma_client and other dependencies
    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma_client,
    ), patch("litellm.proxy.proxy_server.llm_router", None), patch(
        "litellm.proxy.proxy_server.premium_user",
        False,
    ), patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name",
        "admin",
    ):
        # Execute
        result = await _common_key_generation_helper(
            data=key_request,
            user_api_key_dict=mock_admin_auth,
            litellm_changed_by=None,
            team_table=None,
        )

    # Verify object permission creation was called
    mock_object_perm_create.assert_awaited_once()

    # Verify key insertion was called
    assert mock_key_insert.call_count == 1
    key_insert_kwargs = mock_key_insert.call_args.kwargs
    key_data = key_insert_kwargs["data"]

    # Verify object_permission_id is in the key data
    assert key_data.get("object_permission_id") == "objperm_key_456"

    # Verify object_permission dict is NOT in the key data
    assert "object_permission" not in key_data
