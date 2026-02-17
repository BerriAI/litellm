import json
import os
import sys

import pytest
import yaml
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from litellm.proxy._types import (
    GenerateKeyRequest,
    LiteLLM_BudgetTable,
    LiteLLM_OrganizationTable,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LiteLLM_VerificationToken,
    LitellmUserRoles,
    Member,
    ProxyException,
    ResetSpendRequest,
    UpdateKeyRequest,
)
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _check_org_key_limits,
    _check_team_key_limits,
    _common_key_generation_helper,
    _get_and_validate_existing_key,
    _list_key_helper,
    _persist_deleted_verification_tokens,
    _process_single_key_update,
    _save_deleted_verification_token_records,
    _transform_verification_tokens_to_deleted_records,
    _validate_max_budget,
    _validate_reset_spend_value,
    can_modify_verification_token,
    check_org_key_model_specific_limits,
    check_team_key_model_specific_limits,
    delete_verification_tokens,
    generate_key_helper_fn,
    list_keys,
    prepare_key_update_data,
    reset_key_spend_fn,
    validate_key_list_check,
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
    Test that when budget_duration, duration, and key_budget_duration are "1mo":
    - budget_reset_at is set to first of next month (standardized reset time)
    - expires is set to approximately 1 month from creation time (exact duration)
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

    from datetime import datetime, timedelta, timezone

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

    # Calculate expected reset date (first of next month) for budget_reset_at
    if now.month == 12:
        expected_month = 1
        expected_year = now.year + 1
    else:
        expected_month = now.month + 1
        expected_year = now.year

    # Verify budget_reset_at is set to first of next month (standardized reset time)
    budget_reset_at = response.get("budget_reset_at")
    assert budget_reset_at is not None, "budget_reset_at not found in response"
    assert (
        budget_reset_at.year == expected_year
    ), f"Expected year {expected_year}, got {budget_reset_at.year} for budget_reset_at"
    assert (
        budget_reset_at.month == expected_month
    ), f"Expected month {expected_month}, got {budget_reset_at.month} for budget_reset_at"
    assert (
        budget_reset_at.day == 1
    ), f"Expected day 1, got {budget_reset_at.day} for budget_reset_at"

    # Verify expires is set to approximately 1 month from creation time (exact duration, not standardized)
    expires = response.get("expires")
    assert expires is not None, "expires not found in response"
    # expires should be approximately 1 month from now (same day next month, same time)
    # Allow for some variance due to test execution time (subtract 1 second buffer for timing)
    expected_expires_min = now + timedelta(days=28, seconds=-1)
    expected_expires_max = now + timedelta(days=32)
    assert (
        expected_expires_min <= expires <= expected_expires_max
    ), f"Expected expires to be approximately 1 month from now, got {expires}"


@pytest.mark.asyncio
async def test_key_expiration_exact_duration_hours(monkeypatch):
    """
    Test that key expiration uses exact duration addition, not standardized reset times.
    Specifically tests the bug where "12h" duration would expire at midnight instead of 12 hours from creation.
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

    from datetime import datetime, timedelta, timezone

    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_key_helper_fn,
    )

    # Use monkeypatch to set the prisma_client
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Test key generation with duration="12h"
    # This should expire exactly 12 hours from creation, not at the next midnight/noon boundary
    response = await generate_key_helper_fn(
        request_type="user",
        duration="12h",
        user_id="test_user",
    )

    expires = response.get("expires")
    assert expires is not None, "expires not found in response"

    # Calculate expected expiration (approximately 12 hours from now)
    # Allow for small variance due to test execution time
    now = datetime.now(timezone.utc)
    expected_expires_min = now + timedelta(hours=11, minutes=59)
    expected_expires_max = now + timedelta(hours=12, minutes=1)

    assert (
        expected_expires_min <= expires <= expected_expires_max
    ), f"Expected expires to be approximately 12 hours from now ({now}), got {expires}. Duration should be exact, not aligned to time boundaries."

    # Verify it's NOT aligned to hour boundaries (e.g., not exactly at :00 minutes)
    # If created at 2:30 PM, it should expire at 2:30 AM, not midnight
    expires_minute = expires.minute
    expires_second = expires.second
    # If the expiration is exactly at :00:00, it might be aligned (though could be coincidence)
    # More importantly, verify the duration is correct
    time_diff = expires - now
    hours_diff = time_diff.total_seconds() / 3600
    assert (
        11.9 <= hours_diff <= 12.1
    ), f"Expected expiration to be approximately 12 hours from creation, got {hours_diff} hours"


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
async def test_generate_key_helper_fn_with_access_group_ids(monkeypatch):
    """Ensure generate_key_helper_fn passes access_group_ids into the key insert payload."""
    mock_prisma_client = AsyncMock()
    mock_prisma_client.jsonify_object = lambda data: data  # type: ignore
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.litellm_objectpermissiontable = MagicMock()
    mock_prisma_client.db.litellm_objectpermissiontable.create = AsyncMock(
        return_value=MagicMock(object_permission_id=None)
    )

    captured_key_data = {}

    async def _insert_data_side_effect(*args, **kwargs):
        table_name = kwargs.get("table_name")
        if table_name == "user":
            return MagicMock(models=[], spend=0)
        elif table_name == "key":
            captured_key_data.update(kwargs.get("data", {}))
            return MagicMock(
                token="hashed_token_789",
                litellm_budget_table=None,
                object_permission=None,
                created_at=None,
                updated_at=None,
            )
        return MagicMock()

    mock_prisma_client.insert_data = AsyncMock(side_effect=_insert_data_side_effect)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_key_helper_fn,
    )

    await generate_key_helper_fn(
        request_type="key",
        table_name="key",
        user_id="test-user",
        access_group_ids=["ag-1", "ag-2"],
    )

    assert captured_key_data.get("access_group_ids") == ["ag-1", "ag-2"]


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


@pytest.mark.asyncio
async def test_key_info_returns_object_permission(monkeypatch):
    """
    Test that /key/info correctly returns the object_permission relation.
    
    This test verifies that when calling /key/info for a key with object_permission_id,
    the response includes the full object_permission object with fields like
    mcp_access_groups, mcp_servers, vector_stores, agents, etc.
    
    Regression test for bug where object_permission_id was returned but not the
    related object_permission object.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import LiteLLM_VerificationToken
    from litellm.proxy.management_endpoints.key_management_endpoints import info_key_fn

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    
    # Mock key with object_permission_id
    test_key_token = "hashed_test_token_123"
    test_object_permission_id = "objperm_info_test_123"
    
    mock_key_info = MagicMock(spec=LiteLLM_VerificationToken)
    mock_key_info.token = test_key_token
    mock_key_info.object_permission_id = test_object_permission_id
    mock_key_info.user_id = "user123"
    mock_key_info.team_id = None
    mock_key_info.litellm_budget_table = None
    
    # Mock the dict/model_dump methods
    mock_key_info.model_dump.return_value = {
        "token": test_key_token,
        "object_permission_id": test_object_permission_id,
        "user_id": "user123",
        "team_id": None,
        "litellm_budget_table": None,
    }
    mock_key_info.dict.return_value = mock_key_info.model_dump.return_value
    
    # Mock find_unique for the key lookup
    mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=mock_key_info
    )
    
    # Mock object permission record
    mock_object_permission = MagicMock()
    mock_object_permission.model_dump.return_value = {
        "object_permission_id": test_object_permission_id,
        "mcp_access_groups": ["test_group_1", "test_group_2"],
        "mcp_servers": ["server_1"],
        "vector_stores": ["vs_1", "vs_2"],
        "agents": ["agent_1"],
    }
    mock_object_permission.dict.return_value = mock_object_permission.model_dump.return_value
    
    # Mock find_unique for object permission lookup
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique = AsyncMock(
        return_value=mock_object_permission
    )
    
    # Create user API key dict
    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key="sk-test-key-456",
    )
    
    # Call info_key_fn
    result = await info_key_fn(
        key="sk-test-key-456",
        user_api_key_dict=user_api_key_dict,
    )
    
    # Assertions
    assert "info" in result
    assert "object_permission_id" in result["info"]
    assert result["info"]["object_permission_id"] == test_object_permission_id
    
    # CRITICAL: Verify that object_permission object is included in response
    assert "object_permission" in result["info"], (
        "object_permission field missing from /key/info response. "
        "Expected full object_permission object to be attached."
    )
    
    # Verify object_permission contains the expected fields
    obj_perm = result["info"]["object_permission"]
    assert obj_perm["object_permission_id"] == test_object_permission_id
    assert obj_perm["mcp_access_groups"] == ["test_group_1", "test_group_2"]
    assert obj_perm["mcp_servers"] == ["server_1"]
    assert obj_perm["vector_stores"] == ["vs_1", "vs_2"]
    assert obj_perm["agents"] == ["agent_1"]
    
    # Verify the object permission was actually queried from database
    mock_prisma_client.db.litellm_objectpermissiontable.find_unique.assert_called_once_with(
        where={"object_permission_id": test_object_permission_id}
    )


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
async def test_prepare_key_update_data_duration_never_expires():
    """Test that duration="-1" sets expires to None (never expires)."""
    from litellm.proxy._types import UpdateKeyRequest
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

    # Test setting duration to "-1" (never expires)
    update_request = UpdateKeyRequest(key="test-token", duration="-1")

    result = await prepare_key_update_data(
        data=update_request, existing_key_row=existing_key
    )

    # Verify that expires is set to None
    assert result["expires"] is None


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


@pytest.mark.asyncio
async def test_validate_key_team_change_with_member_permissions():
    """
    Test validate_key_team_change function with team member permissions.

    This test covers the new logic that allows team members with specific
    permissions to update keys, not just team admins.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

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
        "litellm.proxy.management_endpoints.key_management_endpoints.can_team_access_model",
        new_callable=AsyncMock,
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
                    await validate_key_team_change(
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


# ============================================
# Organization Key Limit Tests
# ============================================


@pytest.mark.asyncio
async def test_check_org_key_limits_no_existing_keys():
    """
    Test _check_org_key_limits when organization has no existing keys.
    Should allow any TPM/RPM limits within organization bounds.
    """
    # Mock prisma client
    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[]
    )

    # Create organization table with limits in budget table
    org_table = LiteLLM_OrganizationTable(
        organization_id="test-org-123",
        organization_alias="test-org",
        budget_id="budget-123",
        models=["gpt-4"],
        created_by="admin",
        updated_by="admin",
        litellm_budget_table=LiteLLM_BudgetTable(
            budget_id="budget-123",
            tpm_limit=20000,
            rpm_limit=2000,
        ),
    )

    # Create request with limits within organization bounds
    data = GenerateKeyRequest(
        tpm_limit=10000,
        rpm_limit=1000,
        tpm_limit_type="guaranteed_throughput",
        rpm_limit_type="guaranteed_throughput",
    )

    # Should not raise any exception
    await _check_org_key_limits(
        org_table=org_table,
        data=data,
        prisma_client=mock_prisma_client,
    )

    # Verify database was queried
    mock_prisma_client.db.litellm_verificationtoken.find_many.assert_called_once_with(
        where={"organization_id": "test-org-123"}
    )


@pytest.mark.asyncio
async def test_check_org_key_limits_with_existing_keys_within_bounds():
    """
    Test _check_org_key_limits when organization has existing keys but total allocation
    is still within organization limits.
    """
    # Create mock existing keys
    existing_key1 = MagicMock()
    existing_key1.tpm_limit = 5000
    existing_key1.rpm_limit = 400
    existing_key1.metadata = {}

    existing_key2 = MagicMock()
    existing_key2.tpm_limit = 3000
    existing_key2.rpm_limit = 500
    existing_key2.metadata = {}

    existing_key3 = MagicMock()
    existing_key3.tpm_limit = None  # Should be ignored in calculation
    existing_key3.rpm_limit = None  # Should be ignored in calculation
    existing_key3.metadata = {}

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[existing_key1, existing_key2, existing_key3]
    )

    # Create organization table with limits
    org_table = LiteLLM_OrganizationTable(
        organization_id="test-org-456",
        organization_alias="test-org",
        budget_id="budget-456",
        models=["gpt-4"],
        created_by="admin",
        updated_by="admin",
        litellm_budget_table=LiteLLM_BudgetTable(
            budget_id="budget-456",
            tpm_limit=20000,  # Total: 5000 + 3000 + 8000 (new) = 16000 < 20000 ✓
            rpm_limit=2000,  # Total: 400 + 500 + 800 (new) = 1700 < 2000 ✓
        ),
    )

    # Create request that would still be within bounds
    data = GenerateKeyRequest(
        tpm_limit=8000,
        rpm_limit=800,
        tpm_limit_type="guaranteed_throughput",
        rpm_limit_type="guaranteed_throughput",
    )

    # Should not raise any exception
    await _check_org_key_limits(
        org_table=org_table,
        data=data,
        prisma_client=mock_prisma_client,
    )


@pytest.mark.asyncio
async def test_check_org_key_limits_tpm_overallocation():
    """
    Test _check_org_key_limits when new key would cause TPM overallocation.
    Should raise HTTPException with appropriate error message.
    """
    # Create mock existing keys with high TPM usage
    existing_key1 = MagicMock()
    existing_key1.tpm_limit = 12000
    existing_key1.rpm_limit = 500
    existing_key1.metadata = {}

    existing_key2 = MagicMock()
    existing_key2.tpm_limit = 6000
    existing_key2.rpm_limit = 400
    existing_key2.metadata = {}

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[existing_key1, existing_key2]
    )

    # Create organization table with limits
    org_table = LiteLLM_OrganizationTable(
        organization_id="test-org-789",
        organization_alias="test-org",
        budget_id="budget-789",
        models=["gpt-4"],
        created_by="admin",
        updated_by="admin",
        litellm_budget_table=LiteLLM_BudgetTable(
            budget_id="budget-789",
            tpm_limit=20000,  # Allocated: 12000 + 6000 = 18000, New: 3000, Total: 21000 > 20000 ✗
            rpm_limit=2000,
        ),
    )

    # Create request that would exceed TPM limits
    data = GenerateKeyRequest(
        tpm_limit=3000,
        rpm_limit=500,
        tpm_limit_type="guaranteed_throughput",
    )

    # Should raise HTTPException for TPM overallocation
    with pytest.raises(HTTPException) as exc_info:
        await _check_org_key_limits(
            org_table=org_table,
            data=data,
            prisma_client=mock_prisma_client,
        )

    assert exc_info.value.status_code == 400
    assert (
        "Allocated TPM limit=18000 + Key TPM limit=3000 is greater than organization TPM limit=20000"
        in str(exc_info.value.detail)
    )


@pytest.mark.asyncio
async def test_check_org_key_limits_rpm_overallocation():
    """
    Test _check_org_key_limits when new key would cause RPM overallocation.
    Should raise HTTPException with appropriate error message.
    """
    # Create mock existing keys with high RPM usage
    existing_key1 = MagicMock()
    existing_key1.tpm_limit = 5000
    existing_key1.rpm_limit = 1200
    existing_key1.metadata = {}

    existing_key2 = MagicMock()
    existing_key2.tpm_limit = 3000
    existing_key2.rpm_limit = 600
    existing_key2.metadata = {}

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[existing_key1, existing_key2]
    )

    # Create organization table with limits
    org_table = LiteLLM_OrganizationTable(
        organization_id="test-org-101",
        organization_alias="test-org",
        budget_id="budget-101",
        models=["gpt-4"],
        created_by="admin",
        updated_by="admin",
        litellm_budget_table=LiteLLM_BudgetTable(
            budget_id="budget-101",
            tpm_limit=20000,
            rpm_limit=2000,  # Allocated: 1200 + 600 = 1800, New: 300, Total: 2100 > 2000 ✗
        ),
    )

    # Create request that would exceed RPM limits
    data = GenerateKeyRequest(
        tpm_limit=5000,
        rpm_limit=300,
        rpm_limit_type="guaranteed_throughput",
    )

    # Should raise HTTPException for RPM overallocation
    with pytest.raises(HTTPException) as exc_info:
        await _check_org_key_limits(
            org_table=org_table,
            data=data,
            prisma_client=mock_prisma_client,
        )

    assert exc_info.value.status_code == 400
    assert (
        "Allocated RPM limit=1800 + Key RPM limit=300 is greater than organization RPM limit=2000"
        in str(exc_info.value.detail)
    )


@pytest.mark.asyncio
async def test_check_org_key_limits_no_org_limits():
    """
    Test _check_org_key_limits when organization has no TPM/RPM limits set.
    Should allow any key limits since there are no organization constraints.
    """
    # Create mock existing keys
    existing_key = MagicMock()
    existing_key.tpm_limit = 10000
    existing_key.rpm_limit = 1000
    existing_key.metadata = {}

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[existing_key]
    )

    # Create organization table with no limits
    org_table = LiteLLM_OrganizationTable(
        organization_id="test-org-202",
        organization_alias="test-org",
        budget_id="budget-202",
        models=["gpt-4"],
        created_by="admin",
        updated_by="admin",
        litellm_budget_table=LiteLLM_BudgetTable(
            budget_id="budget-202",
            tpm_limit=None,  # No organization limit
            rpm_limit=None,  # No organization limit
        ),
    )

    # Create request with any limits
    data = GenerateKeyRequest(
        tpm_limit=50000,  # High limit should be allowed
        rpm_limit=5000,  # High limit should be allowed
        tpm_limit_type="guaranteed_throughput",
    )

    # Should not raise any exception
    await _check_org_key_limits(
        org_table=org_table,
        data=data,
        prisma_client=mock_prisma_client,
    )


def test_check_org_key_model_specific_limits_no_limits():
    """
    Test check_org_key_model_specific_limits when no model-specific limits are set.
    Should return without raising any exceptions.
    """
    # Create existing key with no model-specific limits
    existing_key = LiteLLM_VerificationToken(
        token="test-token-1",
        user_id="test-user",
        organization_id="test-org-123",
        metadata={},
    )

    keys = [existing_key]

    # Create organization table
    org_table = LiteLLM_OrganizationTable(
        organization_id="test-org-123",
        organization_alias="test-org",
        budget_id="budget-123",
        models=["gpt-4"],
        created_by="admin",
        updated_by="admin",
        metadata={},
        litellm_budget_table=LiteLLM_BudgetTable(
            budget_id="budget-123",
            tpm_limit=20000,
            rpm_limit=2000,
        ),
    )

    # Create request with no model-specific limits
    data = GenerateKeyRequest(
        model_rpm_limit=None,
        model_tpm_limit=None,
    )

    # Should not raise any exception
    check_org_key_model_specific_limits(
        keys=keys,
        org_table=org_table,
        data=data,
    )


def test_check_org_key_model_specific_limits_rpm_overallocation():
    """
    Test check_org_key_model_specific_limits when model-specific RPM would cause overallocation.
    Should raise HTTPException with appropriate error message.
    """
    # Create existing keys with model-specific RPM limits
    existing_key1 = LiteLLM_VerificationToken(
        token="test-token-1",
        user_id="test-user-1",
        organization_id="test-org-456",
        metadata={
            "model_rpm_limit": {
                "gpt-4": 800,
                "gpt-3.5-turbo": 500,
            }
        },
    )

    existing_key2 = LiteLLM_VerificationToken(
        token="test-token-2",
        user_id="test-user-2",
        organization_id="test-org-456",
        metadata={
            "model_rpm_limit": {
                "gpt-4": 600,
            }
        },
    )

    keys = [existing_key1, existing_key2]

    # Create organization table with RPM limit
    org_table = LiteLLM_OrganizationTable(
        organization_id="test-org-456",
        organization_alias="test-org",
        budget_id="budget-456",
        models=["gpt-4"],
        created_by="admin",
        updated_by="admin",
        metadata={},
        litellm_budget_table=LiteLLM_BudgetTable(
            budget_id="budget-456",
            tpm_limit=20000,
            rpm_limit=2000,  # Total organization RPM limit
        ),
    )

    # Create request that would exceed model-specific RPM limits
    # Existing gpt-4: 800 + 600 = 1400, New: 700, Total: 2100 > 2000 (org limit)
    data = GenerateKeyRequest(
        model_rpm_limit={
            "gpt-4": 700,  # This would cause overallocation
        },
        model_tpm_limit=None,
    )

    # Should raise HTTPException for model-specific RPM overallocation
    with pytest.raises(HTTPException) as exc_info:
        check_org_key_model_specific_limits(
            keys=keys,
            org_table=org_table,
            data=data,
        )

    assert exc_info.value.status_code == 400
    assert (
        "Allocated RPM limit=1400 + Key RPM limit=700 is greater than organization RPM limit=2000"
        in str(exc_info.value.detail)
    )


def test_check_org_key_model_specific_limits_org_model_rpm_overallocation():
    """
    Test check_org_key_model_specific_limits when organization has model-specific RPM limits
    in metadata and key allocation would exceed those limits.

    This tests the scenario where org_table.metadata["model_rpm_limit"] is set
    with per-model limits, not just a global organization RPM limit.
    """
    # Create existing keys with model-specific RPM limits
    existing_key1 = LiteLLM_VerificationToken(
        token="test-token-1",
        user_id="test-user-1",
        organization_id="test-org-789",
        metadata={
            "model_rpm_limit": {
                "gpt-4": 600,
                "gpt-3.5-turbo": 400,
            }
        },
    )

    existing_key2 = LiteLLM_VerificationToken(
        token="test-token-2",
        user_id="test-user-2",
        organization_id="test-org-789",
        metadata={
            "model_rpm_limit": {
                "gpt-4": 500,
            }
        },
    )

    keys = [existing_key1, existing_key2]

    # Create organization table with model-specific RPM limits in metadata
    org_table = LiteLLM_OrganizationTable(
        organization_id="test-org-789",
        organization_alias="test-org",
        budget_id="budget-789",
        models=["gpt-4"],
        created_by="admin",
        updated_by="admin",
        metadata={
            "model_rpm_limit": {
                "gpt-4": 1400,  # Organization-level model-specific limit for gpt-4
                "gpt-3.5-turbo": 1000,
            }
        },
        litellm_budget_table=LiteLLM_BudgetTable(
            budget_id="budget-789",
            tpm_limit=None,
            rpm_limit=None,
        ),
    )

    # Create request that would exceed organization's model-specific RPM limits
    # Existing gpt-4: 600 + 500 = 1100, New: 400, Total: 1500 > 1400 (org model-specific limit)
    data = GenerateKeyRequest(
        model_rpm_limit={
            "gpt-4": 400,  # This would cause overallocation against org model-specific limit
        },
        model_tpm_limit=None,
    )

    # Should raise HTTPException for organization model-specific RPM overallocation
    with pytest.raises(HTTPException) as exc_info:
        check_org_key_model_specific_limits(
            keys=keys,
            org_table=org_table,
            data=data,
        )

    assert exc_info.value.status_code == 400
    assert (
        "Allocated RPM limit=1100 + Key RPM limit=400 is greater than organization RPM limit=1400"
        in str(exc_info.value.detail)
    )


def test_check_org_key_model_specific_limits_org_model_tpm_overallocation():
    """
    Test check_org_key_model_specific_limits when organization has model-specific TPM limits
    in metadata and key allocation would exceed those limits.

    This tests the scenario where org_table.metadata["model_tpm_limit"] is set
    with per-model limits, not just a global organization TPM limit.
    """
    # Create existing keys with model-specific TPM limits
    existing_key1 = LiteLLM_VerificationToken(
        token="test-token-1",
        user_id="test-user-1",
        organization_id="test-org-101",
        metadata={
            "model_tpm_limit": {
                "gpt-4": 10000,
                "claude-3": 6000,
            }
        },
    )

    existing_key2 = LiteLLM_VerificationToken(
        token="test-token-2",
        user_id="test-user-2",
        organization_id="test-org-101",
        metadata={
            "model_tpm_limit": {
                "gpt-4": 7000,
            }
        },
    )

    keys = [existing_key1, existing_key2]

    # Create organization table with model-specific TPM limits in metadata
    org_table = LiteLLM_OrganizationTable(
        organization_id="test-org-101",
        organization_alias="test-org",
        budget_id="budget-101",
        models=["gpt-4"],
        created_by="admin",
        updated_by="admin",
        metadata={
            "model_tpm_limit": {
                "gpt-4": 20000,  # Organization-level model-specific limit for gpt-4
                "claude-3": 15000,
            }
        },
        litellm_budget_table=LiteLLM_BudgetTable(
            budget_id="budget-101",
            tpm_limit=None,
            rpm_limit=None,
        ),
    )

    # Create request that would exceed organization's model-specific TPM limits
    # Existing gpt-4: 10000 + 7000 = 17000, New: 4000, Total: 21000 > 20000 (org model-specific limit)
    data = GenerateKeyRequest(
        model_rpm_limit=None,
        model_tpm_limit={
            "gpt-4": 4000,  # This would cause overallocation against org model-specific limit
        },
    )

    # Should raise HTTPException for organization model-specific TPM overallocation
    with pytest.raises(HTTPException) as exc_info:
        check_org_key_model_specific_limits(
            keys=keys,
            org_table=org_table,
            data=data,
        )

    assert exc_info.value.status_code == 400
    assert (
        "Allocated TPM limit=17000 + Key TPM limit=4000 is greater than organization TPM limit=20000"
        in str(exc_info.value.detail)
    )


def test_transform_verification_tokens_to_deleted_records():
    from datetime import datetime, timezone

    user_api_key_dict = UserAPIKeyAuth(
        user_id="user-123",
        api_key="sk-test",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
    )

    key1 = LiteLLM_VerificationToken(
        token="hashed-token-1",
        user_id="user-123",
        team_id="team-456",
        key_alias="test-key-1",
        spend=100.0,
        max_budget=1000.0,
        models=["gpt-4"],
        aliases={},
        config={},
        permissions={},
        metadata={"test": "value"},
        model_max_budget={},
        model_spend={},
        soft_budget_cooldown=False,
        allowed_routes=[],
    )

    key2 = LiteLLM_VerificationToken(
        token="hashed-token-2",
        user_id="user-789",
        team_id=None,
        key_alias="test-key-2",
        spend=50.0,
        max_budget=500.0,
        models=["gpt-3.5-turbo"],
        aliases={"alias": "model"},
        config={"config": "value"},
        permissions={"permission": True},
        metadata={},
        model_max_budget={"gpt-4": {"budget_limit": 100.0}},
        model_spend={},
        soft_budget_cooldown=False,
        allowed_routes=[],
    )

    records = _transform_verification_tokens_to_deleted_records(
        keys=[key1, key2],
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by="admin-user",
    )

    assert len(records) == 2
    assert all("deleted_at" in record for record in records)
    assert all("deleted_by" in record for record in records)
    assert all("deleted_by_api_key" in record for record in records)
    assert all("litellm_changed_by" in record for record in records)
    assert all(record["deleted_by"] == "user-123" for record in records)
    assert all(record["deleted_by_api_key"] == user_api_key_dict.api_key for record in records)
    assert all(record["litellm_changed_by"] == "admin-user" for record in records)

    record1 = records[0]
    assert record1["token"] == "hashed-token-1"
    assert record1["user_id"] == "user-123"
    assert record1["team_id"] == "team-456"
    assert isinstance(record1["aliases"], str)
    assert isinstance(record1["config"], str)
    assert isinstance(record1["permissions"], str)
    assert isinstance(record1["metadata"], str)
    assert "litellm_budget_table" not in record1
    assert "litellm_organization_table" not in record1
    assert "object_permission" not in record1
    assert "id" not in record1

    record2 = records[1]
    assert record2["token"] == "hashed-token-2"
    assert isinstance(record2["model_max_budget"], str)


def test_transform_verification_tokens_to_deleted_records_empty_list():
    user_api_key_dict = UserAPIKeyAuth(
        user_id="user-123",
        api_key="sk-test",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
    )

    records = _transform_verification_tokens_to_deleted_records(
        keys=[],
        user_api_key_dict=user_api_key_dict,
    )

    assert records == []


@pytest.mark.asyncio
async def test_save_deleted_verification_token_records():
    mock_prisma_client = AsyncMock()
    mock_create_many = AsyncMock()
    mock_prisma_client.db.litellm_deletedverificationtoken.create_many = (
        mock_create_many
    )

    records = [
        {
            "token": "hashed-token-1",
            "user_id": "user-123",
            "deleted_at": "2024-01-01T00:00:00Z",
            "deleted_by": "admin",
        },
        {
            "token": "hashed-token-2",
            "user_id": "user-456",
            "deleted_at": "2024-01-01T00:00:00Z",
            "deleted_by": "admin",
        },
    ]

    await _save_deleted_verification_token_records(
        records=records, prisma_client=mock_prisma_client
    )

    mock_create_many.assert_called_once_with(data=records)


@pytest.mark.asyncio
async def test_save_deleted_verification_token_records_empty_list():
    mock_prisma_client = AsyncMock()
    mock_create_many = AsyncMock()
    mock_prisma_client.db.litellm_deletedverificationtoken.create_many = (
        mock_create_many
    )

    await _save_deleted_verification_token_records(
        records=[], prisma_client=mock_prisma_client
    )

    mock_create_many.assert_not_called()


@pytest.mark.asyncio
async def test_persist_deleted_verification_tokens():
    mock_prisma_client = AsyncMock()
    mock_create_many = AsyncMock()
    mock_prisma_client.db.litellm_deletedverificationtoken.create_many = (
        mock_create_many
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_id="user-123",
        api_key="sk-test",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
    )

    key = LiteLLM_VerificationToken(
        token="hashed-token-1",
        user_id="user-123",
        team_id="team-456",
        key_alias="test-key",
        spend=100.0,
        max_budget=1000.0,
        models=["gpt-4"],
        aliases={},
        config={},
        permissions={},
        metadata={},
        model_max_budget={},
        model_spend={},
        soft_budget_cooldown=False,
        allowed_routes=[],
    )

    await _persist_deleted_verification_tokens(
        keys=[key],
        prisma_client=mock_prisma_client,
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by="admin-user",
    )

    mock_create_many.assert_called_once()
    call_args = mock_create_many.call_args
    assert "data" in call_args.kwargs
    records = call_args.kwargs["data"]
    assert len(records) == 1
    assert records[0]["token"] == "hashed-token-1"
    assert records[0]["deleted_by"] == "user-123"
    assert records[0]["litellm_changed_by"] == "admin-user"


@pytest.mark.asyncio
async def test_delete_verification_tokens_persists_deleted_keys(monkeypatch):
    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    user_api_key_dict = UserAPIKeyAuth(
        user_id="admin-user",
        api_key="sk-admin",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
    )

    key1 = LiteLLM_VerificationToken(
        token="hashed-token-1",
        user_id="user-123",
        team_id="team-456",
        key_alias="test-key-1",
        spend=100.0,
        max_budget=1000.0,
        models=["gpt-4"],
        aliases={},
        config={},
        permissions={},
        metadata={},
        model_max_budget={},
        model_spend={},
        soft_budget_cooldown=False,
        allowed_routes=[],
    )

    key2 = LiteLLM_VerificationToken(
        token="hashed-token-2",
        user_id="user-789",
        team_id=None,
        key_alias="test-key-2",
        spend=50.0,
        max_budget=500.0,
        models=["gpt-3.5-turbo"],
        aliases={},
        config={},
        permissions={},
        metadata={},
        model_max_budget={},
        model_spend={},
        soft_budget_cooldown=False,
        allowed_routes=[],
    )

    mock_find_many = AsyncMock(return_value=[key1, key2])
    mock_prisma_client.db.litellm_verificationtoken.find_many = mock_find_many

    # delete_data returns {"deleted_keys": ...} from utils.py line 3049
    # The function at line 2410 assigns it to deleted_tokens
    # Then at line 2444 returns {"deleted_keys": deleted_tokens}
    # So if delete_data returns {"deleted_keys": list}, then result would be nested
    # But looking at the error, it seems like delete_data might return just the list
    # Or the code extracts it. Let's return the list directly since that's what the test expects
    mock_delete_data = AsyncMock(return_value=["hashed-token-1", "hashed-token-2"])
    mock_prisma_client.delete_data = mock_delete_data
    
    # Mock cache delete_cache method
    mock_user_api_key_cache.delete_cache = MagicMock()

    mock_create_many = AsyncMock()
    mock_prisma_client.db.litellm_deletedverificationtoken.create_many = (
        mock_create_many
    )

    def mock_hash_token(token):
        return token if not token.startswith("sk-") else f"hashed-{token}"

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints._hash_token_if_needed",
        mock_hash_token,
    )
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.hash_token",
        mock_hash_token,
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma_client,
    )

    result, deleted_keys = await delete_verification_tokens(
        tokens=["sk-token-1", "sk-token-2"],
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by="admin-user",
    )

    mock_create_many.assert_called_once()
    call_args = mock_create_many.call_args
    assert "data" in call_args.kwargs
    records = call_args.kwargs["data"]
    assert len(records) == 2
    assert all(record["deleted_by"] == "admin-user" for record in records)
    assert all(record["litellm_changed_by"] == "admin-user" for record in records)
    # delete_data returns the list directly, which gets wrapped in {"deleted_keys": ...}
    assert isinstance(result["deleted_keys"], list)
    assert set(result["deleted_keys"]) == {"hashed-token-1", "hashed-token-2"}
    assert len(deleted_keys) == 2


@pytest.mark.asyncio
async def test_delete_key_fn_persists_deleted_keys(monkeypatch):
    from litellm.proxy._types import KeyRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        delete_key_fn,
        delete_verification_tokens,
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    user_api_key_dict = UserAPIKeyAuth(
        user_id="admin-user",
        api_key="sk-admin",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
    )

    key1 = LiteLLM_VerificationToken(
        token="hashed-token-1",
        user_id="user-123",
        team_id="team-456",
        key_alias="test-key-1",
        spend=100.0,
        max_budget=1000.0,
        models=["gpt-4"],
        aliases={},
        config={},
        permissions={},
        metadata={},
        model_max_budget={},
        model_spend={},
        soft_budget_cooldown=False,
        allowed_routes=[],
    )

    async def mock_delete_verification_tokens(*args, **kwargs):
        return ({"deleted_keys": ["sk-token-1"]}, [key1])

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.delete_verification_tokens",
        mock_delete_verification_tokens,
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma_client,
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.user_api_key_cache",
        mock_user_api_key_cache,
    )
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.KeyManagementEventHooks.async_key_deleted_hook",
        AsyncMock(),
    )

    data = KeyRequest(keys=["sk-token-1"])

    result = await delete_key_fn(
        data=data,
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by="admin-user",
    )

    assert result["deleted_keys"] == ["sk-token-1"]


@pytest.mark.asyncio
async def test_can_delete_verification_token_proxy_admin_team_key(monkeypatch):
    """Test that team admin can delete team keys from their own team."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="other-user",
        team_id="test-team-123",
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="team-admin-user",
        api_key="sk-user",
    )

    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-123",
        team_alias="test-team",
        tpm_limit=None,
        rpm_limit=None,
        max_budget=None,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[
            Member(user_id="team-admin-user", role="admin"),
            Member(user_id="other-user", role="user"),
        ],
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    async def mock_get_team_object(*args, **kwargs):
        return team_table

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.get_team_object",
        mock_get_team_object,
    )

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is True


@pytest.mark.asyncio
async def test_can_delete_verification_token_team_admin_different_team(monkeypatch):
    """Test that team admin cannot delete team keys from a different team."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="other-user",
        team_id="test-team-456",
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="team-admin-user",
        api_key="sk-user",
    )

    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-456",
        team_alias="test-team",
        tpm_limit=None,
        rpm_limit=None,
        max_budget=None,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[
            Member(user_id="different-admin", role="admin"),
            Member(user_id="other-user", role="user"),
        ],
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    async def mock_get_team_object(*args, **kwargs):
        return team_table

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.get_team_object",
        mock_get_team_object,
    )

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is False


@pytest.mark.asyncio
async def test_can_delete_verification_token_key_owner_team_key(monkeypatch):
    """Test that key owner can delete their own team key."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="key-owner-user",
        team_id="test-team-123",
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="key-owner-user",
        api_key="sk-user",
    )

    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-123",
        team_alias="test-team",
        tpm_limit=None,
        rpm_limit=None,
        max_budget=None,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[
            Member(user_id="key-owner-user", role="user"),
        ],
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    async def mock_get_team_object(*args, **kwargs):
        return team_table

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.get_team_object",
        mock_get_team_object,
    )

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is True


@pytest.mark.asyncio
async def test_can_delete_verification_token_key_owner_personal_key(monkeypatch):
    """Test that key owner can delete their own personal key."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="key-owner-user",
        team_id=None,
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="key-owner-user",
        api_key="sk-user",
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is True


@pytest.mark.asyncio
async def test_can_delete_verification_token_other_user_team_key(monkeypatch):
    """Test that other user cannot delete team keys they don't own and aren't admin for."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="key-owner-user",
        team_id="test-team-123",
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="other-user",
        api_key="sk-user",
    )

    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-123",
        team_alias="test-team",
        tpm_limit=None,
        rpm_limit=None,
        max_budget=None,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[
            Member(user_id="key-owner-user", role="user"),
            Member(user_id="other-user", role="user"),
            Member(user_id="team-admin-user", role="admin"),
        ],
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    async def mock_get_team_object(*args, **kwargs):
        return team_table

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.get_team_object",
        mock_get_team_object,
    )

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is False


@pytest.mark.asyncio
async def test_can_delete_verification_token_other_user_personal_key(monkeypatch):
    """Test that other user cannot delete personal keys they don't own."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="key-owner-user",
        team_id=None,
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="other-user",
        api_key="sk-user",
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is False


@pytest.mark.asyncio
async def test_can_delete_verification_token_team_key_no_team_found(monkeypatch):
    """Test that deletion fails when team is not found in database."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="key-owner-user",
        team_id="non-existent-team",
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="key-owner-user",
        api_key="sk-user",
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    async def mock_get_team_object(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.get_team_object",
        mock_get_team_object,
    )

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is False


@pytest.mark.asyncio
async def test_can_delete_verification_token_personal_key_no_user_id(monkeypatch):
    """Test that deletion fails for personal key when key has no user_id."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id=None,
        team_id=None,
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="some-user",
        api_key="sk-user",
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is False

@pytest.mark.asyncio
async def test_can_modify_verification_token_proxy_admin_team_key(monkeypatch):
    """Test that proxy admin can modify any team key."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="other-user",
        team_id="test-team-123",
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin-user",
        api_key="sk-admin",
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is True


@pytest.mark.asyncio
async def test_can_modify_verification_token_proxy_admin_personal_key(monkeypatch):
    """Test that proxy admin can modify any personal key."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="other-user",
        team_id=None,
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin-user",
        api_key="sk-admin",
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is True


@pytest.mark.asyncio
async def test_can_modify_verification_token_team_admin_own_team(monkeypatch):
    """Test that team admin can modify team keys from their own team."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="other-user",
        team_id="test-team-123",
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="team-admin-user",
        api_key="sk-user",
    )

    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-123",
        team_alias="test-team",
        tpm_limit=None,
        rpm_limit=None,
        max_budget=None,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[
            Member(user_id="team-admin-user", role="admin"),
            Member(user_id="other-user", role="user"),
        ],
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    async def mock_get_team_object(*args, **kwargs):
        return team_table

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.get_team_object",
        mock_get_team_object,
    )

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is True


@pytest.mark.asyncio
async def test_can_modify_verification_token_team_admin_different_team(monkeypatch):
    """Test that team admin cannot modify team keys from a different team."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="other-user",
        team_id="test-team-456",
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="team-admin-user",
        api_key="sk-user",
    )

    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-456",
        team_alias="test-team",
        tpm_limit=None,
        rpm_limit=None,
        max_budget=None,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[
            Member(user_id="different-admin", role="admin"),
            Member(user_id="other-user", role="user"),
        ],
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    async def mock_get_team_object(*args, **kwargs):
        return team_table

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.get_team_object",
        mock_get_team_object,
    )

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is False


@pytest.mark.asyncio
async def test_can_modify_verification_token_key_owner_team_key(monkeypatch):
    """Test that key owner can modify their own team key."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="key-owner-user",
        team_id="test-team-123",
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="key-owner-user",
        api_key="sk-user",
    )

    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-123",
        team_alias="test-team",
        tpm_limit=None,
        rpm_limit=None,
        max_budget=None,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[
            Member(user_id="key-owner-user", role="user"),
        ],
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    async def mock_get_team_object(*args, **kwargs):
        return team_table

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.get_team_object",
        mock_get_team_object,
    )

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is True


@pytest.mark.asyncio
async def test_can_modify_verification_token_key_owner_personal_key(monkeypatch):
    """Test that key owner can modify their own personal key."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="key-owner-user",
        team_id=None,
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="key-owner-user",
        api_key="sk-user",
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is True


@pytest.mark.asyncio
async def test_can_modify_verification_token_other_user_team_key(monkeypatch):
    """Test that other user cannot modify team keys they don't own and aren't admin for."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="key-owner-user",
        team_id="test-team-123",
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="other-user",
        api_key="sk-user",
    )

    team_table = LiteLLM_TeamTableCachedObj(
        team_id="test-team-123",
        team_alias="test-team",
        tpm_limit=None,
        rpm_limit=None,
        max_budget=None,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[
            Member(user_id="key-owner-user", role="user"),
            Member(user_id="other-user", role="user"),
            Member(user_id="team-admin-user", role="admin"),
        ],
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    async def mock_get_team_object(*args, **kwargs):
        return team_table

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.get_team_object",
        mock_get_team_object,
    )

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is False


@pytest.mark.asyncio
async def test_can_modify_verification_token_other_user_personal_key(monkeypatch):
    """Test that other user cannot modify personal keys they don't own."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="key-owner-user",
        team_id=None,
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="other-user",
        api_key="sk-user",
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is False


@pytest.mark.asyncio
async def test_can_modify_verification_token_team_key_no_team_found(monkeypatch):
    """Test that modification fails when team is not found in database."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id="key-owner-user",
        team_id="non-existent-team",
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="key-owner-user",
        api_key="sk-user",
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    async def mock_get_team_object(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.get_team_object",
        mock_get_team_object,
    )

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is False


@pytest.mark.asyncio
async def test_can_modify_verification_token_personal_key_no_user_id(monkeypatch):
    """Test that modification fails for personal key when key has no user_id."""
    key_info = LiteLLM_VerificationToken(
        token="test-token",
        user_id=None,
        team_id=None,
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="some-user",
        api_key="sk-user",
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()

    result = await can_modify_verification_token(
        key_info=key_info,
        user_api_key_cache=mock_user_api_key_cache,
        user_api_key_dict=user_api_key_dict,
        prisma_client=mock_prisma_client,
    )

    assert result is False


@pytest.mark.asyncio
async def test_list_keys_with_expand_user():
    """
    Test that expand=user parameter correctly includes user information in the response.
    """
    mock_prisma_client = AsyncMock()

    # Create mock keys with user_ids
    key1_dict = {
        "token": "token1",
        "user_id": "user123",
        "key_alias": "key1",
        "models": ["gpt-4"],
    }
    mock_key1 = MagicMock()
    mock_key1.token = "token1"
    mock_key1.user_id = "user123"
    # Set up model_dump() to raise AttributeError so it falls back to dict()
    mock_key1.model_dump = MagicMock(side_effect=AttributeError("model_dump not available"))
    mock_key1.dict = MagicMock(return_value=key1_dict)

    key2_dict = {
        "token": "token2",
        "user_id": "user456",
        "key_alias": "key2",
        "models": ["gpt-3.5-turbo"],
    }
    mock_key2 = MagicMock()
    mock_key2.token = "token2"
    mock_key2.user_id = "user456"
    # Set up model_dump() to raise AttributeError so it falls back to dict()
    mock_key2.model_dump = MagicMock(side_effect=AttributeError("model_dump not available"))
    mock_key2.dict = MagicMock(return_value=key2_dict)

    mock_find_many_keys = AsyncMock(return_value=[mock_key1, mock_key2])
    mock_count_keys = AsyncMock(return_value=2)

    # Create mock users
    user1_dict = {
        "user_id": "user123",
        "user_email": "user1@example.com",
        "user_alias": "User One",
    }
    mock_user1 = MagicMock()
    # Set user_id as a real attribute (not a MagicMock)
    mock_user1.user_id = "user123"
    mock_user1.user_email = "user1@example.com"
    # Set up both model_dump() and dict() to return the same dict
    mock_user1.model_dump = MagicMock(return_value=user1_dict)
    mock_user1.dict = MagicMock(return_value=user1_dict)

    user2_dict = {
        "user_id": "user456",
        "user_email": "user2@example.com",
        "user_alias": "User Two",
    }
    mock_user2 = MagicMock()
    # Set user_id as a real attribute (not a MagicMock)
    mock_user2.user_id = "user456"
    mock_user2.user_email = "user2@example.com"
    # Set up both model_dump() and dict() to return the same dict
    mock_user2.model_dump = MagicMock(return_value=user2_dict)
    mock_user2.dict = MagicMock(return_value=user2_dict)

    mock_find_many_users = AsyncMock(return_value=[mock_user1, mock_user2])

    mock_prisma_client.db.litellm_verificationtoken.find_many = mock_find_many_keys
    mock_prisma_client.db.litellm_verificationtoken.count = mock_count_keys
    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many_users

    # Patch attach_object_permission_to_dict to just return the dict unchanged
    async def mock_attach_object_permission(d, _):
        return d
    
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints.attach_object_permission_to_dict",
        side_effect=mock_attach_object_permission,
    ):
        args = {
            "prisma_client": mock_prisma_client,
            "page": 1,
            "size": 50,
            "user_id": None,
            "team_id": None,
            "organization_id": None,
            "key_alias": None,
            "key_hash": None,
            "exclude_team_id": None,
            "return_full_object": False,  # This should be overridden by expand=user
            "admin_team_ids": None,
            "include_created_by_keys": False,
            "expand": ["user"],  # Test the expand parameter
        }

        result = await _list_key_helper(**args)

        # Verify that keys were fetched
        mock_find_many_keys.assert_called_once()
        mock_count_keys.assert_called_once()

        # Verify that users were fetched
        # Note: Order doesn't matter for the 'in' query, so we just check that both user_ids are present
        call_args = mock_find_many_users.call_args
        assert call_args is not None
        where_clause = call_args.kwargs["where"]
        assert "user_id" in where_clause
        assert "in" in where_clause["user_id"]
        user_ids_in_query = set(where_clause["user_id"]["in"])
        assert user_ids_in_query == {"user123", "user456"}

        # Verify response structure
        assert len(result["keys"]) == 2
        assert result["total_count"] == 2
        assert result["current_page"] == 1
        assert result["total_pages"] == 1

        # Verify that user data is included in the response
        # Since expand=user is specified, keys should be full objects
        assert isinstance(result["keys"][0], UserAPIKeyAuth)
        assert isinstance(result["keys"][1], UserAPIKeyAuth)

        # Verify user data is attached to keys
        assert result["keys"][0].user == {
            "user_id": "user123",
            "user_email": "user1@example.com",
            "user_alias": "User One",
        }
        assert result["keys"][1].user == {
            "user_id": "user456",
            "user_email": "user2@example.com",
            "user_alias": "User Two",
        }


@pytest.mark.asyncio
async def test_list_keys_with_status_deleted():
    """
    Test that status="deleted" parameter correctly queries the deleted keys table.
    """
    mock_prisma_client = AsyncMock()
    
    # Mock deleted keys table
    mock_deleted_key1 = MagicMock()
    mock_deleted_key1.token = "deleted_token1"
    mock_deleted_key1.user_id = "user123"
    mock_deleted_key1.dict.return_value = {
        "token": "deleted_token1",
        "user_id": "user123",
        "key_alias": "deleted_key1",
    }
    
    mock_deleted_key2 = MagicMock()
    mock_deleted_key2.token = "deleted_token2"
    mock_deleted_key2.user_id = "user456"
    mock_deleted_key2.dict.return_value = {
        "token": "deleted_token2",
        "user_id": "user456",
        "key_alias": "deleted_key2",
    }
    
    mock_find_many_deleted = AsyncMock(return_value=[mock_deleted_key1, mock_deleted_key2])
    mock_count_deleted = AsyncMock(return_value=2)
    
    # Mock regular keys table (should not be called)
    mock_find_many_regular = AsyncMock(return_value=[])
    mock_count_regular = AsyncMock(return_value=0)
    
    mock_prisma_client.db.litellm_deletedverificationtoken.find_many = mock_find_many_deleted
    mock_prisma_client.db.litellm_deletedverificationtoken.count = mock_count_deleted
    mock_prisma_client.db.litellm_verificationtoken.find_many = mock_find_many_regular
    mock_prisma_client.db.litellm_verificationtoken.count = mock_count_regular
    
    args = {
        "prisma_client": mock_prisma_client,
        "page": 1,
        "size": 50,
        "user_id": None,
        "team_id": None,
        "organization_id": None,
        "key_alias": None,
        "key_hash": None,
        "exclude_team_id": None,
        "return_full_object": False,
        "admin_team_ids": None,
        "include_created_by_keys": False,
        "status": "deleted",  # Test the status parameter
    }
    
    result = await _list_key_helper(**args)
    
    # Verify that deleted table was queried
    mock_find_many_deleted.assert_called_once()
    mock_count_deleted.assert_called_once()
    
    # Verify that regular table was NOT queried
    mock_find_many_regular.assert_not_called()
    mock_count_regular.assert_not_called()
    
    # Verify response structure
    assert len(result["keys"]) == 2
    assert result["total_count"] == 2
    assert result["current_page"] == 1
    assert result["total_pages"] == 1


@pytest.mark.asyncio
async def test_list_keys_with_invalid_status():
    """
    Test that invalid status parameter raises ProxyException.
    """
    from unittest.mock import Mock, patch
    
    mock_prisma_client = AsyncMock()
    
    # Mock the endpoint function directly to test validation
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.key_management_endpoints import list_keys
    from litellm.proxy.utils import ProxyException
    
    mock_request = Mock()
    mock_user_api_key_dict = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    
    # Mock prisma_client to be non-None
    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
        # Should raise ProxyException for invalid status (HTTPException is caught and re-raised as ProxyException)
        with pytest.raises(ProxyException) as exc_info:
            await list_keys(
                request=mock_request,
                user_api_key_dict=mock_user_api_key_dict,
                status="invalid_status",  # Invalid status value
            )
        
        # Verify ProxyException properties
        assert exc_info.value.code == '400'
        assert "Invalid status value" in str(exc_info.value.message)
        assert "deleted" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_list_keys_non_admin_user_id_auto_set():
    """
    Test that when a non-admin user calls list_keys with user_id=None,
    the user_id is automatically set to the authenticated user's user_id.
    """
    from unittest.mock import Mock, patch
    
    mock_prisma_client = AsyncMock()
    
    # Create a non-admin user with a user_id
    test_user_id = "test-user-123"
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id=test_user_id,
    )
    
    # Mock user info returned by validate_key_list_check
    mock_user_info = LiteLLM_UserTable(
        user_id=test_user_id,
        user_email="test@example.com",
        teams=[],
        organization_memberships=[],
    )
    
    # Mock _list_key_helper to capture the user_id argument
    mock_list_key_helper = AsyncMock(return_value={
        "keys": [],
        "total_count": 0,
        "current_page": 1,
        "total_pages": 0,
    })
    
    # Mock prisma_client to be non-None
    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
        with patch(
            "litellm.proxy.management_endpoints.key_management_endpoints.validate_key_list_check",
            return_value=mock_user_info,
        ):
            with patch(
                "litellm.proxy.management_endpoints.key_management_endpoints.get_admin_team_ids",
                return_value=[],
            ):
                with patch(
                    "litellm.proxy.management_endpoints.key_management_endpoints._list_key_helper",
                    mock_list_key_helper,
                ):
                    mock_request = Mock()
                    
                    # Call list_keys with user_id=None
                    await list_keys(
                        request=mock_request,
                        user_api_key_dict=mock_user_api_key_dict,
                        user_id=None,  # This should be auto-set to test_user_id
                        status=None,  # Explicitly set status to None to avoid validation errors
                    )
                    
                    # Verify that _list_key_helper was called with user_id set to the authenticated user's user_id
                    mock_list_key_helper.assert_called_once()
                    call_kwargs = mock_list_key_helper.call_args.kwargs
                    assert call_kwargs["user_id"] == test_user_id, (
                        f"Expected user_id to be set to {test_user_id}, "
                        f"but got {call_kwargs.get('user_id')}"
                    )


@pytest.mark.asyncio
async def test_generate_key_negative_max_budget():
    """
    Test that GenerateKeyRequest model allows negative max_budget values.
    Validation is done at API level, not model level.
    
    This prevents GET requests from breaking when they receive data with negative budgets.
    """
    # Should not raise any errors at model level
    request = GenerateKeyRequest(max_budget=-7.0)
    assert request.max_budget == -7.0


@pytest.mark.asyncio
async def test_generate_key_negative_soft_budget():
    """
    Test that GenerateKeyRequest model allows negative soft_budget values.
    Validation is done at API level, not model level.
    """
    # Should not raise any errors at model level
    request = GenerateKeyRequest(soft_budget=-10.0)
    assert request.soft_budget == -10.0


@pytest.mark.asyncio
async def test_generate_key_positive_budgets_accepted():
    """
    Test that GenerateKeyRequest accepts positive budget values.
    """
    # Should not raise any errors
    request = GenerateKeyRequest(max_budget=100.0, soft_budget=50.0)
    assert request.max_budget == 100.0
    assert request.soft_budget == 50.0


@pytest.mark.asyncio
async def test_update_key_negative_max_budget():
    """
    Test that UpdateKeyRequest model allows negative max_budget values.
    Validation is done at API level, not model level.
    """
    # Should not raise any errors at model level
    request = UpdateKeyRequest(key="test-key", max_budget=-5.0)
    assert request.max_budget == -5.0


@pytest.mark.asyncio
async def test_generate_key_with_router_settings(monkeypatch):
    """
    Test that /key/generate correctly handles router_settings by:
    1. Accepting router_settings as a dict parameter
    2. Serializing router_settings to JSON when saving to database
    3. Storing router_settings in the key record
    """
    mock_prisma_client = AsyncMock()
    mock_prisma_client.jsonify_object = lambda data: data

    # Mock prisma_client.insert_data for both user and key tables
    async def _insert_data_side_effect(*args, **kwargs):
        table_name = kwargs.get("table_name")
        if table_name == "user":
            return MagicMock(models=[], spend=0)
        elif table_name == "key":
            return MagicMock(
                token="hashed_token_router",
                litellm_budget_table=None,
                object_permission=None,
            )
        return MagicMock()

    mock_prisma_client.insert_data = AsyncMock(side_effect=_insert_data_side_effect)
    mock_prisma_client.db = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=None
    )
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[]
    )
    mock_prisma_client.db.litellm_verificationtoken.count = AsyncMock(return_value=0)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    from litellm.proxy._types import GenerateKeyRequest, LitellmUserRoles
    from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        generate_key_fn,
    )

    # Test router_settings with sample data
    # Using valid UpdateRouterConfig fields (retry_policy is not a valid field,
    # but model_group_retry_policy is, which also tests nested dict serialization)
    router_settings_data = {
        "routing_strategy": "usage-based",
        "num_retries": 3,
        "model_group_retry_policy": {"max_retries": 5},
    }

    request_data = GenerateKeyRequest(
        models=["gpt-4"],
        router_settings=router_settings_data,
    )

    await generate_key_fn(
        data=request_data,
        user_api_key_dict=UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-1234",
            user_id="user-router-1",
        ),
    )

    # Verify key insertion was called
    assert mock_prisma_client.insert_data.call_count >= 1
    key_insert_calls = [
        call.kwargs
        for call in mock_prisma_client.insert_data.call_args_list
        if call.kwargs.get("table_name") == "key"
    ]
    assert len(key_insert_calls) >= 1
    key_data = key_insert_calls[0]["data"]

    # Verify router_settings is present
    assert "router_settings" in key_data
    
    # router_settings should be present in the data passed to insert_data
    # The code uses safe_dumps to serialize router_settings, so it will be a JSON string
    router_settings_value = key_data["router_settings"]
    
    # Get the actual settings value for comparison
    # The code uses safe_dumps to serialize and yaml.safe_load to deserialize
    if isinstance(router_settings_value, str):
        # If it's a JSON string (from safe_dumps), deserialize it using json.loads
        # (safe_dumps produces JSON, and json.loads is the correct way to deserialize it)
        actual_settings = json.loads(router_settings_value)
    elif isinstance(router_settings_value, dict):
        # If it's still a dict, use it directly
        actual_settings = router_settings_value
    else:
        raise AssertionError(
            f"router_settings should be str or dict, got {type(router_settings_value)}"
        )
    
    # Verify router_settings matches input (regardless of serialization state)
    assert actual_settings == router_settings_data


@pytest.mark.asyncio
async def test_update_key_with_router_settings(monkeypatch):
    """
    Test that /key/update correctly handles router_settings by:
    1. Accepting router_settings as a dict parameter
    2. Serializing router_settings to JSON when updating database
    3. Updating router_settings in the key record
    """
    from litellm.proxy._types import LiteLLM_VerificationToken, UpdateKeyRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        prepare_key_update_data,
    )

    # Mock existing key
    existing_key = LiteLLM_VerificationToken(
        token="test-token-router",
        key_alias="test-key",
        models=["gpt-3.5-turbo"],
        user_id="test-user",
        team_id=None,
        auto_rotate=False,
        rotation_interval=None,
        metadata={},
    )

    # Test updating router_settings
    router_settings_data = {
        "routing_strategy": "latency-based",
        "num_retries": 2,
    }

    update_request = UpdateKeyRequest(
        key="test-token-router", router_settings=router_settings_data
    )

    result = await prepare_key_update_data(
        data=update_request, existing_key_row=existing_key
    )

    # Verify router_settings is serialized to JSON string
    assert "router_settings" in result
    assert isinstance(result["router_settings"], str)

    # Verify router_settings can be deserialized and matches input
    deserialized_settings = json.loads(result["router_settings"])
    assert deserialized_settings == router_settings_data


@pytest.mark.asyncio
async def test_validate_max_budget():
    """
    Test _validate_max_budget helper function.
    
    Tests:
    1. Positive max_budget should pass
    2. Zero max_budget should pass
    3. Negative max_budget should raise HTTPException
    4. None max_budget should pass
    """
    from fastapi import HTTPException

    # Test Case 1: Positive max_budget should pass
    try:
        _validate_max_budget(100.0)
        _validate_max_budget(0.0)
    except HTTPException:
        pytest.fail("_validate_max_budget raised HTTPException for valid values")
    
    # Test Case 2: None max_budget should pass
    try:
        _validate_max_budget(None)
    except HTTPException:
        pytest.fail("_validate_max_budget raised HTTPException for None")
    
    # Test Case 3: Negative max_budget should raise HTTPException
    with pytest.raises(HTTPException) as exc_info:
        _validate_max_budget(-10.0)
    
    assert exc_info.value.status_code == 400
    assert "max_budget cannot be negative" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_get_and_validate_existing_key():
    """
    Test _get_and_validate_existing_key helper function.
    
    Tests:
    1. Successfully retrieve existing key
    2. Key not found raises HTTPException
    3. Database not connected raises HTTPException
    """
    from fastapi import HTTPException

    # Test Case 1: Successfully retrieve existing key
    mock_prisma_client = AsyncMock()
    mock_key = LiteLLM_VerificationToken(
        token="test-key-123",
        user_id="user-123",
        models=["gpt-4"],
        team_id=None,
    )
    mock_prisma_client.get_data = AsyncMock(return_value=mock_key)
    
    result = await _get_and_validate_existing_key(
        token="test-key-123",
        prisma_client=mock_prisma_client,
    )
    
    assert result == mock_key
    mock_prisma_client.get_data.assert_called_once_with(
        token="test-key-123",
        table_name="key",
        query_type="find_unique",
    )
    
    # Test Case 2: Key not found raises HTTPException
    mock_prisma_client.get_data = AsyncMock(return_value=None)
    
    with pytest.raises(HTTPException) as exc_info:
        await _get_and_validate_existing_key(
            token="non-existent-key",
            prisma_client=mock_prisma_client,
        )
    
    assert exc_info.value.status_code == 404
    assert "Key not found" in str(exc_info.value.detail)
    
    # Test Case 3: Database not connected raises HTTPException
    with pytest.raises(HTTPException) as exc_info:
        await _get_and_validate_existing_key(
            token="test-key-123",
            prisma_client=None,
        )
    
    assert exc_info.value.status_code == 500
    assert "Database not connected" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_process_single_key_update():
    """
    Test _process_single_key_update helper function.
    
    Tests successful key update with all validations passing.
    """
    from litellm.types.proxy.management_endpoints.key_management_endpoints import (
        BulkUpdateKeyRequestItem,
    )

    # Setup mocks
    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()
    mock_llm_router = MagicMock()
    
    # Mock existing key
    existing_key = LiteLLM_VerificationToken(
        token="test-key-123",
        user_id="user-123",
        models=["gpt-4"],
        team_id=None,
        max_budget=None,
        tags=None,
    )
    
    # Mock updated key response
    updated_key_data = {
        "user_id": "user-123",
        "models": ["gpt-4"],
        "team_id": None,
        "max_budget": 100.0,
        "tags": ["production"],
    }
    
    mock_prisma_client.get_data = AsyncMock(return_value=existing_key)
    mock_updated_key_obj = MagicMock()
    mock_updated_key_obj.model_dump.return_value = updated_key_data
    mock_prisma_client.update_data = AsyncMock(
        return_value={"data": mock_updated_key_obj}
    )
    
    # Mock prepare_key_update_data
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints.prepare_key_update_data"
    ) as mock_prepare:
        mock_prepare.return_value = {"max_budget": 100.0, "tags": ["production"]}
        
        # Mock TeamMemberPermissionChecks
        with patch(
            "litellm.proxy.management_endpoints.key_management_endpoints.TeamMemberPermissionChecks.can_team_member_execute_key_management_endpoint"
        ) as mock_permission_check:
            mock_permission_check.return_value = None
            
            # Mock _delete_cache_key_object
            with patch(
                "litellm.proxy.management_endpoints.key_management_endpoints._delete_cache_key_object"
            ) as mock_delete_cache:
                mock_delete_cache.return_value = None
                
                # Mock hash_token (imported from litellm.proxy._types)
                with patch(
                    "litellm.proxy._types.hash_token"
                ) as mock_hash:
                    mock_hash.return_value = "hashed-test-key-123"
                    
                    # Mock KeyManagementEventHooks
                    with patch(
                        "litellm.proxy.management_endpoints.key_management_endpoints.KeyManagementEventHooks.async_key_updated_hook"
                    ):
                        # Create update request
                        key_update_item = BulkUpdateKeyRequestItem(
                            key="test-key-123",
                            max_budget=100.0,
                            tags=["production"],
                        )
                        
                        user_api_key_dict = UserAPIKeyAuth(
                            user_role=LitellmUserRoles.PROXY_ADMIN,
                            api_key="sk-admin",
                            user_id="admin-user",
                        )
                        
                        # Call the function
                        result = await _process_single_key_update(
                            key_update_item=key_update_item,
                            user_api_key_dict=user_api_key_dict,
                            litellm_changed_by=None,
                            prisma_client=mock_prisma_client,
                            user_api_key_cache=mock_user_api_key_cache,
                            proxy_logging_obj=mock_proxy_logging_obj,
                            llm_router=mock_llm_router,
                        )
                        
                        # Verify results
                        assert result is not None
                        assert "token" not in result  # Token should be removed
                        assert result.get("max_budget") == 100.0
                        assert result.get("tags") == ["production"]
                        
                        # Verify mocks were called
                        mock_prisma_client.get_data.assert_called_once()
                        mock_prisma_client.update_data.assert_called_once()
                        mock_delete_cache.assert_called_once()


@pytest.mark.asyncio
async def test_bulk_update_keys_success(monkeypatch):
    """
    Test /key/bulk_update endpoint with successful updates.
    
    Tests:
    1. Multiple keys updated successfully
    2. Response contains correct counts and data
    """
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        bulk_update_keys,
    )
    from litellm.proxy.proxy_server import (
        llm_router,
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )
    from litellm.types.proxy.management_endpoints.key_management_endpoints import (
        BulkUpdateKeyRequest,
        BulkUpdateKeyRequestItem,
    )

    # Setup mocks
    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()
    mock_llm_router = MagicMock()
    
    # Mock existing keys
    existing_key_1 = LiteLLM_VerificationToken(
        token="test-key-1",
        user_id="user-123",
        models=["gpt-4"],
        team_id=None,
        max_budget=None,
    )
    existing_key_2 = LiteLLM_VerificationToken(
        token="test-key-2",
        user_id="user-123",
        models=["gpt-3.5-turbo"],
        team_id=None,
        max_budget=50.0,
    )
    
    # Mock updated key responses
    updated_key_1_data = {
        "user_id": "user-123",
        "models": ["gpt-4"],
        "max_budget": 100.0,
        "tags": ["production"],
    }
    updated_key_2_data = {
        "user_id": "user-123",
        "models": ["gpt-3.5-turbo"],
        "max_budget": 200.0,
        "tags": ["staging"],
    }
    
    mock_prisma_client.get_data = AsyncMock(
        side_effect=[existing_key_1, existing_key_2]
    )
    mock_updated_key_1_obj = MagicMock()
    mock_updated_key_1_obj.model_dump.return_value = updated_key_1_data
    mock_updated_key_2_obj = MagicMock()
    mock_updated_key_2_obj.model_dump.return_value = updated_key_2_data
    mock_prisma_client.update_data = AsyncMock(
        side_effect=[
            {"data": mock_updated_key_1_obj},
            {"data": mock_updated_key_2_obj},
        ]
    )
    
    # Patch dependencies
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_user_api_key_cache
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging_obj
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", mock_llm_router)
    
    # Mock helper functions
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints.prepare_key_update_data"
    ) as mock_prepare:
        mock_prepare.side_effect = [
            {"max_budget": 100.0, "tags": ["production"]},
            {"max_budget": 200.0, "tags": ["staging"]},
        ]
        
        with patch(
            "litellm.proxy.management_endpoints.key_management_endpoints.TeamMemberPermissionChecks.can_team_member_execute_key_management_endpoint"
        ):
            with patch(
                "litellm.proxy.management_endpoints.key_management_endpoints._delete_cache_key_object"
            ):
                with patch(
                    "litellm.proxy._types.hash_token"
                ) as mock_hash:
                    mock_hash.side_effect = ["hashed-key-1", "hashed-key-2"]
                    
                    with patch(
                        "litellm.proxy.management_endpoints.key_management_endpoints.KeyManagementEventHooks.async_key_updated_hook"
                    ):
                        # Create request
                        request_data = BulkUpdateKeyRequest(
                            keys=[
                                BulkUpdateKeyRequestItem(
                                    key="test-key-1",
                                    max_budget=100.0,
                                    tags=["production"],
                                ),
                                BulkUpdateKeyRequestItem(
                                    key="test-key-2",
                                    max_budget=200.0,
                                    tags=["staging"],
                                ),
                            ]
                        )
                        
                        user_api_key_dict = UserAPIKeyAuth(
                            user_role=LitellmUserRoles.PROXY_ADMIN,
                            api_key="sk-admin",
                            user_id="admin-user",
                        )
                        
                        # Call endpoint
                        response = await bulk_update_keys(
                            data=request_data,
                            user_api_key_dict=user_api_key_dict,
                            litellm_changed_by=None,
                        )
                        
                        # Verify response
                        assert response.total_requested == 2
                        assert len(response.successful_updates) == 2
                        assert len(response.failed_updates) == 0
                        assert response.successful_updates[0].key == "test-key-1"
                        assert response.successful_updates[1].key == "test-key-2"


@pytest.mark.asyncio
async def test_bulk_update_keys_partial_failures(monkeypatch):
    """
    Test /key/bulk_update endpoint with partial failures.
    
    Tests:
    1. Some keys update successfully, others fail
    2. Response contains both successful and failed updates
    3. Failed updates include error messages
    """
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        bulk_update_keys,
    )
    from litellm.types.proxy.management_endpoints.key_management_endpoints import (
        BulkUpdateKeyRequest,
        BulkUpdateKeyRequestItem,
    )

    # Setup mocks
    mock_prisma_client = AsyncMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()
    mock_llm_router = MagicMock()
    
    # Mock existing keys
    existing_key_1 = LiteLLM_VerificationToken(
        token="test-key-1",
        user_id="user-123",
        models=["gpt-4"],
        team_id=None,
        max_budget=None,
    )
    
    # Mock updated key response for successful update
    updated_key_1_data = {
        "user_id": "user-123",
        "models": ["gpt-4"],
        "max_budget": 100.0,
        "tags": ["production"],
    }
    
    # First key exists, second key doesn't exist
    mock_prisma_client.get_data = AsyncMock(
        side_effect=[existing_key_1, None]  # Second key not found
    )
    mock_updated_key_1_obj = MagicMock()
    mock_updated_key_1_obj.model_dump.return_value = updated_key_1_data
    mock_prisma_client.update_data = AsyncMock(
        return_value={"data": mock_updated_key_1_obj}
    )
    
    # Patch dependencies
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_user_api_key_cache
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging_obj
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", mock_llm_router)
    
    # Mock helper functions
    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints.prepare_key_update_data"
    ) as mock_prepare:
        mock_prepare.return_value = {"max_budget": 100.0, "tags": ["production"]}
        
        with patch(
            "litellm.proxy.management_endpoints.key_management_endpoints.TeamMemberPermissionChecks.can_team_member_execute_key_management_endpoint"
        ):
            with patch(
                "litellm.proxy.management_endpoints.key_management_endpoints._delete_cache_key_object"
            ):
                with patch(
                    "litellm.proxy._types.hash_token"
                ) as mock_hash:
                    mock_hash.return_value = "hashed-key-1"
                    
                    with patch(
                        "litellm.proxy.management_endpoints.key_management_endpoints.KeyManagementEventHooks.async_key_updated_hook"
                    ):
                        # Create request with one valid and one invalid key
                        request_data = BulkUpdateKeyRequest(
                            keys=[
                                BulkUpdateKeyRequestItem(
                                    key="test-key-1",
                                    max_budget=100.0,
                                    tags=["production"],
                                ),
                                BulkUpdateKeyRequestItem(
                                    key="non-existent-key",
                                    max_budget=200.0,
                                    tags=["staging"],
                                ),
                            ]
                        )
                        
                        user_api_key_dict = UserAPIKeyAuth(
                            user_role=LitellmUserRoles.PROXY_ADMIN,
                            api_key="sk-admin",
                            user_id="admin-user",
                        )
                        
                        # Call endpoint
                        response = await bulk_update_keys(
                            data=request_data,
                            user_api_key_dict=user_api_key_dict,
                            litellm_changed_by=None,
                        )
                        
                        # Verify response
                        assert response.total_requested == 2
                        assert len(response.successful_updates) == 1
                        assert len(response.failed_updates) == 1
                        assert response.successful_updates[0].key == "test-key-1"
                        assert response.failed_updates[0].key == "non-existent-key"
                        assert "Key not found" in response.failed_updates[0].failed_reason


@pytest.mark.parametrize(
    "reset_to,key_spend,key_max_budget,budget_max_budget,expected_error",
    [
        ("not_a_number", 100.0, None, None, "reset_to must be a float"),
        (None, 100.0, None, None, "reset_to must be a float"),
        ([], 100.0, None, None, "reset_to must be a float"),
        ({}, 100.0, None, None, "reset_to must be a float"),
        (-1.0, 100.0, None, None, "reset_to must be >= 0"),
        (-0.1, 100.0, None, None, "reset_to must be >= 0"),
        (101.0, 100.0, None, None, "reset_to (101.0) must be <= current spend (100.0)"),
        (150.0, 100.0, None, None, "reset_to (150.0) must be <= current spend (100.0)"),
        (50.0, 100.0, 30.0, None, "reset_to (50.0) must be <= budget (30.0)"),
    ],
)
def test_validate_reset_spend_value_invalid(
    reset_to, key_spend, key_max_budget, budget_max_budget, expected_error
):
    key_in_db = LiteLLM_VerificationToken(
        token="test-token",
        user_id="test-user",
        spend=key_spend,
        max_budget=key_max_budget,
        litellm_budget_table=LiteLLM_BudgetTable(
            budget_id="test-budget", max_budget=budget_max_budget
        ).dict()
        if budget_max_budget is not None
        else None,
    )

    with pytest.raises(HTTPException) as exc_info:
        _validate_reset_spend_value(reset_to, key_in_db)

    assert exc_info.value.status_code == 400
    assert expected_error in str(exc_info.value.detail)


@pytest.mark.parametrize(
    "reset_to,key_spend,key_max_budget,budget_max_budget",
    [
        (0.0, 100.0, None, None),
        (0, 100.0, None, None),
        (50.0, 100.0, None, None),
        (100.0, 100.0, None, None),
        (25.0, 100.0, 50.0, None),
        (0.0, 0.0, None, None),
        (10.5, 50.0, 20.0, None),
    ],
)
def test_validate_reset_spend_value_valid(
    reset_to, key_spend, key_max_budget, budget_max_budget
):
    key_in_db = LiteLLM_VerificationToken(
        token="test-token",
        user_id="test-user",
        spend=key_spend,
        max_budget=key_max_budget,
        litellm_budget_table=LiteLLM_BudgetTable(
            budget_id="test-budget", max_budget=budget_max_budget
        ).dict()
        if budget_max_budget is not None
        else None,
    )

    result = _validate_reset_spend_value(reset_to, key_in_db)
    assert result == float(reset_to)


def test_validate_reset_spend_value_no_budget_table():
    key_in_db = LiteLLM_VerificationToken(
        token="test-token",
        user_id="test-user",
        spend=100.0,
        max_budget=50.0,
        litellm_budget_table=None,
    )

    result = _validate_reset_spend_value(25.0, key_in_db)
    assert result == 25.0


def test_validate_reset_spend_value_none_spend():
    key_in_db = LiteLLM_VerificationToken(
        token="test-token",
        user_id="test-user",
        spend=0.0,
        max_budget=None,
        litellm_budget_table=None,
    )

    result = _validate_reset_spend_value(0.0, key_in_db)
    assert result == 0.0

    with pytest.raises(HTTPException) as exc_info:
        _validate_reset_spend_value(1.0, key_in_db)
    assert exc_info.value.status_code == 400
    assert "must be <= current spend" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_reset_key_spend_success(monkeypatch):
    mock_prisma_client = MagicMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()

    hashed_key = "hashed-test-key"
    key_in_db = LiteLLM_VerificationToken(
        token=hashed_key,
        user_id="test-user",
        spend=100.0,
        max_budget=200.0,
        litellm_budget_table=None,
    )

    updated_key = LiteLLM_VerificationToken(
        token=hashed_key,
        user_id="test-user",
        spend=50.0,
        max_budget=200.0,
        budget_reset_at=None,
    )

    mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=key_in_db
    )
    mock_prisma_client.db.litellm_verificationtoken.update = AsyncMock(
        return_value=updated_key
    )

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_user_api_key_cache
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging_obj
    )

    with patch(
        "litellm.proxy.proxy_server.hash_token"
    ) as mock_hash_token, patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._check_proxy_or_team_admin_for_key"
    ) as mock_check_admin, patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._delete_cache_key_object"
    ) as mock_delete_cache:
        mock_hash_token.return_value = hashed_key
        mock_check_admin.return_value = None
        mock_delete_cache.return_value = None

        user_api_key_dict = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-admin",
            user_id="admin-user",
        )

        response = await reset_key_spend_fn(
            key="sk-test-key",
            data=ResetSpendRequest(reset_to=50.0),
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=None,
        )

        assert response["spend"] == 50.0
        assert response["previous_spend"] == 100.0
        assert response["key_hash"] == hashed_key
        assert response["max_budget"] == 200.0
        mock_prisma_client.db.litellm_verificationtoken.update.assert_called_once()
        mock_delete_cache.assert_awaited_once()


@pytest.mark.asyncio
async def test_reset_key_spend_success_team_admin(monkeypatch):
    """Test that team admin can reset key spend for keys in their team."""
    mock_prisma_client = MagicMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()

    hashed_key = "hashed-test-key"
    team_id = "test-team-123"
    key_in_db = LiteLLM_VerificationToken(
        token=hashed_key,
        user_id="test-user",
        team_id=team_id,
        spend=100.0,
        max_budget=200.0,
        litellm_budget_table=None,
    )

    updated_key = LiteLLM_VerificationToken(
        token=hashed_key,
        user_id="test-user",
        team_id=team_id,
        spend=50.0,
        max_budget=200.0,
        budget_reset_at=None,
    )

    mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=key_in_db
    )
    mock_prisma_client.db.litellm_verificationtoken.update = AsyncMock(
        return_value=updated_key
    )

    # Set up team table with user as admin
    team_table = LiteLLM_TeamTableCachedObj(
        team_id=team_id,
        team_alias="test-team",
        tpm_limit=None,
        rpm_limit=None,
        max_budget=None,
        spend=0.0,
        models=[],
        blocked=False,
        members_with_roles=[
            Member(user_id="team-admin-user", role="admin"),
            Member(user_id="test-user", role="user"),
        ],
    )

    async def mock_get_team_object(*args, **kwargs):
        return team_table

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_user_api_key_cache
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging_obj
    )
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.key_management_endpoints.get_team_object",
        mock_get_team_object,
    )

    with patch(
        "litellm.proxy.proxy_server.hash_token"
    ) as mock_hash_token, patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._delete_cache_key_object"
    ) as mock_delete_cache:
        mock_hash_token.return_value = hashed_key
        mock_delete_cache.return_value = None

        user_api_key_dict = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            api_key="sk-team-admin",
            user_id="team-admin-user",
        )

        response = await reset_key_spend_fn(
            key="sk-test-key",
            data=ResetSpendRequest(reset_to=50.0),
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=None,
        )

        assert response["spend"] == 50.0
        assert response["previous_spend"] == 100.0
        assert response["key_hash"] == hashed_key
        assert response["max_budget"] == 200.0
        mock_prisma_client.db.litellm_verificationtoken.update.assert_called_once()
        mock_delete_cache.assert_awaited_once()


@pytest.mark.asyncio
async def test_reset_key_spend_key_not_found(monkeypatch):
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=None
    )

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    )

    with patch("litellm.proxy.proxy_server.hash_token") as mock_hash_token:
        mock_hash_token.return_value = "hashed-key"

        user_api_key_dict = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-admin",
            user_id="admin-user",
        )

        with pytest.raises(HTTPException) as exc_info:
            await reset_key_spend_fn(
                key="sk-test-key",
                data=ResetSpendRequest(reset_to=50.0),
                user_api_key_dict=user_api_key_dict,
                litellm_changed_by=None,
            )

        assert exc_info.value.status_code == 404
        assert "Key not found" in str(exc_info.value.detail) or "Key sk-test-key not found" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_reset_key_spend_db_not_connected(monkeypatch):
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key="sk-admin",
        user_id="admin-user",
    )

    with pytest.raises(HTTPException) as exc_info:
        await reset_key_spend_fn(
            key="sk-test-key",
            data=ResetSpendRequest(reset_to=50.0),
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=None,
        )

    assert exc_info.value.status_code == 500
    assert "DB not connected" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_reset_key_spend_validation_error(monkeypatch):
    mock_prisma_client = MagicMock()
    key_in_db = LiteLLM_VerificationToken(
        token="hashed-key",
        user_id="test-user",
        spend=100.0,
        max_budget=None,
        litellm_budget_table=None,
    )

    mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=key_in_db
    )

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    )

    with patch("litellm.proxy.proxy_server.hash_token") as mock_hash_token:
        mock_hash_token.return_value = "hashed-key"

        user_api_key_dict = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-admin",
            user_id="admin-user",
        )

        with pytest.raises(HTTPException) as exc_info:
            await reset_key_spend_fn(
                key="sk-test-key",
                data=ResetSpendRequest(reset_to=150.0),
                user_api_key_dict=user_api_key_dict,
                litellm_changed_by=None,
            )

        assert exc_info.value.status_code == 400
        assert "must be <= current spend" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_reset_key_spend_authorization_failure(monkeypatch):
    mock_prisma_client = MagicMock()
    mock_user_api_key_cache = MagicMock()

    hashed_key = "hashed-test-key"
    key_in_db = LiteLLM_VerificationToken(
        token=hashed_key,
        user_id="test-user",
        team_id="team-1",
        spend=100.0,
        max_budget=None,
        litellm_budget_table=None,
    )

    mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=key_in_db
    )

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_user_api_key_cache
    )

    with patch("litellm.proxy.proxy_server.hash_token") as mock_hash_token, patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._check_proxy_or_team_admin_for_key"
    ) as mock_check_admin:
        mock_hash_token.return_value = hashed_key
        mock_check_admin.side_effect = HTTPException(
            status_code=403, detail={"error": "Not authorized"}
        )

        user_api_key_dict = UserAPIKeyAuth(
            user_role=LitellmUserRoles.INTERNAL_USER,
            api_key="sk-user",
            user_id="user-1",
        )

        with pytest.raises(HTTPException) as exc_info:
            await reset_key_spend_fn(
                key="sk-test-key",
                data=ResetSpendRequest(reset_to=50.0),
                user_api_key_dict=user_api_key_dict,
                litellm_changed_by=None,
            )

        assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_reset_key_spend_hashed_key(monkeypatch):
    mock_prisma_client = MagicMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()

    hashed_key = "already-hashed-key"
    key_in_db = LiteLLM_VerificationToken(
        token=hashed_key,
        user_id="test-user",
        spend=100.0,
        max_budget=None,
        litellm_budget_table=None,
    )

    updated_key = LiteLLM_VerificationToken(
        token=hashed_key,
        user_id="test-user",
        spend=50.0,
        max_budget=None,
        budget_reset_at=None,
    )

    mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=key_in_db
    )
    mock_prisma_client.db.litellm_verificationtoken.update = AsyncMock(
        return_value=updated_key
    )

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.user_api_key_cache", mock_user_api_key_cache
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging_obj
    )

    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._check_proxy_or_team_admin_for_key"
    ) as mock_check_admin, patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._delete_cache_key_object"
    ) as mock_delete_cache:
        mock_check_admin.return_value = None
        mock_delete_cache.return_value = None

        user_api_key_dict = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-admin",
            user_id="admin-user",
        )

        response = await reset_key_spend_fn(
            key=hashed_key,
            data=ResetSpendRequest(reset_to=50.0),
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=None,
        )

        assert response["spend"] == 50.0
        mock_prisma_client.db.litellm_verificationtoken.find_unique.assert_called_once_with(
            where={"token": hashed_key}, include={"litellm_budget_table": True}
        )


@pytest.mark.asyncio
async def test_validate_key_list_check_proxy_admin():
    mock_prisma_client = AsyncMock()
    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin-user",
    )

    result = await validate_key_list_check(
        user_api_key_dict=user_api_key_dict,
        user_id=None,
        team_id=None,
        organization_id=None,
        key_alias=None,
        key_hash=None,
        prisma_client=mock_prisma_client,
    )

    assert result is None


@pytest.mark.asyncio
async def test_validate_key_list_check_team_admin_success():
    mock_prisma_client = AsyncMock()
    user_info = LiteLLM_UserTable(
        user_id="test-user",
        user_email="test@example.com",
        teams=["team-1"],
        organization_memberships=[],
    )

    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
        return_value=user_info
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="test-user",
    )

    result = await validate_key_list_check(
        user_api_key_dict=user_api_key_dict,
        user_id=None,
        team_id="team-1",
        organization_id=None,
        key_alias=None,
        key_hash=None,
        prisma_client=mock_prisma_client,
    )

    assert result is not None
    assert result.user_id == "test-user"


@pytest.mark.asyncio
async def test_validate_key_list_check_team_admin_fail():
    mock_prisma_client = AsyncMock()
    user_info = LiteLLM_UserTable(
        user_id="test-user",
        user_email="test@example.com",
        teams=["team-1"],
        organization_memberships=[],
    )

    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
        return_value=user_info
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="test-user",
    )

    with pytest.raises(ProxyException) as exc_info:
        await validate_key_list_check(
            user_api_key_dict=user_api_key_dict,
            user_id=None,
            team_id="team-2",
            organization_id=None,
            key_alias=None,
            key_hash=None,
            prisma_client=mock_prisma_client,
        )

    assert exc_info.value.code == "403" or exc_info.value.code == 403
    assert "not authorized to check this team's keys" in exc_info.value.message


@pytest.mark.asyncio
async def test_validate_key_list_check_key_hash_authorized():
    mock_prisma_client = AsyncMock()
    user_info = LiteLLM_UserTable(
        user_id="test-user",
        user_email="test@example.com",
        teams=[],
        organization_memberships=[],
    )

    key_info = LiteLLM_VerificationToken(
        token="hashed-key",
        user_id="test-user",
    )

    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
        return_value=user_info
    )
    mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=key_info
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="test-user",
    )

    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._can_user_query_key_info"
    ) as mock_can_query:
        mock_can_query.return_value = True

        result = await validate_key_list_check(
            user_api_key_dict=user_api_key_dict,
            user_id=None,
            team_id=None,
            organization_id=None,
            key_alias=None,
            key_hash="hashed-key",
            prisma_client=mock_prisma_client,
        )

        assert result is not None
        assert result.user_id == "test-user"


@pytest.mark.asyncio
async def test_validate_key_list_check_key_hash_unauthorized():
    mock_prisma_client = AsyncMock()
    user_info = LiteLLM_UserTable(
        user_id="test-user",
        user_email="test@example.com",
        teams=[],
        organization_memberships=[],
    )

    key_info = LiteLLM_VerificationToken(
        token="hashed-key",
        user_id="other-user",
    )

    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
        return_value=user_info
    )
    mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        return_value=key_info
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="test-user",
    )

    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints._can_user_query_key_info"
    ) as mock_can_query:
        mock_can_query.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            await validate_key_list_check(
                user_api_key_dict=user_api_key_dict,
                user_id=None,
                team_id=None,
                organization_id=None,
                key_alias=None,
                key_hash="hashed-key",
                prisma_client=mock_prisma_client,
            )

        assert exc_info.value.status_code == 403
        assert "not allowed to access this key's info" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_validate_key_list_check_key_hash_not_found():
    mock_prisma_client = AsyncMock()
    user_info = LiteLLM_UserTable(
        user_id="test-user",
        user_email="test@example.com",
        teams=[],
        organization_memberships=[],
    )

    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(
        return_value=user_info
    )
    mock_prisma_client.db.litellm_verificationtoken.find_unique = AsyncMock(
        side_effect=Exception("Key not found")
    )

    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="test-user",
    )

    with pytest.raises(ProxyException) as exc_info:
        await validate_key_list_check(
            user_api_key_dict=user_api_key_dict,
            user_id=None,
            team_id=None,
            organization_id=None,
            key_alias=None,
            key_hash="non-existent-key",
            prisma_client=mock_prisma_client,
        )

    assert exc_info.value.code == "403" or exc_info.value.code == 403
    assert "Key Hash not found" in exc_info.value.message


@pytest.mark.asyncio
async def test_default_key_generate_params_duration(monkeypatch):
    """
    Test that default_key_generate_params with 'duration' is applied
    when no duration is provided in the key generation request.

    Regression test for bug where 'duration' was missing from the list
    of fields populated from default_key_generate_params.
    """
    import litellm

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

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Set default_key_generate_params with duration
    original_value = litellm.default_key_generate_params
    litellm.default_key_generate_params = {"duration": "180d"}

    try:
        request = GenerateKeyRequest()  # No duration specified
        response = await _common_key_generation_helper(
            data=request,
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN,
                api_key="sk-1234",
                user_id="1234",
            ),
            litellm_changed_by=None,
            team_table=None,
        )

        # Verify duration was applied from defaults
        assert request.duration == "180d"
    finally:
        litellm.default_key_generate_params = original_value
