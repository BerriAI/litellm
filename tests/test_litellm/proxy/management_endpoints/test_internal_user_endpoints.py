import json
import os
import sys
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy._types import (
    LiteLLM_UserTableFiltered,
    NewUserRequest,
    ProxyException,
    UpdateUserRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.internal_user_endpoints import (
    LiteLLM_UserTableWithKeyCount,
    _update_internal_user_params,
    get_user_key_counts,
    get_users,
    new_user,
    ui_view_users,
)
from litellm.proxy.proxy_server import app

client = TestClient(app)


@pytest.mark.asyncio
async def test_ui_view_users_with_null_email(mocker, caplog):
    """
    Test that /user/filter/ui endpoint returns users even when they have null email fields
    """
    # Mock the prisma client
    mock_prisma_client = mocker.MagicMock()

    # Create mock user data with null email
    mock_user = mocker.MagicMock()
    mock_user.model_dump.return_value = {
        "user_id": "test-user-null-email",
        "user_email": None,
        "user_role": "proxy_admin",
        "created_at": "2024-01-01T00:00:00Z",
    }

    # Setup the mock find_many response
    # Setup the mock find_many response as an async function
    async def mock_find_many(*args, **kwargs):
        return [mock_user]

    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many

    # Patch the prisma client import in the endpoint
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Call ui_view_users function directly
    response = await ui_view_users(
        user_api_key_dict=UserAPIKeyAuth(user_id="test_user"),
        user_id="test_user",
        user_email=None,
        page=1,
        page_size=50,
    )

    assert response == [
        LiteLLM_UserTableFiltered(user_id="test-user-null-email", user_email=None)
    ]


def test_user_daily_activity_types():
    """
    Assert all fiels in SpendMetrics are reported in DailySpendMetadata as "total_"
    """
    from litellm.proxy.management_endpoints.common_daily_activity import (
        DailySpendMetadata,
        SpendMetrics,
    )

    # Create a SpendMetrics instance
    spend_metrics = SpendMetrics()

    # Create a DailySpendMetadata instance
    daily_spend_metadata = DailySpendMetadata()

    # Assert all fields in SpendMetrics are reported in DailySpendMetadata as "total_"
    for field in spend_metrics.__dict__:
        if field.startswith("total_"):
            assert hasattr(
                daily_spend_metadata, field
            ), f"Field {field} is not reported in DailySpendMetadata"
        else:
            assert not hasattr(
                daily_spend_metadata, field
            ), f"Field {field} is reported in DailySpendMetadata"


@pytest.mark.asyncio
async def test_get_users_includes_timestamps(mocker):
    """
    Test that /user/list endpoint returns users with created_at and updated_at fields.
    """
    # Mock the prisma client
    mock_prisma_client = mocker.MagicMock()

    # Create mock user data with timestamps
    mock_user_data = {
        "user_id": "test-user-timestamps",
        "user_email": "timestamps@example.com",
        "user_role": "internal_user",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    mock_user_row = mocker.MagicMock()
    mock_user_row.model_dump.return_value = mock_user_data

    # Setup the mock find_many response as an async function
    async def mock_find_many(*args, **kwargs):
        return [mock_user_row]

    # Setup the mock count response as an async function
    async def mock_count(*args, **kwargs):
        return 1

    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many
    mock_prisma_client.db.litellm_usertable.count = mock_count

    # Patch the prisma client import in the endpoint
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Mock the helper function get_user_key_counts
    async def mock_get_user_key_counts(*args, **kwargs):
        return {"test-user-timestamps": 0}

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints.get_user_key_counts",
        mock_get_user_key_counts,
    )

    # Call get_users function directly
    response = await get_users(page=1, page_size=1)

    print("user /list response: ", response)

    # Assertions
    assert response is not None
    assert "users" in response
    assert "total" in response
    assert response["total"] == 1
    assert len(response["users"]) == 1

    user_response = response["users"][0]
    assert user_response.user_id == "test-user-timestamps"
    assert user_response.created_at is not None
    assert isinstance(user_response.created_at, datetime)
    assert user_response.updated_at is not None
    assert isinstance(user_response.updated_at, datetime)
    assert user_response.created_at == mock_user_data["created_at"]
    assert user_response.updated_at == mock_user_data["updated_at"]
    assert user_response.key_count == 0


def test_validate_sort_params():
    """
    Test that validate_sort_params returns None if sort_by is None
    """
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _validate_sort_params,
    )

    assert _validate_sort_params(None, "asc") is None
    assert _validate_sort_params(None, "desc") is None
    assert _validate_sort_params("user_id", "asc") == {"user_id": "asc"}
    assert _validate_sort_params("user_id", "desc") == {"user_id": "desc"}
    with pytest.raises(Exception):
        _validate_sort_params("user_id", "invalid")


def test_update_user_request_pydantic_object():
    """
    Test that _update_internal_user_params correctly processes an email-only update
    """
    data = UpdateUserRequest(user_email="test@example.com")

    data_json = data.model_dump(exclude_unset=True)

    assert data_json == {"user_email": "test@example.com"}


def test_update_internal_user_params_email():
    """
    Test that _update_internal_user_params correctly processes an email-only update
    """
    from litellm.proxy._types import UpdateUserRequest
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _update_internal_user_params,
    )

    # Create test data with only email update
    data_json = {"user_email": "test@example.com"}
    data = UpdateUserRequest(user_email="test@example.com")

    # Call the function
    non_default_values = _update_internal_user_params(data_json=data_json, data=data)

    # Assertions
    assert len(non_default_values) == 1  # Should only contain email
    assert "user_email" in non_default_values
    assert non_default_values["user_email"] == "test@example.com"
    assert "user_id" not in non_default_values  # Should not add user_id if not provided
    assert "max_budget" not in non_default_values  # Should not add default values
    assert "budget_duration" not in non_default_values  # Should not add default values


def test_update_internal_user_params_reset_spend_and_max_budget():
    """
    Relevant Issue: https://github.com/BerriAI/litellm/issues/10495
    """
    from litellm.proxy._types import UpdateUserRequest
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _update_internal_user_params,
    )

    # Create test data with only email update
    data = UpdateUserRequest(spend=0, max_budget=0, user_id="test_user_id")
    data_json = data.model_dump(exclude_unset=True)

    # Call the function
    non_default_values = _update_internal_user_params(data_json=data_json, data=data)

    # Assertions
    assert len(non_default_values) == 3  # Should only contain email
    assert "spend" in non_default_values
    assert non_default_values["spend"] == 0
    assert "max_budget" in non_default_values
    assert non_default_values["max_budget"] == 0
    assert "user_id" in non_default_values  # Should not add user_id if not provided
    assert non_default_values["user_id"] == "test_user_id"
    assert "budget_duration" not in non_default_values  # Should not add default values


@pytest.mark.asyncio
async def test_new_user_license_over_limit(mocker):
    """
    Test that /user/new endpoint raises an error when license is over the user limit
    """
    from fastapi import HTTPException

    from litellm.proxy._types import NewUserRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.internal_user_endpoints import new_user

    # Mock the prisma client
    mock_prisma_client = mocker.MagicMock()

    # Setup the mock count response to return a high number of users
    async def mock_count(*args, **kwargs):
        return 1000  # High user count

    mock_prisma_client.db.litellm_usertable.count = mock_count

    # Mock check_duplicate_user_email to pass
    async def mock_check_duplicate_user_email(*args, **kwargs):
        return None  # No duplicate found

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints._check_duplicate_user_email",
        mock_check_duplicate_user_email,
    )

    # Mock the license check to return True (over limit)
    mock_license_check = mocker.MagicMock()
    mock_license_check.is_over_limit.return_value = True

    # Patch the imports in the endpoint
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    mocker.patch("litellm.proxy.proxy_server._license_check", mock_license_check)

    # Create test request data
    user_request = NewUserRequest(
        user_email="test@example.com", user_role="internal_user"
    )

    # Mock user_api_key_dict
    mock_user_api_key_dict = UserAPIKeyAuth(user_id="test_admin")

    # Call new_user function and expect HTTPException
    with pytest.raises(ProxyException) as exc_info:
        await new_user(data=user_request, user_api_key_dict=mock_user_api_key_dict)

    # Verify the exception details
    assert exc_info.value.code == 403 or exc_info.value.code == "403"
    assert "License is over limit" in str(exc_info.value.message)
    assert "support@berri.ai" in str(exc_info.value.message)

    # Verify that the license check was called with the correct user count
    mock_license_check.is_over_limit.assert_called_once_with(total_users=1000)


@pytest.mark.asyncio
async def test_user_info_url_encoding_plus_character(mocker):
    """
    Test that /user/info endpoint properly handles email addresses with + characters
    when passed in the URL query parameters.

    Issue: + characters in emails get converted to spaces due to URL encoding
    Solution: Parse the raw query string to preserve + characters
    """
    from fastapi import Request

    from litellm.proxy._types import LiteLLM_UserTable, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.internal_user_endpoints import user_info

    # Mock the prisma client
    mock_prisma_client = mocker.MagicMock()
    
    # Create a real LiteLLM_UserTable instance (BaseModel) so isinstance check passes
    mock_user = LiteLLM_UserTable(
        user_id="machine-user+alp-air-admin-b58-b@tempus.com",
        user_email="machine-user+alp-air-admin-b58-b@tempus.com",
        teams=[],
    )
    
    # Mock get_data to return user when called with user_id, empty list for keys
    async def mock_get_data(*args, **kwargs):
        if kwargs.get("table_name") == "key":
            return []
        elif kwargs.get("table_name") == "team":
            return []
        elif kwargs.get("user_id") is not None:
            return mock_user
        return None
    
    mock_prisma_client.get_data = mocker.AsyncMock(side_effect=mock_get_data)

    # Mock list_team to return None (patch it from where it's imported)
    mock_list_team = mocker.AsyncMock(return_value=None)
    mocker.patch(
        "litellm.proxy.management_endpoints.team_endpoints.list_team",
        mock_list_team,
    )

    # Patch the prisma client import in the endpoint
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Create a mock request with the raw query string containing +
    mock_request = mocker.MagicMock(spec=Request)
    mock_request.url.query = "user_id=machine-user+alp-air-admin-b58-b@tempus.com"

    # Mock user_api_key_dict
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_id="test_admin", user_role="proxy_admin"
    )

    # Call user_info function with the URL-decoded user_id (as FastAPI would pass it)
    # FastAPI would normally convert + to space, but our fix should handle this
    decoded_user_id = (
        "machine-user alp-air-admin-b58-b@tempus.com"  # What FastAPI gives us
    )
    expected_user_id = "machine-user+alp-air-admin-b58-b@tempus.com"
    
    response = await user_info(
        user_id=decoded_user_id,
        user_api_key_dict=mock_user_api_key_dict,
        request=mock_request,
    )

    # Verify that the response contains the correct user data
    # Check that get_data was called with the correct user_id (first call should be for user)
    user_call = None
    for call in mock_prisma_client.get_data.call_args_list:
        if call.kwargs.get("user_id") and not call.kwargs.get("table_name"):
            user_call = call
            break
    
    assert user_call is not None, "get_data should be called with user_id"
    assert user_call.kwargs["user_id"] == expected_user_id


@pytest.mark.asyncio
async def test_user_info_nonexistent_user(mocker):
    """
    Test that /user/info endpoint returns 404 when a non-existent user_id is provided.
    """
    from fastapi import Request

    from litellm.proxy._types import ProxyException, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.internal_user_endpoints import user_info

    # Mock the prisma client
    mock_prisma_client = mocker.MagicMock()
    
    # Mock get_data to return None (user doesn't exist)
    async def mock_get_data(*args, **kwargs):
        if kwargs.get("table_name") == "key":
            return []
        elif kwargs.get("user_id") is not None:
            return None  # User not found
        return None
    
    mock_prisma_client.get_data = mocker.AsyncMock(side_effect=mock_get_data)

    # Patch the prisma client import in the endpoint
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Create a mock request
    mock_request = mocker.MagicMock(spec=Request)

    # Mock user_api_key_dict
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_id="test_admin", user_role="proxy_admin"
    )

    # Call user_info function with a non-existent user_id
    nonexistent_user_id = "nonexistent-user@example.com"
    
    # Should raise ProxyException with 404 status code (HTTPException is converted by decorator)
    with pytest.raises(ProxyException) as exc_info:
        await user_info(
            user_id=nonexistent_user_id,
            user_api_key_dict=mock_user_api_key_dict,
            request=mock_request,
        )

    # Verify the exception details
    assert exc_info.value.code == "404"  # ProxyException.code is a string
    assert f"User {nonexistent_user_id} not found" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_new_user_default_teams_flow(mocker):
    """
    Test that when teams are set via default_internal_user_params:
    - Teams are NOT sent to generate_key_helper_fn
    - Teams ARE sent to _add_user_to_team
    """
    import litellm
    from litellm.proxy._types import NewUserRequest, NewUserRequestTeam, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.internal_user_endpoints import new_user

    # Mock the prisma client
    mock_prisma_client = mocker.MagicMock()

    # Setup the mock count response (under license limit)
    async def mock_count(*args, **kwargs):
        return 5  # Low user count, under limit

    mock_prisma_client.db.litellm_usertable.count = mock_count

    # Mock check_duplicate_user_email to pass
    async def mock_check_duplicate_user_email(*args, **kwargs):
        return None  # No duplicate found

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints._check_duplicate_user_email",
        mock_check_duplicate_user_email,
    )

    # Mock the license check to return False (under limit)
    mock_license_check = mocker.MagicMock()
    mock_license_check.is_over_limit.return_value = False

    # Mock generate_key_helper_fn
    mock_generate_key_helper_fn = mocker.AsyncMock()
    mock_generate_key_helper_fn.return_value = {
        "user_id": "test-user-123",
        "token": "sk-test-token-123",
        "expires": None,
        "max_budget": 100,
    }

    # Mock _add_user_to_team
    mock_add_user_to_team = mocker.AsyncMock()

    # Mock UserManagementEventHooks.async_user_created_hook
    mock_user_created_hook = mocker.AsyncMock()

    # Setup default_internal_user_params with teams
    original_default_params = getattr(litellm, "default_internal_user_params", None)
    litellm.default_internal_user_params = {
        "teams": [
            {
                "team_id": "96fed65b-0182-4ff4-8429-2721cd7d42af",
                "max_budget_in_team": 100,
                "user_role": "user",
            }
        ]
    }

    try:
        # Patch all the imports
        mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
        mocker.patch("litellm.proxy.proxy_server._license_check", mock_license_check)
        mocker.patch(
            "litellm.proxy.management_endpoints.internal_user_endpoints.generate_key_helper_fn",
            mock_generate_key_helper_fn,
        )
        mocker.patch(
            "litellm.proxy.management_endpoints.internal_user_endpoints._add_user_to_team",
            mock_add_user_to_team,
        )
        mocker.patch(
            "litellm.proxy.management_endpoints.internal_user_endpoints.UserManagementEventHooks.async_user_created_hook",
            mock_user_created_hook,
        )

        # Create test request data WITHOUT teams (teams should come from defaults)
        user_request = NewUserRequest(
            user_email="test@example.com", user_role="internal_user"
        )

        # Mock user_api_key_dict
        mock_user_api_key_dict = UserAPIKeyAuth(user_id="test_admin")

        # Call new_user function
        response = await new_user(
            data=user_request, user_api_key_dict=mock_user_api_key_dict
        )

        # Verify generate_key_helper_fn was called WITHOUT teams
        mock_generate_key_helper_fn.assert_called_once()
        call_kwargs = mock_generate_key_helper_fn.call_args.kwargs

        # Teams should be removed from the data passed to generate_key_helper_fn
        assert (
            "teams" not in call_kwargs
        ), "Teams should not be passed to generate_key_helper_fn"
        assert call_kwargs["request_type"] == "user"
        assert call_kwargs["user_email"] == "test@example.com"
        assert call_kwargs["user_role"] == "internal_user"

        # Verify _add_user_to_team was called with the default team
        mock_add_user_to_team.assert_called_once()
        team_call_kwargs = mock_add_user_to_team.call_args.kwargs

        assert team_call_kwargs["user_id"] == "test-user-123"
        assert team_call_kwargs["team_id"] == "96fed65b-0182-4ff4-8429-2721cd7d42af"
        assert team_call_kwargs["user_email"] == "test@example.com"
        assert team_call_kwargs["user_role"] == "user"

        # Verify response structure
        assert response.user_id == "test-user-123"
        assert response.key == "sk-test-token-123"

    finally:
        # Restore original default params
        if original_default_params is not None:
            litellm.default_internal_user_params = original_default_params
        else:
            if hasattr(litellm, "default_internal_user_params"):
                delattr(litellm, "default_internal_user_params")


def test_update_internal_new_user_params_proxy_admin_role():
    """
    Test that default_internal_user_params are NOT applied when user_role is PROXY_ADMIN
    """
    import litellm
    from litellm.proxy._types import LitellmUserRoles, NewUserRequest
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _update_internal_new_user_params,
    )

    # Set up default_internal_user_params
    original_default_params = getattr(litellm, "default_internal_user_params", None)
    litellm.default_internal_user_params = {
        "max_budget": 1000,
        "models": ["gpt-3.5-turbo", "gpt-4"],
        "tpm_limit": 5000,
    }

    try:
        # Create test data with PROXY_ADMIN role
        data = NewUserRequest(
            user_email="admin@example.com", user_role=LitellmUserRoles.PROXY_ADMIN.value
        )
        data_json = data.model_dump(exclude_unset=True)

        # Call the function
        result = _update_internal_new_user_params(data_json=data_json, data=data)

        # Assertions - default params should NOT be applied for PROXY_ADMIN
        assert (
            "max_budget" not in result
        ), "Default max_budget should NOT be applied to PROXY_ADMIN"
        assert (
            "models" not in result
        ), "Default models should NOT be applied to PROXY_ADMIN"
        assert (
            "tpm_limit" not in result
        ), "Default tpm_limit should NOT be applied to PROXY_ADMIN"

        # These should still work
        assert result["user_email"] == "admin@example.com"
        assert result["user_role"] == LitellmUserRoles.PROXY_ADMIN.value

    finally:
        # Restore original default params
        if original_default_params is not None:
            litellm.default_internal_user_params = original_default_params
        else:
            if hasattr(litellm, "default_internal_user_params"):
                delattr(litellm, "default_internal_user_params")


def test_update_internal_new_user_params_no_role_specified():
    """
    Test that default_internal_user_params ARE applied when user_role is not set
    """
    import litellm
    from litellm.proxy._types import NewUserRequest
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _update_internal_new_user_params,
    )

    # Set up default_internal_user_params
    original_default_params = getattr(litellm, "default_internal_user_params", None)
    litellm.default_internal_user_params = {
        "max_budget": 1000,
        "models": ["gpt-3.5-turbo", "gpt-4"],
        "tpm_limit": 5000,
    }

    try:
        # Create test data without specifying user_role
        data = NewUserRequest(user_email="user@example.com")  # No user_role specified
        data_json = data.model_dump(exclude_unset=True)

        # Call the function
        result = _update_internal_new_user_params(data_json=data_json, data=data)

        # Assertions - default params should be applied when no role is specified
        assert result.get("max_budget") == 1000
        assert result.get("models") == ["gpt-3.5-turbo", "gpt-4"]
        assert result.get("tpm_limit") == 5000
        assert result["user_email"] == "user@example.com"

    finally:
        # Restore original default params
        if original_default_params is not None:
            litellm.default_internal_user_params = original_default_params
        else:
            if hasattr(litellm, "default_internal_user_params"):
                delattr(litellm, "default_internal_user_params")


def test_update_internal_new_user_params_internal_user_role():
    """
    Test that default_internal_user_params ARE applied when user_role is INTERNAL_USER
    """
    import litellm
    from litellm.proxy._types import LitellmUserRoles, NewUserRequest
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _update_internal_new_user_params,
    )

    # Set up default_internal_user_params
    original_default_params = getattr(litellm, "default_internal_user_params", None)
    litellm.default_internal_user_params = {
        "max_budget": 1000,
        "models": ["gpt-3.5-turbo", "gpt-4"],
        "tpm_limit": 5000,
    }

    try:
        # Create test data with INTERNAL_USER role
        data = NewUserRequest(
            user_email="internaluser@example.com",
            user_role=LitellmUserRoles.INTERNAL_USER.value,
        )
        data_json = data.model_dump(exclude_unset=True)

        # Call the function
        result = _update_internal_new_user_params(data_json=data_json, data=data)

        # Assertions - default params should be applied for INTERNAL_USER
        assert result.get("max_budget") == 1000
        assert result.get("models") == ["gpt-3.5-turbo", "gpt-4"]
        assert result.get("tpm_limit") == 5000
        assert result["user_email"] == "internaluser@example.com"
        assert result["user_role"] == LitellmUserRoles.INTERNAL_USER.value

    finally:
        # Restore original default params
        if original_default_params is not None:
            litellm.default_internal_user_params = original_default_params
        else:
            if hasattr(litellm, "default_internal_user_params"):
                delattr(litellm, "default_internal_user_params")


@pytest.mark.asyncio
async def test_check_duplicate_user_email_case_insensitive(mocker):
    """
    Test that _check_duplicate_user_email performs case insensitive email matching.

    This ensures that emails like 'User@Example.com' and 'user@example.com'
    are treated as the same user, preventing duplicate accounts.
    """
    from fastapi import HTTPException

    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _check_duplicate_user_email,
    )

    # Mock the prisma client
    mock_prisma_client = mocker.MagicMock()

    # Test Case 1: Duplicate found with different case
    # Mock existing user with uppercase email
    mock_existing_user = mocker.MagicMock()
    mock_existing_user.user_email = "User@Example.com"

    async def mock_find_first_duplicate(*args, **kwargs):
        # Verify that the query uses case insensitive matching
        where_clause = kwargs.get("where", {})
        user_email_clause = where_clause.get("user_email", {})

        # Check that the query structure is correct for case insensitive search
        assert (
            "equals" in user_email_clause
        ), "Query should use 'equals' for case insensitive search"
        assert (
            user_email_clause.get("mode") == "insensitive"
        ), "Query should use 'insensitive' mode"
        assert (
            user_email_clause.get("equals") == "user@example.com"
        ), "Query should search for the provided email"

        return mock_existing_user  # Return existing user to simulate duplicate

    mock_prisma_client.db.litellm_usertable.find_first = mock_find_first_duplicate

    # Should raise HTTPException when duplicate is found
    with pytest.raises(HTTPException) as exc_info:
        await _check_duplicate_user_email("user@example.com", mock_prisma_client)

    assert exc_info.value.status_code == 400
    assert "User with email User@Example.com already exists" in str(
        exc_info.value.detail
    )

    # Test Case 2: No duplicate found
    async def mock_find_first_no_duplicate(*args, **kwargs):
        # Verify the query structure again
        where_clause = kwargs.get("where", {})
        user_email_clause = where_clause.get("user_email", {})

        assert "equals" in user_email_clause
        assert user_email_clause.get("mode") == "insensitive"
        assert user_email_clause.get("equals") == "newuser@example.com"

        return None  # No existing user found

    mock_prisma_client.db.litellm_usertable.find_first = mock_find_first_no_duplicate

    # Should not raise any exception when no duplicate is found
    try:
        await _check_duplicate_user_email("newuser@example.com", mock_prisma_client)
        # If we reach here, no exception was raised (which is expected)
        assert True
    except Exception as e:
        pytest.fail(f"Should not raise exception when no duplicate found, but got: {e}")

    # Test Case 3: None email should not cause issues
    await _check_duplicate_user_email(
        None, mock_prisma_client
    )  # Should not raise exception


def test_process_keys_for_user_info_filters_dashboard_keys(monkeypatch):
    """
    Test that _process_keys_for_user_info filters out keys with team_id='litellm-dashboard'
    
    UI session tokens (team_id='litellm-dashboard') should be excluded from user info responses
    to prevent confusion, as these are automatically created during dashboard login.
    """
    from unittest.mock import MagicMock

    from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _process_keys_for_user_info,
    )

    # Create mock keys with different team_ids
    mock_key_dashboard = MagicMock()
    mock_key_dashboard.model_dump.return_value = {
        "token": "sk-dashboard-token",
        "team_id": UI_SESSION_TOKEN_TEAM_ID,
        "user_id": "test-user",
        "key_alias": "dashboard-session-key",
    }
    
    mock_key_regular = MagicMock()
    mock_key_regular.model_dump.return_value = {
        "token": "sk-regular-token",
        "team_id": "regular-team",
        "user_id": "test-user",
        "key_alias": "regular-key",
    }
    
    mock_key_no_team = MagicMock()
    mock_key_no_team.model_dump.return_value = {
        "token": "sk-no-team-token",
        "team_id": None,
        "user_id": "test-user",
        "key_alias": "no-team-key",
    }

    keys = [mock_key_dashboard, mock_key_regular, mock_key_no_team]

    # Mock general_settings and litellm_master_key_hash (they're imported from proxy_server)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings",
        {},
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.litellm_master_key_hash",
        "different-hash",
    )

    # Call the function
    result = _process_keys_for_user_info(keys=keys, all_teams=None)

    # Verify that dashboard key is filtered out
    assert len(result) == 2, "Should return 2 keys (dashboard key filtered out)"
    
    # Verify dashboard key is not in results
    result_team_ids = [key.get("team_id") for key in result]
    assert UI_SESSION_TOKEN_TEAM_ID not in result_team_ids, "Dashboard key should be filtered out"
    
    # Verify regular keys are included
    assert "regular-team" in result_team_ids, "Regular team key should be included"
    assert None in result_team_ids, "No-team key should be included"
    
    # Verify the correct keys are returned
    result_tokens = [key.get("token") for key in result]
    assert "sk-regular-token" in result_tokens, "Regular key should be included"
    assert "sk-no-team-token" in result_tokens, "No-team key should be included"
    assert "sk-dashboard-token" not in result_tokens, "Dashboard key should not be included"


def test_process_keys_for_user_info_handles_none_keys(monkeypatch):
    """
    Test that _process_keys_for_user_info handles None keys gracefully
    """
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _process_keys_for_user_info,
    )

    # Mock general_settings and litellm_master_key_hash (they're imported from proxy_server)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings",
        {},
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.litellm_master_key_hash",
        "different-hash",
    )

    # Call with None keys
    result = _process_keys_for_user_info(keys=None, all_teams=None)

    # Should return empty list
    assert result == [], "Should return empty list when keys is None"


def test_process_keys_for_user_info_handles_empty_keys(monkeypatch):
    """
    Test that _process_keys_for_user_info handles empty keys list
    """
    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _process_keys_for_user_info,
    )

    # Mock general_settings and litellm_master_key_hash (they're imported from proxy_server)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.general_settings",
        {},
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.litellm_master_key_hash",
        "different-hash",
    )

    # Call with empty list
    result = _process_keys_for_user_info(keys=[], all_teams=None)

    # Should return empty list
    assert result == [], "Should return empty list when keys is empty"


@pytest.mark.asyncio
async def test_get_users_user_id_partial_match(mocker):
    """
    Test that /user/list endpoint uses partial matching for single user_id
    and exact matching for multiple user_ids.
    """
    from litellm.proxy._types import UserAPIKeyAuth

    mock_prisma_client = mocker.MagicMock()

    mock_user_data = {
        "user_id": "test-user-partial-match",
        "user_email": "test@example.com",
        "user_role": "internal_user",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    mock_user_row = mocker.MagicMock()
    mock_user_row.model_dump.return_value = mock_user_data

    captured_where_conditions = {}

    async def mock_find_many(*args, **kwargs):
        if "where" in kwargs:
            captured_where_conditions.update(kwargs["where"])
        return [mock_user_row]

    async def mock_count(*args, **kwargs):
        return 1

    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many
    mock_prisma_client.db.litellm_usertable.count = mock_count

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    async def mock_get_user_key_counts(*args, **kwargs):
        return {"test-user-partial-match": 0}

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints.get_user_key_counts",
        mock_get_user_key_counts,
    )

    captured_where_conditions.clear()
    await get_users(user_ids="test-user", page=1, page_size=1)

    assert "user_id" in captured_where_conditions
    assert "contains" in captured_where_conditions["user_id"]
    assert captured_where_conditions["user_id"]["contains"] == "test-user"
    assert captured_where_conditions["user_id"]["mode"] == "insensitive"

    captured_where_conditions.clear()
    await get_users(user_ids="user1,user2,user3", page=1, page_size=1)

    assert "user_id" in captured_where_conditions
    assert "in" in captured_where_conditions["user_id"]
    assert captured_where_conditions["user_id"]["in"] == ["user1", "user2", "user3"]
