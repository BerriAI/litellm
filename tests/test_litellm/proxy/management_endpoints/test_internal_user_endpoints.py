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
    LitellmUserRoles,
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
    Test that /user/filter/ui endpoint returns users even when they have null email fields.
    Uses proxy admin so no org filtering is applied.
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

    async def mock_find_many(*args, **kwargs):
        return [mock_user]

    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many

    # Flag OFF by default
    mocker.patch(
        "litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints.get_ui_settings_cached",
        return_value={},
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Proxy admin: no org filter, no get_user_object call
    response = await ui_view_users(
        user_api_key_dict=UserAPIKeyAuth(
            user_id="test_user", user_role=LitellmUserRoles.PROXY_ADMIN
        ),
        user_id="test_user",
        user_email=None,
        team_id=None,
        page=1,
        page_size=50,
    )

    assert response == [
        LiteLLM_UserTableFiltered(user_id="test-user-null-email", user_email=None)
    ]


@pytest.mark.asyncio
async def test_ui_view_users_proxy_admin_no_org_filter(mocker):
    """
    Proxy admin: find_many is called without organization_memberships in where.
    """
    mock_prisma_client = mocker.MagicMock()
    async def mock_find_many(*args, **kwargs):
        assert "organization_memberships" not in (kwargs.get("where") or {})
        return []

    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many

    # Flag OFF by default
    mocker.patch(
        "litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints.get_ui_settings_cached",
        return_value={},
    )
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    await ui_view_users(
        user_api_key_dict=UserAPIKeyAuth(
            user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN
        ),
        user_id=None,
        user_email="foo",
        team_id=None,
        page=1,
        page_size=50,
    )


@pytest.mark.asyncio
async def test_ui_view_users_org_admin_filtered_by_org(mocker):
    """
    Org admin with scope_user_search_to_org ON: find_many is called with
    organization_memberships filter so only users in the caller's org(s) are returned.
    """
    from litellm.proxy._types import LiteLLM_OrganizationMembershipTable

    mock_prisma_client = mocker.MagicMock()
    org_id = "org-123"

    async def mock_find_many(*args, **kwargs):
        where = kwargs.get("where") or {}
        assert "organization_memberships" in where
        assert where["organization_memberships"] == {
            "some": {"organization_id": {"in": [org_id]}}
        }
        return []

    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many

    # Flag ON
    mocker.patch(
        "litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints.get_ui_settings_cached",
        return_value={"scope_user_search_to_org": True},
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    mocker.patch("litellm.proxy.proxy_server.user_api_key_cache", mocker.MagicMock())
    mocker.patch("litellm.proxy.proxy_server.proxy_logging_obj", mocker.MagicMock())

    caller_user = mocker.MagicMock()
    caller_user.organization_memberships = [
        LiteLLM_OrganizationMembershipTable(
            user_id="org-admin",
            organization_id=org_id,
            user_role=LitellmUserRoles.ORG_ADMIN.value,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    ]

    async def mock_get_user_object(*args, **kwargs):
        return caller_user

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints.get_user_object",
        side_effect=mock_get_user_object,
    )

    response = await ui_view_users(
        user_api_key_dict=UserAPIKeyAuth(user_id="org-admin", user_role=None),
        user_id=None,
        user_email="u",
        team_id=None,
        page=1,
        page_size=50,
    )

    assert response == []


@pytest.mark.asyncio
async def test_ui_view_users_non_org_admin_returns_403(mocker):
    """
    Flag ON, caller is not proxy admin and not org admin, no team_id: endpoint returns 403.
    """
    from fastapi import HTTPException

    mock_prisma_client = mocker.MagicMock()

    # Flag ON
    mocker.patch(
        "litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints.get_ui_settings_cached",
        return_value={"scope_user_search_to_org": True},
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    mocker.patch("litellm.proxy.proxy_server.user_api_key_cache", mocker.MagicMock())
    mocker.patch("litellm.proxy.proxy_server.proxy_logging_obj", mocker.MagicMock())

    # Caller has no org admin membership
    caller_user = mocker.MagicMock()
    caller_user.organization_memberships = []  # not an org admin

    async def mock_get_user_object(*args, **kwargs):
        return caller_user

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints.get_user_object",
        side_effect=mock_get_user_object,
    )

    with pytest.raises(HTTPException) as exc_info:
        await ui_view_users(
            user_api_key_dict=UserAPIKeyAuth(user_id="internal_user", user_role=None),
            user_id=None,
            user_email="u",
            team_id=None,
            page=1,
            page_size=50,
        )

    assert exc_info.value.status_code == 403
    assert "scope_user_search_to_org is enabled" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_ui_view_users_flag_off_internal_user_can_search(mocker):
    """
    Flag OFF (default): any authenticated user can search all users without org filtering.
    """
    mock_prisma_client = mocker.MagicMock()

    async def mock_find_many(*args, **kwargs):
        where = kwargs.get("where") or {}
        assert "organization_memberships" not in where
        return []

    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many

    # Flag OFF
    mocker.patch(
        "litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints.get_ui_settings_cached",
        return_value={},
    )
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    response = await ui_view_users(
        user_api_key_dict=UserAPIKeyAuth(user_id="internal_user", user_role=None),
        user_id=None,
        user_email="foo",
        team_id=None,
        page=1,
        page_size=50,
    )

    assert response == []


@pytest.mark.asyncio
async def test_ui_view_users_flag_on_team_admin_org_team(mocker):
    """
    Flag ON, team admin for org-bound team: org filter is applied using team's org.
    """
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj

    mock_prisma_client = mocker.MagicMock()
    org_id = "org-456"
    tid = "team-789"

    async def mock_find_many(*args, **kwargs):
        where = kwargs.get("where") or {}
        assert "organization_memberships" in where
        assert where["organization_memberships"] == {
            "some": {"organization_id": {"in": [org_id]}}
        }
        return []

    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many

    # Flag ON
    mocker.patch(
        "litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints.get_ui_settings_cached",
        return_value={"scope_user_search_to_org": True},
    )

    # Mock get_team_object
    team_obj = LiteLLM_TeamTableCachedObj(
        team_id=tid,
        team_alias="test-team",
        organization_id=org_id,
        members_with_roles=[{"user_id": "team-admin-user", "role": "admin"}],
    )

    async def mock_get_team_object(*args, **kwargs):
        return team_obj

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints.get_team_object",
        side_effect=mock_get_team_object,
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    mocker.patch("litellm.proxy.proxy_server.user_api_key_cache", mocker.MagicMock())
    mocker.patch("litellm.proxy.proxy_server.proxy_logging_obj", mocker.MagicMock())

    # Caller is not org admin
    caller_user = mocker.MagicMock()
    caller_user.organization_memberships = []

    async def mock_get_user_object(*args, **kwargs):
        return caller_user

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints.get_user_object",
        side_effect=mock_get_user_object,
    )

    response = await ui_view_users(
        user_api_key_dict=UserAPIKeyAuth(user_id="team-admin-user", user_role=None),
        user_id=None,
        user_email="u",
        team_id=tid,
        page=1,
        page_size=50,
    )

    assert response == []


@pytest.mark.asyncio
async def test_ui_view_users_flag_on_team_admin_non_org_team_403(mocker):
    """
    Flag ON, team admin for non-org team: returns 403.
    """
    from fastapi import HTTPException
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj

    mock_prisma_client = mocker.MagicMock()
    tid = "team-no-org"

    # Flag ON
    mocker.patch(
        "litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints.get_ui_settings_cached",
        return_value={"scope_user_search_to_org": True},
    )

    # Mock get_team_object — team has no organization_id
    team_obj = LiteLLM_TeamTableCachedObj(
        team_id=tid,
        team_alias="no-org-team",
        organization_id=None,
        members_with_roles=[{"user_id": "team-admin-user", "role": "admin"}],
    )

    async def mock_get_team_object(*args, **kwargs):
        return team_obj

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints.get_team_object",
        side_effect=mock_get_team_object,
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    mocker.patch("litellm.proxy.proxy_server.user_api_key_cache", mocker.MagicMock())
    mocker.patch("litellm.proxy.proxy_server.proxy_logging_obj", mocker.MagicMock())

    # Caller is not org admin
    caller_user = mocker.MagicMock()
    caller_user.organization_memberships = []

    async def mock_get_user_object(*args, **kwargs):
        return caller_user

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints.get_user_object",
        side_effect=mock_get_user_object,
    )

    with pytest.raises(HTTPException) as exc_info:
        await ui_view_users(
            user_api_key_dict=UserAPIKeyAuth(
                user_id="team-admin-user", user_role=None
            ),
            user_id=None,
            user_email="u",
            team_id=tid,
            page=1,
            page_size=50,
        )

    assert exc_info.value.status_code == 403
    assert "not part of an organization" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_ui_view_users_flag_on_non_admin_no_team_id_403(mocker):
    """
    Flag ON, non-admin caller without team_id: returns 403.
    """
    from fastapi import HTTPException

    mock_prisma_client = mocker.MagicMock()

    # Flag ON
    mocker.patch(
        "litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints.get_ui_settings_cached",
        return_value={"scope_user_search_to_org": True},
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    mocker.patch("litellm.proxy.proxy_server.user_api_key_cache", mocker.MagicMock())
    mocker.patch("litellm.proxy.proxy_server.proxy_logging_obj", mocker.MagicMock())

    # Caller is not org admin
    caller_user = mocker.MagicMock()
    caller_user.organization_memberships = []

    async def mock_get_user_object(*args, **kwargs):
        return caller_user

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints.get_user_object",
        side_effect=mock_get_user_object,
    )

    with pytest.raises(HTTPException) as exc_info:
        await ui_view_users(
            user_api_key_dict=UserAPIKeyAuth(user_id="internal_user", user_role=None),
            user_id=None,
            user_email="u",
            team_id=None,
            page=1,
            page_size=50,
        )

    assert exc_info.value.status_code == 403
    assert "scope_user_search_to_org is enabled" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_ui_view_users_flag_on_team_admin_org_member_no_team_id(mocker):
    """
    Flag ON, team admin who is an org member (not org admin), no team_id param:
    should succeed and filter by the user's org membership.
    """
    mock_prisma_client = mocker.MagicMock()
    org_id = "org-member-org"

    async def mock_find_many(*args, **kwargs):
        where = kwargs.get("where") or {}
        assert "organization_memberships" in where
        assert where["organization_memberships"] == {
            "some": {"organization_id": {"in": [org_id]}}
        }
        return []

    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many

    mocker.patch(
        "litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints.get_ui_settings_cached",
        return_value={"scope_user_search_to_org": True},
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    mocker.patch("litellm.proxy.proxy_server.user_api_key_cache", mocker.MagicMock())
    mocker.patch("litellm.proxy.proxy_server.proxy_logging_obj", mocker.MagicMock())

    # Caller is org member (internal_user role, not org admin)
    membership = mocker.MagicMock()
    membership.organization_id = org_id
    membership.user_role = "internal_user"

    caller_user = mocker.MagicMock()
    caller_user.organization_memberships = [membership]

    async def mock_get_user_object(*args, **kwargs):
        return caller_user

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints.get_user_object",
        side_effect=mock_get_user_object,
    )

    response = await ui_view_users(
        user_api_key_dict=UserAPIKeyAuth(user_id="team-admin-in-org", user_role=None),
        user_id=None,
        user_email="u",
        team_id=None,
        page=1,
        page_size=50,
    )

    assert response == []


@pytest.mark.asyncio
async def test_ui_view_users_flag_on_team_admin_not_in_org_resolves_via_key_team(
    mocker,
):
    """
    Flag ON, team admin NOT in any org, no team_id query param but
    user_api_key_dict.team_id is set: resolves org via the key's team.
    """
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj

    mock_prisma_client = mocker.MagicMock()
    org_id = "org-from-team"
    tid = "key-team-id"

    async def mock_find_many(*args, **kwargs):
        where = kwargs.get("where") or {}
        assert "organization_memberships" in where
        assert where["organization_memberships"] == {
            "some": {"organization_id": {"in": [org_id]}}
        }
        return []

    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many

    mocker.patch(
        "litellm.proxy.ui_crud_endpoints.proxy_setting_endpoints.get_ui_settings_cached",
        return_value={"scope_user_search_to_org": True},
    )

    team_obj = LiteLLM_TeamTableCachedObj(
        team_id=tid,
        team_alias="key-team",
        organization_id=org_id,
        members_with_roles=[{"user_id": "team-admin-no-org", "role": "admin"}],
    )

    async def mock_get_team_object(*args, **kwargs):
        return team_obj

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints.get_team_object",
        side_effect=mock_get_team_object,
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    mocker.patch("litellm.proxy.proxy_server.user_api_key_cache", mocker.MagicMock())
    mocker.patch("litellm.proxy.proxy_server.proxy_logging_obj", mocker.MagicMock())

    # Caller has no org memberships
    caller_user = mocker.MagicMock()
    caller_user.organization_memberships = []

    async def mock_get_user_object(*args, **kwargs):
        return caller_user

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints.get_user_object",
        side_effect=mock_get_user_object,
    )

    # No team_id query param, but team_id on the API key
    response = await ui_view_users(
        user_api_key_dict=UserAPIKeyAuth(
            user_id="team-admin-no-org", user_role=None, team_id=tid
        ),
        user_id=None,
        user_email="u",
        team_id=None,
        page=1,
        page_size=50,
    )

    assert response == []


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

    # Call get_users function directly with proxy admin auth
    admin_key = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
    response = await get_users(page=1, page_size=1, user_api_key_dict=admin_key, organization_ids=None)

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

    # Mock duplicate checks to pass
    async def mock_check_duplicate_user_email(*args, **kwargs):
        return None  # No duplicate found

    async def mock_check_duplicate_user_id(*args, **kwargs):
        return None  # No duplicate found

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints._check_duplicate_user_email",
        mock_check_duplicate_user_email,
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints._check_duplicate_user_id",
        mock_check_duplicate_user_id,
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
async def test_new_user_non_admin_cannot_create_admin(mocker):
    """
    Test that non-admin users cannot create administrative users (PROXY_ADMIN or PROXY_ADMIN_VIEW_ONLY).
    This prevents privilege escalation vulnerabilities.
    """
    from litellm.proxy.management_endpoints.internal_user_endpoints import new_user

    # Mock the prisma client
    mock_prisma_client = mocker.MagicMock()

    # Setup the mock count response (under license limit)
    async def mock_count(*args, **kwargs):
        return 5  # Low user count, under limit

    mock_prisma_client.db.litellm_usertable.count = mock_count

    # Mock duplicate checks to pass
    async def mock_check_duplicate_user_email(*args, **kwargs):
        return None  # No duplicate found

    async def mock_check_duplicate_user_id(*args, **kwargs):
        return None  # No duplicate found

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints._check_duplicate_user_email",
        mock_check_duplicate_user_email,
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints._check_duplicate_user_id",
        mock_check_duplicate_user_id,
    )

    # Mock the license check to return False (under limit)
    mock_license_check = mocker.MagicMock()
    mock_license_check.is_over_limit.return_value = False

    # Patch the imports in the endpoint
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    mocker.patch("litellm.proxy.proxy_server._license_check", mock_license_check)

    # Test Case 1: INTERNAL_USER trying to create PROXY_ADMIN
    user_request = NewUserRequest(
        user_email="admin@example.com", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    # Mock user_api_key_dict with non-admin role
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_id="test_internal_user", user_role=LitellmUserRoles.INTERNAL_USER
    )

    # Call new_user function and expect ProxyException
    with pytest.raises(ProxyException) as exc_info:
        await new_user(data=user_request, user_api_key_dict=mock_user_api_key_dict)

    # Verify the exception details
    assert exc_info.value.code == 403 or exc_info.value.code == "403"
    assert "Only proxy admins can create administrative users" in str(exc_info.value.message)
    assert "proxy_admin" in str(exc_info.value.message)
    assert "proxy_admin_viewer" in str(exc_info.value.message)
    assert str(LitellmUserRoles.PROXY_ADMIN) in str(exc_info.value.message)
    assert str(LitellmUserRoles.INTERNAL_USER) in str(exc_info.value.message)

    # Test Case 2: INTERNAL_USER trying to create PROXY_ADMIN_VIEW_ONLY
    user_request_viewer = NewUserRequest(
        user_email="admin_viewer@example.com",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    )

    with pytest.raises(ProxyException) as exc_info2:
        await new_user(
            data=user_request_viewer, user_api_key_dict=mock_user_api_key_dict
        )

    # Verify the exception details
    assert exc_info2.value.code == 403 or exc_info2.value.code == "403"
    assert "Only proxy admins can create administrative users" in str(
        exc_info2.value.message
    )
    assert str(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY) in str(exc_info2.value.message)


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

    # Mock duplicate checks to pass
    async def mock_check_duplicate_user_email(*args, **kwargs):
        return None  # No duplicate found

    async def mock_check_duplicate_user_id(*args, **kwargs):
        return None  # No duplicate found

    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints._check_duplicate_user_email",
        mock_check_duplicate_user_email,
    )
    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints._check_duplicate_user_id",
        mock_check_duplicate_user_id,
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
        # Restore original default params (always assign, never delattr — the attribute
        # is defined in litellm/__init__.py and delattr-ing it breaks parallel tests)
        litellm.default_internal_user_params = original_default_params


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
        litellm.default_internal_user_params = original_default_params


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
        litellm.default_internal_user_params = original_default_params


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
        litellm.default_internal_user_params = original_default_params


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

    assert exc_info.value.status_code == 409
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


@pytest.mark.asyncio
async def test_check_duplicate_user_id(mocker):
    """
    Test that _check_duplicate_user_id detects duplicates and does not use case insensitive matching.
    """
    from fastapi import HTTPException

    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        _check_duplicate_user_id,
    )

    mock_prisma_client = mocker.MagicMock()

    # Duplicate user_id should raise
    mock_existing_user = mocker.MagicMock()
    mock_existing_user.user_id = "existing-user-id"

    async def mock_find_first_duplicate(*args, **kwargs):
        where_clause = kwargs.get("where", {})
        user_id_clause = where_clause.get("user_id", {})
        assert user_id_clause.get("equals") == "existing-user-id"
        assert "mode" not in user_id_clause
        return mock_existing_user

    mock_prisma_client.db.litellm_usertable.find_first = mock_find_first_duplicate

    with pytest.raises(HTTPException) as exc_info:
        await _check_duplicate_user_id("existing-user-id", mock_prisma_client)

    assert exc_info.value.status_code == 409
    assert "User with id existing-user-id already exists" in str(
        exc_info.value.detail
    )

    # No duplicate should pass
    async def mock_find_first_no_duplicate(*args, **kwargs):
        where_clause = kwargs.get("where", {})
        user_id_clause = where_clause.get("user_id", {})
        assert user_id_clause.get("equals") == "new-user-id"
        assert "mode" not in user_id_clause
        return None

    mock_prisma_client.db.litellm_usertable.find_first = mock_find_first_no_duplicate

    await _check_duplicate_user_id("new-user-id", mock_prisma_client)

    # None user_id should no-op
    await _check_duplicate_user_id(None, mock_prisma_client)


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

    admin_key = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)

    captured_where_conditions.clear()
    await get_users(user_ids="test-user", page=1, page_size=1, user_api_key_dict=admin_key, organization_ids=None)

    assert "user_id" in captured_where_conditions
    assert "contains" in captured_where_conditions["user_id"]
    assert captured_where_conditions["user_id"]["contains"] == "test-user"
    assert captured_where_conditions["user_id"]["mode"] == "insensitive"

    captured_where_conditions.clear()
    await get_users(user_ids="user1,user2,user3", page=1, page_size=1, user_api_key_dict=admin_key, organization_ids=None)

    assert "user_id" in captured_where_conditions
    assert "in" in captured_where_conditions["user_id"]
    assert captured_where_conditions["user_id"]["in"] == ["user1", "user2", "user3"]


def test_update_internal_user_params_reset_max_budget_with_none():
    """
    Test that _update_internal_user_params allows setting max_budget to None.
    This verifies the fix for unsetting/resetting the budget to unlimited.
    """
    
    # Case 1: max_budget is explicitly None in the input dictionary
    data_json = {"max_budget": None, "user_id": "test_user"}
    data = UpdateUserRequest(max_budget=None, user_id="test_user")

    # Call the function
    non_default_values = _update_internal_user_params(data_json=data_json, data=data)

    # Assertions
    assert "max_budget" in non_default_values
    assert non_default_values["max_budget"] is None
    assert non_default_values["user_id"] == "test_user"


def test_update_internal_user_params_ignores_other_nones():
    """
    Test that other fields are still filtered out if None
    """
    # Create test data with other None fields
    data_json = {"user_alias": None, "user_id": "test_user", "max_budget": 100.0}
    data = UpdateUserRequest(user_alias=None, user_id="test_user", max_budget=100.0)

    # Call the function
    non_default_values = _update_internal_user_params(data_json=data_json, data=data)

    # Assertions
    assert "user_alias" not in non_default_values
    assert non_default_values["max_budget"] == 100.0


def test_update_internal_user_params_keeps_original_max_budget_when_not_provided():
    """
    Test that _update_internal_user_params does not include max_budget 
    when it's not provided in the request (should keep original value).
    """
    # Create test data without max_budget
    data_json = {"user_id": "test_user", "user_alias": "test_alias"}
    data = UpdateUserRequest(user_id="test_user", user_alias="test_alias")

    # Call the function
    non_default_values = _update_internal_user_params(data_json=data_json, data=data)

    # Assertions: max_budget should NOT be in non_default_values
    assert "max_budget" not in non_default_values
    assert "user_id" in non_default_values
    assert "user_alias" in non_default_values


def test_generate_request_base_validator():
    """
    Test that GenerateRequestBase validator converts empty string to None for max_budget
    """
    from litellm.proxy._types import GenerateRequestBase
    
    # Test with empty string
    req = GenerateRequestBase(max_budget="")
    assert req.max_budget is None

    # Test with actual float
    req = GenerateRequestBase(max_budget=100.0)
    assert req.max_budget == 100.0

    # Test with None
    req = GenerateRequestBase(max_budget=None)
    assert req.max_budget is None


@pytest.mark.asyncio
async def test_get_user_daily_activity_non_admin_cannot_view_other_users(monkeypatch):
    """
    Test that non-admin users cannot view another user's daily activity data.
    The endpoint should raise 403 when user_id does not match the caller's own user_id.
    Also verifies that omitting user_id defaults to the caller's own user_id.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi import HTTPException

    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        get_user_daily_activity,
    )

    # Mock the prisma client so the DB-not-connected check passes
    mock_prisma_client = MagicMock()
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    )

    # Non-admin caller
    non_admin_key_dict = UserAPIKeyAuth(
        user_id="regular-user-123",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    # Case 1: Non-admin tries to view a different user's data — should get 403
    with pytest.raises(HTTPException) as exc_info:
        await get_user_daily_activity(
            start_date="2025-01-01",
            end_date="2025-01-31",
            model=None,
            api_key=None,
            user_id="other-user-456",
            page=1,
            page_size=50,
            timezone=None,
            user_api_key_dict=non_admin_key_dict,
        )

    assert exc_info.value.status_code == 403
    assert "Non-admin users can only view their own spend data" in str(
        exc_info.value.detail
    )

    # Case 2: Non-admin omits user_id — should default to their own user_id
    mock_response = MagicMock()
    with patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints.get_daily_activity",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_get_daily:
        result = await get_user_daily_activity(
            start_date="2025-01-01",
            end_date="2025-01-31",
            model=None,
            api_key=None,
            user_id=None,
            page=1,
            page_size=50,
            timezone=None,
            user_api_key_dict=non_admin_key_dict,
        )

        # Verify it called get_daily_activity with the caller's own user_id
        mock_get_daily.assert_called_once()
        call_kwargs = mock_get_daily.call_args
        assert call_kwargs.kwargs["entity_id"] == "regular-user-123"


@pytest.mark.asyncio
async def test_get_user_daily_activity_aggregated_admin_global_view(monkeypatch):
    """
    Test that admin users can call the aggregated endpoint without a user_id
    to get a global view. Also verifies that the correct arguments are forwarded
    to the underlying get_daily_activity_aggregated helper.
    """
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy.management_endpoints.internal_user_endpoints import (
        get_user_daily_activity_aggregated,
    )

    # Mock the prisma client
    mock_prisma_client = MagicMock()
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    )

    # Mock the downstream helper so we don't need a real DB
    mock_response = MagicMock()
    mock_get_daily_agg = AsyncMock(return_value=mock_response)
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.internal_user_endpoints.get_daily_activity_aggregated",
        mock_get_daily_agg,
    )

    # Admin caller
    admin_key_dict = UserAPIKeyAuth(
        user_id="admin-user-001",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )

    # Admin calls without user_id → global view (entity_id=None)
    result = await get_user_daily_activity_aggregated(
        start_date="2025-02-01",
        end_date="2025-02-28",
        model="gpt-4",
        api_key=None,
        user_id=None,
        timezone=480,
        user_api_key_dict=admin_key_dict,
    )

    assert result is mock_response

    # Verify the helper was called with the right parameters
    mock_get_daily_agg.assert_called_once_with(
        prisma_client=mock_prisma_client,
        table_name="litellm_dailyuserspend",
        entity_id_field="user_id",
        entity_id=None,  # global view: no user_id filter
        entity_metadata_field=None,
        start_date="2025-02-01",
        end_date="2025-02-28",
        model="gpt-4",
        api_key=None,
        timezone_offset_minutes=480,
    )


@pytest.mark.asyncio
async def test_delete_user_cleans_up_created_by_invitation_links(mocker):
    """
    Test that delete_user removes invitation links where the deleted user is the
    creator (created_by) or updater (updated_by), not just the invited person (user_id).

    This prevents FK constraint violations when deleting a user who created pending invites.
    """
    from litellm.proxy._types import DeleteUserRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.internal_user_endpoints import delete_user

    mock_prisma_client = mocker.MagicMock()

    # Mock user lookup
    mock_user_row = mocker.MagicMock()
    mock_user_row.user_id = "admin-creator"
    mock_user_row.user_email = "admin@example.com"
    mock_user_row.teams = []
    mock_user_row.json.return_value = "{}"
    mock_user_row.model_dump.return_value = {
        "user_id": "admin-creator",
        "user_email": "admin@example.com",
        "teams": [],
    }

    async def mock_find_unique(*args, **kwargs):
        return mock_user_row

    mock_prisma_client.db.litellm_usertable.find_unique = mocker.AsyncMock(
        side_effect=mock_find_unique
    )

    # Mock find_many for teams (no teams)
    mock_prisma_client.db.litellm_teamtable.find_many = mocker.AsyncMock(
        return_value=[]
    )

    # Mock all delete_many calls
    mock_prisma_client.db.litellm_verificationtoken.delete_many = mocker.AsyncMock(
        return_value=0
    )
    mock_prisma_client.db.litellm_invitationlink.delete_many = mocker.AsyncMock(
        return_value=1
    )
    mock_prisma_client.db.litellm_organizationmembership.delete_many = mocker.AsyncMock(
        return_value=0
    )
    mock_prisma_client.db.litellm_teammembership.delete_many = mocker.AsyncMock(
        return_value=0
    )
    mock_prisma_client.db.litellm_usertable.delete_many = mocker.AsyncMock(
        return_value=1
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Call delete_user
    data = DeleteUserRequest(user_ids=["admin-creator"])
    user_api_key_dict = UserAPIKeyAuth(
        user_id="proxy-admin", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    await delete_user(data=data, user_api_key_dict=user_api_key_dict)

    # Verify invitation link deletion uses OR with user_id, created_by, updated_by
    mock_prisma_client.db.litellm_invitationlink.delete_many.assert_called_once()
    call_kwargs = mock_prisma_client.db.litellm_invitationlink.delete_many.call_args
    where_clause = call_kwargs.kwargs.get("where") or call_kwargs[1].get("where")

    assert "OR" in where_clause, "Should use OR to match user_id, created_by, and updated_by"
    or_conditions = where_clause["OR"]
    assert len(or_conditions) == 3, "Should have 3 OR conditions"

    # Verify all three FK fields are covered
    condition_keys = [list(c.keys())[0] for c in or_conditions]
    assert "user_id" in condition_keys
    assert "created_by" in condition_keys
    assert "updated_by" in condition_keys

    # Verify each condition uses {"in": ["admin-creator"]}
    for condition in or_conditions:
        field = list(condition.keys())[0]
        assert condition[field] == {"in": ["admin-creator"]}


# =====================================================================
# /v2/user/info endpoint tests
# =====================================================================


@pytest.mark.asyncio
async def test_user_info_v2_proxy_admin_can_query_any_user(mocker):
    """
    Test that proxy admin can query any user via /v2/user/info.
    """
    from fastapi import Request

    from litellm.proxy._types import UserInfoV2Response
    from litellm.proxy.management_endpoints.internal_user_endpoints import user_info_v2

    mock_prisma_client = mocker.MagicMock()

    mock_user_row = mocker.MagicMock()
    mock_user_row.model_dump.return_value = {
        "user_id": "target-user-123",
        "user_email": "target@example.com",
        "user_alias": "Target User",
        "user_role": "internal_user",
        "spend": 42.5,
        "max_budget": 100.0,
        "models": ["gpt-4"],
        "budget_duration": "30d",
        "budget_reset_at": None,
        "metadata": {"team": "engineering"},
        "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2024, 6, 1, tzinfo=timezone.utc),
        "sso_user_id": "sso-abc",
        "teams": ["team-1", "team-2"],
    }

    async def mock_find_unique(*args, **kwargs):
        if kwargs.get("where", {}).get("user_id") == "target-user-123":
            return mock_user_row
        return None

    mock_prisma_client.db.litellm_usertable.find_unique = mocker.AsyncMock(
        side_effect=mock_find_unique
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mock_request = mocker.MagicMock(spec=Request)

    admin_key = UserAPIKeyAuth(
        user_id="admin-user", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    response = await user_info_v2(
        request=mock_request,
        user_id="target-user-123",
        user_api_key_dict=admin_key,
    )

    assert isinstance(response, UserInfoV2Response)
    assert response.user_id == "target-user-123"
    assert response.user_email == "target@example.com"
    assert response.user_alias == "Target User"
    assert response.user_role == "internal_user"
    assert response.spend == 42.5
    assert response.max_budget == 100.0
    assert response.models == ["gpt-4"]
    assert response.teams == ["team-1", "team-2"]
    assert response.sso_user_id == "sso-abc"
    assert response.metadata == {"team": "engineering"}


@pytest.mark.asyncio
async def test_user_info_v2_internal_user_can_query_self(mocker):
    """
    Test that an internal user can query their own info.
    """
    from fastapi import Request

    from litellm.proxy._types import UserInfoV2Response
    from litellm.proxy.management_endpoints.internal_user_endpoints import user_info_v2

    mock_prisma_client = mocker.MagicMock()

    mock_user_row = mocker.MagicMock()
    mock_user_row.model_dump.return_value = {
        "user_id": "self-user",
        "user_email": "self@example.com",
        "user_alias": None,
        "user_role": "internal_user",
        "spend": 10.0,
        "max_budget": None,
        "models": [],
        "budget_duration": None,
        "budget_reset_at": None,
        "metadata": None,
        "created_at": None,
        "updated_at": None,
        "sso_user_id": None,
        "teams": [],
    }

    async def mock_find_unique(*args, **kwargs):
        if kwargs.get("where", {}).get("user_id") == "self-user":
            return mock_user_row
        return None

    mock_prisma_client.db.litellm_usertable.find_unique = mocker.AsyncMock(
        side_effect=mock_find_unique
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mock_request = mocker.MagicMock(spec=Request)

    user_key = UserAPIKeyAuth(
        user_id="self-user", user_role=LitellmUserRoles.INTERNAL_USER
    )

    response = await user_info_v2(
        request=mock_request,
        user_id="self-user",
        user_api_key_dict=user_key,
    )

    assert isinstance(response, UserInfoV2Response)
    assert response.user_id == "self-user"
    assert response.user_email == "self@example.com"
    assert response.spend == 10.0


@pytest.mark.asyncio
async def test_user_info_v2_internal_user_cannot_query_other(mocker):
    """
    Test that an internal user cannot query another user - returns 404.
    """
    from fastapi import Request

    from litellm.proxy.management_endpoints.internal_user_endpoints import user_info_v2

    mock_prisma_client = mocker.MagicMock()

    # Caller user has no teams (so no team admin access)
    mock_caller_row = mocker.MagicMock()
    mock_caller_row.teams = []

    async def mock_find_unique(*args, **kwargs):
        user_id = kwargs.get("where", {}).get("user_id")
        if user_id == "caller-user":
            return mock_caller_row
        return None

    mock_prisma_client.db.litellm_usertable.find_unique = mocker.AsyncMock(
        side_effect=mock_find_unique
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mock_request = mocker.MagicMock(spec=Request)

    user_key = UserAPIKeyAuth(
        user_id="caller-user", user_role=LitellmUserRoles.INTERNAL_USER
    )

    with pytest.raises(ProxyException) as exc_info:
        await user_info_v2(
            request=mock_request,
            user_id="other-user-456",
            user_api_key_dict=user_key,
        )

    assert exc_info.value.code == "404"


@pytest.mark.asyncio
async def test_user_info_v2_no_user_id_defaults_to_self(mocker):
    """
    Test that omitting user_id defaults to the caller's own user info.
    """
    from fastapi import Request

    from litellm.proxy._types import UserInfoV2Response
    from litellm.proxy.management_endpoints.internal_user_endpoints import user_info_v2

    mock_prisma_client = mocker.MagicMock()

    mock_user_row = mocker.MagicMock()
    mock_user_row.model_dump.return_value = {
        "user_id": "my-user-id",
        "user_email": "me@example.com",
        "user_alias": None,
        "user_role": "internal_user",
        "spend": 0.0,
        "max_budget": None,
        "models": [],
        "budget_duration": None,
        "budget_reset_at": None,
        "metadata": None,
        "created_at": None,
        "updated_at": None,
        "sso_user_id": None,
        "teams": [],
    }

    async def mock_find_unique(*args, **kwargs):
        if kwargs.get("where", {}).get("user_id") == "my-user-id":
            return mock_user_row
        return None

    mock_prisma_client.db.litellm_usertable.find_unique = mocker.AsyncMock(
        side_effect=mock_find_unique
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mock_request = mocker.MagicMock(spec=Request)

    user_key = UserAPIKeyAuth(
        user_id="my-user-id", user_role=LitellmUserRoles.INTERNAL_USER
    )

    # Call without user_id
    response = await user_info_v2(
        request=mock_request,
        user_id=None,
        user_api_key_dict=user_key,
    )

    assert isinstance(response, UserInfoV2Response)
    assert response.user_id == "my-user-id"
    assert response.user_email == "me@example.com"


@pytest.mark.asyncio
async def test_user_info_v2_nonexistent_user_returns_404(mocker):
    """
    Test that querying a nonexistent user returns 404.
    """
    from fastapi import Request

    from litellm.proxy.management_endpoints.internal_user_endpoints import user_info_v2

    mock_prisma_client = mocker.MagicMock()

    async def mock_find_unique(*args, **kwargs):
        return None

    mock_prisma_client.db.litellm_usertable.find_unique = mocker.AsyncMock(
        side_effect=mock_find_unique
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mock_request = mocker.MagicMock(spec=Request)

    admin_key = UserAPIKeyAuth(
        user_id="admin-user", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    with pytest.raises(ProxyException) as exc_info:
        await user_info_v2(
            request=mock_request,
            user_id="nonexistent-user-id",
            user_api_key_dict=admin_key,
        )

    assert exc_info.value.code == "404"
    assert "nonexistent-user-id" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_user_info_v2_response_shape(mocker):
    """
    Test that the response shape contains expected fields and
    does NOT contain keys or teams objects (only team IDs).
    """
    from fastapi import Request

    from litellm.proxy._types import UserInfoV2Response
    from litellm.proxy.management_endpoints.internal_user_endpoints import user_info_v2

    mock_prisma_client = mocker.MagicMock()

    mock_user_row = mocker.MagicMock()
    mock_user_row.model_dump.return_value = {
        "user_id": "shape-test-user",
        "user_email": "shape@example.com",
        "user_alias": "Shape Test",
        "user_role": "internal_user",
        "spend": 5.0,
        "max_budget": 50.0,
        "models": ["gpt-3.5-turbo"],
        "budget_duration": "7d",
        "budget_reset_at": datetime(2024, 7, 1, tzinfo=timezone.utc),
        "metadata": {"env": "test"},
        "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
        "updated_at": datetime(2024, 6, 1, tzinfo=timezone.utc),
        "sso_user_id": None,
        "teams": ["team-a", "team-b"],
    }

    async def mock_find_unique(*args, **kwargs):
        return mock_user_row

    mock_prisma_client.db.litellm_usertable.find_unique = mocker.AsyncMock(
        side_effect=mock_find_unique
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mock_request = mocker.MagicMock(spec=Request)

    admin_key = UserAPIKeyAuth(
        user_id="admin-user", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    response = await user_info_v2(
        request=mock_request,
        user_id="shape-test-user",
        user_api_key_dict=admin_key,
    )

    assert isinstance(response, UserInfoV2Response)

    # Verify all expected fields are present
    response_dict = response.model_dump()
    expected_fields = {
        "user_id", "user_email", "user_alias", "user_role", "spend",
        "max_budget", "models", "budget_duration", "budget_reset_at",
        "metadata", "created_at", "updated_at", "sso_user_id", "teams",
    }
    assert set(response_dict.keys()) == expected_fields

    # Verify teams is a list of strings (team IDs), not team objects
    assert isinstance(response.teams, list)
    assert all(isinstance(t, str) for t in response.teams)
    assert response.teams == ["team-a", "team-b"]

    # Verify models is a list of strings
    assert isinstance(response.models, list)
    assert response.models == ["gpt-3.5-turbo"]


@pytest.mark.asyncio
async def test_user_info_v2_team_admin_can_query_team_member(mocker):
    """
    Test that a team admin can query info of a user in their team.
    """
    from fastapi import Request

    from litellm.proxy._types import LiteLLM_TeamTable, UserInfoV2Response
    from litellm.proxy.management_endpoints.internal_user_endpoints import user_info_v2

    mock_prisma_client = mocker.MagicMock()

    # Caller (team admin)
    mock_caller = mocker.MagicMock()
    mock_caller.teams = ["shared-team-id"]

    # Target user (team member)
    mock_target = mocker.MagicMock()
    mock_target.teams = ["shared-team-id"]
    mock_target.model_dump.return_value = {
        "user_id": "target-member",
        "user_email": "member@example.com",
        "user_alias": None,
        "user_role": "internal_user",
        "spend": 0.0,
        "max_budget": None,
        "models": [],
        "budget_duration": None,
        "budget_reset_at": None,
        "metadata": None,
        "created_at": None,
        "updated_at": None,
        "sso_user_id": None,
        "teams": ["shared-team-id"],
    }

    async def mock_find_unique(*args, **kwargs):
        uid = kwargs.get("where", {}).get("user_id")
        if uid == "team-admin-user":
            return mock_caller
        elif uid == "target-member":
            return mock_target
        return None

    mock_prisma_client.db.litellm_usertable.find_unique = mocker.AsyncMock(
        side_effect=mock_find_unique
    )

    # Mock team with caller as admin
    mock_team = mocker.MagicMock()
    mock_team.team_id = "shared-team-id"
    mock_team.model_dump.return_value = {
        "team_id": "shared-team-id",
        "team_alias": "Shared Team",
        "members_with_roles": [
            {"user_id": "team-admin-user", "role": "admin"},
            {"user_id": "target-member", "role": "user"},
        ],
    }

    async def mock_find_many_teams(*args, **kwargs):
        return [mock_team]

    mock_prisma_client.db.litellm_teamtable.find_many = mocker.AsyncMock(
        side_effect=mock_find_many_teams
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mock_request = mocker.MagicMock(spec=Request)

    team_admin_key = UserAPIKeyAuth(
        user_id="team-admin-user", user_role=LitellmUserRoles.INTERNAL_USER
    )

    response = await user_info_v2(
        request=mock_request,
        user_id="target-member",
        user_api_key_dict=team_admin_key,
    )

    assert isinstance(response, UserInfoV2Response)
    assert response.user_id == "target-member"
    assert response.user_email == "member@example.com"


@pytest.mark.asyncio
async def test_user_info_v2_team_admin_cannot_query_non_team_member(mocker):
    """
    Test that a team admin cannot query a user NOT in their team - returns 404.
    """
    from fastapi import Request

    from litellm.proxy.management_endpoints.internal_user_endpoints import user_info_v2

    mock_prisma_client = mocker.MagicMock()

    # Caller (team admin of team-A)
    mock_caller = mocker.MagicMock()
    mock_caller.teams = ["team-A"]

    # Target user (in team-B only)
    mock_target = mocker.MagicMock()
    mock_target.teams = ["team-B"]

    async def mock_find_unique(*args, **kwargs):
        uid = kwargs.get("where", {}).get("user_id")
        if uid == "team-admin-user":
            return mock_caller
        elif uid == "non-member-user":
            return mock_target
        return None

    mock_prisma_client.db.litellm_usertable.find_unique = mocker.AsyncMock(
        side_effect=mock_find_unique
    )

    # Mock team where caller is admin
    mock_team = mocker.MagicMock()
    mock_team.team_id = "team-A"
    mock_team.model_dump.return_value = {
        "team_id": "team-A",
        "team_alias": "Team A",
        "members_with_roles": [
            {"user_id": "team-admin-user", "role": "admin"},
        ],
    }

    async def mock_find_many_teams(*args, **kwargs):
        return [mock_team]

    mock_prisma_client.db.litellm_teamtable.find_many = mocker.AsyncMock(
        side_effect=mock_find_many_teams
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mock_request = mocker.MagicMock(spec=Request)

    team_admin_key = UserAPIKeyAuth(
        user_id="team-admin-user", user_role=LitellmUserRoles.INTERNAL_USER
    )

    with pytest.raises(ProxyException) as exc_info:
        await user_info_v2(
            request=mock_request,
            user_id="non-member-user",
            user_api_key_dict=team_admin_key,
        )

    assert exc_info.value.code == "404"


@pytest.mark.asyncio
async def test_user_info_v2_url_encoding_plus_character(mocker):
    """
    Test that /v2/user/info properly handles email addresses with + characters.
    """
    from fastapi import Request

    from litellm.proxy._types import UserInfoV2Response
    from litellm.proxy.management_endpoints.internal_user_endpoints import user_info_v2

    mock_prisma_client = mocker.MagicMock()

    expected_user_id = "machine-user+admin@example.com"

    mock_user_row = mocker.MagicMock()
    mock_user_row.model_dump.return_value = {
        "user_id": expected_user_id,
        "user_email": expected_user_id,
        "user_alias": None,
        "user_role": "internal_user",
        "spend": 0.0,
        "max_budget": None,
        "models": [],
        "budget_duration": None,
        "budget_reset_at": None,
        "metadata": None,
        "created_at": None,
        "updated_at": None,
        "sso_user_id": None,
        "teams": [],
    }

    async def mock_find_unique(*args, **kwargs):
        uid = kwargs.get("where", {}).get("user_id")
        if uid == expected_user_id:
            return mock_user_row
        return None

    mock_prisma_client.db.litellm_usertable.find_unique = mocker.AsyncMock(
        side_effect=mock_find_unique
    )

    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    mock_request = mocker.MagicMock(spec=Request)
    mock_request.url.query = f"user_id={expected_user_id}"

    admin_key = UserAPIKeyAuth(
        user_id="admin-user", user_role=LitellmUserRoles.PROXY_ADMIN
    )

    # Simulate FastAPI converting + to space
    decoded_user_id = "machine-user admin@example.com"

    response = await user_info_v2(
        request=mock_request,
        user_id=decoded_user_id,
        user_api_key_dict=admin_key,
    )

    assert isinstance(response, UserInfoV2Response)
    assert response.user_id == expected_user_id