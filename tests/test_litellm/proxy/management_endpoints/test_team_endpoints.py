import asyncio
import json
import os
import sys
import uuid
from typing import Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../")
)  # Adds the parent directory to the system path
from litellm.proxy._types import UserAPIKeyAuth  # Import UserAPIKeyAuth
from litellm.proxy._types import (
    LiteLLM_OrganizationTable,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    Member,
    ProxyErrorTypes,
    ProxyException,
    TeamMemberAddRequest,
)
from litellm.proxy.management_endpoints.team_endpoints import (
    user_api_key_auth,  # Assuming this dependency is needed
)
from litellm.proxy.management_endpoints.team_endpoints import (
    GetTeamMemberPermissionsResponse,
    UpdateTeamMemberPermissionsRequest,
    router,
    team_member_add_duplication_check,
    validate_team_org_change,
)
from litellm.proxy.management_helpers.team_member_permission_checks import (
    TeamMemberPermissionChecks,
)
from litellm.proxy.proxy_server import app
from litellm.router import Router

# Setup TestClient
client = TestClient(app)

# Mock prisma_client
mock_prisma_client = MagicMock()


# Fixture to provide the mock prisma client
@pytest.fixture(autouse=True)
def mock_db_client():
    with patch(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    ):  # Mock in both places if necessary
        yield mock_prisma_client
    mock_prisma_client.reset_mock()


# Fixture to provide a mock admin user auth object
@pytest.fixture
def mock_admin_auth():
    mock_auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    return mock_auth


# Test for validate_team_org_change when organization IDs match
@pytest.mark.asyncio
async def test_validate_team_org_change_same_org_id():
    """
    Test that validate_team_org_change returns True without performing any checks
    when the team and organization have the same organization_id.

    This is a user issue, a user was editing their team and this function raised an exception even when they were not changing the organization.
    """
    # Create mock team and organization with same org ID
    org_id = "test-org-123"

    # Mock team
    team = MagicMock(spec=LiteLLM_TeamTable)
    team.organization_id = org_id
    team.models = ["gpt-4", "claude-2"]
    team.max_budget = 100.0
    team.tpm_limit = 1000
    team.rpm_limit = 100
    team.members_with_roles = []

    # Mock organization
    organization = MagicMock(spec=LiteLLM_OrganizationTable)
    organization.organization_id = org_id
    organization.models = []
    organization.litellm_budget_table = MagicMock()
    organization.litellm_budget_table.max_budget = (
        50.0  # This would normally fail validation
    )
    organization.litellm_budget_table.tpm_limit = (
        500  # This would normally fail validation
    )
    organization.litellm_budget_table.rpm_limit = (
        50  # This would normally fail validation
    )
    organization.users = []

    # Mock Router
    mock_router = MagicMock(spec=Router)

    # Use patch to ensure the model access check is never called
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.can_org_access_model"
    ) as mock_access_check:
        result = validate_team_org_change(
            team=team, organization=organization, llm_router=mock_router
        )

        # Assert the function returns True without checking anything
        assert result is True
        mock_access_check.assert_not_called()  # Ensure access check wasn't called


# Test for /team/permissions_list endpoint (GET)
@pytest.mark.asyncio
async def test_get_team_permissions_list_success(mock_db_client, mock_admin_auth):
    """
    Test successful retrieval of team member permissions.
    """
    test_team_id = "test-team-123"
    permissions = ["/key/generate", "/key/update"]
    mock_team_data = {
        "team_id": test_team_id,
        "team_alias": "Test Team",
        "team_member_permissions": permissions,
        "spend": 0.0,
    }
    mock_team_row = MagicMock()
    mock_team_row.model_dump.return_value = mock_team_data

    # Set attributes directly on the mock object
    mock_team_row.team_id = test_team_id
    mock_team_row.team_alias = "Test Team"
    mock_team_row.team_member_permissions = permissions
    mock_team_row.spend = 0.0

    # Mock the get_team_object function used in the endpoint
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        new_callable=AsyncMock,
        return_value=mock_team_row,
    ):
        # Override the dependency for this test
        app.dependency_overrides[user_api_key_auth] = lambda: mock_admin_auth

        response = client.get(f"/team/permissions_list?team_id={test_team_id}")

        assert response.status_code == 200
        response_data = response.json()
        assert response_data["team_id"] == test_team_id
        assert (
            response_data["team_member_permissions"]
            == mock_team_data["team_member_permissions"]
        )
        assert (
            response_data["all_available_permissions"]
            == TeamMemberPermissionChecks.get_all_available_team_member_permissions()
        )

        # Clean up dependency override
        app.dependency_overrides = {}


# Test for /team/permissions_update endpoint (POST)
@pytest.mark.asyncio
async def test_update_team_permissions_success(mock_db_client, mock_admin_auth):
    """
    Test successful update of team member permissions by an admin.
    """
    test_team_id = "test-team-456"
    update_permissions = ["/key/generate", "/key/update"]
    update_payload = {
        "team_id": test_team_id,
        "team_member_permissions": update_permissions,
    }

    existing_permissions = ["/key/list"]
    mock_existing_team_data = {
        "team_id": test_team_id,
        "team_alias": "Existing Team",
        "team_member_permissions": existing_permissions,
        "spend": 0.0,
        "models": [],
    }
    mock_updated_team_data = {
        **mock_existing_team_data,
        "team_member_permissions": update_payload["team_member_permissions"],
    }

    mock_existing_team_row = MagicMock(spec=LiteLLM_TeamTable)
    mock_existing_team_row.model_dump.return_value = mock_existing_team_data

    # Set attributes directly on the existing team mock
    mock_existing_team_row.team_id = test_team_id
    mock_existing_team_row.team_alias = "Existing Team"
    mock_existing_team_row.team_member_permissions = existing_permissions
    mock_existing_team_row.spend = 0.0
    mock_existing_team_row.models = []

    mock_updated_team_row = MagicMock(spec=LiteLLM_TeamTable)
    mock_updated_team_row.model_dump.return_value = mock_updated_team_data

    # Set attributes directly on the updated team mock
    mock_updated_team_row.team_id = test_team_id
    mock_updated_team_row.team_alias = "Existing Team"
    mock_updated_team_row.team_member_permissions = update_permissions
    mock_updated_team_row.spend = 0.0
    mock_updated_team_row.models = []

    # Mock the get_team_object function used in the endpoint
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        new_callable=AsyncMock,
        return_value=mock_existing_team_row,
    ):
        # Mock the database update function
        mock_db_client.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team_row
        )

        # Override the dependency for this test
        app.dependency_overrides[user_api_key_auth] = lambda: mock_admin_auth

        response = client.post("/team/permissions_update", json=update_payload)

        assert response.status_code == 200
        response_data = response.json()

        # Use model_dump for comparison if the endpoint returns the Prisma model directly
        assert response_data == mock_updated_team_row.model_dump()

        mock_db_client.db.litellm_teamtable.update.assert_awaited_once_with(
            where={"team_id": test_team_id},
            data={"team_member_permissions": update_payload["team_member_permissions"]},
        )

        # Clean up dependency override
        app.dependency_overrides = {}


@pytest.mark.asyncio
async def test_new_team_with_object_permission(mock_db_client, mock_admin_auth):
    """Ensure /team/new correctly handles `object_permission` by
    1. Creating a record in litellm_objectpermissiontable
    2. Passing the returned `object_permission_id` into the team insert payload
    """
    # --- Configure mocked prisma client ---
    # Helper identity converters used by team logic
    mock_db_client.jsonify_team_object = lambda db_data: db_data  # type: ignore
    mock_db_client.get_data = AsyncMock(return_value=None)
    mock_db_client.update_data = AsyncMock(return_value=MagicMock())

    # Mock DB structure under prisma_client.db
    mock_db_client.db = MagicMock()

    # 1. Mock object permission table creation
    mock_object_perm_create = AsyncMock(
        return_value=MagicMock(object_permission_id="objperm123")
    )
    mock_db_client.db.litellm_objectpermissiontable = MagicMock()
    mock_db_client.db.litellm_objectpermissiontable.create = mock_object_perm_create

    # 2. Mock model table creation (may be skipped but provided for safety)
    mock_db_client.db.litellm_modeltable = MagicMock()
    mock_db_client.db.litellm_modeltable.create = AsyncMock(
        return_value=MagicMock(id="model123")
    )

    # 3. Capture team table creation and count
    team_create_result = MagicMock(
        team_id="team-456",
        object_permission_id="objperm123",
    )
    team_create_result.model_dump.return_value = {
        "team_id": "team-456",
        "object_permission_id": "objperm123",
    }
    mock_team_create = AsyncMock(return_value=team_create_result)
    mock_team_count = AsyncMock(
        return_value=0
    )  # Mock count to return 0 (no existing teams)
    mock_db_client.db.litellm_teamtable = MagicMock()
    mock_db_client.db.litellm_teamtable.create = mock_team_create
    mock_db_client.db.litellm_teamtable.count = mock_team_count

    # 4. Mock user table update behaviour (called for each member)
    mock_db_client.db.litellm_usertable = MagicMock()
    mock_db_client.db.litellm_usertable.update = AsyncMock(return_value=MagicMock())

    # --- Import after mocks applied ---
    from fastapi import Request

    from litellm.proxy._types import LiteLLM_ObjectPermissionBase, NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Build request objects
    team_request = NewTeamRequest(
        team_alias="my-team",
        object_permission=LiteLLM_ObjectPermissionBase(vector_stores=["my-vector"]),
    )

    # Pass a dummy FastAPI Request object
    dummy_request = MagicMock(spec=Request)

    # Execute the endpoint function
    await new_team(
        data=team_request,
        http_request=dummy_request,
        user_api_key_dict=mock_admin_auth,
    )

    # --- Assertions ---
    # 1. Object permission creation should be called exactly once
    mock_object_perm_create.assert_awaited_once()

    # 2. Team creation payload should include the generated object_permission_id
    assert mock_team_create.call_count == 1
    created_team_kwargs = mock_team_create.call_args.kwargs
    assert created_team_kwargs["data"].get("object_permission_id") == "objperm123"


@pytest.mark.asyncio
async def test_team_update_object_permissions_existing_permission(monkeypatch):
    """
    Test updating object permissions when a team already has an existing object_permission_id.

    This test verifies that when updating vector stores for a team that already has an
    object_permission_id, the existing LiteLLM_ObjectPermissionTable record is updated
    with the new permissions and the object_permission_id remains the same.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import LiteLLM_ObjectPermissionBase, LiteLLM_TeamTable
    from litellm.proxy.management_endpoints.team_endpoints import (
        handle_update_object_permission,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    # Mock existing team with object_permission_id
    existing_team_row = LiteLLM_TeamTable(
        team_id="test_team_id",
        object_permission_id="existing_perm_id_123",
        team_alias="test_team",
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
        "team_alias": "updated_team",
    }

    # Call the function
    result = await handle_update_object_permission(
        data_json=data_json,
        existing_team_row=existing_team_row,
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
async def test_team_update_object_permissions_no_existing_permission(monkeypatch):
    """
    Test creating object permissions when a team has no existing object_permission_id.

    This test verifies that when updating object permissions for a team that has
    object_permission_id set to None, a new entry is created in the
    LiteLLM_ObjectPermissionTable and the team is updated with the new object_permission_id.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import LiteLLM_ObjectPermissionBase, LiteLLM_TeamTable
    from litellm.proxy.management_endpoints.team_endpoints import (
        handle_update_object_permission,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    existing_team_row_no_perm = LiteLLM_TeamTable(
        team_id="test_team_id_2",
        object_permission_id=None,
        team_alias="test_team_2",
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
        "team_alias": "updated_team_2",
    }

    result = await handle_update_object_permission(
        data_json=data_json,
        existing_team_row=existing_team_row_no_perm,
    )

    # Verify new object_permission_id was set
    assert "object_permission" not in result
    assert result["object_permission_id"] == "new_perm_id_456"

    # Verify upsert was called to create new record
    mock_prisma_client.db.litellm_objectpermissiontable.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_team_update_object_permissions_missing_permission_record(monkeypatch):
    """
    Test creating object permissions when existing object_permission_id record is not found.

    This test verifies that when updating object permissions for a team that has an
    object_permission_id but the corresponding record cannot be found in the database,
    a new entry is created in the LiteLLM_ObjectPermissionTable with the new permissions.
    """
    from unittest.mock import AsyncMock, MagicMock

    import pytest

    from litellm.proxy._types import LiteLLM_ObjectPermissionBase, LiteLLM_TeamTable
    from litellm.proxy.management_endpoints.team_endpoints import (
        handle_update_object_permission,
    )

    # Mock prisma client
    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)

    existing_team_row_missing_perm = LiteLLM_TeamTable(
        team_id="test_team_id_3",
        object_permission_id="missing_perm_id_789",
        team_alias="test_team_3",
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
        "team_alias": "updated_team_3",
    }

    result = await handle_update_object_permission(
        data_json=data_json,
        existing_team_row=existing_team_row_missing_perm,
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


def test_team_member_add_duplication_check_raises_proxy_exception():
    """
    Test that team_member_add_duplication_check raises ProxyException when a user is already in the team
    """
    # Create a mock team with existing members
    existing_team_row = MagicMock(spec=LiteLLM_TeamTable)
    existing_team_row.team_id = "test-team-123"
    existing_team_row.members_with_roles = [
        Member(user_id="existing-user-id", role="user"),
        Member(user_id="another-user-id", role="admin"),
    ]

    # Create a request to add a member who is already in the team
    duplicate_member = Member(user_id="existing-user-id", role="user")
    data = TeamMemberAddRequest(
        team_id="test-team-123",
        member=duplicate_member,
    )

    # Test that ProxyException is raised with the correct error type
    with pytest.raises(ProxyException) as exc_info:
        team_member_add_duplication_check(
            data=data,
            existing_team_row=existing_team_row,
        )

    # Verify the exception details
    assert exc_info.value.type == ProxyErrorTypes.team_member_already_in_team
    assert exc_info.value.param == "user_id"
    assert exc_info.value.code == "400"
    assert "existing-user-id" in str(exc_info.value.message)
    assert "already in team" in str(exc_info.value.message)


def test_team_member_add_duplication_check_allows_new_member():
    """
    Test that team_member_add_duplication_check allows adding a new member who is not already in the team
    """
    # Create a mock team with existing members
    existing_team_row = MagicMock(spec=LiteLLM_TeamTable)
    existing_team_row.team_id = "test-team-123"
    existing_team_row.members_with_roles = [
        Member(user_id="existing-user-id", role="user"),
        Member(user_id="another-user-id", role="admin"),
    ]

    # Create a request to add a member who is NOT already in the team
    new_member = Member(user_id="new-user-id", role="user")
    data = TeamMemberAddRequest(
        team_id="test-team-123",
        member=new_member,
    )

    # Test that no exception is raised for a new member
    try:
        team_member_add_duplication_check(
            data=data,
            existing_team_row=existing_team_row,
        )
        # If we reach here, no exception was raised, which is expected
        assert True
    except ProxyException:
        # If a ProxyException is raised, the test should fail
        pytest.fail("ProxyException should not be raised for a new member")


@pytest.mark.asyncio
async def test_add_team_member_budget_table_success():
    """
    Test _add_team_member_budget_table when budget is found successfully
    """
    from litellm.proxy._types import TeamInfoResponseObjectTeamTable
    from litellm.proxy.management_endpoints.team_endpoints import (
        _add_team_member_budget_table,
    )

    # Mock prisma client
    mock_prisma_client = MagicMock()

    # Mock budget record
    mock_budget_record = MagicMock()
    mock_budget_record.budget_id = "budget-123"
    mock_budget_record.max_budget = 1000.0

    mock_prisma_client.db.litellm_budgettable.find_unique = AsyncMock(
        return_value=mock_budget_record
    )

    # Create team info response object
    team_info_response = TeamInfoResponseObjectTeamTable(
        team_id="test-team-123", team_alias="Test Team"
    )

    # Call the function
    result = await _add_team_member_budget_table(
        team_member_budget_id="budget-123",
        prisma_client=mock_prisma_client,
        team_info_response_object=team_info_response,
    )

    # Verify the result
    assert result == team_info_response
    assert result.team_member_budget_table == mock_budget_record

    # Verify database call was made correctly
    mock_prisma_client.db.litellm_budgettable.find_unique.assert_called_once_with(
        where={"budget_id": "budget-123"}
    )


@pytest.mark.asyncio
async def test_add_team_member_budget_table_exception_handling():
    """
    Test _add_team_member_budget_table when an exception occurs during budget lookup
    """
    from litellm.proxy._types import TeamInfoResponseObjectTeamTable
    from litellm.proxy.management_endpoints.team_endpoints import (
        _add_team_member_budget_table,
    )

    # Mock prisma client to raise an exception
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_budgettable.find_unique = AsyncMock(
        side_effect=Exception("Database connection failed")
    )

    # Create team info response object
    team_info_response = TeamInfoResponseObjectTeamTable(
        team_id="test-team-456", team_alias="Test Team 2"
    )

    # Mock the verbose_proxy_logger to capture log calls
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.verbose_proxy_logger"
    ) as mock_logger:
        # Call the function
        result = await _add_team_member_budget_table(
            team_member_budget_id="nonexistent-budget-456",
            prisma_client=mock_prisma_client,
            team_info_response_object=team_info_response,
        )

        # Verify the result is returned even when exception occurs
        assert result == team_info_response

        # Verify team_member_budget_table is not set when exception occurs
        assert (
            not hasattr(result, "team_member_budget_table")
            or result.team_member_budget_table is None
        )

        # Verify the error was logged
        mock_logger.info.assert_called_once_with(
            "Team member budget table not found, passed team_member_budget_id=nonexistent-budget-456"
        )

        # Verify database call was attempted
        mock_prisma_client.db.litellm_budgettable.find_unique.assert_called_once_with(
            where={"budget_id": "nonexistent-budget-456"}
        )


@pytest.mark.asyncio
async def test_add_team_member_budget_table_budget_not_found():
    """
    Test _add_team_member_budget_table when budget record is not found (returns None)
    """
    from litellm.proxy._types import TeamInfoResponseObjectTeamTable
    from litellm.proxy.management_endpoints.team_endpoints import (
        _add_team_member_budget_table,
    )

    # Mock prisma client to return None (budget not found)
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_budgettable.find_unique = AsyncMock(return_value=None)

    # Create team info response object
    team_info_response = TeamInfoResponseObjectTeamTable(
        team_id="test-team-789", team_alias="Test Team 3"
    )

    # Call the function
    result = await _add_team_member_budget_table(
        team_member_budget_id="nonexistent-budget-789",
        prisma_client=mock_prisma_client,
        team_info_response_object=team_info_response,
    )

    # Verify the result
    assert result == team_info_response
    assert result.team_member_budget_table is None

    # Verify database call was made correctly
    mock_prisma_client.db.litellm_budgettable.find_unique.assert_called_once_with(
        where={"budget_id": "nonexistent-budget-789"}
    )


def test_add_new_models_to_team():
    """
    Test add_new_models_to_team function
    """
    from litellm.proxy._types import SpecialModelNames
    from litellm.proxy.management_endpoints.team_endpoints import add_new_models_to_team

    team_obj = MagicMock(spec=LiteLLM_TeamTable)
    team_obj.models = []
    new_models = ["model4", "model5"]
    updated_models = add_new_models_to_team(team_obj=team_obj, new_models=new_models)
    assert (
        updated_models.sort()
        == [
            SpecialModelNames.all_proxy_models.value,
            "model4",
            "model5",
        ].sort()
    )
