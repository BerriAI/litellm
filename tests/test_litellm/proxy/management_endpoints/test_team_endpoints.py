import asyncio
import json
import os
import sys
from typing import Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from litellm._uuid import uuid

sys.path.insert(
    0, os.path.abspath("../../../")
)  # Adds the parent directory to the system path
from litellm.proxy._types import UserAPIKeyAuth  # Import UserAPIKeyAuth
from litellm.proxy._types import (
    LiteLLM_OrganizationMembershipTable,
    LiteLLM_OrganizationTable,
    LiteLLM_OrganizationTableWithMembers,
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
from litellm.types.proxy.management_endpoints.team_endpoints import (
    BulkTeamMemberAddRequest,
    BulkTeamMemberAddResponse,
    TeamMemberAddResult,
)

# Setup TestClient
client = TestClient(app)

# Mock prisma_client
mock_prisma_client = MagicMock()
# Set up async mock for db operations
mock_prisma_client.db = MagicMock()
mock_prisma_client.db.litellm_teamtable = MagicMock()
mock_prisma_client.db.litellm_teamtable.update = AsyncMock()


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
    organization = MagicMock(spec=LiteLLM_OrganizationTableWithMembers)
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
    organization.members = []

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


@pytest.mark.asyncio
async def test_validate_team_org_change_members_in_org():
    """
    Test that validate_team_org_change passes when team members are in organization.members.

    This tests the fix for issue #17552 where membership was incorrectly checked against
    organization.users (deprecated) instead of organization.members (correct).
    """
    team_org_id = "team-org-123"
    new_org_id = "new-org-456"
    user_id_1 = "user-123"
    user_id_2 = "user-456"

    # Mock team with members
    team = MagicMock(spec=LiteLLM_TeamTable)
    team.organization_id = team_org_id
    team.models = []
    team.max_budget = None
    team.tpm_limit = None
    team.rpm_limit = None

    # Create mock team members
    team_member_1 = MagicMock()
    team_member_1.user_id = user_id_1
    team_member_2 = MagicMock()
    team_member_2.user_id = user_id_2
    team.members_with_roles = [team_member_1, team_member_2]

    # Mock organization with members (using LiteLLM_OrganizationMembershipTable structure)
    organization = MagicMock(spec=LiteLLM_OrganizationTableWithMembers)
    organization.organization_id = new_org_id
    organization.models = []
    organization.litellm_budget_table = None

    # Create mock organization members - these should match team members
    org_member_1 = MagicMock(spec=LiteLLM_OrganizationMembershipTable)
    org_member_1.user_id = user_id_1
    org_member_2 = MagicMock(spec=LiteLLM_OrganizationMembershipTable)
    org_member_2.user_id = user_id_2
    organization.members = [org_member_1, org_member_2]

    # Mock Router
    mock_router = MagicMock(spec=Router)

    # Test should pass - all team members are in org members
    result = validate_team_org_change(
        team=team, organization=organization, llm_router=mock_router
    )
    assert result is True


@pytest.mark.asyncio
async def test_validate_team_org_change_member_not_in_org():
    """
    Test that validate_team_org_change raises HTTPException when team members
    are NOT in organization.members.

    This tests the fix for issue #17552 where membership was incorrectly checked against
    organization.users (deprecated) instead of organization.members (correct).
    """
    team_org_id = "team-org-123"
    new_org_id = "new-org-456"
    user_id_1 = "user-123"
    user_id_2 = "user-456"
    user_id_not_in_org = "user-not-in-org-789"

    # Mock team with members (including one not in org)
    team = MagicMock(spec=LiteLLM_TeamTable)
    team.organization_id = team_org_id
    team.models = []
    team.max_budget = None
    team.tpm_limit = None
    team.rpm_limit = None

    # Create mock team members - user_id_not_in_org is not in the org
    team_member_1 = MagicMock()
    team_member_1.user_id = user_id_1
    team_member_2 = MagicMock()
    team_member_2.user_id = user_id_not_in_org
    team.members_with_roles = [team_member_1, team_member_2]

    # Mock organization with members (missing user_id_not_in_org)
    organization = MagicMock(spec=LiteLLM_OrganizationTableWithMembers)
    organization.organization_id = new_org_id
    organization.models = []
    organization.litellm_budget_table = None

    # Create mock organization members - only user_id_1 and user_id_2 are members
    org_member_1 = MagicMock(spec=LiteLLM_OrganizationMembershipTable)
    org_member_1.user_id = user_id_1
    org_member_2 = MagicMock(spec=LiteLLM_OrganizationMembershipTable)
    org_member_2.user_id = user_id_2
    organization.members = [org_member_1, org_member_2]

    # Mock Router
    mock_router = MagicMock(spec=Router)

    # Test should fail - user_id_not_in_org is not in org members
    with pytest.raises(HTTPException) as exc_info:
        validate_team_org_change(
            team=team, organization=organization, llm_router=mock_router
        )

    assert exc_info.value.status_code == 403
    assert "not a member of the organization" in str(exc_info.value.detail)
    assert user_id_not_in_org in str(exc_info.value.detail)


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
    """
    Test that /team/new correctly handles object_permission by:
    1. Creating a record in litellm_objectpermissiontable
    2. Passing the returned object_permission_id into the team insert payload
    3. NOT passing the object_permission dict to the team table
    """
    # Configure mocked prisma client
    mock_db_client.jsonify_team_object = lambda db_data: db_data
    mock_db_client.get_data = AsyncMock(return_value=None)
    mock_db_client.update_data = AsyncMock(return_value=MagicMock())
    mock_db_client.db = MagicMock()

    # Mock object permission table creation
    mock_object_perm_create = AsyncMock(
        return_value=MagicMock(object_permission_id="objperm123")
    )
    mock_db_client.db.litellm_objectpermissiontable = MagicMock()
    mock_db_client.db.litellm_objectpermissiontable.create = mock_object_perm_create

    # Mock model table creation
    mock_db_client.db.litellm_modeltable = MagicMock()
    mock_db_client.db.litellm_modeltable.create = AsyncMock(
        return_value=MagicMock(id="model123")
    )

    # Capture team table creation
    team_create_result = MagicMock(
        team_id="team-456",
        object_permission_id="objperm123",
    )
    team_create_result.model_dump.return_value = {
        "team_id": "team-456",
        "object_permission_id": "objperm123",
    }
    mock_team_create = AsyncMock(return_value=team_create_result)
    mock_team_count = AsyncMock(return_value=0)
    mock_db_client.db.litellm_teamtable = MagicMock()
    mock_db_client.db.litellm_teamtable.create = mock_team_create
    mock_db_client.db.litellm_teamtable.count = mock_team_count
    mock_db_client.db.litellm_teamtable.update = AsyncMock(
        return_value=team_create_result
    )

    # Mock user table
    mock_db_client.db.litellm_usertable = MagicMock()
    mock_db_client.db.litellm_usertable.update = AsyncMock(return_value=MagicMock())

    from fastapi import Request

    from litellm.proxy._types import LiteLLM_ObjectPermissionBase, NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Build request with object_permission
    team_request = NewTeamRequest(
        team_alias="my-team",
        object_permission=LiteLLM_ObjectPermissionBase(vector_stores=["my-vector"]),
    )

    dummy_request = MagicMock(spec=Request)

    # Execute the endpoint function
    await new_team(
        data=team_request,
        http_request=dummy_request,
        user_api_key_dict=mock_admin_auth,
    )

    # Verify object permission creation was called
    mock_object_perm_create.assert_awaited_once()

    # Verify team creation was called
    assert mock_team_create.call_count == 1
    created_team_kwargs = mock_team_create.call_args.kwargs
    team_data = created_team_kwargs["data"]
    
    # Verify object_permission_id is in the team data
    assert team_data.get("object_permission_id") == "objperm123"
    
    # Verify object_permission dict is NOT in the team data
    assert "object_permission" not in team_data


@pytest.mark.asyncio
async def test_new_team_with_mcp_tool_permissions(mock_db_client, mock_admin_auth):
    """
    Test that /team/new correctly handles mcp_tool_permissions in object_permission.
    
    This test verifies that:
    1. mcp_tool_permissions is accepted in the object_permission field
    2. The field is properly stored in the LiteLLM_ObjectPermissionTable
    3. The team is correctly linked to the object_permission record
    """
    # Configure mocked prisma client
    mock_db_client.jsonify_team_object = lambda db_data: db_data
    mock_db_client.get_data = AsyncMock(return_value=None)
    mock_db_client.update_data = AsyncMock(return_value=MagicMock())
    mock_db_client.db = MagicMock()

    # Track what data is passed to object permission create
    created_permission_data = {}

    async def mock_obj_perm_create(**kwargs):
        created_permission_data.update(kwargs.get("data", {}))
        return MagicMock(object_permission_id="objperm_team_mcp_456")

    mock_db_client.db.litellm_objectpermissiontable = MagicMock()
    mock_db_client.db.litellm_objectpermissiontable.create = mock_obj_perm_create

    # Mock model table
    mock_db_client.db.litellm_modeltable = MagicMock()
    mock_db_client.db.litellm_modeltable.create = AsyncMock(
        return_value=MagicMock(id="model456")
    )

    # Mock team table
    team_create_result = MagicMock(
        team_id="team-mcp-789",
        object_permission_id="objperm_team_mcp_456",
    )
    team_create_result.model_dump.return_value = {
        "team_id": "team-mcp-789",
        "object_permission_id": "objperm_team_mcp_456",
    }
    mock_db_client.db.litellm_teamtable = MagicMock()
    mock_db_client.db.litellm_teamtable.create = AsyncMock(return_value=team_create_result)
    mock_db_client.db.litellm_teamtable.count = AsyncMock(return_value=0)
    mock_db_client.db.litellm_teamtable.update = AsyncMock(return_value=team_create_result)

    # Mock user table
    mock_db_client.db.litellm_usertable = MagicMock()
    mock_db_client.db.litellm_usertable.update = AsyncMock(return_value=MagicMock())

    from fastapi import Request

    from litellm.proxy._types import LiteLLM_ObjectPermissionBase, NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Create team with mcp_tool_permissions
    team_request = NewTeamRequest(
        team_alias="mcp-team",
        object_permission=LiteLLM_ObjectPermissionBase(
            mcp_servers=["server_a", "server_b"],
            mcp_tool_permissions={
                "server_a": ["read_wiki_structure", "read_wiki_contents"],
                "server_b": ["ask_question"],
            },
        ),
    )

    dummy_request = MagicMock(spec=Request)

    await new_team(
        data=team_request,
        http_request=dummy_request,
        user_api_key_dict=mock_admin_auth,
    )

    # Verify mcp_tool_permissions was stored
    import json
    assert "mcp_tool_permissions" in created_permission_data
    # mcp_tool_permissions is stored as a JSON string
    assert json.loads(created_permission_data["mcp_tool_permissions"]) == {
        "server_a": ["read_wiki_structure", "read_wiki_contents"],
        "server_b": ["ask_question"],
    }
    assert created_permission_data["mcp_servers"] == ["server_a", "server_b"]


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
    assert exc_info.value.param == "member"
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


@pytest.mark.asyncio
async def test_validate_team_member_add_permissions_admin():
    """
    Test _validate_team_member_add_permissions allows proxy admin
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_team_member_add_permissions,
    )

    # Create admin user
    admin_user = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    # Create mock team
    team = MagicMock(spec=LiteLLM_TeamTable)
    team.team_id = "test-team-123"

    # Should not raise any exception for admin
    await _validate_team_member_add_permissions(
        user_api_key_dict=admin_user,
        complete_team_data=team,
    )


@pytest.mark.asyncio
async def test_validate_team_member_add_permissions_non_admin():
    """
    Test _validate_team_member_add_permissions raises exception for non-admin non-team-admin
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_team_member_add_permissions,
    )

    # Create non-admin user
    regular_user = UserAPIKeyAuth(
        user_id="regular-user",
        user_role=LitellmUserRoles.INTERNAL_USER,
        team_id="different-team",
    )

    # Create mock team
    team = MagicMock(spec=LiteLLM_TeamTable)
    team.team_id = "test-team-123"
    team.members_with_roles = []

    # Mock the helper functions to return False
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
        return_value=False,
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._is_available_team",
        return_value=False,
    ):
        # Should raise HTTPException for non-admin
        with pytest.raises(HTTPException) as exc_info:
            await _validate_team_member_add_permissions(
                user_api_key_dict=regular_user,
                complete_team_data=team,
            )

        assert exc_info.value.status_code == 403
        assert "not proxy admin OR team admin" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_process_team_members_single_member():
    """
    Test _process_team_members with a single member
    """
    from litellm.proxy._types import LiteLLM_TeamMembership, LiteLLM_UserTable
    from litellm.proxy.management_endpoints.team_endpoints import _process_team_members

    # Mock dependencies
    mock_prisma_client = MagicMock()
    mock_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_team.metadata = {"team_member_budget_id": "budget-123"}

    # Mock user and membership objects
    mock_user = MagicMock(spec=LiteLLM_UserTable)
    mock_user.user_id = "new-user-123"
    mock_membership = MagicMock(spec=LiteLLM_TeamMembership)

    # Create request with single member
    single_member = Member(user_email="new@example.com", role="user")
    request_data = TeamMemberAddRequest(
        team_id="test-team-123",
        member=single_member,
    )

    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.add_new_member",
        new_callable=AsyncMock,
        return_value=(mock_user, mock_membership),
    ) as mock_add_member:
        users, memberships = await _process_team_members(
            data=request_data,
            complete_team_data=mock_team,
            prisma_client=mock_prisma_client,
            user_api_key_dict=UserAPIKeyAuth(),
            litellm_proxy_admin_name="admin",
        )

        # Verify results
        assert len(users) == 1
        assert len(memberships) == 1
        assert users[0] == mock_user
        assert memberships[0] == mock_membership

        # Verify add_new_member was called correctly
        mock_add_member.assert_called_once_with(
            new_member=single_member,
            max_budget_in_team=None,
            prisma_client=mock_prisma_client,
            user_api_key_dict=UserAPIKeyAuth(),
            litellm_proxy_admin_name="admin",
            team_id="test-team-123",
            default_team_budget_id="budget-123",
        )


@pytest.mark.asyncio
async def test_process_team_members_multiple_members():
    """
    Test _process_team_members with multiple members
    """
    from litellm.proxy._types import LiteLLM_TeamMembership, LiteLLM_UserTable
    from litellm.proxy.management_endpoints.team_endpoints import _process_team_members

    # Mock dependencies
    mock_prisma_client = MagicMock()
    mock_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_team.metadata = None

    # Create multiple members as dictionaries (they will be converted to Member objects)
    members = [
        Member(user_email="user1@example.com", role="user"),
        Member(user_email="user2@example.com", role="admin"),
    ]
    request_data = TeamMemberAddRequest(
        team_id="test-team-123",
        member=members,
        max_budget_in_team=100.0,
    )

    # Mock different users and memberships for each call
    mock_users = [MagicMock(spec=LiteLLM_UserTable) for _ in range(2)]
    mock_memberships = [MagicMock(spec=LiteLLM_TeamMembership) for _ in range(2)]

    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.add_new_member",
        new_callable=AsyncMock,
        side_effect=[
            (mock_users[0], mock_memberships[0]),
            (mock_users[1], mock_memberships[1]),
        ],
    ) as mock_add_member:
        users, memberships = await _process_team_members(
            data=request_data,
            complete_team_data=mock_team,
            prisma_client=mock_prisma_client,
            user_api_key_dict=UserAPIKeyAuth(),
            litellm_proxy_admin_name="admin",
        )

        # Verify results
        assert len(users) == 2
        assert len(memberships) == 2
        assert users == mock_users
        assert memberships == mock_memberships

        # Verify add_new_member was called for each member
        assert mock_add_member.call_count == 2


@pytest.mark.asyncio
async def test_update_team_members_list_single_member():
    """
    Test _update_team_members_list with a single member
    """
    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.management_endpoints.team_endpoints import (
        _update_team_members_list,
    )

    # Create mock team with existing members
    mock_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_team.members_with_roles = [Member(user_id="existing-user", role="admin")]

    # Create new member without user_id
    new_member = Member(user_email="new@example.com", role="user")
    request_data = TeamMemberAddRequest(
        team_id="test-team-123",
        member=new_member,
    )

    # Create mock user with matching email
    mock_user = MagicMock(spec=LiteLLM_UserTable)
    mock_user.user_id = "new-user-123"
    mock_user.user_email = "new@example.com"

    await _update_team_members_list(
        data=request_data,
        complete_team_data=mock_team,
        updated_users=[mock_user],
    )

    # Verify member was added
    assert len(mock_team.members_with_roles) == 2
    added_member = mock_team.members_with_roles[1]
    assert added_member.user_id == "new-user-123"
    assert added_member.user_email == "new@example.com"
    assert added_member.role == "user"


@pytest.mark.asyncio
async def test_update_team_members_list_duplicate_prevention():
    """
    Test _update_team_members_list prevents duplicate members
    """
    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.management_endpoints.team_endpoints import (
        _update_team_members_list,
    )

    # Create mock team with existing members
    mock_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_team.members_with_roles = [
        Member(user_id="existing-user", user_email="existing@example.com", role="admin")
    ]

    # Try to add the same member again
    duplicate_member = Member(user_id="existing-user", role="user")
    request_data = TeamMemberAddRequest(
        team_id="test-team-123",
        member=duplicate_member,
    )

    # Create mock user
    mock_user = MagicMock(spec=LiteLLM_UserTable)
    mock_user.user_id = "existing-user"
    mock_user.user_email = "existing@example.com"

    await _update_team_members_list(
        data=request_data,
        complete_team_data=mock_team,
        updated_users=[mock_user],
    )

    # Verify member was NOT added (still only 1 member)
    assert len(mock_team.members_with_roles) == 1


def test_add_new_models_to_team_with_existing_models():
    """
    Test add_new_models_to_team function with existing models
    """
    from litellm.proxy._types import SpecialModelNames
    from litellm.proxy.management_endpoints.team_endpoints import add_new_models_to_team

    team_obj = MagicMock(spec=LiteLLM_TeamTable)
    team_obj.models = ["model1", "model2"]
    new_models = ["model3", "model4"]

    updated_models = add_new_models_to_team(
        team_obj=team_obj,
        new_models=new_models,
    )

    assert updated_models.sort() == ["model1", "model2", "model3", "model4"].sort()


@pytest.mark.asyncio
async def test_update_team_team_member_budget_not_passed_to_db():
    """
    Test that 'team_member_budget' is never passed to prisma_client.db.litellm_teamtable.update
    regardless of whether the value is set or None.

    This ensures that team_member_budget is properly handled via the separate budget table
    and not accidentally passed to the team table update operation.
    """
    from unittest.mock import AsyncMock, MagicMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import LitellmUserRoles, UpdateTeamRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Mock dependencies
    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test_user_id"
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client, patch(
        "litellm.proxy.proxy_server.llm_router"
    ) as mock_llm_router, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.proxy_logging_obj"
    ) as mock_logging, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.auth.auth_checks._cache_team_object"
    ) as mock_cache_team, patch(
        "litellm.proxy.management_endpoints.team_endpoints.TeamMemberBudgetHandler.upsert_team_member_budget_table"
    ) as mock_upsert_budget:

        # Setup mock prisma client
        mock_existing_team = MagicMock()
        mock_existing_team.model_dump.return_value = {
            "team_id": "test_team_id",
            "team_alias": "test_team",
            "metadata": {"team_member_budget_id": "budget_123"},
        }
        mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )

        # Mock the update return value
        mock_updated_team = MagicMock()
        mock_updated_team.team_id = "test_team_id"
        mock_updated_team.model_dump.return_value = {"team_id": "test_team_id"}
        mock_prisma_client.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )
        mock_prisma_client.jsonify_team_object = MagicMock(
            side_effect=lambda db_data: db_data
        )

        # Mock budget upsert to return updated_kv without team_member_budget
        def mock_upsert_side_effect(
            team_table, user_api_key_dict, updated_kv, team_member_budget=None, team_member_rpm_limit=None, team_member_tpm_limit=None
        ):
            # Remove team_member_budget from updated_kv as the real function does
            result_kv = updated_kv.copy()
            result_kv.pop("team_member_budget", None)
            return result_kv

        mock_upsert_budget.side_effect = mock_upsert_side_effect

        # Test Case 1: team_member_budget is set (not None)
        update_request_with_budget = UpdateTeamRequest(
            team_id="test_team_id", team_member_budget=100.0, team_alias="updated_alias"
        )

        result = await update_team(
            data=update_request_with_budget,
            http_request=mock_request,
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Verify update was called
        assert mock_prisma_client.db.litellm_teamtable.update.called

        # Get the call arguments
        call_args = mock_prisma_client.db.litellm_teamtable.update.call_args
        update_data = call_args[1]["data"]  # data parameter from the update call

        # Verify team_member_budget is NOT in the update data
        assert (
            "team_member_budget" not in update_data
        ), f"team_member_budget should not be in update data, but found: {update_data}"

        # Verify other fields are present (team_alias should be there)
        assert "team_alias" in update_data or "team_id" in str(
            call_args
        ), "Expected team update fields should be present"

        # Reset mock for second test
        mock_prisma_client.db.litellm_teamtable.update.reset_mock()

        # Test Case 2: team_member_budget is None
        update_request_without_budget = UpdateTeamRequest(
            team_id="test_team_id",
            team_member_budget=None,
            team_alias="updated_alias_2",
        )

        result = await update_team(
            data=update_request_without_budget,
            http_request=mock_request,
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Verify update was called again
        assert mock_prisma_client.db.litellm_teamtable.update.called

        # Get the call arguments for second call
        call_args = mock_prisma_client.db.litellm_teamtable.update.call_args
        update_data = call_args[1]["data"]  # data parameter from the update call

        # Verify team_member_budget is NOT in the update data
        assert (
            "team_member_budget" not in update_data
        ), f"team_member_budget should not be in update data, but found: {update_data}"

        # Test Case 3: No team_member_budget field at all (excluded from request)
        mock_prisma_client.db.litellm_teamtable.update.reset_mock()

        update_request_no_budget_field = UpdateTeamRequest(
            team_id="test_team_id",
            team_alias="updated_alias_3",
            # team_member_budget not specified at all
        )

        result = await update_team(
            data=update_request_no_budget_field,
            http_request=mock_request,
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Verify update was called again
        assert mock_prisma_client.db.litellm_teamtable.update.called

        # Get the call arguments for third call
        call_args = mock_prisma_client.db.litellm_teamtable.update.call_args
        update_data = call_args[1]["data"]  # data parameter from the update call

        # Verify team_member_budget is NOT in the update data
        assert (
            "team_member_budget" not in update_data
        ), f"team_member_budget should not be in update data, but found: {update_data}"

        print(
            "✅ All test cases passed: team_member_budget is properly excluded from database update operations"
        )


@pytest.mark.asyncio
async def test_bulk_team_member_add_success():
    """
    Test bulk_team_member_add with successful addition of multiple members
    """
    from litellm.proxy._types import (
        LiteLLM_TeamMembership,
        LiteLLM_UserTable,
        TeamAddMemberResponse,
    )
    from litellm.proxy.management_endpoints.team_endpoints import bulk_team_member_add

    # Create test data
    test_members = [
        Member(user_email="user1@example.com", role="user"),
        Member(user_email="user2@example.com", role="admin"),
    ]

    bulk_request = BulkTeamMemberAddRequest(
        team_id="test-team-123",
        members=test_members,
        max_budget_in_team=100.0,
    )

    # Mock successful team_member_add response using MagicMock for simplicity
    mock_user_1 = MagicMock(spec=LiteLLM_UserTable)
    mock_user_1.user_id = "user-1"
    mock_user_1.user_email = "user1@example.com"
    mock_user_1.model_dump.return_value = {
        "user_id": "user-1",
        "user_email": "user1@example.com",
    }

    mock_user_2 = MagicMock(spec=LiteLLM_UserTable)
    mock_user_2.user_id = "user-2"
    mock_user_2.user_email = "user2@example.com"
    mock_user_2.model_dump.return_value = {
        "user_id": "user-2",
        "user_email": "user2@example.com",
    }

    mock_updated_users = [mock_user_1, mock_user_2]

    mock_membership_1 = MagicMock(spec=LiteLLM_TeamMembership)
    mock_membership_1.user_id = "user-1"
    mock_membership_1.team_id = "test-team-123"
    mock_membership_1.model_dump.return_value = {
        "user_id": "user-1",
        "team_id": "test-team-123",
    }

    mock_membership_2 = MagicMock(spec=LiteLLM_TeamMembership)
    mock_membership_2.user_id = "user-2"
    mock_membership_2.team_id = "test-team-123"
    mock_membership_2.model_dump.return_value = {
        "user_id": "user-2",
        "team_id": "test-team-123",
    }

    mock_updated_memberships = [mock_membership_1, mock_membership_2]

    # Create a mock response that has model_dump method
    mock_team_response = MagicMock()
    mock_team_response.team_id = "test-team-123"
    mock_team_response.team_alias = "Test Team"
    mock_team_response.updated_users = mock_updated_users
    mock_team_response.updated_team_memberships = mock_updated_memberships
    mock_team_response.model_dump.return_value = {
        "team_id": "test-team-123",
        "team_alias": "Test Team",
        "updated_users": [u.model_dump() for u in mock_updated_users],
        "updated_team_memberships": [m.model_dump() for m in mock_updated_memberships],
    }

    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.team_member_add",
        new_callable=AsyncMock,
        return_value=mock_team_response,
    ) as mock_team_member_add:

        mock_auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

        result = await bulk_team_member_add(
            data=bulk_request,
            user_api_key_dict=mock_auth,
        )

        # Verify the result structure
        assert isinstance(result, BulkTeamMemberAddResponse)
        assert result.team_id == "test-team-123"
        assert result.total_requested == 2
        assert result.successful_additions == 2
        assert result.failed_additions == 0
        assert len(result.results) == 2

        # Verify individual results
        for i, member_result in enumerate(result.results):
            assert isinstance(member_result, TeamMemberAddResult)
            assert member_result.success is True
            assert member_result.error is None
            assert member_result.user_email == test_members[i].user_email

        # Verify team_member_add was called with correct data
        mock_team_member_add.assert_called_once()
        call_args = mock_team_member_add.call_args[1]["data"]
        assert call_args.team_id == "test-team-123"
        assert call_args.member == test_members
        assert call_args.max_budget_in_team == 100.0


@pytest.mark.asyncio
async def test_bulk_team_member_add_no_members_error():
    """
    Test bulk_team_member_add raises error when no members provided
    """
    from litellm.proxy.management_endpoints.team_endpoints import bulk_team_member_add

    bulk_request = BulkTeamMemberAddRequest(
        team_id="test-team-123",
        members=[],  # Empty list
    )

    mock_auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    with pytest.raises(HTTPException) as exc_info:
        await bulk_team_member_add(
            data=bulk_request,
            user_api_key_dict=mock_auth,
        )

    assert exc_info.value.status_code == 400
    assert "At least one member is required" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_bulk_team_member_add_batch_size_limit():
    """
    Test bulk_team_member_add enforces maximum batch size limit
    """
    from litellm.proxy.management_endpoints.team_endpoints import bulk_team_member_add

    # Create more than 500 members (the max batch size)
    large_member_list = [
        Member(user_email=f"user{i}@example.com", role="user") for i in range(501)
    ]

    bulk_request = BulkTeamMemberAddRequest(
        team_id="test-team-123",
        members=large_member_list,
    )

    mock_auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    with pytest.raises(HTTPException) as exc_info:
        await bulk_team_member_add(
            data=bulk_request,
            user_api_key_dict=mock_auth,
        )

    assert exc_info.value.status_code == 400
    assert "Maximum 500 members can be added at once" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_bulk_team_member_add_all_users_flag():
    """
    Test bulk_team_member_add with all_users flag set to True
    """
    from litellm.proxy._types import LiteLLM_UserTable, TeamAddMemberResponse
    from litellm.proxy.management_endpoints.team_endpoints import bulk_team_member_add

    bulk_request = BulkTeamMemberAddRequest(
        team_id="test-team-123",
        all_users=True,
        max_budget_in_team=50.0,
    )

    # Mock database users
    mock_db_users = [
        MagicMock(user_id="user-1", user_email="user1@example.com"),
        MagicMock(user_id="user-2", user_email="user2@example.com"),
    ]

    mock_team_response = TeamAddMemberResponse(
        team_id="test-team-123",
        team_alias="Test Team",
        updated_users=[],
        updated_team_memberships=[],
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.management_endpoints.team_endpoints.team_member_add",
        new_callable=AsyncMock,
        return_value=mock_team_response,
    ) as mock_team_member_add:

        # Mock the database find_many call
        mock_prisma.db.litellm_usertable.find_many = AsyncMock(
            return_value=mock_db_users
        )

        mock_auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

        result = await bulk_team_member_add(
            data=bulk_request,
            user_api_key_dict=mock_auth,
        )

        # Verify that find_many was called to get all users
        mock_prisma.db.litellm_usertable.find_many.assert_called_once_with(
            order={"created_at": "desc"}
        )

        # Verify team_member_add was called with users from database
        mock_team_member_add.assert_called_once()
        call_args = mock_team_member_add.call_args[1]["data"]
        assert call_args.team_id == "test-team-123"
        assert len(call_args.member) == 2  # Should have 2 members from mock_db_users
        assert call_args.max_budget_in_team == 50.0


@pytest.mark.asyncio
async def test_bulk_team_member_add_failure_scenario():
    """
    Test bulk_team_member_add handles failures gracefully
    """
    from litellm.proxy.management_endpoints.team_endpoints import bulk_team_member_add

    test_members = [
        Member(user_email="user1@example.com", role="user"),
        Member(user_email="user2@example.com", role="admin"),
    ]

    bulk_request = BulkTeamMemberAddRequest(
        team_id="test-team-123",
        members=test_members,
    )

    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.team_member_add",
        new_callable=AsyncMock,
        side_effect=Exception("Database connection failed"),
    ) as mock_team_member_add:

        mock_auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

        result = await bulk_team_member_add(
            data=bulk_request,
            user_api_key_dict=mock_auth,
        )

        # Verify failure response structure
        assert isinstance(result, BulkTeamMemberAddResponse)
        assert result.team_id == "test-team-123"
        assert result.total_requested == 2
        assert result.successful_additions == 0
        assert result.failed_additions == 2
        assert result.updated_team is None

        # Verify all members marked as failed
        assert len(result.results) == 2
        for member_result in result.results:
            assert member_result.success is False
            assert member_result.error == "Database connection failed"


@pytest.mark.asyncio
async def test_bulk_team_member_add_no_db_connection():
    """
    Test bulk_team_member_add handles missing database connection
    """
    from litellm.proxy.management_endpoints.team_endpoints import bulk_team_member_add

    bulk_request = BulkTeamMemberAddRequest(
        team_id="test-team-123",
        members=[Member(user_email="user1@example.com", role="user")],
    )

    with patch("litellm.proxy.proxy_server.prisma_client", None):
        mock_auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

        with pytest.raises(HTTPException) as exc_info:
            await bulk_team_member_add(
                data=bulk_request,
                user_api_key_dict=mock_auth,
            )

        assert exc_info.value.status_code == 500
        assert "DB not connected" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_list_team_v2_security_check_non_admin_user():
    """
    Test that list_team_v2 properly checks route permissions for non-admin users.
    Non-admin users should only be able to query their own teams.
    """
    from unittest.mock import AsyncMock, Mock, patch

    from fastapi import HTTPException, Request

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    # Mock request
    mock_request = Mock(spec=Request)

    # Test Case 1: Non-admin user trying to query all teams (user_id=None)
    mock_user_api_key_dict_non_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="non_admin_user_123",
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client:
        mock_prisma_client.return_value = MagicMock()  # Mock non-None prisma client

        # Should raise HTTPException with 401 status
        with pytest.raises(HTTPException) as exc_info:
            await list_team_v2(
                http_request=mock_request,
                user_id=None,  # Non-admin trying to query all teams
                user_api_key_dict=mock_user_api_key_dict_non_admin,
            )

        assert exc_info.value.status_code == 401
        assert "Only admin users can query all teams/other teams" in str(
            exc_info.value.detail
        )
        assert LitellmUserRoles.INTERNAL_USER.value in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_list_team_v2_security_check_non_admin_user_other_user():
    """
    Test that list_team_v2 properly checks route permissions for non-admin users
    trying to query other users' teams.
    """
    from unittest.mock import AsyncMock, Mock, patch

    from fastapi import HTTPException, Request

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    # Mock request
    mock_request = Mock(spec=Request)

    # Test Case 2: Non-admin user trying to query another user's teams
    mock_user_api_key_dict_non_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="non_admin_user_123",
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client:
        mock_prisma_client.return_value = MagicMock()  # Mock non-None prisma client

        # Should raise HTTPException with 401 status
        with pytest.raises(HTTPException) as exc_info:
            await list_team_v2(
                http_request=mock_request,
                user_id="other_user_456",  # Non-admin trying to query other user's teams
                user_api_key_dict=mock_user_api_key_dict_non_admin,
            )

        assert exc_info.value.status_code == 401
        assert "Only admin users can query all teams/other teams" in str(
            exc_info.value.detail
        )


@pytest.mark.asyncio
async def test_list_team_v2_security_check_non_admin_user_own_teams():
    """
    Test that list_team_v2 allows non-admin users to query their own teams.
    """
    from unittest.mock import AsyncMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    # Mock request
    mock_request = Mock(spec=Request)

    # Test Case 3: Non-admin user querying their own teams (should be allowed)
    mock_user_api_key_dict_non_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="non_admin_user_123",
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client:
        # Mock prisma client and database operations
        mock_db = Mock()
        mock_prisma_client.db = mock_db
        
        # Mock user lookup
        mock_user_object = Mock()
        mock_user_object.model_dump.return_value = {
            "user_id": "non_admin_user_123",
            "teams": ["team_1", "team_2"],
        }
        mock_db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user_object)
        
        # Mock team lookup
        mock_teams = [
            Mock(model_dump=lambda: {"team_id": "team_1", "team_alias": "Team 1"}),
            Mock(model_dump=lambda: {"team_id": "team_2", "team_alias": "Team 2"}),
        ]
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=mock_teams)
        mock_db.litellm_teamtable.count = AsyncMock(return_value=2)

        # Should NOT raise an exception
        result = await list_team_v2(
            http_request=mock_request,
            user_id="non_admin_user_123",  # Non-admin querying their own teams
            user_api_key_dict=mock_user_api_key_dict_non_admin,
            team_id=None,
            page=1,
            page_size=10,
        )

        # Should return results without error
        assert "teams" in result
        assert "total" in result
        assert result["total"] == 2


@pytest.mark.asyncio
async def test_list_team_v2_security_check_admin_user():
    """
    Test that list_team_v2 allows admin users to query any teams.
    """
    from unittest.mock import AsyncMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    # Mock request
    mock_request = Mock(spec=Request)

    # Test Case 4: Admin user querying all teams (should be allowed)
    mock_user_api_key_dict_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user_123",
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client:
        # Mock prisma client and database operations
        mock_db = Mock()
        mock_prisma_client.db = mock_db
        
        # Mock team lookup
        mock_teams = [
            Mock(model_dump=lambda: {"team_id": "team_1", "team_alias": "Team 1"}),
            Mock(model_dump=lambda: {"team_id": "team_2", "team_alias": "Team 2"}),
        ]
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=mock_teams)
        mock_db.litellm_teamtable.count = AsyncMock(return_value=2)

        # Should NOT raise an exception
        result = await list_team_v2(
            http_request=mock_request,
            user_id=None,  # Admin querying all teams
            user_api_key_dict=mock_user_api_key_dict_admin,
            page=1,
            page_size=10,
        )

        # Should return results without error
        assert "teams" in result
        assert "total" in result
        assert result["total"] == 2


@pytest.mark.asyncio
async def test_team_member_delete_cleans_membership(mock_db_client, mock_admin_auth):
    """
    Verify that /team/member_delete removes the corresponding LiteLLM_TeamMembership row
    so the same user can be re-added without unique constraint issues.
    """
    from litellm.proxy._types import TeamMemberDeleteRequest
    from litellm.proxy.management_endpoints.team_endpoints import team_member_delete

    test_team_id = "team-del-123"
    test_user_id = "user@example.com"

    # Mock Team row with the user as a member
    mock_team_row = MagicMock()
    mock_team_row.model_dump.return_value = {
        "team_id": test_team_id,
        "members_with_roles": [
            {"user_id": test_user_id, "user_email": None, "role": "user"}
        ],
        "team_member_permissions": [],
        "metadata": {},
        "models": [],
        "spend": 0.0,
    }

    # Configure DB mocks used by team_member_delete
    mock_db_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_team_row)
    mock_db_client.db.litellm_teamtable.update = AsyncMock(return_value=mock_team_row)

    # User row to allow removal from user's teams list
    mock_user_row = MagicMock()
    mock_user_row.user_id = test_user_id
    mock_user_row.teams = [test_team_id]
    mock_db_client.db.litellm_usertable.find_many = AsyncMock(return_value=[mock_user_row])
    mock_db_client.db.litellm_usertable.update = AsyncMock(return_value=MagicMock())

    # Membership deletion should be called
    mock_db_client.db.litellm_teammembership = MagicMock()
    mock_db_client.db.litellm_teammembership.delete_many = AsyncMock(return_value=MagicMock())

    # Verification token deletion should be called
    mock_db_client.db.litellm_verificationtoken = MagicMock()
    mock_db_client.db.litellm_verificationtoken.delete_many = AsyncMock(return_value=MagicMock())

    # Execute
    await team_member_delete(
        data=TeamMemberDeleteRequest(team_id=test_team_id, user_id=test_user_id),
        user_api_key_dict=mock_admin_auth,
    )

    # Assert membership cleanup executed
    mock_db_client.db.litellm_teammembership.delete_many.assert_awaited_with(
        where={"team_id": test_team_id, "user_id": test_user_id}
    )
    

@pytest.mark.asyncio
async def test_team_member_delete_cleans_verification_tokens(mock_db_client, mock_admin_auth):
    from litellm.proxy._types import TeamMemberDeleteRequest
    from litellm.proxy.management_endpoints.team_endpoints import team_member_delete

    test_team_id = "team-del-tokens-123"
    test_user_id = "user-tokens@example.com"

    mock_team_row = MagicMock()
    mock_team_row.model_dump.return_value = {
        "team_id": test_team_id,
        "members_with_roles": [
            {"user_id": test_user_id, "user_email": None, "role": "user"}
        ],
        "team_member_permissions": [],
        "metadata": {},
        "models": [],
        "spend": 0.0,
    }

    mock_db_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_team_row)
    mock_db_client.db.litellm_teamtable.update = AsyncMock(return_value=mock_team_row)

    mock_user_row = MagicMock()
    mock_user_row.user_id = test_user_id
    mock_user_row.teams = [test_team_id]
    mock_db_client.db.litellm_usertable.find_many = AsyncMock(return_value=[mock_user_row])
    mock_db_client.db.litellm_usertable.update = AsyncMock(return_value=MagicMock())

    mock_db_client.db.litellm_teammembership = MagicMock()
    mock_db_client.db.litellm_teammembership.delete_many = AsyncMock(return_value=MagicMock())

    mock_db_client.db.litellm_verificationtoken = MagicMock()
    mock_db_client.db.litellm_verificationtoken.delete_many = AsyncMock(return_value=MagicMock())

    await team_member_delete(
        data=TeamMemberDeleteRequest(team_id=test_team_id, user_id=test_user_id),
        user_api_key_dict=mock_admin_auth,
    )

    mock_db_client.db.litellm_verificationtoken.delete_many.assert_awaited_once_with(
        where={
            "user_id": {"in": [test_user_id]},
            "team_id": test_team_id,
        }
    )


@pytest.mark.asyncio
async def test_new_team_max_budget_exceeds_user_max_budget():
    """
    Test that /team/new raises ProxyException when max_budget exceeds user's end_user_max_budget.
    
    This validates the budget enforcement logic where non-admin users cannot create teams
    with budgets higher than their personal maximum budget limit.
    """
    from fastapi import Request

    from litellm.proxy._types import NewTeamRequest, ProxyException, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Create non-admin user with user_max_budget set to 100.0
    non_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="non-admin-user-123",
        user_max_budget=100.0,
    )

    # Create team request with max_budget (200.0) exceeding user's limit (100.0)
    team_request = NewTeamRequest(
        team_alias="high-budget-team",
        max_budget=200.0,  # Exceeds user's user_max_budget
    )

    dummy_request = MagicMock(spec=Request)

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server._license_check"
    ) as mock_license, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ) as mock_audit:
        # Setup basic mocks
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_license.is_team_count_over_limit.return_value = False
        mock_prisma.get_data = AsyncMock(return_value=None)
        
        # Mock user cache to return a user object with max_budget=100.0
        from litellm.proxy._types import LiteLLM_UserTable
        mock_user_obj = LiteLLM_UserTable(
            user_id="non-admin-user-123",
            max_budget=100.0,
        )
        mock_cache.async_get_cache = AsyncMock(return_value=mock_user_obj)
        
        # Should raise ProxyException (HTTPException gets converted by handle_exception_on_proxy)
        with pytest.raises(ProxyException) as exc_info:
            await new_team(
                data=team_request,
                http_request=dummy_request,
                user_api_key_dict=non_admin_user,
            )

        # Verify exception details
        # ProxyException stores status_code in 'code' attribute
        assert exc_info.value.code == '400'
        assert "max budget higher than user max" in str(exc_info.value.message)
        assert "100.0" in str(exc_info.value.message)  # User's user_max_budget should be mentioned
        assert LitellmUserRoles.INTERNAL_USER.value in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_new_team_max_budget_within_user_limit():
    """
    Test that /team/new succeeds when max_budget is within user's user_max_budget.
    
    This ensures that users can create teams with budgets at or below their personal limit.
    """
    from fastapi import Request

    from litellm.proxy._types import NewTeamRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Create non-admin user with user_max_budget set to 100.0
    non_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="non-admin-user-456",
        user_max_budget=100.0,
        models=[],  # Empty models list to bypass model validation
    )

    # Create team request with max_budget (50.0) within user's limit (100.0)
    team_request = NewTeamRequest(
        team_alias="within-budget-team",
        max_budget=50.0,  # Within user's user_max_budget
    )

    dummy_request = MagicMock(spec=Request)

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server._license_check"
    ) as mock_license, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ) as mock_audit:

        # Setup mocks
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_license.is_team_count_over_limit.return_value = False
        mock_prisma.jsonify_team_object = lambda db_data: db_data
        mock_prisma.get_data = AsyncMock(return_value=None)
        mock_prisma.update_data = AsyncMock()
        
        # Mock user cache to return a user object with max_budget=100.0
        from litellm.proxy._types import LiteLLM_UserTable
        mock_user_obj = LiteLLM_UserTable(
            user_id="non-admin-user-456",
            max_budget=100.0,
        )
        mock_cache.async_get_cache = AsyncMock(return_value=mock_user_obj)
        
        # Mock team creation
        mock_created_team = MagicMock()
        mock_created_team.team_id = "team-within-budget-789"
        mock_created_team.team_alias = "within-budget-team"
        mock_created_team.max_budget = 50.0
        mock_created_team.members_with_roles = []
        mock_created_team.metadata = None
        mock_created_team.model_dump.return_value = {
            "team_id": "team-within-budget-789",
            "team_alias": "within-budget-team",
            "max_budget": 50.0,
            "members_with_roles": [],
        }
        mock_prisma.db.litellm_teamtable.create = AsyncMock(return_value=mock_created_team)
        mock_prisma.db.litellm_teamtable.update = AsyncMock(return_value=mock_created_team)
        
        # Mock model table
        mock_prisma.db.litellm_modeltable = MagicMock()
        mock_prisma.db.litellm_modeltable.create = AsyncMock(return_value=MagicMock(id="model123"))
        
        # Mock user table operations for adding the creator as a member
        mock_user = MagicMock()
        mock_user.user_id = "non-admin-user-456"
        mock_user.model_dump.return_value = {"user_id": "non-admin-user-456", "teams": ["team-within-budget-789"]}
        mock_prisma.db.litellm_usertable = MagicMock()
        mock_prisma.db.litellm_usertable.upsert = AsyncMock(return_value=mock_user)
        mock_prisma.db.litellm_usertable.update = AsyncMock(return_value=mock_user)
        
        # Mock team membership table
        mock_membership = MagicMock()
        mock_membership.model_dump.return_value = {
            "team_id": "team-within-budget-789",
            "user_id": "non-admin-user-456",
            "budget_id": None,
        }
        mock_prisma.db.litellm_teammembership = MagicMock()
        mock_prisma.db.litellm_teammembership.create = AsyncMock(return_value=mock_membership)

        # Should NOT raise an exception
        result = await new_team(
            data=team_request,
            http_request=dummy_request,
            user_api_key_dict=non_admin_user,
        )

        # Verify the team was created successfully
        assert result is not None
        assert result["team_id"] == "team-within-budget-789"
        assert result["max_budget"] == 50.0


@pytest.mark.asyncio
async def test_new_team_org_scoped_budget_bypasses_user_limit():
    """
    Test that /team/new with organization_id does NOT validate budget against user's personal max_budget.

    This is the bug fix for: When an org admin creates an org-scoped team, the team's budget should
    be validated against the organization's limits, not the user's personal limits.

    Scenario:
    - Organization has max_budget=$100
    - User (org admin) has personal max_budget=$3
    - Team is created with organization_id and max_budget=$50
    - Expected: Should succeed (within org's $100 limit)
    - Bug behavior: Would fail with "max budget higher than user max. User max budget=3.0"
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_OrganizationTable,
        LiteLLM_UserTable,
        NewTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Create non-admin user with very restrictive personal budget ($3)
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-user-123",
        user_max_budget=3.0,  # Restrictive personal budget
        models=[],  # Empty models list to bypass model validation
    )

    # Create team request with budget ($50) that's within org's limit but exceeds user's personal limit
    team_request = NewTeamRequest(
        team_alias="org-scoped-team",
        max_budget=50.0,  # Within org's $100 limit, but exceeds user's $3 limit
        organization_id="test-org-123",  # This makes it an org-scoped team
    )

    dummy_request = MagicMock(spec=Request)

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server._license_check"
    ) as mock_license, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ) as mock_audit, patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_org_object"
    ) as mock_get_org:

        # Setup mocks
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_license.is_team_count_over_limit.return_value = False
        mock_prisma.jsonify_team_object = lambda db_data: db_data
        mock_prisma.get_data = AsyncMock(return_value=None)
        mock_prisma.update_data = AsyncMock()

        # Mock organization with $100 budget
        mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
        mock_org.organization_id = "test-org-123"
        mock_org.max_budget = 100.0
        mock_org.models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"]
        mock_org.litellm_budget_table = None  # No budget table for this test
        mock_get_org.return_value = mock_org

        # Mock user cache to return user with restrictive personal budget
        mock_user_obj = LiteLLM_UserTable(
            user_id="org-admin-user-123",
            max_budget=3.0,  # Restrictive personal budget
        )
        mock_cache.async_get_cache = AsyncMock(return_value=mock_user_obj)

        # Mock team creation
        mock_created_team = MagicMock()
        mock_created_team.team_id = "team-org-scoped-789"
        mock_created_team.team_alias = "org-scoped-team"
        mock_created_team.max_budget = 50.0
        mock_created_team.organization_id = "test-org-123"
        mock_created_team.members_with_roles = []
        mock_created_team.metadata = None
        mock_created_team.model_dump.return_value = {
            "team_id": "team-org-scoped-789",
            "team_alias": "org-scoped-team",
            "max_budget": 50.0,
            "organization_id": "test-org-123",
            "members_with_roles": [],
        }
        mock_prisma.db.litellm_teamtable.create = AsyncMock(return_value=mock_created_team)
        mock_prisma.db.litellm_teamtable.update = AsyncMock(return_value=mock_created_team)

        # Mock model table
        mock_prisma.db.litellm_modeltable = MagicMock()
        mock_prisma.db.litellm_modeltable.create = AsyncMock(return_value=MagicMock(id="model123"))

        # Mock user table operations
        mock_user = MagicMock()
        mock_user.user_id = "org-admin-user-123"
        mock_user.model_dump.return_value = {"user_id": "org-admin-user-123", "teams": ["team-org-scoped-789"]}
        mock_prisma.db.litellm_usertable = MagicMock()
        mock_prisma.db.litellm_usertable.upsert = AsyncMock(return_value=mock_user)
        mock_prisma.db.litellm_usertable.update = AsyncMock(return_value=mock_user)

        # Mock team membership table
        mock_membership = MagicMock()
        mock_membership.model_dump.return_value = {
            "team_id": "team-org-scoped-789",
            "user_id": "org-admin-user-123",
            "budget_id": None,
        }
        mock_prisma.db.litellm_teammembership = MagicMock()
        mock_prisma.db.litellm_teammembership.create = AsyncMock(return_value=mock_membership)

        # Should NOT raise an exception - the fix should bypass user budget validation for org-scoped teams
        result = await new_team(
            data=team_request,
            http_request=dummy_request,
            user_api_key_dict=org_admin_user,
        )

        # Verify the team was created successfully with the higher budget
        assert result is not None
        assert result["team_id"] == "team-org-scoped-789"
        assert result["max_budget"] == 50.0
        assert result["organization_id"] == "test-org-123"


@pytest.mark.asyncio
async def test_new_team_org_scoped_models_bypasses_user_limit():
    """
    Test that /team/new with organization_id does NOT validate models against user's personal models.

    This is the bug fix for: When an org admin creates an org-scoped team, the team's models should
    be validated against the organization's models, not the user's personal models.

    Scenario:
    - Organization has models=['gpt-4', 'gpt-3.5-turbo', 'claude-3-opus']
    - User (org admin) has personal models=['no-default-models']
    - Team is created with organization_id and models=['gpt-4']
    - Expected: Should succeed (within org's allowed models)
    - Bug behavior: Would fail with "Model not in allowed user models"
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_OrganizationTable,
        LiteLLM_UserTable,
        NewTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Create non-admin user with restrictive personal models
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-user-456",
        user_max_budget=None,  # No budget restriction for this test
        models=["no-default-models"],  # Restrictive personal models
    )

    # Create team request with models that are within org's allowed models but not user's
    team_request = NewTeamRequest(
        team_alias="org-scoped-models-team",
        models=["gpt-4"],  # Within org's allowed models, but not in user's personal models
        organization_id="test-org-456",  # This makes it an org-scoped team
    )

    dummy_request = MagicMock(spec=Request)

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server._license_check"
    ) as mock_license, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ) as mock_audit, patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_org_object"
    ) as mock_get_org:

        # Setup mocks
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_license.is_team_count_over_limit.return_value = False
        mock_prisma.jsonify_team_object = lambda db_data: db_data
        mock_prisma.get_data = AsyncMock(return_value=None)
        mock_prisma.update_data = AsyncMock()

        # Mock organization with allowed models
        mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
        mock_org.organization_id = "test-org-456"
        mock_org.max_budget = 100.0
        mock_org.models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"]
        mock_org.litellm_budget_table = None
        mock_get_org.return_value = mock_org

        # Mock user cache
        mock_user_obj = LiteLLM_UserTable(
            user_id="org-admin-user-456",
            max_budget=None,
        )
        mock_cache.async_get_cache = AsyncMock(return_value=mock_user_obj)

        # Mock team creation
        mock_created_team = MagicMock()
        mock_created_team.team_id = "team-org-scoped-models-789"
        mock_created_team.team_alias = "org-scoped-models-team"
        mock_created_team.max_budget = None
        mock_created_team.organization_id = "test-org-456"
        mock_created_team.models = ["gpt-4"]
        mock_created_team.members_with_roles = []
        mock_created_team.metadata = None
        mock_created_team.model_dump.return_value = {
            "team_id": "team-org-scoped-models-789",
            "team_alias": "org-scoped-models-team",
            "max_budget": None,
            "organization_id": "test-org-456",
            "models": ["gpt-4"],
            "members_with_roles": [],
        }
        mock_prisma.db.litellm_teamtable.create = AsyncMock(return_value=mock_created_team)
        mock_prisma.db.litellm_teamtable.update = AsyncMock(return_value=mock_created_team)

        # Mock model table
        mock_prisma.db.litellm_modeltable = MagicMock()
        mock_prisma.db.litellm_modeltable.create = AsyncMock(return_value=MagicMock(id="model123"))

        # Mock user table operations
        mock_user = MagicMock()
        mock_user.user_id = "org-admin-user-456"
        mock_user.model_dump.return_value = {"user_id": "org-admin-user-456", "teams": ["team-org-scoped-models-789"]}
        mock_prisma.db.litellm_usertable = MagicMock()
        mock_prisma.db.litellm_usertable.upsert = AsyncMock(return_value=mock_user)
        mock_prisma.db.litellm_usertable.update = AsyncMock(return_value=mock_user)

        # Mock team membership table
        mock_membership = MagicMock()
        mock_membership.model_dump.return_value = {
            "team_id": "team-org-scoped-models-789",
            "user_id": "org-admin-user-456",
            "budget_id": None,
        }
        mock_prisma.db.litellm_teammembership = MagicMock()
        mock_prisma.db.litellm_teammembership.create = AsyncMock(return_value=mock_membership)

        # Should NOT raise an exception - the fix should bypass user model validation for org-scoped teams
        result = await new_team(
            data=team_request,
            http_request=dummy_request,
            user_api_key_dict=org_admin_user,
        )

        # Verify the team was created successfully with the org's models
        assert result is not None
        assert result["team_id"] == "team-org-scoped-models-789"
        assert result["models"] == ["gpt-4"]
        assert result["organization_id"] == "test-org-456"


@pytest.mark.asyncio
async def test_new_team_standalone_validates_against_user_models():
    """
    Test that /team/new WITHOUT organization_id still validates models against user's personal models.

    This ensures that standalone teams (not org-scoped) still use user-level validation.

    Scenario:
    - User has personal models=['no-default-models']
    - Team is created WITHOUT organization_id and models=['gpt-4']
    - Expected: Should fail with "Model not in allowed user models"
    """
    from fastapi import Request

    from litellm.proxy._types import NewTeamRequest, ProxyException, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Create non-admin user with restrictive personal models
    non_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="non-admin-user-789",
        user_max_budget=None,
        models=["no-default-models"],  # Restrictive personal models
    )

    # Create standalone team request (no organization_id) with models not in user's list
    team_request = NewTeamRequest(
        team_alias="standalone-team",
        models=["gpt-4"],  # Not in user's allowed models
        # Note: No organization_id - this is a standalone team
    )

    dummy_request = MagicMock(spec=Request)

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server._license_check"
    ) as mock_license, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ) as mock_audit:
        # Setup basic mocks
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_license.is_team_count_over_limit.return_value = False
        mock_prisma.get_data = AsyncMock(return_value=None)

        # Should raise ProxyException because gpt-4 is not in user's allowed models
        with pytest.raises(ProxyException) as exc_info:
            await new_team(
                data=team_request,
                http_request=dummy_request,
                user_api_key_dict=non_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == '400'
        assert "Model not in allowed user models" in str(exc_info.value.message)
        assert "no-default-models" in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_new_team_standalone_validates_against_user_budget():
    """
    Test that /team/new WITHOUT organization_id still validates budget against user's personal max_budget.

    This ensures that standalone teams (not org-scoped) still use user-level validation.
    This is essentially the same as test_new_team_max_budget_exceeds_user_max_budget but
    explicitly showing the contrast with org-scoped teams.

    Scenario:
    - User has personal max_budget=$3
    - Team is created WITHOUT organization_id and max_budget=$50
    - Expected: Should fail with "max budget higher than user max"
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_UserTable,
        NewTeamRequest,
        ProxyException,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Create non-admin user with restrictive personal budget
    non_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="non-admin-user-budget-789",
        user_max_budget=100.0,  # This is for key auth, actual budget is from user object
        models=[],  # Empty models list to bypass model validation
    )

    # Create standalone team request (no organization_id) with budget exceeding user's limit
    team_request = NewTeamRequest(
        team_alias="standalone-budget-team",
        max_budget=50.0,  # Exceeds user's personal budget
        # Note: No organization_id - this is a standalone team
    )

    dummy_request = MagicMock(spec=Request)

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server._license_check"
    ) as mock_license, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ) as mock_audit:
        # Setup basic mocks
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_license.is_team_count_over_limit.return_value = False
        mock_prisma.get_data = AsyncMock(return_value=None)

        # Mock user cache to return user with restrictive personal budget ($3)
        mock_user_obj = LiteLLM_UserTable(
            user_id="non-admin-user-budget-789",
            max_budget=3.0,  # Restrictive personal budget
        )
        mock_cache.async_get_cache = AsyncMock(return_value=mock_user_obj)

        # Should raise ProxyException because budget exceeds user's max_budget
        with pytest.raises(ProxyException) as exc_info:
            await new_team(
                data=team_request,
                http_request=dummy_request,
                user_api_key_dict=non_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == '400'
        assert "max budget higher than user max" in str(exc_info.value.message)
        assert "3.0" in str(exc_info.value.message)  # User's max_budget should be mentioned


@pytest.mark.asyncio
async def test_new_team_org_scoped_budget_exceeds_org_limit():
    """
    Test that /team/new with organization_id fails when team budget exceeds organization's max_budget.

    Scenario:
    - Organization has max_budget=$100
    - Team is created with organization_id and max_budget=$150
    - Expected: Should fail with error about exceeding org budget
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_OrganizationTable,
        NewTeamRequest,
        ProxyException,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Create user (org admin)
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-user-budget-test",
        models=[],
    )

    # Create team request with budget ($150) that exceeds org's limit ($100)
    team_request = NewTeamRequest(
        team_alias="org-team-exceeds-budget",
        max_budget=150.0,  # Exceeds org's $100 limit
        organization_id="test-org-budget-limit",
    )

    dummy_request = MagicMock(spec=Request)

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server._license_check"
    ) as mock_license, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ) as mock_audit, patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_org_object"
    ) as mock_get_org:

        # Setup mocks
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_license.is_team_count_over_limit.return_value = False
        mock_prisma.get_data = AsyncMock(return_value=None)

        # Mock organization with $100 budget limit
        mock_budget_table = MagicMock(spec=LiteLLM_BudgetTable)
        mock_budget_table.max_budget = 100.0

        mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
        mock_org.organization_id = "test-org-budget-limit"
        mock_org.models = ["gpt-4", "gpt-3.5-turbo"]
        mock_org.litellm_budget_table = mock_budget_table
        mock_get_org.return_value = mock_org

        # Should raise ProxyException because team budget exceeds org budget
        with pytest.raises(ProxyException) as exc_info:
            await new_team(
                data=team_request,
                http_request=dummy_request,
                user_api_key_dict=org_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == '400'
        assert "exceeds organization" in str(exc_info.value.message).lower() or "organization" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_new_team_org_scoped_models_not_in_org_models():
    """
    Test that /team/new with organization_id fails when team models are not in organization's allowed models.

    Scenario:
    - Organization has models=['gpt-4', 'gpt-3.5-turbo']
    - Team is created with organization_id and models=['claude-3-opus']
    - Expected: Should fail with error about model not in org's allowed models
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_OrganizationTable,
        NewTeamRequest,
        ProxyException,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Create user (org admin)
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-user-models-test",
        models=[],
    )

    # Create team request with model not in org's allowed list
    team_request = NewTeamRequest(
        team_alias="org-team-invalid-model",
        models=["claude-3-opus"],  # Not in org's allowed models
        organization_id="test-org-models-limit",
    )

    dummy_request = MagicMock(spec=Request)

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server._license_check"
    ) as mock_license, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ) as mock_audit, patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_org_object"
    ) as mock_get_org:

        # Setup mocks
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_license.is_team_count_over_limit.return_value = False
        mock_prisma.get_data = AsyncMock(return_value=None)

        # Mock organization with specific allowed models (not including claude-3-opus)
        mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
        mock_org.organization_id = "test-org-models-limit"
        mock_org.models = ["gpt-4", "gpt-3.5-turbo"]  # claude-3-opus is NOT allowed
        mock_org.litellm_budget_table = None
        mock_get_org.return_value = mock_org

        # Should raise ProxyException because claude-3-opus is not in org's allowed models
        with pytest.raises(ProxyException) as exc_info:
            await new_team(
                data=team_request,
                http_request=dummy_request,
                user_api_key_dict=org_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == '400'
        assert "claude-3-opus" in str(exc_info.value.message) or "organization" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_update_team_standalone_budget_exceeds_user_limit():
    """
    Test that /team/update for a standalone team fails when new budget exceeds user's max_budget.

    Scenario:
    - User has personal max_budget=$50
    - Standalone team exists (no organization_id)
    - User tries to update team budget to $100
    - Expected: Should fail with error about exceeding user budget
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_UserTable,
        ProxyException,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Create non-admin user with restrictive personal budget
    non_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="non-admin-update-test",
        models=[],
    )

    # Create update request with budget exceeding user's limit
    update_request = UpdateTeamRequest(
        team_id="standalone-team-123",
        max_budget=100.0,  # Exceeds user's $50 limit
    )

    dummy_request = MagicMock(spec=Request)

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ) as mock_audit:

        # Mock existing standalone team (no organization_id)
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "standalone-team-123"
        mock_existing_team.organization_id = None  # Standalone team
        mock_existing_team.max_budget = 30.0
        mock_existing_team.model_dump.return_value = {
            "team_id": "standalone-team-123",
            "organization_id": None,
            "max_budget": 30.0,
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_existing_team)

        # Mock user cache to return user with restrictive budget
        mock_user_obj = LiteLLM_UserTable(
            user_id="non-admin-update-test",
            max_budget=50.0,  # User's budget limit
        )
        mock_cache.async_get_cache = AsyncMock(return_value=mock_user_obj)

        # Should raise ProxyException because new budget exceeds user's max_budget
        with pytest.raises(ProxyException) as exc_info:
            await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=non_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == '400'
        assert "budget" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_update_team_org_scoped_budget_exceeds_org_limit():
    """
    Test that /team/update for an org-scoped team fails when new budget exceeds organization's max_budget.

    Scenario:
    - Organization has max_budget=$100
    - Org-scoped team exists
    - User tries to update team budget to $150
    - Expected: Should fail with error about exceeding org budget
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_OrganizationTable,
        ProxyException,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Create user (org admin)
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-update-test",
        models=[],
    )

    # Create update request with budget exceeding org's limit
    update_request = UpdateTeamRequest(
        team_id="org-team-456",
        max_budget=150.0,  # Exceeds org's $100 limit
    )

    dummy_request = MagicMock(spec=Request)

    # Mock organization with $100 budget limit
    mock_budget_table = MagicMock(spec=LiteLLM_BudgetTable)
    mock_budget_table.max_budget = 100.0

    mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
    mock_org.organization_id = "test-org-update"
    mock_org.models = ["gpt-4"]
    mock_org.litellm_budget_table = mock_budget_table

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ) as mock_audit, patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
        new=AsyncMock(return_value=mock_org)
    ) as mock_get_org:

        # Mock existing org-scoped team
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "org-team-456"
        mock_existing_team.organization_id = "test-org-update"
        mock_existing_team.max_budget = 80.0
        mock_existing_team.model_dump.return_value = {
            "team_id": "org-team-456",
            "organization_id": "test-org-update",
            "max_budget": 80.0,
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_existing_team)

        # Should raise ProxyException because new budget exceeds org's max_budget
        with pytest.raises(ProxyException) as exc_info:
            await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=org_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == '400'
        assert "organization" in str(exc_info.value.message).lower() or "budget" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_update_team_standalone_models_exceeds_user_limit():
    """
    Test that /team/update for a standalone team fails when models are not in user's allowed models.

    Scenario:
    - User has personal models=['gpt-3.5-turbo']
    - Standalone team exists (no organization_id)
    - User tries to update team models to ['gpt-4'] (not in user's allowed models)
    - Expected: Should fail with error about model not in user's allowed models
    """
    from fastapi import Request

    from litellm.proxy._types import ProxyException, UpdateTeamRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Create non-admin user with restrictive personal models
    non_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="non-admin-update-models-test",
        models=["gpt-3.5-turbo"],  # Restrictive model list
    )

    # Create update request with model not in user's allowed list
    update_request = UpdateTeamRequest(
        team_id="standalone-team-models-123",
        models=["gpt-4"],  # Not in user's allowed models
    )

    dummy_request = MagicMock(spec=Request)

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ) as mock_audit:

        # Mock existing standalone team (no organization_id)
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "standalone-team-models-123"
        mock_existing_team.organization_id = None  # Standalone team
        mock_existing_team.models = ["gpt-3.5-turbo"]
        mock_existing_team.model_dump.return_value = {
            "team_id": "standalone-team-models-123",
            "organization_id": None,
            "models": ["gpt-3.5-turbo"],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_existing_team)

        # Should raise ProxyException because model not in user's allowed models
        with pytest.raises(ProxyException) as exc_info:
            await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=non_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == '400'
        assert "model" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_update_team_org_scoped_budget_bypasses_user_limit():
    """
    Test that /team/update for an org-scoped team does NOT validate budget against user's personal max_budget.

    Scenario:
    - Organization has max_budget=$100
    - User (org admin) has personal max_budget=$3
    - Org-scoped team exists with current budget=$30
    - User tries to update team budget to $50 (within org limit, exceeds user limit)
    - Expected: Should succeed (validated against org, not user)
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_OrganizationTable,
        LiteLLM_UserTable,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Create user with very restrictive personal budget ($3)
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-update-budget-test",
        models=[],
    )

    # Create update request with budget within org limit but exceeding user limit
    update_request = UpdateTeamRequest(
        team_id="org-team-update-budget-123",
        max_budget=50.0,  # Within org's $100 limit, exceeds user's $3 limit
    )

    dummy_request = MagicMock(spec=Request)

    # Mock organization with $100 budget limit
    mock_budget_table = MagicMock(spec=LiteLLM_BudgetTable)
    mock_budget_table.max_budget = 100.0

    mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
    mock_org.organization_id = "test-org-update-budget"
    mock_org.models = ["gpt-4", "gpt-3.5-turbo"]
    mock_org.litellm_budget_table = mock_budget_table

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ) as mock_audit, patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
        new=AsyncMock(return_value=mock_org)
    ) as mock_get_org:

        # Mock existing org-scoped team
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "org-team-update-budget-123"
        mock_existing_team.organization_id = "test-org-update-budget"
        mock_existing_team.max_budget = 30.0
        mock_existing_team.model_id = None
        mock_existing_team.model_dump.return_value = {
            "team_id": "org-team-update-budget-123",
            "organization_id": "test-org-update-budget",
            "max_budget": 30.0,
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_existing_team)
        mock_prisma.jsonify_team_object = lambda db_data: db_data

        # Mock user cache to return user with restrictive budget
        mock_user_obj = LiteLLM_UserTable(
            user_id="org-admin-update-budget-test",
            max_budget=3.0,  # Restrictive personal budget
        )
        mock_cache.async_get_cache = AsyncMock(return_value=mock_user_obj)
        mock_cache.async_set_cache = AsyncMock()  # Mock cache set for _cache_team_object

        # Mock team update
        mock_updated_team = MagicMock()
        mock_updated_team.team_id = "org-team-update-budget-123"
        mock_updated_team.organization_id = "test-org-update-budget"
        mock_updated_team.max_budget = 50.0
        mock_updated_team.litellm_model_table = None
        mock_updated_team.model_dump.return_value = {
            "team_id": "org-team-update-budget-123",
            "organization_id": "test-org-update-budget",
            "max_budget": 50.0,
        }
        mock_prisma.db.litellm_teamtable.update = AsyncMock(return_value=mock_updated_team)

        # Should NOT raise an exception - bypass user budget validation for org-scoped teams
        result = await update_team(
            data=update_request,
            http_request=dummy_request,
            user_api_key_dict=org_admin_user,
        )

        # Verify the team was updated successfully with the higher budget
        assert result is not None
        assert result["data"].max_budget == 50.0


@pytest.mark.asyncio
async def test_update_team_org_scoped_models_bypasses_user_limit():
    """
    Test that /team/update for an org-scoped team does NOT validate models against user's personal models.

    Scenario:
    - Organization has models=['gpt-4', 'gpt-3.5-turbo', 'claude-3-opus']
    - User (org admin) has personal models=['no-default-models']
    - Org-scoped team exists
    - User tries to update team models to ['gpt-4'] (in org's allowed, not in user's)
    - Expected: Should succeed (validated against org, not user)
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_OrganizationTable,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Create user with very restrictive personal models
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-update-models-test",
        models=["no-default-models"],  # Restrictive model list
    )

    # Create update request with models in org's allowed but not in user's
    update_request = UpdateTeamRequest(
        team_id="org-team-update-models-123",
        models=["gpt-4"],  # In org's allowed, not in user's
    )

    dummy_request = MagicMock(spec=Request)

    # Mock organization with generous model list
    mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
    mock_org.organization_id = "test-org-update-models"
    mock_org.models = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"]
    mock_org.litellm_budget_table = None

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ) as mock_audit, patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
        new=AsyncMock(return_value=mock_org)
    ) as mock_get_org:

        # Mock existing org-scoped team
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "org-team-update-models-123"
        mock_existing_team.organization_id = "test-org-update-models"
        mock_existing_team.models = ["gpt-3.5-turbo"]
        mock_existing_team.model_id = None
        mock_existing_team.model_dump.return_value = {
            "team_id": "org-team-update-models-123",
            "organization_id": "test-org-update-models",
            "models": ["gpt-3.5-turbo"],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_existing_team)
        mock_prisma.jsonify_team_object = lambda db_data: db_data
        mock_cache.async_set_cache = AsyncMock()  # Mock cache set for _cache_team_object

        # Mock team update
        mock_updated_team = MagicMock()
        mock_updated_team.team_id = "org-team-update-models-123"
        mock_updated_team.organization_id = "test-org-update-models"
        mock_updated_team.models = ["gpt-4"]
        mock_updated_team.litellm_model_table = None
        mock_updated_team.model_dump.return_value = {
            "team_id": "org-team-update-models-123",
            "organization_id": "test-org-update-models",
            "models": ["gpt-4"],
        }
        mock_prisma.db.litellm_teamtable.update = AsyncMock(return_value=mock_updated_team)

        # Should NOT raise an exception - bypass user models validation for org-scoped teams
        result = await update_team(
            data=update_request,
            http_request=dummy_request,
            user_api_key_dict=org_admin_user,
        )

        # Verify the team was updated successfully with the new models
        assert result is not None
        assert result["data"].models == ["gpt-4"]


@pytest.mark.asyncio
async def test_update_team_org_scoped_models_not_in_org_models():
    """
    Test that /team/update for an org-scoped team fails when models are not in organization's allowed models.

    Scenario:
    - Organization has models=['gpt-4', 'gpt-3.5-turbo']
    - Org-scoped team exists
    - User tries to update team models to ['claude-3-opus'] (not in org's allowed models)
    - Expected: Should fail with error about model not in org's allowed models
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_OrganizationTable,
        ProxyException,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Create user (org admin)
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-update-models-fail-test",
        models=[],
    )

    # Create update request with model not in org's allowed list
    update_request = UpdateTeamRequest(
        team_id="org-team-update-models-fail-123",
        models=["claude-3-opus"],  # Not in org's allowed models
    )

    dummy_request = MagicMock(spec=Request)

    # Mock organization with restricted model list (no claude-3-opus)
    mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
    mock_org.organization_id = "test-org-update-models-fail"
    mock_org.models = ["gpt-4", "gpt-3.5-turbo"]  # claude-3-opus is NOT allowed
    mock_org.litellm_budget_table = None

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ) as mock_audit, patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
        new=AsyncMock(return_value=mock_org)
    ) as mock_get_org:

        # Mock existing org-scoped team
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "org-team-update-models-fail-123"
        mock_existing_team.organization_id = "test-org-update-models-fail"
        mock_existing_team.models = ["gpt-4"]
        mock_existing_team.model_dump.return_value = {
            "team_id": "org-team-update-models-fail-123",
            "organization_id": "test-org-update-models-fail",
            "models": ["gpt-4"],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_existing_team)

        # Should raise ProxyException because claude-3-opus is not in org's allowed models
        with pytest.raises(ProxyException) as exc_info:
            await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=org_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == '400'
        assert "claude-3-opus" in str(exc_info.value.message) or "organization" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_update_team_tpm_limit_exceeds_user_limit():
    """
    Test that /team/update fails when TPM limit exceeds user's TPM limit.

    Scenario:
    - User has tpm_limit=1000
    - User tries to update team with tpm_limit=5000
    - Expected: Should fail with error about exceeding user TPM limit
    """
    from fastapi import Request

    from litellm.proxy._types import ProxyException, UpdateTeamRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Create non-admin user with TPM limit
    non_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="tpm-limit-user",
        models=[],
        tpm_limit=1000,  # User's TPM limit
    )

    # Create update request with TPM exceeding user's limit
    update_request = UpdateTeamRequest(
        team_id="team-tpm-test-123",
        tpm_limit=5000,  # Exceeds user's 1000 limit
    )

    dummy_request = MagicMock(spec=Request)

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ):

        # Mock existing standalone team
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "team-tpm-test-123"
        mock_existing_team.organization_id = None
        mock_existing_team.tpm_limit = 500
        mock_existing_team.model_dump.return_value = {
            "team_id": "team-tpm-test-123",
            "organization_id": None,
            "tpm_limit": 500,
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_existing_team)

        # Should raise ProxyException because new TPM exceeds user's limit
        with pytest.raises(ProxyException) as exc_info:
            await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=non_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == '400'
        assert "tpm" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_update_team_rpm_limit_exceeds_user_limit():
    """
    Test that /team/update fails when RPM limit exceeds user's RPM limit.

    Scenario:
    - User has rpm_limit=100
    - User tries to update team with rpm_limit=500
    - Expected: Should fail with error about exceeding user RPM limit
    """
    from fastapi import Request

    from litellm.proxy._types import ProxyException, UpdateTeamRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Create non-admin user with RPM limit
    non_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="rpm-limit-user",
        models=[],
        rpm_limit=100,  # User's RPM limit
    )

    # Create update request with RPM exceeding user's limit
    update_request = UpdateTeamRequest(
        team_id="team-rpm-test-123",
        rpm_limit=500,  # Exceeds user's 100 limit
    )

    dummy_request = MagicMock(spec=Request)

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ):

        # Mock existing standalone team
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "team-rpm-test-123"
        mock_existing_team.organization_id = None
        mock_existing_team.rpm_limit = 50
        mock_existing_team.model_dump.return_value = {
            "team_id": "team-rpm-test-123",
            "organization_id": None,
            "rpm_limit": 50,
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_existing_team)

        # Should raise ProxyException because new RPM exceeds user's limit
        with pytest.raises(ProxyException) as exc_info:
            await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=non_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == '400'
        assert "rpm" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_new_team_org_scoped_tpm_exceeds_org_limit():
    """
    Test that /team/new for an org-scoped team fails when TPM exceeds organization's TPM limit.

    Scenario:
    - Organization has tpm_limit=10000
    - User tries to create org-scoped team with tpm_limit=20000
    - Expected: Should fail with error about exceeding org TPM limit
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_OrganizationTable,
        NewTeamRequest,
        ProxyException,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Create user (with restrictive personal TPM limit that should be bypassed)
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-tpm-test",
        models=[],
        tpm_limit=1000,  # User's personal limit (should be bypassed for org teams)
    )

    # Create team request with TPM exceeding org's limit
    team_request = NewTeamRequest(
        team_alias="org-tpm-test-team",
        organization_id="test-org-tpm",
        tpm_limit=20000,  # Exceeds org's 10000 limit
    )

    dummy_request = MagicMock(spec=Request)

    # Mock organization with TPM limit
    mock_budget_table = MagicMock(spec=LiteLLM_BudgetTable)
    mock_budget_table.tpm_limit = 10000  # Org's TPM limit
    mock_budget_table.rpm_limit = None
    mock_budget_table.max_budget = None

    mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
    mock_org.organization_id = "test-org-tpm"
    mock_org.models = ["gpt-4"]
    mock_org.litellm_budget_table = mock_budget_table

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server._license_check"
    ) as mock_license, patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
        new=AsyncMock(return_value=mock_org)
    ):
        mock_license.is_team_count_over_limit.return_value = False
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_prisma.get_data = AsyncMock(return_value=None)

        # Should raise ProxyException because TPM exceeds org limit
        with pytest.raises(ProxyException) as exc_info:
            await new_team(
                data=team_request,
                http_request=dummy_request,
                user_api_key_dict=org_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == '400'
        assert "tpm" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_new_team_org_scoped_rpm_exceeds_org_limit():
    """
    Test that /team/new for an org-scoped team fails when RPM exceeds organization's RPM limit.

    Scenario:
    - Organization has rpm_limit=1000
    - User tries to create org-scoped team with rpm_limit=2000
    - Expected: Should fail with error about exceeding org RPM limit
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_OrganizationTable,
        NewTeamRequest,
        ProxyException,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Create user (with restrictive personal RPM limit that should be bypassed)
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-rpm-test",
        models=[],
        rpm_limit=100,  # User's personal limit (should be bypassed for org teams)
    )

    # Create team request with RPM exceeding org's limit
    team_request = NewTeamRequest(
        team_alias="org-rpm-test-team",
        organization_id="test-org-rpm",
        rpm_limit=2000,  # Exceeds org's 1000 limit
    )

    dummy_request = MagicMock(spec=Request)

    # Mock organization with RPM limit
    mock_budget_table = MagicMock(spec=LiteLLM_BudgetTable)
    mock_budget_table.tpm_limit = None
    mock_budget_table.rpm_limit = 1000  # Org's RPM limit
    mock_budget_table.max_budget = None

    mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
    mock_org.organization_id = "test-org-rpm"
    mock_org.models = ["gpt-4"]
    mock_org.litellm_budget_table = mock_budget_table

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server._license_check"
    ) as mock_license, patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
        new=AsyncMock(return_value=mock_org)
    ):
        mock_license.is_team_count_over_limit.return_value = False
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_prisma.get_data = AsyncMock(return_value=None)

        # Should raise ProxyException because RPM exceeds org limit
        with pytest.raises(ProxyException) as exc_info:
            await new_team(
                data=team_request,
                http_request=dummy_request,
                user_api_key_dict=org_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == '400'
        assert "rpm" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_new_team_org_scoped_tpm_rpm_bypasses_user_limit():
    """
    Test that /team/new for an org-scoped team bypasses user's TPM/RPM limits.

    Scenario:
    - User has tpm_limit=1000, rpm_limit=100
    - Organization has tpm_limit=50000, rpm_limit=5000
    - User creates org-scoped team with tpm_limit=10000, rpm_limit=1000
    - Expected: Should succeed (bypasses user limits, within org limits)
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_OrganizationTable,
        LiteLLM_TeamTable,
        NewTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Create user with restrictive personal limits
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-bypass-test",
        models=[],
        tpm_limit=1000,  # Restrictive user TPM limit
        rpm_limit=100,   # Restrictive user RPM limit
    )

    # Create team request exceeding user limits but within org limits
    team_request = NewTeamRequest(
        team_alias="org-bypass-test-team",
        organization_id="test-org-bypass",
        tpm_limit=10000,  # Exceeds user's 1000 but within org's 50000
        rpm_limit=1000,   # Exceeds user's 100 but within org's 5000
    )

    dummy_request = MagicMock(spec=Request)

    # Mock organization with generous limits
    mock_budget_table = MagicMock(spec=LiteLLM_BudgetTable)
    mock_budget_table.tpm_limit = 50000  # Generous org TPM limit
    mock_budget_table.rpm_limit = 5000   # Generous org RPM limit
    mock_budget_table.max_budget = None

    mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
    mock_org.organization_id = "test-org-bypass"
    mock_org.models = ["gpt-4"]
    mock_org.litellm_budget_table = mock_budget_table

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server._license_check"
    ) as mock_license, patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
        new=AsyncMock(return_value=mock_org)
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._add_team_members_to_team",
        new=AsyncMock()
    ):
        mock_license.is_team_count_over_limit.return_value = False
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_prisma.get_data = AsyncMock(return_value=None)

        # Mock team creation
        mock_created_team = MagicMock(spec=LiteLLM_TeamTable)
        mock_created_team.team_id = "new-bypass-team-id"
        mock_created_team.team_alias = "org-bypass-test-team"
        mock_created_team.tpm_limit = 10000
        mock_created_team.rpm_limit = 1000
        mock_created_team.metadata = None
        mock_created_team.members_with_roles = []
        mock_created_team.model_dump.return_value = {
            "team_id": "new-bypass-team-id",
            "team_alias": "org-bypass-test-team",
            "tpm_limit": 10000,
            "rpm_limit": 1000,
            "metadata": None,
            "members_with_roles": [],
        }
        mock_prisma.db.litellm_teamtable.create = AsyncMock(return_value=mock_created_team)
        mock_prisma.db.litellm_teamtable.update = AsyncMock(return_value=mock_created_team)
        mock_prisma.jsonify_team_object = MagicMock(side_effect=lambda db_data: db_data)

        # Should succeed - bypasses user limits since org-scoped
        result = await new_team(
            data=team_request,
            http_request=dummy_request,
            user_api_key_dict=org_admin_user,
        )

        # Verify team was created
        assert result["team_id"] == "new-bypass-team-id"


@pytest.mark.asyncio
async def test_update_team_org_scoped_tpm_exceeds_org_limit():
    """
    Test that /team/update for an org-scoped team fails when TPM exceeds organization's TPM limit.

    Scenario:
    - Organization has tpm_limit=10000
    - User tries to update org-scoped team with tpm_limit=20000
    - Expected: Should fail with error about exceeding org TPM limit
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_OrganizationTable,
        ProxyException,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Create user (with restrictive personal TPM limit that should be bypassed)
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-update-tpm-test",
        models=[],
        tpm_limit=1000,  # User's personal limit (should be bypassed for org teams)
    )

    # Create update request with TPM exceeding org's limit
    update_request = UpdateTeamRequest(
        team_id="org-team-update-tpm-123",
        tpm_limit=20000,  # Exceeds org's 10000 limit
    )

    dummy_request = MagicMock(spec=Request)

    # Mock organization with TPM limit
    mock_budget_table = MagicMock(spec=LiteLLM_BudgetTable)
    mock_budget_table.tpm_limit = 10000  # Org's TPM limit
    mock_budget_table.rpm_limit = None
    mock_budget_table.max_budget = None

    mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
    mock_org.organization_id = "test-org-update-tpm"
    mock_org.models = ["gpt-4"]
    mock_org.litellm_budget_table = mock_budget_table

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
        new=AsyncMock(return_value=mock_org)
    ):

        # Mock existing org-scoped team
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "org-team-update-tpm-123"
        mock_existing_team.organization_id = "test-org-update-tpm"
        mock_existing_team.tpm_limit = 5000
        mock_existing_team.model_dump.return_value = {
            "team_id": "org-team-update-tpm-123",
            "organization_id": "test-org-update-tpm",
            "tpm_limit": 5000,
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_existing_team)

        # Should raise ProxyException because TPM exceeds org limit
        with pytest.raises(ProxyException) as exc_info:
            await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=org_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == '400'
        assert "tpm" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_update_team_org_scoped_rpm_exceeds_org_limit():
    """
    Test that /team/update for an org-scoped team fails when RPM exceeds organization's RPM limit.

    Scenario:
    - Organization has rpm_limit=1000
    - User tries to update org-scoped team with rpm_limit=2000
    - Expected: Should fail with error about exceeding org RPM limit
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_OrganizationTable,
        ProxyException,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Create user (with restrictive personal RPM limit that should be bypassed)
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-update-rpm-test",
        models=[],
        rpm_limit=100,  # User's personal limit (should be bypassed for org teams)
    )

    # Create update request with RPM exceeding org's limit
    update_request = UpdateTeamRequest(
        team_id="org-team-update-rpm-123",
        rpm_limit=2000,  # Exceeds org's 1000 limit
    )

    dummy_request = MagicMock(spec=Request)

    # Mock organization with RPM limit
    mock_budget_table = MagicMock(spec=LiteLLM_BudgetTable)
    mock_budget_table.tpm_limit = None
    mock_budget_table.rpm_limit = 1000  # Org's RPM limit
    mock_budget_table.max_budget = None

    mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
    mock_org.organization_id = "test-org-update-rpm"
    mock_org.models = ["gpt-4"]
    mock_org.litellm_budget_table = mock_budget_table

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
        new=AsyncMock(return_value=mock_org)
    ):

        # Mock existing org-scoped team
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "org-team-update-rpm-123"
        mock_existing_team.organization_id = "test-org-update-rpm"
        mock_existing_team.rpm_limit = 500
        mock_existing_team.model_dump.return_value = {
            "team_id": "org-team-update-rpm-123",
            "organization_id": "test-org-update-rpm",
            "rpm_limit": 500,
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_existing_team)

        # Should raise ProxyException because RPM exceeds org limit
        with pytest.raises(ProxyException) as exc_info:
            await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=org_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == '400'
        assert "rpm" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_update_team_org_scoped_tpm_rpm_bypasses_user_limit():
    """
    Test that /team/update for an org-scoped team bypasses user's TPM/RPM limits.

    Scenario:
    - User has tpm_limit=1000, rpm_limit=100
    - Organization has tpm_limit=50000, rpm_limit=5000
    - User updates org-scoped team with tpm_limit=10000, rpm_limit=1000
    - Expected: Should succeed (bypasses user limits, within org limits)
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_BudgetTable,
        LiteLLM_OrganizationTable,
        LiteLLM_TeamTable,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Create user with restrictive personal limits
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-update-bypass-test",
        models=[],
        tpm_limit=1000,  # Restrictive user TPM limit
        rpm_limit=100,   # Restrictive user RPM limit
    )

    # Create update request exceeding user limits but within org limits
    update_request = UpdateTeamRequest(
        team_id="org-team-update-bypass-123",
        tpm_limit=10000,  # Exceeds user's 1000 but within org's 50000
        rpm_limit=1000,   # Exceeds user's 100 but within org's 5000
    )

    dummy_request = MagicMock(spec=Request)

    # Mock organization with generous limits
    mock_budget_table = MagicMock(spec=LiteLLM_BudgetTable)
    mock_budget_table.tpm_limit = 50000  # Generous org TPM limit
    mock_budget_table.rpm_limit = 5000   # Generous org RPM limit
    mock_budget_table.max_budget = None

    mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
    mock_org.organization_id = "test-org-update-bypass"
    mock_org.models = ["gpt-4"]
    mock_org.litellm_budget_table = mock_budget_table

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.proxy_logging_obj"
    ) as mock_logging, patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
        new=AsyncMock(return_value=mock_org)
    ):

        # Mock existing org-scoped team
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "org-team-update-bypass-123"
        mock_existing_team.organization_id = "test-org-update-bypass"
        mock_existing_team.tpm_limit = 5000
        mock_existing_team.rpm_limit = 500
        mock_existing_team.model_id = None
        mock_existing_team.model_dump.return_value = {
            "team_id": "org-team-update-bypass-123",
            "organization_id": "test-org-update-bypass",
            "tpm_limit": 5000,
            "rpm_limit": 500,
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=mock_existing_team)
        mock_cache.async_set_cache = AsyncMock()

        # Mock team update
        mock_updated_team = MagicMock(spec=LiteLLM_TeamTable)
        mock_updated_team.team_id = "org-team-update-bypass-123"
        mock_updated_team.tpm_limit = 10000
        mock_updated_team.rpm_limit = 1000
        mock_updated_team.model_dump.return_value = {
            "team_id": "org-team-update-bypass-123",
            "tpm_limit": 10000,
            "rpm_limit": 1000,
        }
        mock_prisma.db.litellm_teamtable.update = AsyncMock(return_value=mock_updated_team)
        mock_prisma.jsonify_team_object = MagicMock(side_effect=lambda db_data: db_data)

        # Should succeed - bypasses user limits since org-scoped
        result = await update_team(
            data=update_request,
            http_request=dummy_request,
            user_api_key_dict=org_admin_user,
        )

        # Verify team was updated
        assert result["team_id"] == "org-team-update-bypass-123"


@pytest.mark.asyncio
async def test_update_team_guardrails_with_org_id():
    """
    Test that updating team guardrails works when team has an organization_id.
    The fix ensures 'teams' field is included when fetching organization data.
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_OrganizationTable,
        LiteLLM_TeamTable,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Create user (org admin)
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-guardrails-test",
        models=[],
    )

    # Update request to add guardrails to team
    update_request = UpdateTeamRequest(
        team_id="team-guardrails-123",
        guardrails=["aporia-pre-call", "aporia-post-call"],
        organization_id="test-org-guardrails",  # Changing org triggers fetch_and_validate_organization
    )

    dummy_request = MagicMock(spec=Request)

    # Mock organization with all required fields including teams (the fix)
    from datetime import datetime
    mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
    mock_org.organization_id = "test-org-guardrails"
    mock_org.models = ["gpt-4", "gpt-3.5-turbo"]
    mock_org.budget_id = "budget-123"
    mock_org.created_by = "admin"
    mock_org.updated_by = "admin"
    mock_org.created_at = datetime(2024, 1, 1)
    mock_org.updated_at = datetime(2024, 1, 1)
    mock_org.litellm_budget_table = None
    mock_org.members = []
    mock_org.teams = []  # Must be a list, not None
    mock_org.model_dump.return_value = {
        "organization_id": "test-org-guardrails",
        "models": ["gpt-4", "gpt-3.5-turbo"],
        "budget_id": "budget-123",
        "created_by": "admin",
        "updated_by": "admin",
        "created_at": datetime(2024, 1, 1),
        "updated_at": datetime(2024, 1, 1),
        "litellm_budget_table": None,
        "members": [],
        "teams": [],
    }

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma, patch(
        "litellm.proxy.proxy_server.user_api_key_cache"
    ) as mock_cache, patch(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"
    ), patch(
        "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
    ), patch(
        "litellm.proxy.proxy_server.premium_user", True  # Required for guardrails feature
    ):
        # Mock existing team - must have compatible models with organization
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "team-guardrails-123"
        mock_existing_team.organization_id = None
        mock_existing_team.metadata = {}
        mock_existing_team.model_id = None
        mock_existing_team.models = ["gpt-4"]  # Subset of org models to pass validation
        mock_existing_team.max_budget = None
        mock_existing_team.tpm_limit = None
        mock_existing_team.rpm_limit = None
        mock_existing_team.model_dump.return_value = {
            "team_id": "team-guardrails-123",
            "organization_id": None,
            "metadata": {},
            "models": ["gpt-4"],
            "max_budget": None,
            "tpm_limit": None,
            "rpm_limit": None,
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )
        mock_cache.async_set_cache = AsyncMock()

        # Mock organization fetch - this is where the bug occurred
        # The fix ensures 'teams: True' is in the include clause
        mock_prisma.db.litellm_organizationtable.find_unique = AsyncMock(
            return_value=mock_org
        )

        # Mock team update
        mock_updated_team = MagicMock(spec=LiteLLM_TeamTable)
        mock_updated_team.team_id = "team-guardrails-123"
        mock_updated_team.organization_id = "test-org-guardrails"
        mock_updated_team.metadata = {"guardrails": ["aporia-pre-call", "aporia-post-call"]}
        mock_updated_team.litellm_model_table = None
        mock_updated_team.model_dump.return_value = {
            "team_id": "team-guardrails-123",
            "organization_id": "test-org-guardrails",
            "metadata": {"guardrails": ["aporia-pre-call", "aporia-post-call"]},
        }
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )
        mock_prisma.jsonify_team_object = MagicMock(side_effect=lambda db_data: db_data)

        # Mock llm_router
        mock_router = MagicMock()
        with patch("litellm.proxy.proxy_server.llm_router", mock_router):
            # This should succeed without Pydantic validation error
            result = await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=org_admin_user,
            )

            # Verify the team was updated successfully with guardrails
            assert result is not None
            assert result["data"].organization_id == "test-org-guardrails"
            assert result["data"].metadata["guardrails"] == ["aporia-pre-call", "aporia-post-call"]

            # Verify that organization fetch was called with proper include clause
            # The function is called twice: once by fetch_and_validate_organization (with include)
            # and once by get_org_object (without include). We verify the first call has 'teams'.
            assert mock_prisma.db.litellm_organizationtable.find_unique.call_count >= 1
            
            # Get the first call (from fetch_and_validate_organization)
            first_call_kwargs = mock_prisma.db.litellm_organizationtable.find_unique.call_args_list[0].kwargs
            
            # Verify that 'teams' is included in the fetch
            assert "include" in first_call_kwargs
            assert "teams" in first_call_kwargs["include"]
            assert first_call_kwargs["include"]["teams"] is True
