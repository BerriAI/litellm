import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final, Optional, cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock, call, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from litellm._uuid import uuid

from litellm.proxy._types import UserAPIKeyAuth  # Import UserAPIKeyAuth
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_BudgetTableFull,
    LiteLLM_ModelTable,
    LiteLLM_OrganizationMembershipTable,
    LiteLLM_OrganizationTable,
    LiteLLM_OrganizationTableWithMembers,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    LitellmUserRoles,
    Member,
    ProxyErrorTypes,
    ProxyException,
    ResetSpendRequest,
    TeamMemberAddRequest,
    TeamMemberUpdateRequest,
    UpdateTeamRequest,
)
from litellm.proxy.management_endpoints.team_endpoints import (
    user_api_key_auth,  # Assuming this dependency is needed
)
from litellm.proxy.management_endpoints.team_endpoints import (
    GetTeamMemberPermissionsResponse,
    UpdateTeamMemberPermissionsRequest,
    _STRIP_DELETED_TEAM_FROM_USERS_SQL,
    _persist_deleted_team_records,
    _save_deleted_team_records,
    _transform_teams_to_deleted_records,
    _update_model_table,
    _validate_and_populate_member_user_info,
    _validate_team_member_reset_spend_value,
    _verify_team_access,
    delete_team,
    list_available_teams,
    reset_team_member_spend_fn,
    router,
    team_member_add_duplication_check,
    team_member_delete,
    team_member_update,
    update_team,
    validate_team_org_change,
)
from litellm.proxy.management_helpers.access_group_team_sync import (
    TEAM_ADVISORY_LOCK_SQL,
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


def _wire_team_create_tx(prisma_client):
    """`/team/new` inserts the team and mirrors it onto the access groups in one transaction,
    so a mocked client has to hand its team table back out of `db.tx()`.

    A `/team/new` carrying members then adds them under the team's advisory lock, and those
    writes run on that lock's transaction, so `tx()` has to hand back the mocked tables too
    for the per-table assertions on `prisma_client.db.*` to keep seeing them."""

    @asynccontextmanager
    async def _tx():
        yield SimpleNamespace(
            litellm_teamtable=prisma_client.db.litellm_teamtable,
            query_raw=AsyncMock(return_value=[]),
        )

    prisma_client.db.tx = lambda *_args, **_kwargs: _tx()
    _wire_member_add_tx(prisma_client)


def _wire_member_add_tx(prisma_client):
    """/team/member_add takes the team's advisory lock, re-reads the roster under it, and runs
    the user, budget, and membership writes on that same transaction, so a mocked client has
    to hand its own table mocks back out of `tx()`.

    Tables resolve on access, not here, since tests routinely replace `db.<table>` after
    wiring the transaction."""

    class _Tx:
        query_raw = AsyncMock(return_value=[{"members_with_roles": []}])

        def __getattr__(self, table_name):
            return getattr(prisma_client.db, table_name)

    tx = _Tx()
    tx_cm = MagicMock()
    tx_cm.__aenter__ = AsyncMock(return_value=tx)
    tx_cm.__aexit__ = AsyncMock(return_value=None)
    prisma_client.tx = MagicMock(return_value=tx_cm)


def _wire_member_delete_tx(prisma_client):
    """/team/member_delete's four cleanups, plus the advisory-lock re-read that now guards
    them, run inside one transaction, so a mocked client has to hand back its own table
    mocks (and a `query_raw` that answers the locked re-read from the same team row the
    test already configured on `find_unique`) out of `tx()` for the existing per-table
    assertions to keep seeing the calls."""

    async def _query_raw(sql, team_id):
        if sql != TEAM_ADVISORY_LOCK_SQL:
            team_row = await prisma_client.db.litellm_teamtable.find_unique(where={"team_id": team_id})
            if team_row is not None:
                return [{"members_with_roles": team_row.model_dump()["members_with_roles"]}]
        return []

    class _Tx:
        query_raw = staticmethod(_query_raw)

        def __getattr__(self, table_name):
            return getattr(prisma_client.db, table_name)

    tx = _Tx()
    tx_cm = MagicMock()
    tx_cm.__aenter__ = AsyncMock(return_value=tx)
    tx_cm.__aexit__ = AsyncMock(return_value=None)
    prisma_client.tx = MagicMock(return_value=tx_cm)


def _wire_team_delete_tx(prisma_client):
    """`/team/delete` deletes the team rows and runs its post-delete reference sweep under
    every team's advisory lock in one transaction, so a mocked client has to hand its own
    table mocks (and db-level execute_raw) back out of `tx()` for existing per-table
    assertions on `prisma_client.db.*` to keep seeing those calls."""
    tx = SimpleNamespace(
        litellm_teamtable=prisma_client.db.litellm_teamtable,
        litellm_teammembership=prisma_client.db.litellm_teammembership,
        query_raw=AsyncMock(return_value=[]),
        execute_raw=prisma_client.db.execute_raw,
    )
    tx_cm = MagicMock()
    tx_cm.__aenter__ = AsyncMock(return_value=tx)
    tx_cm.__aexit__ = AsyncMock(return_value=None)
    prisma_client.tx = MagicMock(return_value=tx_cm)


# Mock prisma_client
mock_prisma_client = MagicMock()
# Set up async mock for db operations
mock_prisma_client.db = MagicMock()
mock_prisma_client.db.litellm_teamtable = MagicMock()
mock_prisma_client.db.litellm_teamtable.update = AsyncMock()
mock_prisma_client.db.litellm_auditlog = MagicMock()
mock_prisma_client.db.litellm_auditlog.create = AsyncMock()


# Fixture to provide the mock prisma client
@pytest.fixture(autouse=True)
def mock_db_client():
    with patch(
        "litellm.proxy.proxy_server.prisma_client", mock_prisma_client
    ):  # Mock in both places if necessary
        yield mock_prisma_client
    mock_prisma_client.reset_mock()


@pytest.fixture
def disable_audit_logging_for_mocked_team(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("litellm.store_audit_logs", False)


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
@pytest.mark.parametrize("field", ["budget_duration", "team_member_budget_duration"])
@pytest.mark.parametrize("bad_duration", ["0s", "-5m"])
async def test_new_team_rejects_a_duration_that_never_advances(
    mock_db_client, mock_admin_auth, field, bad_duration
):
    """A zero-length window resets to "now", so the team row is due again the
    moment it is written. The reset job re-reads such rows on every tick, and a
    tenant with enough of them fills each batch and starves other tenants.
    """
    from fastapi import Request

    from litellm.proxy._types import NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    mock_db_client.db = MagicMock()
    mock_team_create = AsyncMock()
    mock_db_client.db.litellm_teamtable = MagicMock()
    mock_db_client.db.litellm_teamtable.create = mock_team_create
    _wire_team_create_tx(mock_db_client)

    with pytest.raises(ProxyException) as exc_info:
        await new_team(
            data=NewTeamRequest(team_alias="my-team", **{field: bad_duration}),
            http_request=MagicMock(spec=Request),
            user_api_key_dict=mock_admin_auth,
        )

    assert str(exc_info.value.code) == "400"
    assert "Invalid budget_duration" in str(exc_info.value.message)
    mock_team_create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["budget_duration", "team_member_budget_duration"])
async def test_update_team_rejects_a_duration_that_never_advances(
    mock_db_client, mock_admin_auth, field
):
    """/team/update must reject the same never-advancing durations /team/new does."""
    from fastapi import Request

    from litellm.proxy._types import UpdateTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    mock_db_client.db = MagicMock()
    mock_find_unique = AsyncMock(return_value=None)
    mock_db_client.db.litellm_teamtable = MagicMock()
    mock_db_client.db.litellm_teamtable.find_unique = mock_find_unique

    with pytest.raises(ProxyException) as exc_info:
        await update_team(
            data=UpdateTeamRequest(team_id="team-1", **{field: "0s"}),
            http_request=MagicMock(spec=Request),
            user_api_key_dict=mock_admin_auth,
        )

    assert str(exc_info.value.code) == "400"
    assert "Invalid budget_duration" in str(exc_info.value.message)
    mock_find_unique.assert_not_awaited()


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
    _wire_team_create_tx(mock_db_client)
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
    mock_db_client.db.litellm_teamtable.create = AsyncMock(
        return_value=team_create_result
    )
    _wire_team_create_tx(mock_db_client)
    mock_db_client.db.litellm_teamtable.count = AsyncMock(return_value=0)
    mock_db_client.db.litellm_teamtable.update = AsyncMock(
        return_value=team_create_result
    )

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


@pytest.mark.parametrize(
    "user_role,user_id,flag_value,expected",
    [
        (LitellmUserRoles.PROXY_ADMIN, "admin-1", True, False),
        (LitellmUserRoles.PROXY_ADMIN, "admin-1", False, True),
        (LitellmUserRoles.PROXY_ADMIN, "admin-1", None, True),
        (LitellmUserRoles.INTERNAL_USER, "user-1", True, True),
        (LitellmUserRoles.ORG_ADMIN, "org-admin-1", True, True),
        (LitellmUserRoles.PROXY_ADMIN, None, False, False),
    ],
)
def test_should_auto_add_team_creator(user_role, user_id, flag_value, expected):
    from litellm.proxy.management_endpoints.team_endpoints import (
        _should_auto_add_team_creator,
    )

    general_settings = (
        {} if flag_value is None else {"disable_auto_add_proxy_admin_to_teams": flag_value}
    )
    auth = UserAPIKeyAuth(user_role=user_role, user_id=user_id)
    assert _should_auto_add_team_creator(auth, general_settings) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "disable_flag,expect_creator_added", [(True, False), (False, True)]
)
async def test_new_team_disable_auto_add_proxy_admin_flag(
    mock_db_client, disable_flag, expect_creator_added
):
    """
    When general_settings.disable_auto_add_proxy_admin_to_teams is True, a proxy
    admin calling /team/new must NOT be auto-added to the team's members. When
    the flag is off, the creator is auto-added as a team admin (default
    behavior, regression guard for LIT-3739).
    """
    mock_db_client.jsonify_team_object = lambda db_data: db_data
    mock_db_client.get_data = AsyncMock(return_value=None)
    mock_db_client.update_data = AsyncMock(return_value=MagicMock())
    mock_db_client.db = MagicMock()

    team_create_result = MagicMock(team_id="team-789")
    team_create_result.model_dump.return_value = {"team_id": "team-789"}
    mock_db_client.db.litellm_teamtable = MagicMock()
    mock_db_client.db.litellm_teamtable.create = AsyncMock(
        return_value=team_create_result
    )
    _wire_team_create_tx(mock_db_client)
    mock_db_client.db.litellm_teamtable.count = AsyncMock(return_value=0)
    mock_db_client.db.litellm_usertable = MagicMock()
    mock_db_client.db.litellm_usertable.update = AsyncMock(return_value=MagicMock())

    from fastapi import Request

    from litellm.proxy._types import NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    admin_auth = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-user-1"
    )

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"disable_auto_add_proxy_admin_to_teams": disable_flag},
    ), patch(
        "litellm.proxy.management_endpoints.team_endpoints._add_team_members_to_team",
        new_callable=AsyncMock,
    ) as mock_add_members:
        await new_team(
            data=NewTeamRequest(team_alias="flag-test-team"),
            http_request=MagicMock(spec=Request),
            user_api_key_dict=admin_auth,
        )

    mock_add_members.assert_called_once()
    member_add_request = mock_add_members.call_args.kwargs["data"]
    member_user_ids = [m.user_id for m in member_add_request.member]
    assert ("admin-user-1" in member_user_ids) is expect_creator_added


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
    assert result.team_member_budget_table == mock_budget_record
    assert result == team_info_response.model_copy(
        update={"team_member_budget_table": mock_budget_record}
    )
    assert team_info_response.team_member_budget_table is None

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
            "Team member budget table not found, passed team_member_budget_id=%s",
            "nonexistent-budget-456",
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


def _make_team_member_add_request(
    member_user_id: Optional[str] = "regular-user",
    role: str = "user",
    team_id: str = "test-team-123",
):
    """Build a TeamMemberAddRequest with one Member entry for tests below."""
    from litellm.proxy._types import Member, TeamMemberAddRequest

    return TeamMemberAddRequest(
        team_id=team_id,
        member=Member(role=role, user_id=member_user_id),
    )


@pytest.mark.asyncio
async def test_validate_team_member_add_permissions_admin():
    """
    Test _validate_team_member_add_permissions allows proxy admin
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_team_member_add_permissions,
    )

    admin_user = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)

    team = MagicMock(spec=LiteLLM_TeamTable)
    team.team_id = "test-team-123"

    await _validate_team_member_add_permissions(
        user_api_key_dict=admin_user,
        complete_team_data=team,
        data=_make_team_member_add_request(member_user_id="any-user", role="admin"),
    )


@pytest.mark.asyncio
async def test_validate_team_member_add_permissions_non_admin():
    """
    Test _validate_team_member_add_permissions raises exception for non-admin non-team-admin
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_team_member_add_permissions,
    )

    regular_user = UserAPIKeyAuth(
        user_id="regular-user",
        user_role=LitellmUserRoles.INTERNAL_USER,
        team_id="different-team",
    )

    team = MagicMock(spec=LiteLLM_TeamTable)
    team.team_id = "test-team-123"
    team.members_with_roles = []
    team.organization_id = None

    with (
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
            return_value=False,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_available_team",
            return_value=False,
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _validate_team_member_add_permissions(
                user_api_key_dict=regular_user,
                complete_team_data=team,
                data=_make_team_member_add_request(),
            )

        assert exc_info.value.status_code == 403
        assert "not proxy admin OR team admin" in str(exc_info.value.detail)


# ── VERIA-56 regression tests for _is_available_team self-join enforcement ───


@pytest.mark.asyncio
async def test_available_team_self_join_with_caller_user_id_allowed():
    """A standard user adding themselves to an available team with role=user
    is the only legitimate use of the available-team bypass."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_team_member_add_permissions,
    )

    user = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    team = MagicMock(spec=LiteLLM_TeamTable)
    team.team_id = "public-team"
    team.members_with_roles = []
    team.organization_id = None

    with (
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
            return_value=False,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_available_team",
            return_value=True,
        ),
    ):
        await _validate_team_member_add_permissions(
            user_api_key_dict=user,
            complete_team_data=team,
            data=_make_team_member_add_request(member_user_id="alice", role="user"),
        )


@pytest.mark.asyncio
async def test_available_team_self_join_blocks_admin_role():
    """Privesc shape from VERIA-56: caller adds themselves with role=admin
    via the available-team bypass.  Must be rejected."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_team_member_add_permissions,
    )

    user = UserAPIKeyAuth(user_id="alice", user_role=LitellmUserRoles.INTERNAL_USER)
    team = MagicMock(spec=LiteLLM_TeamTable)
    team.team_id = "public-team"
    team.members_with_roles = []
    team.organization_id = None

    with (
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
            return_value=False,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_available_team",
            return_value=True,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await _validate_team_member_add_permissions(
            user_api_key_dict=user,
            complete_team_data=team,
            data=_make_team_member_add_request(member_user_id="alice", role="admin"),
        )

    assert exc_info.value.status_code == 403
    assert "admin" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_available_team_self_join_blocks_other_user_id():
    """Cross-user-injection shape from VERIA-56: caller adds someone else
    via the available-team bypass.  Must be rejected."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_team_member_add_permissions,
    )

    user = UserAPIKeyAuth(user_id="alice", user_role=LitellmUserRoles.INTERNAL_USER)
    team = MagicMock(spec=LiteLLM_TeamTable)
    team.team_id = "public-team"
    team.members_with_roles = []
    team.organization_id = None

    with (
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
            return_value=False,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_available_team",
            return_value=True,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await _validate_team_member_add_permissions(
            user_api_key_dict=user,
            complete_team_data=team,
            data=_make_team_member_add_request(
                member_user_id="bob-victim", role="user"
            ),
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_available_team_self_join_blocks_when_caller_has_no_user_id():
    """If the auth context has no user_id we cannot prove self-join, so the
    bypass must fail closed."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_team_member_add_permissions,
    )

    user = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER)  # no user_id
    team = MagicMock(spec=LiteLLM_TeamTable)
    team.team_id = "public-team"
    team.members_with_roles = []
    team.organization_id = None

    with (
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
            return_value=False,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_available_team",
            return_value=True,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await _validate_team_member_add_permissions(
            user_api_key_dict=user,
            complete_team_data=team,
            data=_make_team_member_add_request(member_user_id="alice", role="user"),
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_available_team_self_join_blocks_email_only_member():
    """An email-only member entry can't be safely self-join-validated; the
    caller must use their own user_id explicitly."""
    from litellm.proxy._types import Member, TeamMemberAddRequest
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_team_member_add_permissions,
    )

    user = UserAPIKeyAuth(user_id="alice", user_role=LitellmUserRoles.INTERNAL_USER)
    team = MagicMock(spec=LiteLLM_TeamTable)
    team.team_id = "public-team"
    team.members_with_roles = []
    team.organization_id = None

    data = TeamMemberAddRequest(
        team_id="public-team",
        member=Member(role="user", user_email="alice@example.com"),
    )

    with (
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
            return_value=False,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_available_team",
            return_value=True,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await _validate_team_member_add_permissions(
            user_api_key_dict=user,
            complete_team_data=team,
            data=data,
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_available_team_self_join_blocks_admin_role_in_member_list():
    """Bulk shape: list of members where one has role=admin must be rejected
    even if the caller's own entry is correct."""
    from litellm.proxy._types import Member, TeamMemberAddRequest
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_team_member_add_permissions,
    )

    user = UserAPIKeyAuth(user_id="alice", user_role=LitellmUserRoles.INTERNAL_USER)
    team = MagicMock(spec=LiteLLM_TeamTable)
    team.team_id = "public-team"
    team.members_with_roles = []
    team.organization_id = None

    data = TeamMemberAddRequest(
        team_id="public-team",
        member=[
            Member(role="user", user_id="alice"),
            Member(role="admin", user_id="alice"),
        ],
    )

    with (
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
            return_value=False,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_available_team",
            return_value=True,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await _validate_team_member_add_permissions(
            user_api_key_dict=user,
            complete_team_data=team,
            data=data,
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget_control",
    [
        {"max_budget_in_team": 1000.0},
        {"budget_duration": "1h"},
        {"allowed_models": ["gpt-4o"]},
    ],
)
async def test_available_team_self_join_blocks_member_budget_controls(budget_control):
    """A self-joining non-admin must not be able to set their own per-member
    budget or model controls via the available-team bypass; only proxy/team/org
    admins may. Without this guard a self-joiner could shorten their budget
    reset window or widen their cap/model scope past the team default."""
    from litellm.proxy._types import Member, TeamMemberAddRequest
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_team_member_add_permissions,
    )

    user = UserAPIKeyAuth(user_id="alice", user_role=LitellmUserRoles.INTERNAL_USER)
    team = MagicMock(spec=LiteLLM_TeamTable)
    team.team_id = "public-team"
    team.members_with_roles = []
    team.organization_id = None

    data = TeamMemberAddRequest(
        team_id="public-team",
        member=Member(role="user", user_id="alice"),
        **budget_control,
    )

    with (
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
            return_value=False,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_available_team",
            return_value=True,
        ),
        pytest.raises(HTTPException) as exc_info,
    ):
        await _validate_team_member_add_permissions(
            user_api_key_dict=user,
            complete_team_data=team,
            data=data,
        )

    assert exc_info.value.status_code == 403
    assert "admin-only" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_available_team_self_join_allows_no_budget_controls():
    """The clean self-join (no per-member budget/model controls) must still be
    permitted, so the new guard does not break the legitimate join path."""
    from litellm.proxy._types import Member, TeamMemberAddRequest
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_team_member_add_permissions,
    )

    user = UserAPIKeyAuth(user_id="alice", user_role=LitellmUserRoles.INTERNAL_USER)
    team = MagicMock(spec=LiteLLM_TeamTable)
    team.team_id = "public-team"
    team.members_with_roles = []
    team.organization_id = None

    data = TeamMemberAddRequest(
        team_id="public-team",
        member=Member(role="user", user_id="alice"),
    )

    with (
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
            return_value=False,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_available_team",
            return_value=True,
        ),
    ):
        await _validate_team_member_add_permissions(
            user_api_key_dict=user,
            complete_team_data=team,
            data=data,
        )


@pytest.mark.asyncio
async def test_update_team_member_permissions_blocks_non_admin_via_available_team(
    mock_db_client,
):
    """A non-admin caller invoking /team/permissions_update on an available
    team must be rejected.  The previous code path delegated to
    ``_is_available_team`` and accepted the write; this PR removes that
    bypass entirely so the result is 403 even with the bypass mocked True."""
    test_team_id = "public-team"
    update_payload = {
        "team_id": test_team_id,
        "team_member_permissions": ["/key/generate"],
    }

    existing_row = MagicMock(spec=LiteLLM_TeamTable)
    existing_row.model_dump.return_value = {
        "team_id": test_team_id,
        "team_alias": "Public Team",
        "team_member_permissions": [],
        "spend": 0.0,
        "models": [],
    }
    existing_row.team_id = test_team_id
    existing_row.members_with_roles = []
    existing_row.organization_id = None

    non_admin_auth = UserAPIKeyAuth(
        user_id="alice",
        user_role=LitellmUserRoles.INTERNAL_USER,
    )

    with (
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
            new_callable=AsyncMock,
            return_value=existing_row,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
            return_value=False,
        ),
        patch(
            # Even with the available-team bypass mocked True, the endpoint
            # must NOT consult it any more — the gate should reject the
            # non-admin caller outright.
            "litellm.proxy.management_endpoints.team_endpoints._is_available_team",
            return_value=True,
        ),
    ):
        app.dependency_overrides[user_api_key_auth] = lambda: non_admin_auth
        try:
            response = client.post("/team/permissions_update", json=update_payload)
        finally:
            app.dependency_overrides = {}

    assert response.status_code == 403
    body = response.json()
    assert "permissions_update" in str(body) or "not proxy admin" in str(body)


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
    mock_team.default_team_member_models = None

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
            allowed_models=None,
            budget_duration=None,
            tx=None,
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
    mock_team.default_team_member_models = None

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


@pytest.mark.asyncio
async def test_add_team_members_reconciles_against_freshly_locked_row():
    """
    Regression: _add_team_members_to_team must build the new members_with_roles
    from the row it re-reads under the team's advisory lock, not from the stale
    complete_team_data snapshot captured at the start of the request.

    Two concurrent /team/member_add calls for the same team read the same
    snapshot; without the locked re-read the losing write rewrites the whole
    JSON array from its stale copy and silently drops the member the other call
    already committed. Here the snapshot holds only "zed", a concurrent writer
    has already committed "alice" (returned by the locked SELECT), and this call
    adds "bob". The write must contain all three.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        _add_team_members_to_team,
    )

    stale_snapshot = LiteLLM_TeamTable(
        team_id="test-team-lock",
        members_with_roles=[Member(user_id="zed", role="user")],
    )

    freshly_committed = [
        {"user_id": "zed", "user_email": None, "role": "user"},
        {"user_id": "alice", "user_email": None, "role": "user"},
    ]

    captured: dict = {}

    async def _capture_update(where, data):
        captured["data"] = data
        return LiteLLM_TeamTable(
            team_id="test-team-lock",
            members_with_roles=json.loads(data["members_with_roles"]),
        )

    tx = MagicMock()
    tx.query_raw = AsyncMock(return_value=[{"members_with_roles": freshly_committed}])
    tx.litellm_teamtable.update = AsyncMock(side_effect=_capture_update)

    tx_cm = MagicMock()
    tx_cm.__aenter__ = AsyncMock(return_value=tx)
    tx_cm.__aexit__ = AsyncMock(return_value=None)

    prisma_client = MagicMock()
    prisma_client.tx = MagicMock(return_value=tx_cm)

    with patch(
        "litellm.proxy.management_endpoints.team_endpoints._process_team_members",
        new=AsyncMock(return_value=([], [])),
    ):
        updated_team, _, _ = await _add_team_members_to_team(
            data=TeamMemberAddRequest(
                team_id="test-team-lock",
                member=Member(user_id="bob", role="user"),
            ),
            complete_team_data=stale_snapshot,
            prisma_client=cast(object, prisma_client),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
            litellm_proxy_admin_name="admin",
        )

    written_ids = sorted(m["user_id"] for m in json.loads(captured["data"]["members_with_roles"]))
    assert written_ids == ["alice", "bob", "zed"]

    assert tx.query_raw.call_args_list[0].args == (TEAM_ADVISORY_LOCK_SQL, "test-team-lock"), (
        "expected the team's advisory lock to be acquired before the members_with_roles read"
    )
    assert not any("FOR UPDATE" in str(call.args[0]) for call in tx.query_raw.call_args_list), (
        "a row lock here can deadlock with the access-group endpoints; only the advisory lock is safe"
    )

    assert [m.user_id for m in updated_team.members_with_roles] == ["zed", "alice", "bob"]


@pytest.mark.asyncio
async def test_add_team_members_runs_member_writes_on_the_lock_holding_transaction():
    """
    Regression pin against exhausting the connection pool with advisory-lock waiters.

    Every concurrent /team/member_add for one team holds a pooled connection while it waits
    on the team's advisory lock. If the holder's member writes went to the regular client,
    it would need a second connection to finish, so enough concurrent adds fill the pool
    with waiters and the holder can never commit or release the lock. The member writes
    therefore have to run on the transaction that already owns the connection.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        _add_team_members_to_team,
    )

    added_user = MagicMock()
    added_user.user_id = "bob"
    added_user.model_dump.return_value = {"user_id": "bob", "teams": ["team-pool"]}
    created_budget = MagicMock()
    created_budget.budget_id = "budget-pool"
    membership = MagicMock()
    membership.model_dump.return_value = {
        "team_id": "team-pool",
        "user_id": "bob",
        "budget_id": "budget-pool",
        "litellm_budget_table": None,
    }

    tx = MagicMock()
    tx.query_raw = AsyncMock(return_value=[{"members_with_roles": []}])
    tx.litellm_teamtable.update = AsyncMock(
        return_value=LiteLLM_TeamTable(team_id="team-pool", members_with_roles=[])
    )
    tx.litellm_usertable.upsert = AsyncMock(return_value=added_user)
    tx.litellm_usertable.update_many = AsyncMock()
    tx.litellm_budgettable.create = AsyncMock(return_value=created_budget)
    tx.litellm_teammembership.create = AsyncMock(return_value=membership)

    tx_cm = MagicMock()
    tx_cm.__aenter__ = AsyncMock(return_value=tx)
    tx_cm.__aexit__ = AsyncMock(return_value=None)

    prisma_client = MagicMock()
    prisma_client.tx = MagicMock(return_value=tx_cm)
    type(prisma_client).db = PropertyMock(
        side_effect=AssertionError("member writes must not reach for a second pooled connection")
    )

    _, updated_users, updated_team_memberships = await _add_team_members_to_team(
        data=TeamMemberAddRequest(
            team_id="team-pool",
            member=Member(user_id="bob", role="user"),
            max_budget_in_team=50.0,
        ),
        complete_team_data=LiteLLM_TeamTable(team_id="team-pool", members_with_roles=[]),
        prisma_client=cast(object, prisma_client),
        user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        litellm_proxy_admin_name="admin",
    )

    assert [user.user_id for user in updated_users] == ["bob"]
    assert [tm.budget_id for tm in updated_team_memberships] == ["budget-pool"]


@pytest.mark.asyncio
async def test_add_team_members_writes_nothing_when_the_team_is_deleted_mid_request():
    """
    Regression pin for the /team/member_add vs /team/delete race.

    The advisory lock is acquired, and the team is gone, before any write is attempted:
    the empty locked SELECT is proof a /team/delete already committed under the same
    lock, so this request must fail without writing the user or membership rows in the
    first place, rather than writing them and then trying to sweep them back out.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        _add_team_members_to_team,
    )

    tx = MagicMock()
    tx.query_raw = AsyncMock(return_value=[])
    tx.litellm_teamtable.update = AsyncMock()

    tx_cm = MagicMock()
    tx_cm.__aenter__ = AsyncMock(return_value=tx)
    tx_cm.__aexit__ = AsyncMock(return_value=None)

    prisma_client = MagicMock()
    prisma_client.tx = MagicMock(return_value=tx_cm)
    prisma_client.db.execute_raw = AsyncMock()
    prisma_client.db.litellm_teammembership.delete_many = AsyncMock()

    process_team_members = AsyncMock(return_value=([], []))
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints._process_team_members",
        new=process_team_members,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _add_team_members_to_team(
                data=TeamMemberAddRequest(
                    team_id="team-deleted-mid-add",
                    member=Member(user_id="bob", role="user"),
                ),
                complete_team_data=LiteLLM_TeamTable(team_id="team-deleted-mid-add", members_with_roles=[]),
                prisma_client=cast(object, prisma_client),
                user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
                litellm_proxy_admin_name="admin",
            )

    assert exc_info.value.status_code == 404
    process_team_members.assert_not_awaited()
    tx.litellm_teamtable.update.assert_not_awaited()
    prisma_client.db.execute_raw.assert_not_awaited()
    prisma_client.db.litellm_teammembership.delete_many.assert_not_awaited()


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
@pytest.mark.parametrize(
    "endpoint_name",
    ["team_model_add", "team_model_delete"],
)
async def test_team_model_add_delete_refresh_team_cache(endpoint_name):
    """
    Regression pin for LIT-3244 vector-store BYOK 403.

    `team_model_add` and `team_model_delete` mutate `team.models` in the
    DB. Without a cache refresh, the in-memory `LiteLLM_TeamTableCachedObj`
    used by `common_checks` stays stale and team members 403 on a model
    the DB has just granted (or, symmetrically, keep using a model the DB
    has just revoked).

    Pin: after the DB update, the endpoint must call `_cache_team_object`
    with the updated team row so the cached team stays in sync.
    """
    from unittest.mock import AsyncMock, MagicMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import (
        LitellmUserRoles,
        TeamModelAddRequest,
        TeamModelDeleteRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import (
        team_model_add,
        team_model_delete,
    )

    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test_user_id"
    )

    existing_team = MagicMock()
    existing_team.model_dump.return_value = {
        "team_id": "team-1234",
        "models": ["bedrock-claude-sonnet-4", "openai/*"],
        "object_permission_id": "op-1234",
        "object_permission": {
            "object_permission_id": "op-1234",
            "search_tools": ["allowed-tool-A"],
        },
    }

    updated_team = MagicMock()
    updated_team.team_id = "team-1234"
    updated_team.model_dump.return_value = {
        "team_id": "team-1234",
        "models": ["bedrock-claude-sonnet-4", "openai/*", "team-byok-1"],
        # The Prisma update must come back with `object_permission` populated
        # (via `include={"object_permission": True}`), otherwise the cache
        # write below would null it out — see LIT-3244 follow-up.
        "object_permission_id": "op-1234",
        "object_permission": {
            "object_permission_id": "op-1234",
            "search_tools": ["allowed-tool-A"],
        },
    }

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._cache_team_object",
            new_callable=AsyncMock,
        ) as mock_cache_team,
    ):
        mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=existing_team
        )
        mock_prisma_client.db.litellm_teamtable.update = AsyncMock(
            return_value=updated_team
        )
        mock_prisma_client.db.execute_raw = AsyncMock(return_value=None)

        if endpoint_name == "team_model_add":
            await team_model_add(
                data=TeamModelAddRequest(team_id="team-1234", models=["team-byok-1"]),
                http_request=mock_request,
                user_api_key_dict=mock_user_api_key_dict,
            )
        else:
            await team_model_delete(
                data=TeamModelDeleteRequest(team_id="team-1234", models=["openai/*"]),
                http_request=mock_request,
                user_api_key_dict=mock_user_api_key_dict,
            )

        # The pin: cache refresh must run with the updated team row.
        assert mock_cache_team.await_count == 1, (
            f"{endpoint_name} must call _cache_team_object exactly once "
            f"after the DB update (LIT-3244 regression pin); "
            f"got await_count={mock_cache_team.await_count}"
        )
        call_kwargs = mock_cache_team.await_args.kwargs
        assert call_kwargs["team_id"] == "team-1234"
        # The cached object must be built from the *updated* row, not the
        # pre-mutation `existing_team` — that's the whole point. Both rows
        # share team_id, so the only assertion that actually pins this is
        # against the field that differs between them: `models`.
        assert call_kwargs["team_table"].team_id == "team-1234"
        assert call_kwargs["team_table"].models == [
            "bedrock-claude-sonnet-4",
            "openai/*",
            "team-byok-1",
        ]
        # And the cached object MUST carry the `object_permission` relation
        # (LIT-3244 follow-up). If the Prisma update were missing
        # `include={"object_permission": True}`, the cached team would have
        # object_permission=None, and downstream consumers like
        # `validate_key_search_tools_against_team` would treat that as
        # "no team-level restriction" and stop enforcing the team's
        # search-tool allowlist on key issuance.
        assert call_kwargs["team_table"].object_permission is not None
        assert call_kwargs["team_table"].object_permission.search_tools == [
            "allowed-tool-A"
        ]
        # Pin the Prisma call shape too — the regression is in *what the
        # update returns*, so the contract that the update asks for
        # `object_permission` belongs in this test.
        update_call_kwargs = (
            mock_prisma_client.db.litellm_teamtable.update.call_args.kwargs
        )
        assert update_call_kwargs.get("include", {}).get("object_permission") is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint_name",
    ["team_model_add", "team_model_delete", "update_team_member_permissions"],
)
async def test_team_write_404s_when_row_vanishes_before_update(endpoint_name):
    """A team deleted between the read and the write must 404.

    Prisma's `update` returns None when no row matches `where`, and the team
    row can be deleted between the read these endpoints do first and the
    update that follows it. Without the guard, `team_model_add` /
    `team_model_delete` hand that None to `_refresh_cached_team` (which
    reads `team_row.team_id`) and `/team/permissions_update` returns None
    out of a route declared to return a team, so a plain race turns into a
    500 instead of the 404 every other not-found path in this file raises.
    """
    from unittest.mock import AsyncMock, MagicMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import (
        LitellmUserRoles,
        TeamModelAddRequest,
        TeamModelDeleteRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import (
        team_model_add,
        team_model_delete,
        update_team_member_permissions,
    )

    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test_user_id"
    )

    existing_team = MagicMock()
    existing_team.team_id = "team-1234"
    existing_team.model_dump.return_value = {
        "team_id": "team-1234",
        "models": ["bedrock-claude-sonnet-4", "openai/*"],
        "team_member_permissions": [],
        "spend": 0.0,
    }

    call_endpoint_under_test: Final = {
        "team_model_add": lambda: team_model_add(
            data=TeamModelAddRequest(team_id="team-1234", models=["team-byok-1"]),
            http_request=mock_request,
            user_api_key_dict=mock_user_api_key_dict,
        ),
        "team_model_delete": lambda: team_model_delete(
            data=TeamModelDeleteRequest(team_id="team-1234", models=["openai/*"]),
            http_request=mock_request,
            user_api_key_dict=mock_user_api_key_dict,
        ),
        "update_team_member_permissions": lambda: update_team_member_permissions(
            data=UpdateTeamMemberPermissionsRequest(
                team_id="team-1234",
                team_member_permissions=["/key/generate"],
            ),
            http_request=mock_request,
            user_api_key_dict=mock_user_api_key_dict,
        ),
    }[endpoint_name]

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client,  # test-quality-ok: proxy_server module global is the endpoint's only injection point
        patch("litellm.proxy.proxy_server.user_api_key_cache"),  # test-quality-ok: proxy_server module global is the endpoint's only injection point
        patch("litellm.proxy.proxy_server.proxy_logging_obj"),  # test-quality-ok: proxy_server module global is the endpoint's only injection point
        patch(  # test-quality-ok: stubs the cache write so the test observes only the DB result handling
            "litellm.proxy.management_endpoints.team_endpoints._cache_team_object",
            new_callable=AsyncMock,
        ),
        patch(  # test-quality-ok: stubs the collaborator so the test pins the endpoint's own error contract
            "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
            new_callable=AsyncMock,
            return_value=existing_team,
        ),
    ):
        mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=existing_team
        )
        mock_prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=None)
        mock_prisma_client.db.execute_raw = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await call_endpoint_under_test()

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == {"error": "Team not found, passed team_id=team-1234"}


@pytest.mark.asyncio
async def test_update_team_team_member_budget_not_passed_to_db(
    disable_audit_logging_for_mocked_team,
):
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client,
        patch("litellm.proxy.proxy_server.llm_router") as mock_llm_router,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._cache_team_object"
        ) as mock_cache_team,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.TeamMemberBudgetHandler.upsert_team_member_budget_table"
        ) as mock_upsert_budget,
    ):
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
            team_table,
            user_api_key_dict,
            updated_kv,
            team_member_budget=None,
            team_member_rpm_limit=None,
            team_member_tpm_limit=None,
            team_member_budget_duration=None,
            explicitly_set_fields=frozenset(),
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


def test_clean_team_member_fields():
    """
    Test that _clean_team_member_fields removes all team member fields from a dictionary.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    data_dict = {
        "team_id": "test_team",
        "team_alias": "Test Team",
        "team_member_budget": 100.0,
        "team_member_budget_duration": "30d",
        "team_member_rpm_limit": 50,
        "team_member_tpm_limit": 1000,
        "other_field": "should_remain",
    }

    TeamMemberBudgetHandler._clean_team_member_fields(data_dict)

    assert "team_member_budget" not in data_dict
    assert "team_member_budget_duration" not in data_dict
    assert "team_member_rpm_limit" not in data_dict
    assert "team_member_tpm_limit" not in data_dict
    assert data_dict["team_id"] == "test_team"
    assert data_dict["team_alias"] == "Test Team"
    assert data_dict["other_field"] == "should_remain"


def test_clean_team_member_fields_with_missing_fields():
    """
    Test that _clean_team_member_fields handles dictionaries without team member fields gracefully.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    data_dict = {
        "team_id": "test_team",
        "team_alias": "Test Team",
    }

    TeamMemberBudgetHandler._clean_team_member_fields(data_dict)

    assert data_dict["team_id"] == "test_team"
    assert data_dict["team_alias"] == "Test Team"


@pytest.mark.asyncio
async def test_create_team_member_budget_table():
    """
    Test that create_team_member_budget_table creates a budget and adds it to metadata.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.proxy._types import LitellmUserRoles, NewTeamRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test_user_id"
    )

    data = NewTeamRequest(
        team_id="test_team_id",
        team_alias="Test Team",
        budget_duration="1mo",
    )
    new_team_data_json = {
        "team_id": "test_team_id",
        "team_alias": "Test Team",
        "team_member_budget": 100.0,
        "team_member_budget_duration": "30d",
        "team_member_rpm_limit": 50,
        "team_member_tpm_limit": 1000,
    }

    mock_budget_response = MagicMock()
    mock_budget_response.budget_id = "budget_123"

    with patch(
        "litellm.proxy.management_endpoints.budget_management_endpoints.new_budget",
        new_callable=AsyncMock,
    ) as mock_new_budget:
        mock_new_budget.return_value = mock_budget_response

        result = await TeamMemberBudgetHandler.create_team_member_budget_table(
            data=data,
            new_team_data_json=new_team_data_json,
            user_api_key_dict=mock_user_api_key_dict,
            team_member_budget=100.0,
            team_member_rpm_limit=50,
            team_member_tpm_limit=1000,
            team_member_budget_duration="30d",
        )

        assert mock_new_budget.called
        call_args = mock_new_budget.call_args
        budget_request = call_args[1]["budget_obj"]

        assert budget_request.max_budget == 100.0
        assert budget_request.rpm_limit == 50
        assert budget_request.tpm_limit == 1000
        assert budget_request.budget_duration == "30d"
        assert budget_request.budget_id is not None
        assert "team-" in budget_request.budget_id

        assert "team_member_budget_id" in result["metadata"]
        assert result["metadata"]["team_member_budget_id"] == "budget_123"

        assert "team_member_budget" not in result
        assert "team_member_budget_duration" not in result
        assert "team_member_rpm_limit" not in result
        assert "team_member_tpm_limit" not in result


@pytest.mark.asyncio
async def test_create_team_member_budget_table_without_team_alias():
    """
    Test that create_team_member_budget_table generates budget_id correctly when team_alias is None.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.proxy._types import LitellmUserRoles, NewTeamRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test_user_id"
    )

    data = NewTeamRequest(team_id="test_team_id")
    new_team_data_json = {
        "team_id": "test_team_id",
        "team_member_budget": 100.0,
    }

    mock_budget_response = MagicMock()
    mock_budget_response.budget_id = "budget_123"

    with patch(
        "litellm.proxy.management_endpoints.budget_management_endpoints.new_budget",
        new_callable=AsyncMock,
    ) as mock_new_budget:
        mock_new_budget.return_value = mock_budget_response

        result = await TeamMemberBudgetHandler.create_team_member_budget_table(
            data=data,
            new_team_data_json=new_team_data_json,
            user_api_key_dict=mock_user_api_key_dict,
            team_member_budget=100.0,
        )

        assert mock_new_budget.called
        call_args = mock_new_budget.call_args
        budget_request = call_args[1]["budget_obj"]

        assert budget_request.budget_id is not None
        assert budget_request.budget_id.startswith("team-budget-")


@pytest.mark.asyncio
async def test_upsert_team_member_budget_table_existing_budget():
    """
    Test that upsert_team_member_budget_table updates an existing budget when team_member_budget_id exists.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.proxy._types import LitellmUserRoles, LiteLLM_TeamTable, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test_user_id"
    )

    team_table = MagicMock(spec=LiteLLM_TeamTable)
    team_table.metadata = {"team_member_budget_id": "existing_budget_123"}

    updated_kv = {
        "team_id": "test_team_id",
        "team_member_budget": 200.0,
        "team_member_budget_duration": "60d",
        "team_member_rpm_limit": 100,
    }

    mock_budget_response = MagicMock()
    mock_budget_response.budget_id = "existing_budget_123"

    with patch(
        "litellm.proxy.management_endpoints.budget_management_endpoints.update_budget",
        new_callable=AsyncMock,
    ) as mock_update_budget:
        mock_update_budget.return_value = mock_budget_response

        result = await TeamMemberBudgetHandler.upsert_team_member_budget_table(
            team_table=team_table,
            user_api_key_dict=mock_user_api_key_dict,
            updated_kv=updated_kv,
            team_member_budget=200.0,
            team_member_budget_duration="60d",
            team_member_rpm_limit=100,
        )

        assert mock_update_budget.called
        call_args = mock_update_budget.call_args
        budget_request = call_args[1]["budget_obj"]

        assert budget_request.budget_id == "existing_budget_123"
        assert budget_request.max_budget == 200.0
        assert budget_request.budget_duration == "60d"
        assert budget_request.rpm_limit == 100

        assert "team_member_budget_id" in result["metadata"]
        assert result["metadata"]["team_member_budget_id"] == "existing_budget_123"

        assert "team_member_budget" not in result
        assert "team_member_budget_duration" not in result
        assert "team_member_rpm_limit" not in result


@pytest.mark.asyncio
async def test_upsert_team_member_budget_table_no_existing_budget():
    """
    Test that upsert_team_member_budget_table creates a new budget when team_member_budget_id does not exist.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.proxy._types import LitellmUserRoles, LiteLLM_TeamTable, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test_user_id"
    )

    team_table = MagicMock(spec=LiteLLM_TeamTable)
    team_table.metadata = {}
    team_table.team_alias = "Test Team"
    team_table.budget_duration = None

    updated_kv = {
        "team_id": "test_team_id",
        "team_member_budget": 150.0,
        "team_member_budget_duration": "45d",
    }

    mock_budget_response = MagicMock()
    mock_budget_response.budget_id = "new_budget_456"

    with patch(
        "litellm.proxy.management_endpoints.budget_management_endpoints.new_budget",
        new_callable=AsyncMock,
    ) as mock_new_budget:
        mock_new_budget.return_value = mock_budget_response

        result = await TeamMemberBudgetHandler.upsert_team_member_budget_table(
            team_table=team_table,
            user_api_key_dict=mock_user_api_key_dict,
            updated_kv=updated_kv,
            team_member_budget=150.0,
            team_member_budget_duration="45d",
        )

        assert mock_new_budget.called
        assert "team_member_budget_id" in result["metadata"]
        assert result["metadata"]["team_member_budget_id"] == "new_budget_456"

        assert "team_member_budget" not in result
        assert "team_member_budget_duration" not in result


@pytest.mark.asyncio
async def test_upsert_team_member_budget_table_clears_duration_kept_budget(mock_db_client):
    """
    A request that keeps team_member_budget but explicitly nulls
    team_member_budget_duration must clear the reset period and its reset time.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test_user_id"
    )

    team_table = MagicMock(spec=LiteLLM_TeamTable)
    team_table.metadata = {"team_member_budget_id": "existing_budget_123"}

    mock_db_client.db.litellm_budgettable.update = AsyncMock(
        side_effect=lambda where, data: SimpleNamespace(**data)
    )

    result = await TeamMemberBudgetHandler.upsert_team_member_budget_table(
        team_table=team_table,
        user_api_key_dict=mock_user_api_key_dict,
        updated_kv={
            "team_id": "test_team_id",
            "team_member_budget": 100.0,
            "team_member_budget_duration": None,
        },
        team_member_budget=100.0,
        team_member_budget_duration=None,
        explicitly_set_fields={
            "team_member_budget",
            "team_member_budget_duration",
        },
    )

    written = mock_db_client.db.litellm_budgettable.update.call_args.kwargs["data"]
    assert written["max_budget"] == 100.0
    assert written["budget_duration"] is None
    assert written["budget_reset_at"] is None
    assert "rpm_limit" not in written
    assert "tpm_limit" not in written
    assert result["metadata"]["team_member_budget_id"] == "existing_budget_123"
    assert "team_member_budget" not in result
    assert "team_member_budget_duration" not in result


@pytest.mark.asyncio
async def test_create_team_member_budget_table_explicit_null_duration_does_not_inherit_team_duration(
    mock_db_client,
):
    """
    A first-time member budget with an explicitly null duration must never
    reset, even when the team itself has a reset period.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test_user_id"
    )

    team_table = MagicMock(spec=LiteLLM_TeamTable)
    team_table.metadata = {}
    team_table.team_alias = "Test Team"
    team_table.budget_duration = "30d"

    mock_db_client.db.litellm_budgettable.create = AsyncMock(
        side_effect=lambda data: SimpleNamespace(**data)
    )

    result = await TeamMemberBudgetHandler.create_team_member_budget_table(
        data=team_table,
        new_team_data_json={"team_id": "test_team_id"},
        user_api_key_dict=mock_user_api_key_dict,
        team_member_budget=100.0,
        team_member_budget_duration=None,
        explicitly_set_fields={
            "team_member_budget",
            "team_member_budget_duration",
        },
    )

    written = mock_db_client.db.litellm_budgettable.create.call_args.kwargs["data"]
    assert written["max_budget"] == 100.0
    assert "budget_duration" not in written
    assert "budget_reset_at" not in written
    assert result["metadata"]["team_member_budget_id"] == written["budget_id"]
    assert "team_member_budget" not in result


@pytest.mark.asyncio
async def test_create_team_member_budget_table_inherits_team_duration_when_duration_omitted(
    mock_db_client,
):
    """
    Omitting team_member_budget_duration keeps the existing inheritance of the
    team's own reset period.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test_user_id"
    )

    team_table = MagicMock(spec=LiteLLM_TeamTable)
    team_table.metadata = {}
    team_table.team_alias = "Test Team"
    team_table.budget_duration = "30d"

    mock_db_client.db.litellm_budgettable.create = AsyncMock(
        side_effect=lambda data: SimpleNamespace(**data)
    )

    result = await TeamMemberBudgetHandler.create_team_member_budget_table(
        data=team_table,
        new_team_data_json={"team_id": "test_team_id"},
        user_api_key_dict=mock_user_api_key_dict,
        team_member_budget=100.0,
        explicitly_set_fields={"team_member_budget"},
    )

    written = mock_db_client.db.litellm_budgettable.create.call_args.kwargs["data"]
    assert written["budget_duration"] == "30d"
    assert written["budget_reset_at"] is not None
    assert result["metadata"]["team_member_budget_id"] == written["budget_id"]


@pytest.mark.asyncio
async def test_update_team_with_team_member_budget_duration(
    disable_audit_logging_for_mocked_team,
):
    """
    Test that team/update endpoint properly handles team_member_budget_duration.
    """
    from unittest.mock import AsyncMock, MagicMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import LitellmUserRoles, UpdateTeamRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="test_user_id"
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client,
        patch("litellm.proxy.proxy_server.llm_router") as mock_llm_router,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._cache_team_object"
        ) as mock_cache_team,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.TeamMemberBudgetHandler.upsert_team_member_budget_table"
        ) as mock_upsert_budget,
    ):
        mock_existing_team = MagicMock()
        mock_existing_team.model_dump.return_value = {
            "team_id": "test_team_id",
            "team_alias": "test_team",
            "metadata": {"team_member_budget_id": "budget_123"},
        }
        mock_existing_team.metadata = {"team_member_budget_id": "budget_123"}
        mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )

        mock_updated_team = MagicMock()
        mock_updated_team.team_id = "test_team_id"
        mock_updated_team.model_dump.return_value = {"team_id": "test_team_id"}
        mock_prisma_client.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )
        mock_prisma_client.jsonify_team_object = MagicMock(
            side_effect=lambda db_data: db_data
        )

        def mock_upsert_side_effect(
            team_table,
            user_api_key_dict,
            updated_kv,
            team_member_budget=None,
            team_member_rpm_limit=None,
            team_member_tpm_limit=None,
            team_member_budget_duration=None,
            explicitly_set_fields=frozenset(),
        ):
            result_kv = updated_kv.copy()
            result_kv.pop("team_member_budget", None)
            result_kv.pop("team_member_budget_duration", None)
            return result_kv

        mock_upsert_budget.side_effect = mock_upsert_side_effect

        update_request = UpdateTeamRequest(
            team_id="test_team_id",
            team_alias="updated_alias",
            team_member_budget=100.0,
            team_member_budget_duration="30d",
        )

        result = await update_team(
            data=update_request,
            http_request=mock_request,
            user_api_key_dict=mock_user_api_key_dict,
        )

        assert mock_upsert_budget.called
        call_args = mock_upsert_budget.call_args
        assert call_args[1]["team_member_budget"] == 100.0
        assert call_args[1]["team_member_budget_duration"] == "30d"

        assert mock_prisma_client.db.litellm_teamtable.update.called
        update_call_args = mock_prisma_client.db.litellm_teamtable.update.call_args
        update_data = update_call_args[1]["data"]

        assert "team_member_budget" not in update_data
        assert "team_member_budget_duration" not in update_data


@pytest.mark.asyncio
async def test_backfill_team_member_budget_entries_creates_missing_memberships():
    """
    When backfill_team_member_budget_entries is called, it should create
    team_memberships rows only for members that don't already have one.

    Regression test for: https://github.com/BerriAI/litellm/issues/25506
    """
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._types import Member
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    team_id = "team-abc"
    budget_id = "budget-xyz"

    # user-A already has a membership; user-B does not
    existing_membership = MagicMock()
    existing_membership.user_id = "user-A"

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teammembership.find_many = AsyncMock(
        return_value=[existing_membership]
    )
    mock_prisma.db.litellm_teammembership.create_many = AsyncMock(return_value=None)
    mock_prisma.db.litellm_teammembership.update_many = AsyncMock(return_value=0)

    # Test with Member instances
    members = [
        Member(user_id="user-A", role="user"),
        Member(user_id="user-B", role="user"),
    ]

    await TeamMemberBudgetHandler.backfill_team_member_budget_entries(
        team_id=team_id,
        members_with_roles=members,
        team_member_budget_id=budget_id,
        prisma_client=mock_prisma,
    )

    # find_many should have been called to fetch existing memberships
    mock_prisma.db.litellm_teammembership.find_many.assert_awaited_once_with(
        where={"team_id": team_id}
    )

    # create_many should only create an entry for user-B (user-A already has one)
    mock_prisma.db.litellm_teammembership.create_many.assert_awaited_once_with(
        data=[{"team_id": team_id, "user_id": "user-B", "budget_id": budget_id}],
        skip_duplicates=True,
    )

    # Also test with raw dicts (members_with_roles may be dicts when deserialized from DB)
    mock_prisma.db.litellm_teammembership.find_many.reset_mock()
    mock_prisma.db.litellm_teammembership.create_many.reset_mock()
    mock_prisma.db.litellm_teammembership.update_many.reset_mock()

    members_as_dicts = [
        {"user_id": "user-A", "role": "user"},
        {"user_id": "user-B", "role": "user"},
    ]

    await TeamMemberBudgetHandler.backfill_team_member_budget_entries(
        team_id=team_id,
        members_with_roles=members_as_dicts,
        team_member_budget_id=budget_id,
        prisma_client=mock_prisma,
    )

    mock_prisma.db.litellm_teammembership.create_many.assert_awaited_once_with(
        data=[{"team_id": team_id, "user_id": "user-B", "budget_id": budget_id}],
        skip_duplicates=True,
    )


@pytest.mark.asyncio
async def test_backfill_team_member_budget_entries_no_op_when_all_exist():
    """
    backfill_team_member_budget_entries should not call create_many when all
    members already have a team_memberships entry.
    """
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._types import Member
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    team_id = "team-abc"
    budget_id = "budget-xyz"

    existing_a = MagicMock()
    existing_a.user_id = "user-A"
    existing_b = MagicMock()
    existing_b.user_id = "user-B"

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teammembership.find_many = AsyncMock(
        return_value=[existing_a, existing_b]
    )
    mock_prisma.db.litellm_teammembership.create_many = AsyncMock(return_value=None)
    mock_prisma.db.litellm_teammembership.update_many = AsyncMock(return_value=0)

    members = [
        Member(user_id="user-A", role="user"),
        Member(user_id="user-B", role="user"),
    ]

    await TeamMemberBudgetHandler.backfill_team_member_budget_entries(
        team_id=team_id,
        members_with_roles=members,
        team_member_budget_id=budget_id,
        prisma_client=mock_prisma,
    )

    mock_prisma.db.litellm_teammembership.create_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_backfill_team_member_budget_entries_populates_null_budget_id_on_existing_rows():
    """
    backfill_team_member_budget_entries should populate budget_id on
    existing TeamMembership rows where it is currently NULL, so admins
    can configure a team member budget after members have already joined
    and have enforcement apply to those pre-existing members.
    """
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._types import Member
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    team_id = "team-abc"
    budget_id = "budget-xyz"

    # Both members already have rows, so create_many must not fire;
    # update_many must fire with the NULL-budget_id filter.
    existing_a = MagicMock()
    existing_a.user_id = "user-A"
    existing_b = MagicMock()
    existing_b.user_id = "user-B"

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teammembership.find_many = AsyncMock(
        return_value=[existing_a, existing_b]
    )
    mock_prisma.db.litellm_teammembership.create_many = AsyncMock(return_value=None)
    mock_prisma.db.litellm_teammembership.update_many = AsyncMock(return_value=2)

    await TeamMemberBudgetHandler.backfill_team_member_budget_entries(
        team_id=team_id,
        members_with_roles=[
            Member(user_id="user-A", role="user"),
            Member(user_id="user-B", role="user"),
        ],
        team_member_budget_id=budget_id,
        prisma_client=mock_prisma,
    )

    mock_prisma.db.litellm_teammembership.create_many.assert_not_awaited()
    mock_prisma.db.litellm_teammembership.update_many.assert_awaited_once_with(
        where={"team_id": team_id, "budget_id": None},
        data={"budget_id": budget_id},
    )


@pytest.mark.asyncio
async def test_backfill_team_member_budget_entries_empty_members():
    """
    backfill_team_member_budget_entries should be a no-op when the member list
    is empty (no DB queries at all).
    """
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teammembership.find_many = AsyncMock(return_value=[])
    mock_prisma.db.litellm_teammembership.create_many = AsyncMock(return_value=None)

    await TeamMemberBudgetHandler.backfill_team_member_budget_entries(
        team_id="team-abc",
        members_with_roles=[],
        team_member_budget_id="budget-xyz",
        prisma_client=mock_prisma,
    )

    mock_prisma.db.litellm_teammembership.find_many.assert_not_awaited()
    mock_prisma.db.litellm_teammembership.create_many.assert_not_awaited()


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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.team_member_add",
            new_callable=AsyncMock,
            return_value=mock_team_response,
        ) as mock_team_member_add,
    ):
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client,
        patch("litellm.proxy.proxy_server.user_api_key_cache"),
        patch("litellm.proxy.proxy_server.proxy_logging_obj"),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        mock_prisma_client.return_value = MagicMock()  # Mock non-None prisma client

        # Should raise HTTPException with 401 status
        with pytest.raises(HTTPException) as exc_info:
            await list_team_v2(
                http_request=mock_request,
                user_id=None,  # Non-admin trying to query all teams
                user_api_key_dict=mock_user_api_key_dict_non_admin,
                status=None,
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client,
        patch("litellm.proxy.proxy_server.user_api_key_cache"),
        patch("litellm.proxy.proxy_server.proxy_logging_obj"),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        mock_prisma_client.return_value = MagicMock()  # Mock non-None prisma client

        # Should raise HTTPException with 401 status
        with pytest.raises(HTTPException) as exc_info:
            await list_team_v2(
                http_request=mock_request,
                user_id="other_user_456",  # Non-admin trying to query other user's teams
                user_api_key_dict=mock_user_api_key_dict_non_admin,
                status=None,
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client,
        patch("litellm.proxy.proxy_server.user_api_key_cache"),
        patch("litellm.proxy.proxy_server.proxy_logging_obj"),
    ):
        # Mock prisma client and database operations
        mock_db = Mock()
        mock_prisma_client.db = mock_db

        # Mock get_user_object to return a user with teams
        from litellm.proxy._types import LiteLLM_UserTable

        mock_user = LiteLLM_UserTable(
            user_id="non_admin_user_123",
            teams=["team_1", "team_2"],
        )

        # Mock team lookup
        mock_teams = [
            Mock(model_dump=lambda: {"team_id": "team_1", "team_alias": "Team 1"}),
            Mock(model_dump=lambda: {"team_id": "team_2", "team_alias": "Team 2"}),
        ]
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=mock_teams)
        mock_db.litellm_teamtable.count = AsyncMock(return_value=2)
        mock_db.litellm_verificationtoken.group_by = AsyncMock(return_value=[])

        with patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
            new_callable=AsyncMock,
            return_value=mock_user,
        ):
            # Should NOT raise an exception
            result = await list_team_v2(
                http_request=mock_request,
                user_id="non_admin_user_123",  # Non-admin querying their own teams
                user_api_key_dict=mock_user_api_key_dict_non_admin,
                team_id=None,
                page=1,
                page_size=10,
                status=None,
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
        mock_db.litellm_verificationtoken.group_by = AsyncMock(return_value=[])

        # Should NOT raise an exception
        result = await list_team_v2(
            http_request=mock_request,
            user_id=None,  # Admin querying all teams
            user_api_key_dict=mock_user_api_key_dict_admin,
            page=1,
            page_size=10,
            status=None,
        )

        # Should return results without error
        assert "teams" in result
        assert "total" in result
        assert result["total"] == 2


@pytest.mark.asyncio
async def test_list_team_v2_with_status_deleted():
    """
    Test that status="deleted" parameter correctly queries the deleted teams table.
    """
    from unittest.mock import AsyncMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    # Mock request
    mock_request = Mock(spec=Request)

    # Mock admin user
    mock_user_api_key_dict_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user_123",
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client:
        # Mock prisma client and database operations
        mock_db = Mock()
        mock_prisma_client.db = mock_db

        # Mock deleted teams
        mock_deleted_team1 = Mock(
            model_dump=lambda: {"team_id": "team_1", "team_alias": "Deleted Team 1"}
        )
        mock_deleted_team2 = Mock(
            model_dump=lambda: {"team_id": "team_2", "team_alias": "Deleted Team 2"}
        )

        # Mock deleted teams table (should be called)
        mock_db.litellm_deletedteamtable.find_many = AsyncMock(
            return_value=[mock_deleted_team1, mock_deleted_team2]
        )
        mock_db.litellm_deletedteamtable.count = AsyncMock(return_value=2)

        # Mock regular teams table (should NOT be called)
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=[])
        mock_db.litellm_teamtable.count = AsyncMock(return_value=0)

        # Should NOT raise an exception
        result = await list_team_v2(
            http_request=mock_request,
            user_id=None,  # Admin querying all teams
            user_api_key_dict=mock_user_api_key_dict_admin,
            page=1,
            page_size=10,
            status="deleted",  # Test the status parameter
        )

        # Verify that deleted table was queried
        mock_db.litellm_deletedteamtable.find_many.assert_called_once()
        mock_db.litellm_deletedteamtable.count.assert_called_once()

        # Verify that regular table was NOT queried
        mock_db.litellm_teamtable.find_many.assert_not_called()
        mock_db.litellm_teamtable.count.assert_not_called()

        # Should return results without error
        assert "teams" in result
        assert "total" in result
        assert result["total"] == 2
        assert len(result["teams"]) == 2


@pytest.mark.asyncio
async def test_list_team_v2_org_admin_sees_org_teams():
    """
    Test that an org admin (internal_user role with org_admin membership)
    can list teams scoped to their organisations without getting a 401.
    """
    from datetime import datetime
    from unittest.mock import AsyncMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_OrganizationMembershipTable,
        LiteLLM_UserTable,
        LitellmUserRoles,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org_admin_user",
    )

    mock_user = LiteLLM_UserTable(
        user_id="org_admin_user",
        teams=[],
        organization_memberships=[
            LiteLLM_OrganizationMembershipTable(
                user_id="org_admin_user",
                organization_id="org_A",
                user_role="org_admin",
                spend=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ],
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache"),
        patch("litellm.proxy.proxy_server.proxy_logging_obj"),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
    ):
        mock_db = Mock()
        mock_prisma.db = mock_db

        mock_team = Mock()
        mock_team.model_dump.return_value = {
            "team_id": "team_in_org_A",
            "team_alias": "Org A Team",
            "organization_id": "org_A",
            "members_with_roles": [{"user_id": "u1", "role": "user"}],
        }
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=[mock_team])
        mock_db.litellm_teamtable.count = AsyncMock(return_value=1)
        mock_db.litellm_verificationtoken.group_by = AsyncMock(return_value=[])

        result = await list_team_v2(
            http_request=mock_request,
            user_id=None,
            organization_id=None,
            team_id=None,
            team_alias=None,
            user_api_key_dict=mock_user_api_key_dict,
            page=1,
            page_size=10,
            sort_by=None,
            sort_order="asc",
            status=None,
        )

        assert result["total"] == 1
        assert len(result["teams"]) == 1
        assert result["teams"][0].members_count == 1

        # Verify org-scoped where clause
        where = mock_db.litellm_teamtable.find_many.call_args.kwargs["where"]
        assert where["organization_id"] == {"in": ["org_A"]}


@pytest.mark.asyncio
async def test_list_team_v2_org_admin_own_user_id_sees_all_org_teams():
    """
    Test that an org admin whose own user_id is sent (as the UI does for
    non-Admin roles) still sees all teams in their organization, not just
    teams they are a direct member of.

    Regression test for https://github.com/BerriAI/litellm/issues/30215
    """
    from datetime import datetime
    from unittest.mock import AsyncMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_OrganizationMembershipTable,
        LiteLLM_UserTable,
        LitellmUserRoles,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org_admin_user",
    )

    mock_user = LiteLLM_UserTable(
        user_id="org_admin_user",
        teams=["team_1"],  # direct member of only 1 team
        organization_memberships=[
            LiteLLM_OrganizationMembershipTable(
                user_id="org_admin_user",
                organization_id="org_A",
                user_role="org_admin",
                spend=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ],
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache"),
        patch("litellm.proxy.proxy_server.proxy_logging_obj"),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
    ):
        mock_db = Mock()
        mock_prisma.db = mock_db

        mock_team_1 = Mock()
        mock_team_1.model_dump.return_value = {
            "team_id": "team_1",
            "team_alias": "Team One",
            "organization_id": "org_A",
            "members_with_roles": [{"user_id": "org_admin_user", "role": "admin"}],
        }
        mock_team_2 = Mock()
        mock_team_2.model_dump.return_value = {
            "team_id": "team_2",
            "team_alias": "Team Two",
            "organization_id": "org_A",
            "members_with_roles": [{"user_id": "other_user", "role": "user"}],
        }
        mock_db.litellm_teamtable.find_many = AsyncMock(
            return_value=[mock_team_1, mock_team_2]
        )
        mock_db.litellm_teamtable.count = AsyncMock(return_value=2)
        mock_db.litellm_verificationtoken.group_by = AsyncMock(return_value=[])

        # UI sends the caller's own user_id for non-Admin roles
        result = await list_team_v2(
            http_request=mock_request,
            user_id="org_admin_user",  # same as caller — UI sends this
            organization_id=None,
            team_id=None,
            team_alias=None,
            user_api_key_dict=mock_user_api_key_dict,
            page=1,
            page_size=10,
            sort_by=None,
            sort_order="asc",
            status=None,
        )

        assert result["total"] == 2
        assert len(result["teams"]) == 2

        # Verify the where clause scopes by org only — no team_id filter
        where = mock_db.litellm_teamtable.find_many.call_args.kwargs["where"]
        assert where["organization_id"] == {"in": ["org_A"]}
        assert "team_id" not in where


@pytest.mark.asyncio
async def test_list_team_v2_org_admin_cannot_view_other_orgs():
    """
    Test that an org admin is rejected with 403 when filtering by an
    organisation they do not administer.
    """
    from datetime import datetime
    from unittest.mock import AsyncMock, Mock, patch

    from fastapi import HTTPException, Request

    from litellm.proxy._types import (
        LiteLLM_OrganizationMembershipTable,
        LiteLLM_UserTable,
        LitellmUserRoles,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org_admin_user",
    )

    mock_user = LiteLLM_UserTable(
        user_id="org_admin_user",
        teams=[],
        organization_memberships=[
            LiteLLM_OrganizationMembershipTable(
                user_id="org_admin_user",
                organization_id="org_A",
                user_role="org_admin",
                spend=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ],
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache"),
        patch("litellm.proxy.proxy_server.proxy_logging_obj"),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
    ):
        mock_prisma.db = Mock()

        with pytest.raises(HTTPException) as exc_info:
            await list_team_v2(
                http_request=mock_request,
                user_id=None,
                organization_id="org_B",  # not their org
                team_id=None,
                team_alias=None,
                user_api_key_dict=mock_user_api_key_dict,
                page=1,
                page_size=10,
                sort_by=None,
                sort_order="asc",
                status=None,
            )

        assert exc_info.value.status_code == 403
        assert (
            "only view teams within your organizations"
            in str(exc_info.value.detail).lower()
        )


@pytest.mark.asyncio
async def test_list_team_v2_org_admin_with_user_id_returns_user_teams():
    """
    Test that an org admin passing user_id gets that user's direct team
    memberships (not all org teams).
    """
    from datetime import datetime
    from unittest.mock import AsyncMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_OrganizationMembershipTable,
        LiteLLM_UserTable,
        LitellmUserRoles,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org_admin_user",
    )

    mock_org_admin = LiteLLM_UserTable(
        user_id="org_admin_user",
        teams=["team_1"],
        organization_memberships=[
            LiteLLM_OrganizationMembershipTable(
                user_id="org_admin_user",
                organization_id="org_A",
                user_role="org_admin",
                spend=0.0,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ],
    )

    # The target user whose teams we want to list
    mock_target_user = LiteLLM_UserTable(
        user_id="target_user",
        teams=["team_X", "team_Y"],
    )

    call_count = 0

    async def mock_get_user_object(**kwargs):
        nonlocal call_count
        call_count += 1
        # First call: org admin lookup in list_team_v2
        # Second call: target user lookup in _build_team_list_where_conditions
        if call_count == 1:
            return mock_org_admin
        return mock_target_user

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache"),
        patch("litellm.proxy.proxy_server.proxy_logging_obj"),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
            side_effect=mock_get_user_object,
        ),
    ):
        mock_db = Mock()
        mock_prisma.db = mock_db

        mock_team = Mock()
        mock_team.model_dump.return_value = {
            "team_id": "team_X",
            "team_alias": "Target Team",
            "members_with_roles": [{"user_id": "target_user", "role": "user"}],
        }
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=[mock_team])
        mock_db.litellm_teamtable.count = AsyncMock(return_value=1)
        mock_db.litellm_verificationtoken.group_by = AsyncMock(return_value=[])

        result = await list_team_v2(
            http_request=mock_request,
            user_id="target_user",
            organization_id=None,
            team_id=None,
            team_alias=None,
            user_api_key_dict=mock_user_api_key_dict,
            page=1,
            page_size=10,
            sort_by=None,
            sort_order="asc",
            status=None,
        )

        assert result["total"] == 1

        # Verify the where clause filters by user's teams AND org scope
        where = mock_db.litellm_teamtable.find_many.call_args.kwargs["where"]
        assert where["team_id"] == {"in": ["team_X", "team_Y"]}
        assert where["organization_id"] == {"in": ["org_A"]}


@pytest.mark.asyncio
async def test_list_team_v2_with_invalid_status():
    """
    Test that invalid status parameter raises HTTPException.
    """
    from unittest.mock import Mock, patch

    from fastapi import HTTPException, Request

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    # Mock request
    mock_request = Mock(spec=Request)

    # Mock admin user
    mock_user_api_key_dict_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user_123",
    )

    mock_prisma_client = Mock()

    # Mock prisma_client to be non-None
    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
        # Should raise HTTPException for invalid status
        with pytest.raises(HTTPException) as exc_info:
            await list_team_v2(
                http_request=mock_request,
                user_id=None,
                user_api_key_dict=mock_user_api_key_dict_admin,
                page=1,
                page_size=10,
                status="invalid_status",  # Invalid status value
            )

        assert exc_info.value.status_code == 400
        assert "Invalid status value" in str(exc_info.value.detail)
        assert "deleted" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_list_team_v2_search_builds_or_clause():
    """
    `search` should be passed as a Prisma OR across an exact team_id match and a
    case-insensitive team_alias contains, so the UI needs one backend filter.
    Exact id matching is the documented default and must not change.
    """
    from unittest.mock import AsyncMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    mock_request = Mock(spec=Request)
    mock_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client:
        mock_db = Mock()
        mock_prisma_client.db = mock_db
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=[])
        mock_db.litellm_teamtable.count = AsyncMock(return_value=0)

        await list_team_v2(
            http_request=mock_request,
            user_id=None,
            organization_id=None,
            team_id=None,
            team_alias=None,
            search="platform",
            user_api_key_dict=mock_admin,
            page=1,
            page_size=10,
            status=None,
        )

        find_many_kwargs = mock_db.litellm_teamtable.find_many.call_args.kwargs
        assert find_many_kwargs["where"] == {
            "OR": [
                {"team_id": "platform"},
                {"team_alias": {"contains": "platform", "mode": "insensitive"}},
            ]
        }


@pytest.mark.asyncio
async def test_list_team_v2_search_team_id_match_prefix():
    """
    Opting into `search_team_id_match="prefix"` should widen the team_id side of
    the search OR to an index-friendly prefix match, so the first characters of a
    team id quoted in a proxy error find the team.
    """
    from unittest.mock import AsyncMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    mock_request = Mock(spec=Request)
    mock_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin_user"
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client:
        mock_db = Mock()
        mock_prisma_client.db = mock_db
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=[])
        mock_db.litellm_teamtable.count = AsyncMock(return_value=0)

        await list_team_v2(
            http_request=mock_request,
            user_id=None,
            organization_id=None,
            team_id=None,
            team_alias=None,
            search="66c432fa",
            search_team_id_match="prefix",
            user_api_key_dict=mock_admin,
            page=1,
            page_size=10,
            status=None,
        )

        find_many_kwargs = mock_db.litellm_teamtable.find_many.call_args.kwargs
        assert find_many_kwargs["where"] == {
            "OR": [
                {"team_id": {"startsWith": "66c432fa"}},
                {"team_alias": {"contains": "66c432fa", "mode": "insensitive"}},
            ]
        }


@pytest.mark.asyncio
async def test_list_team_v2_search_composes_with_user_id_filter():
    """
    For non-admin users, `search` must compose with the membership filter:
    the resulting where clause should AND `team_id IN <user's teams>` with
    the search OR clause, so users still only see their own teams.
    """
    from datetime import datetime
    from unittest.mock import AsyncMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_UserTable,
        LitellmUserRoles,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    mock_request = Mock(spec=Request)
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="member_user"
    )

    mock_user = LiteLLM_UserTable(
        user_id="member_user",
        teams=["team_a", "team_b"],
        organization_memberships=[],
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
            new=AsyncMock(return_value=mock_user),
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._get_org_admin_org_ids",
            new=AsyncMock(return_value=None),
        ),
    ):
        mock_db = Mock()
        mock_prisma.db = mock_db
        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=[])
        mock_db.litellm_teamtable.count = AsyncMock(return_value=0)

        await list_team_v2(
            http_request=mock_request,
            user_id="member_user",
            organization_id=None,
            team_id=None,
            team_alias=None,
            search="team_a",
            search_team_id_match="prefix",
            user_api_key_dict=mock_user_api_key_dict,
            page=1,
            page_size=10,
            status=None,
        )

        find_many_kwargs = mock_db.litellm_teamtable.find_many.call_args.kwargs
        where = find_many_kwargs["where"]
        assert where["OR"] == [
            {"team_id": {"startsWith": "team_a"}},
            {"team_alias": {"contains": "team_a", "mode": "insensitive"}},
        ]
        assert where["team_id"] == {"in": ["team_a", "team_b"]}


@pytest.mark.asyncio
async def test_list_team_v2_populates_keys_count():
    """
    Test that list_team_v2 returns a keys_count per team derived from a single
    batched group_by against LiteLLM_VerificationToken.
    """
    from unittest.mock import AsyncMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    mock_request = Mock(spec=Request)
    mock_user_api_key_dict_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user_123",
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client:
        mock_db = Mock()
        mock_prisma_client.db = mock_db

        team_a = Mock()
        team_a.team_id = "team_a"
        team_a.model_dump = lambda: {
            "team_id": "team_a",
            "team_alias": "Team A",
            "members_with_roles": [{"user_id": "u1", "role": "user"}],
        }
        team_b = Mock()
        team_b.team_id = "team_b"
        team_b.model_dump = lambda: {
            "team_id": "team_b",
            "team_alias": "Team B",
            "members_with_roles": [],
        }

        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=[team_a, team_b])
        mock_db.litellm_teamtable.count = AsyncMock(return_value=2)
        mock_db.litellm_verificationtoken.group_by = AsyncMock(
            return_value=[
                {"team_id": "team_a", "_count": {"team_id": 3}},
                # team_b intentionally absent → expect 0
            ]
        )

        result = await list_team_v2(
            http_request=mock_request,
            user_id=None,
            user_api_key_dict=mock_user_api_key_dict_admin,
            page=1,
            page_size=10,
            status=None,
        )

        assert result["total"] == 2
        by_id = {t.team_id: t for t in result["teams"]}
        assert by_id["team_a"].keys_count == 3
        assert by_id["team_b"].keys_count == 0

        # The aggregate is one batched query, filtered by the page's team IDs.
        group_by_kwargs = mock_db.litellm_verificationtoken.group_by.call_args.kwargs
        assert group_by_kwargs["by"] == ["team_id"]
        assert group_by_kwargs["where"] == {"team_id": {"in": ["team_a", "team_b"]}}
        assert group_by_kwargs["count"] == {"team_id": True}


@pytest.mark.asyncio
async def test_list_team_v2_keys_count_skipped_for_empty_page():
    """
    When the page has no teams, the keys-count group_by must not be issued.
    """
    from unittest.mock import AsyncMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    mock_request = Mock(spec=Request)
    mock_user_api_key_dict_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user_123",
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client:
        mock_db = Mock()
        mock_prisma_client.db = mock_db

        mock_db.litellm_teamtable.find_many = AsyncMock(return_value=[])
        mock_db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_db.litellm_verificationtoken.group_by = AsyncMock(return_value=[])

        result = await list_team_v2(
            http_request=mock_request,
            user_id=None,
            user_api_key_dict=mock_user_api_key_dict_admin,
            page=1,
            page_size=10,
            status=None,
        )

        assert result["total"] == 0
        assert result["teams"] == []
        mock_db.litellm_verificationtoken.group_by.assert_not_called()


@pytest.mark.asyncio
async def test_list_team_v2_keys_count_skipped_for_deleted_status():
    """
    The deleted-table branch returns LiteLLM_DeletedTeamTable items, which do
    not carry keys_count — group_by must not be issued.
    """
    from unittest.mock import AsyncMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import list_team_v2

    mock_request = Mock(spec=Request)
    mock_user_api_key_dict_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user_123",
    )

    with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client:
        mock_db = Mock()
        mock_prisma_client.db = mock_db

        mock_deleted = Mock()
        mock_deleted.team_id = "team_d"
        mock_deleted.model_dump = lambda: {
            "team_id": "team_d",
            "team_alias": "Deleted Team",
        }

        mock_db.litellm_deletedteamtable.find_many = AsyncMock(
            return_value=[mock_deleted]
        )
        mock_db.litellm_deletedteamtable.count = AsyncMock(return_value=1)
        mock_db.litellm_verificationtoken.group_by = AsyncMock(return_value=[])

        result = await list_team_v2(
            http_request=mock_request,
            user_id=None,
            user_api_key_dict=mock_user_api_key_dict_admin,
            page=1,
            page_size=10,
            status="deleted",
        )

        assert result["total"] == 1
        mock_db.litellm_verificationtoken.group_by.assert_not_called()


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
    mock_db_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team_row
    )
    mock_db_client.db.litellm_teamtable.update = AsyncMock(return_value=mock_team_row)

    # User row to allow removal from user's teams list
    mock_user_row = MagicMock()
    mock_user_row.user_id = test_user_id
    mock_user_row.teams = [test_team_id]
    mock_db_client.db.litellm_usertable.find_many = AsyncMock(
        return_value=[mock_user_row]
    )
    mock_db_client.db.litellm_usertable.update = AsyncMock(return_value=MagicMock())

    # Membership deletion should be called
    mock_db_client.db.litellm_teammembership = MagicMock()
    mock_db_client.db.litellm_teammembership.delete_many = AsyncMock(
        return_value=MagicMock()
    )

    # Verification token deletion should be called
    mock_db_client.db.litellm_verificationtoken = MagicMock()
    mock_db_client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    mock_db_client.db.litellm_verificationtoken.delete_many = AsyncMock(
        return_value=MagicMock()
    )

    _wire_member_delete_tx(mock_db_client)

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
async def test_team_member_delete_cleans_verification_tokens(
    mock_db_client, mock_admin_auth
):
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

    mock_db_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team_row
    )
    mock_db_client.db.litellm_teamtable.update = AsyncMock(return_value=mock_team_row)

    mock_user_row = MagicMock()
    mock_user_row.user_id = test_user_id
    mock_user_row.teams = [test_team_id]
    mock_db_client.db.litellm_usertable.find_many = AsyncMock(
        return_value=[mock_user_row]
    )
    mock_db_client.db.litellm_usertable.update = AsyncMock(return_value=MagicMock())

    mock_db_client.db.litellm_teammembership = MagicMock()
    mock_db_client.db.litellm_teammembership.delete_many = AsyncMock(
        return_value=MagicMock()
    )

    mock_db_client.db.litellm_verificationtoken = MagicMock()
    mock_db_client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    mock_db_client.db.litellm_verificationtoken.delete_many = AsyncMock(
        return_value=MagicMock()
    )

    _wire_member_delete_tx(mock_db_client)

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
async def test_team_member_delete_reads_on_the_lock_holding_transaction(
    mock_db_client, mock_admin_auth
):
    """
    Regression pin against exhausting the connection pool with advisory-lock waiters.

    Every concurrent removal for one team holds a pooled connection while it waits on the
    team's advisory lock, and /team/delete fans its per-member removals out concurrently.
    A holder whose reads went to the regular client would need a second connection to
    finish, so enough waiters fill the pool and the holder can never release the lock.
    Both reads therefore have to run on the transaction that already owns the connection.
    """
    from litellm.proxy._types import TeamMemberDeleteRequest
    from litellm.proxy.management_endpoints.team_endpoints import team_member_delete

    test_team_id = "team-del-pool-123"
    test_user_id = "user-del-pool-123"
    roster_entry = {"user_id": test_user_id, "user_email": None, "role": "user"}

    mock_team_row = MagicMock()
    mock_team_row.model_dump.return_value = {
        "team_id": test_team_id,
        "members_with_roles": [roster_entry],
        "team_member_permissions": [],
        "metadata": {},
        "models": [],
        "spend": 0.0,
    }
    mock_db_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team_row
    )

    user_row = MagicMock()
    user_row.user_id = test_user_id
    user_row.teams = [test_team_id]

    # Both are wired to answer, so the endpoint completes either way and the awaits below
    # are what tells which connection it read on.
    pooled_user_read = AsyncMock(return_value=[user_row])
    pooled_token_read = AsyncMock(return_value=[])
    mock_db_client.db.litellm_usertable.find_many = pooled_user_read
    mock_db_client.db.litellm_verificationtoken.find_many = pooled_token_read

    tx = MagicMock()
    tx.query_raw = AsyncMock(return_value=[{"members_with_roles": [roster_entry]}])
    tx.litellm_teamtable.update = AsyncMock(return_value=mock_team_row)
    tx.litellm_usertable.find_many = AsyncMock(return_value=[user_row])
    tx.litellm_usertable.update = AsyncMock()
    tx.litellm_teammembership.delete_many = AsyncMock()
    tx.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    tx.litellm_verificationtoken.delete_many = AsyncMock()

    tx_cm = MagicMock()
    tx_cm.__aenter__ = AsyncMock(return_value=tx)
    tx_cm.__aexit__ = AsyncMock(return_value=None)
    mock_db_client.tx = MagicMock(return_value=tx_cm)

    await team_member_delete(
        data=TeamMemberDeleteRequest(team_id=test_team_id, user_id=test_user_id),
        user_api_key_dict=mock_admin_auth,
    )

    tx.litellm_usertable.find_many.assert_awaited_once_with(
        where={"user_id": {"in": [test_user_id]}}
    )
    tx.litellm_verificationtoken.find_many.assert_awaited_once_with(
        where={"user_id": {"in": [test_user_id]}, "team_id": test_team_id}
    )
    pooled_user_read.assert_not_awaited()
    pooled_token_read.assert_not_awaited()

    tx.litellm_usertable.update.assert_awaited_once_with(
        where={"user_id": test_user_id}, data={"teams": {"set": []}}
    )
    tx.litellm_teammembership.delete_many.assert_awaited_once_with(
        where={"team_id": test_team_id, "user_id": test_user_id}
    )


@pytest.mark.parametrize(
    "roster_email",
    ["Alice@Example.com", "alice-invited-as@example.com"],
    ids=["case_variant_of_the_row_email", "email_the_row_never_carried"],
)
@pytest.mark.parametrize("user_row_exists", [True, False])
@pytest.mark.asyncio
async def test_team_member_delete_by_email_the_user_row_does_not_carry(
    user_row_exists, roster_email, mock_db_client, mock_admin_auth
):
    """
    Removing a member addressed by user_email drove its user-row and membership cleanup off that raw
    email instead of off the user_id the roster entry already carries, so an email the user row does
    not literally hold matched nothing and both cleanups silently no-opped behind a 200.

    Both roster emails here are reachable over plain HTTP. /team/member_add resolves an email to a
    user case-insensitively but stores the caller's casing in members_with_roles, which produces the
    case variant; it also leaves an unmatched email on the entry when no user row carries it at all,
    which produces the second. Both converge on the same lookup, so they are parametrized inputs
    rather than separate paths, and each one has to detect the bug on its own.

    The user table below is case-sensitive like Postgres, so only a lookup driven by the resolved
    user_id finds the row. The user_row_exists=False leg pins the second half on its own: the
    membership row has to go even when no user row is left to resolve it from.
    """
    from litellm.proxy._types import TeamMemberDeleteRequest
    from litellm.proxy.management_endpoints.team_endpoints import team_member_delete

    test_team_id = "team-del-email-case-123"
    test_user_id = "user-del-email-case-123"
    user_row_email = "alice@example.com"

    mock_team_row = MagicMock()
    mock_team_row.model_dump.return_value = {
        "team_id": test_team_id,
        "members_with_roles": [
            {"user_id": test_user_id, "user_email": roster_email, "role": "user"}
        ],
        "team_member_permissions": [],
        "metadata": {},
        "models": [],
        "spend": 0.0,
    }

    mock_db_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team_row
    )
    mock_db_client.db.litellm_teamtable.update = AsyncMock(return_value=mock_team_row)

    mock_user_row = MagicMock()
    mock_user_row.user_id = test_user_id
    mock_user_row.user_email = user_row_email
    mock_user_row.teams = [test_team_id]

    async def find_user_rows(where):
        if not user_row_exists:
            return []
        user_id_filter = where.get("user_id")
        if isinstance(user_id_filter, dict) and test_user_id in user_id_filter.get(
            "in", []
        ):
            return [mock_user_row]
        if where.get("user_email") == user_row_email:
            return [mock_user_row]
        return []

    mock_db_client.db.litellm_usertable.find_many = AsyncMock(
        side_effect=find_user_rows
    )
    mock_db_client.db.litellm_usertable.update = AsyncMock(return_value=MagicMock())

    mock_db_client.db.litellm_teammembership = MagicMock()
    mock_db_client.db.litellm_teammembership.delete_many = AsyncMock(
        return_value=MagicMock()
    )

    mock_db_client.db.litellm_verificationtoken = MagicMock()
    mock_db_client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    mock_db_client.db.litellm_verificationtoken.delete_many = AsyncMock(
        return_value=MagicMock()
    )

    _wire_member_delete_tx(mock_db_client)

    await team_member_delete(
        data=TeamMemberDeleteRequest(team_id=test_team_id, user_email=roster_email),
        user_api_key_dict=mock_admin_auth,
    )

    if user_row_exists:
        mock_db_client.db.litellm_usertable.update.assert_awaited_once_with(
            where={"user_id": test_user_id},
            data={"teams": {"set": []}},
        )
    else:
        mock_db_client.db.litellm_usertable.update.assert_not_awaited()

    mock_db_client.db.litellm_teammembership.delete_many.assert_awaited_once_with(
        where={"team_id": test_team_id, "user_id": test_user_id}
    )


class _InjectedMemberDeleteFailure(Exception):
    pass


@pytest.mark.asyncio
async def test_team_member_delete_is_atomic_across_its_four_writes(
    mock_db_client, mock_admin_auth
):
    """
    /team/member_delete's four cleanups (team roster, user.teams, team
    membership, verification tokens) run as one transaction, so a failure
    partway through must not leave the removal half applied.

    Failing the second write (the user's ``teams`` update) pins two things a
    non-transactional implementation gets wrong: the roster write that already
    ran has to land on the SAME transaction client the failure raises on (so a
    real database rolls it back too), and the writes still queued behind the
    failure (membership delete, token delete) must never be attempted at all.
    """
    from litellm.proxy._types import TeamMemberDeleteRequest
    from litellm.proxy.management_endpoints.team_endpoints import team_member_delete

    test_team_id = "team-del-atomic-123"
    test_user_id = "user-atomic@example.com"

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
    mock_db_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team_row
    )
    mock_db_client.db.litellm_teamtable.update = AsyncMock(return_value=mock_team_row)

    mock_user_row = MagicMock()
    mock_user_row.user_id = test_user_id
    mock_user_row.teams = [test_team_id]
    mock_db_client.db.litellm_usertable.find_many = AsyncMock(
        return_value=[mock_user_row]
    )
    mock_db_client.db.litellm_usertable.update = AsyncMock(
        side_effect=_InjectedMemberDeleteFailure("boom between writes 1 and 2")
    )

    mock_db_client.db.litellm_teammembership = MagicMock()
    mock_db_client.db.litellm_teammembership.delete_many = AsyncMock()

    mock_db_client.db.litellm_verificationtoken = MagicMock()
    mock_db_client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    mock_db_client.db.litellm_verificationtoken.delete_many = AsyncMock()

    _wire_member_delete_tx(mock_db_client)

    with pytest.raises(_InjectedMemberDeleteFailure):
        await team_member_delete(
            data=TeamMemberDeleteRequest(team_id=test_team_id, user_id=test_user_id),
            user_api_key_dict=mock_admin_auth,
        )

    # The roster write ran, but on the transaction the injected failure also raised on.
    mock_db_client.db.litellm_teamtable.update.assert_awaited_once()
    mock_db_client.tx.assert_called_once()
    aexit_args = mock_db_client.tx.return_value.__aexit__.await_args.args
    assert aexit_args[0] is _InjectedMemberDeleteFailure

    # Writes queued behind the failure inside that same transaction never ran.
    mock_db_client.db.litellm_teammembership.delete_many.assert_not_awaited()
    mock_db_client.db.litellm_verificationtoken.delete_many.assert_not_awaited()


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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server._license_check") as mock_license,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
    ):
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
        assert exc_info.value.code == "400"
        assert "max budget higher than user max" in str(exc_info.value.message)
        assert "100.0" in str(
            exc_info.value.message
        )  # User's user_max_budget should be mentioned
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server._license_check") as mock_license,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
    ):
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
        mock_created_team.default_team_member_models = None
        mock_created_team.model_dump.return_value = {
            "team_id": "team-within-budget-789",
            "team_alias": "within-budget-team",
            "max_budget": 50.0,
            "members_with_roles": [],
        }
        mock_prisma.db.litellm_teamtable.create = AsyncMock(
            return_value=mock_created_team
        )
        _wire_team_create_tx(mock_prisma)
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_created_team
        )

        # Mock model table
        mock_prisma.db.litellm_modeltable = MagicMock()
        mock_prisma.db.litellm_modeltable.create = AsyncMock(
            return_value=MagicMock(id="model123")
        )

        # Mock user table operations for adding the creator as a member
        mock_user = MagicMock()
        mock_user.user_id = "non-admin-user-456"
        mock_user.model_dump.return_value = {
            "user_id": "non-admin-user-456",
            "teams": ["team-within-budget-789"],
        }
        mock_prisma.db.litellm_usertable = MagicMock()
        mock_prisma.db.litellm_usertable.upsert = AsyncMock(return_value=mock_user)
        mock_prisma.db.litellm_usertable.update_many = AsyncMock()
        mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)
        mock_prisma.db.litellm_usertable.update = AsyncMock(return_value=mock_user)

        # Mock team membership table
        mock_membership = MagicMock()
        mock_membership.model_dump.return_value = {
            "team_id": "team-within-budget-789",
            "user_id": "non-admin-user-456",
            "budget_id": None,
        }
        mock_prisma.db.litellm_teammembership = MagicMock()
        mock_prisma.db.litellm_teammembership.create = AsyncMock(
            return_value=mock_membership
        )

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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server._license_check") as mock_license,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_org_object"
        ) as mock_get_org,
    ):
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
        mock_created_team.default_team_member_models = None
        mock_created_team.model_dump.return_value = {
            "team_id": "team-org-scoped-789",
            "team_alias": "org-scoped-team",
            "max_budget": 50.0,
            "organization_id": "test-org-123",
            "members_with_roles": [],
        }
        mock_prisma.db.litellm_teamtable.create = AsyncMock(
            return_value=mock_created_team
        )
        _wire_team_create_tx(mock_prisma)
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_created_team
        )

        # Mock model table
        mock_prisma.db.litellm_modeltable = MagicMock()
        mock_prisma.db.litellm_modeltable.create = AsyncMock(
            return_value=MagicMock(id="model123")
        )

        # Mock user table operations
        mock_user = MagicMock()
        mock_user.user_id = "org-admin-user-123"
        mock_user.model_dump.return_value = {
            "user_id": "org-admin-user-123",
            "teams": ["team-org-scoped-789"],
        }
        mock_prisma.db.litellm_usertable = MagicMock()
        mock_prisma.db.litellm_usertable.upsert = AsyncMock(return_value=mock_user)
        mock_prisma.db.litellm_usertable.update_many = AsyncMock()
        mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)
        mock_prisma.db.litellm_usertable.update = AsyncMock(return_value=mock_user)

        # Mock team membership table
        mock_membership = MagicMock()
        mock_membership.model_dump.return_value = {
            "team_id": "team-org-scoped-789",
            "user_id": "org-admin-user-123",
            "budget_id": None,
        }
        mock_prisma.db.litellm_teammembership = MagicMock()
        mock_prisma.db.litellm_teammembership.create = AsyncMock(
            return_value=mock_membership
        )

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
        models=[
            "gpt-4"
        ],  # Within org's allowed models, but not in user's personal models
        organization_id="test-org-456",  # This makes it an org-scoped team
    )

    dummy_request = MagicMock(spec=Request)

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server._license_check") as mock_license,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_org_object"
        ) as mock_get_org,
    ):
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
        mock_created_team.default_team_member_models = None
        mock_created_team.model_dump.return_value = {
            "team_id": "team-org-scoped-models-789",
            "team_alias": "org-scoped-models-team",
            "max_budget": None,
            "organization_id": "test-org-456",
            "models": ["gpt-4"],
            "members_with_roles": [],
        }
        mock_prisma.db.litellm_teamtable.create = AsyncMock(
            return_value=mock_created_team
        )
        _wire_team_create_tx(mock_prisma)
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_created_team
        )

        # Mock model table
        mock_prisma.db.litellm_modeltable = MagicMock()
        mock_prisma.db.litellm_modeltable.create = AsyncMock(
            return_value=MagicMock(id="model123")
        )

        # Mock user table operations
        mock_user = MagicMock()
        mock_user.user_id = "org-admin-user-456"
        mock_user.model_dump.return_value = {
            "user_id": "org-admin-user-456",
            "teams": ["team-org-scoped-models-789"],
        }
        mock_prisma.db.litellm_usertable = MagicMock()
        mock_prisma.db.litellm_usertable.upsert = AsyncMock(return_value=mock_user)
        mock_prisma.db.litellm_usertable.update_many = AsyncMock()
        mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)
        mock_prisma.db.litellm_usertable.update = AsyncMock(return_value=mock_user)

        # Mock team membership table
        mock_membership = MagicMock()
        mock_membership.model_dump.return_value = {
            "team_id": "team-org-scoped-models-789",
            "user_id": "org-admin-user-456",
            "budget_id": None,
        }
        mock_prisma.db.litellm_teammembership = MagicMock()
        mock_prisma.db.litellm_teammembership.create = AsyncMock(
            return_value=mock_membership
        )

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
async def test_new_team_standalone_validates_against_user_models(monkeypatch):
    """
    Test that /team/new WITHOUT organization_id still validates models against user's personal models.

    This ensures that standalone teams (not org-scoped) still use user-level validation.

    Scenario:
    - User has personal models=['no-default-models']
    - Team is created WITHOUT organization_id and models=['gpt-4']
    - Expected: Should fail with "Model not in allowed user models"
    """
    import litellm
    from fastapi import Request

    from litellm.proxy._types import NewTeamRequest, ProxyException, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Avoid injecting max_budget via global defaults; that path calls get_user_object and
    # needs cache/DB mocks — this test only covers model validation.
    monkeypatch.setattr(litellm, "default_team_settings", None)
    monkeypatch.setattr(litellm, "default_team_params", None)

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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server._license_check") as mock_license,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
    ):
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
        assert exc_info.value.code == "400"
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server._license_check") as mock_license,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
    ):
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
        assert exc_info.value.code == "400"
        assert "max budget higher than user max" in str(exc_info.value.message)
        assert "3.0" in str(
            exc_info.value.message
        )  # User's max_budget should be mentioned


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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server._license_check") as mock_license,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_org_object"
        ) as mock_get_org,
    ):
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
        assert exc_info.value.code == "400"
        assert (
            "exceeds organization" in str(exc_info.value.message).lower()
            or "organization" in str(exc_info.value.message).lower()
        )


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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server._license_check") as mock_license,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_org_object"
        ) as mock_get_org,
    ):
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
        assert exc_info.value.code == "400"
        assert (
            "claude-3-opus" in str(exc_info.value.message)
            or "organization" in str(exc_info.value.message).lower()
        )


@pytest.mark.asyncio
async def test_update_team_standalone_budget_raise_blocked_for_team_admin():
    """
    Test that /team/update for a standalone team blocks a non-proxy-admin
    (team admin) from RAISING the team budget above the team's current value.

    Raising a team's spend ceiling is a budget-authority action reserved for
    proxy admins. The rejection is NOT based on the caller's personal budget.

    Scenario:
    - Team admin (internal_user) manages the team
    - Standalone team exists with current budget=$30
    - Admin tries to raise team budget to $100
    - Expected: 403 (only a proxy admin may raise the team budget)
    """
    from fastapi import Request

    from litellm.proxy._types import (
        ProxyException,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    team_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="non-admin-update-test",
        models=[],
    )

    update_request = UpdateTeamRequest(
        team_id="standalone-team-123",
        max_budget=100.0,  # Raise above the team's current $30
    )

    dummy_request = MagicMock(spec=Request)

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ),
    ):
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "standalone-team-123"
        mock_existing_team.organization_id = None  # Standalone team
        mock_existing_team.max_budget = 30.0
        mock_existing_team.model_id = None
        mock_existing_team.model_dump.return_value = {
            "team_id": "standalone-team-123",
            "organization_id": None,
            "max_budget": 30.0,
            "members_with_roles": [
                {"user_id": "non-admin-update-test", "role": "admin"}
            ],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )
        mock_cache.async_get_cache = AsyncMock(return_value=None)

        with pytest.raises(ProxyException) as exc_info:
            await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=team_admin_user,
            )

        assert exc_info.value.code == "403"
        assert "proxy admin" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_update_team_standalone_budget_raise_allowed_for_proxy_admin(
    disable_audit_logging_for_mocked_team,
):
    """
    Test that a proxy admin CAN raise a standalone team's budget on /team/update.

    Scenario:
    - Caller is a proxy admin
    - Standalone team exists with current budget=$30
    - Proxy admin raises team budget to $100
    - Expected: Should succeed (proxy admin holds budget authority)
    """
    from fastapi import Request

    from litellm.proxy._types import UpdateTeamRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    proxy_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="proxy-admin-update-test",
        models=[],
    )

    update_request = UpdateTeamRequest(
        team_id="standalone-team-123",
        max_budget=100.0,  # Raise above the team's current $30
    )

    dummy_request = MagicMock(spec=Request)

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ),
    ):
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "standalone-team-123"
        mock_existing_team.organization_id = None
        mock_existing_team.max_budget = 30.0
        mock_existing_team.model_id = None
        mock_existing_team.model_dump.return_value = {
            "team_id": "standalone-team-123",
            "organization_id": None,
            "max_budget": 30.0,
            "members_with_roles": [
                {"user_id": "proxy-admin-update-test", "role": "admin"}
            ],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )
        mock_prisma.jsonify_team_object = lambda db_data: db_data
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_cache.async_set_cache = AsyncMock()

        mock_updated_team = MagicMock()
        mock_updated_team.team_id = "standalone-team-123"
        mock_updated_team.organization_id = None
        mock_updated_team.max_budget = 100.0
        mock_updated_team.litellm_model_table = None
        mock_updated_team.model_dump.return_value = {
            "team_id": "standalone-team-123",
            "organization_id": None,
            "max_budget": 100.0,
        }
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )

        result = await update_team(
            data=update_request,
            http_request=dummy_request,
            user_api_key_dict=proxy_admin,
        )

        assert result is not None
        assert result["data"].max_budget == 100.0


@pytest.mark.asyncio
async def test_update_team_standalone_budget_removal_blocked_for_team_admin():
    """
    A team admin must not be able to REMOVE a team's spend ceiling
    (max_budget=null), which is the strongest possible raise (finite -> unlimited).

    Scenario:
    - Team admin (internal_user) manages a team with current budget=$500
    - Admin explicitly sets max_budget=None to strip the cap
    - Expected: 403 (only a proxy admin can remove the team budget)
    """
    from fastapi import Request

    from litellm.proxy._types import (
        ProxyException,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    team_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="budget-removal-admin",
        models=[],
    )

    # Explicitly set max_budget=None so it lands in model_fields_set and would be
    # persisted by data.json(exclude_unset=True).
    update_request = UpdateTeamRequest(
        team_id="standalone-team-123",
        max_budget=None,
    )
    assert "max_budget" in update_request.model_fields_set

    dummy_request = MagicMock(spec=Request)

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ),
    ):
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "standalone-team-123"
        mock_existing_team.organization_id = None
        mock_existing_team.max_budget = 500.0
        mock_existing_team.model_id = None
        mock_existing_team.model_dump.return_value = {
            "team_id": "standalone-team-123",
            "organization_id": None,
            "max_budget": 500.0,
            "members_with_roles": [
                {"user_id": "budget-removal-admin", "role": "admin"}
            ],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )
        mock_cache.async_get_cache = AsyncMock(return_value=None)

        with pytest.raises(ProxyException) as exc_info:
            await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=team_admin_user,
            )

        assert exc_info.value.code == "403"
        assert "remove" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_update_team_standalone_uncapped_team_admin_sets_finite_allowed(
    disable_audit_logging_for_mocked_team,
):
    """
    When a team currently has NO cap (max_budget=None / unlimited), a team admin
    setting a finite max_budget is a RESTRICTION, not a raise, and is
    intentionally allowed.

    Scenario:
    - Team admin manages a team with current max_budget=None (unlimited)
    - Admin sets max_budget=1000 (unlimited -> finite is more restrictive)
    - Expected: 200
    """
    from fastapi import Request

    from litellm.proxy._types import UpdateTeamRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    team_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="uncapped-team-admin",
        models=[],
    )

    update_request = UpdateTeamRequest(
        team_id="standalone-uncapped-123",
        max_budget=1000.0,
    )

    dummy_request = MagicMock(spec=Request)

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ),
    ):
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "standalone-uncapped-123"
        mock_existing_team.organization_id = None
        mock_existing_team.max_budget = None  # team has no cap
        mock_existing_team.model_id = None
        mock_existing_team.model_dump.return_value = {
            "team_id": "standalone-uncapped-123",
            "organization_id": None,
            "max_budget": None,
            "members_with_roles": [
                {"user_id": "uncapped-team-admin", "role": "admin"}
            ],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )
        mock_prisma.jsonify_team_object = lambda db_data: db_data
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_cache.async_set_cache = AsyncMock()

        mock_updated_team = MagicMock()
        mock_updated_team.team_id = "standalone-uncapped-123"
        mock_updated_team.organization_id = None
        mock_updated_team.max_budget = 1000.0
        mock_updated_team.litellm_model_table = None
        mock_updated_team.model_dump.return_value = {
            "team_id": "standalone-uncapped-123",
            "organization_id": None,
            "max_budget": 1000.0,
        }
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )

        result = await update_team(
            data=update_request,
            http_request=dummy_request,
            user_api_key_dict=team_admin_user,
        )

        assert result is not None
        assert result["data"].max_budget == 1000.0


@pytest.mark.asyncio
async def test_update_team_standalone_unchanged_budget_allowed(
    disable_audit_logging_for_mocked_team,
):
    """
    Test that /team/update for a standalone team does NOT compare against the
    caller's personal max_budget when the budget is unchanged.

    This is the LiteLLM UI scenario: the UI sends the full team object on every
    update (including the unchanged max_budget). A team admin only changing
    tpm_limit should not be blocked by a budget the team already has.

    Scenario:
    - User (team admin) has personal max_budget=$100
    - Standalone team exists with current budget=$500
    - User updates tpm_limit and re-sends the unchanged max_budget=$500
    - Expected: Should succeed (budget unchanged, not an increase)
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_UserTable,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    team_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="standalone-unchanged-budget-admin",
        models=[],
    )

    # UI re-sends the unchanged max_budget alongside the tpm_limit change.
    update_request = UpdateTeamRequest(
        team_id="standalone-unchanged-budget-123",
        max_budget=500.0,  # Unchanged from the team's current budget
        tpm_limit=50000,
    )

    dummy_request = MagicMock(spec=Request)

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
    ):
        # Mock existing standalone team (no organization_id) with budget=$500
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "standalone-unchanged-budget-123"
        mock_existing_team.organization_id = None
        mock_existing_team.max_budget = 500.0
        mock_existing_team.model_id = None
        mock_existing_team.model_dump.return_value = {
            "team_id": "standalone-unchanged-budget-123",
            "organization_id": None,
            "max_budget": 500.0,
            "members_with_roles": [
                {"user_id": "standalone-unchanged-budget-admin", "role": "admin"}
            ],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )
        mock_prisma.jsonify_team_object = lambda db_data: db_data

        # User has a restrictive personal budget that is lower than the team's.
        mock_user_obj = LiteLLM_UserTable(
            user_id="standalone-unchanged-budget-admin",
            max_budget=100.0,
        )
        mock_cache.async_get_cache = AsyncMock(return_value=mock_user_obj)
        mock_cache.async_set_cache = AsyncMock()

        mock_updated_team = MagicMock()
        mock_updated_team.team_id = "standalone-unchanged-budget-123"
        mock_updated_team.organization_id = None
        mock_updated_team.max_budget = 500.0
        mock_updated_team.litellm_model_table = None
        mock_updated_team.model_dump.return_value = {
            "team_id": "standalone-unchanged-budget-123",
            "organization_id": None,
            "max_budget": 500.0,
            "tpm_limit": 50000,
        }
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )

        # Should NOT raise - unchanged budget skips the personal-budget check.
        result = await update_team(
            data=update_request,
            http_request=dummy_request,
            user_api_key_dict=team_admin_user,
        )

        assert result is not None
        assert result["data"].max_budget == 500.0


@pytest.mark.asyncio
async def test_update_team_standalone_lower_budget_allowed(
    disable_audit_logging_for_mocked_team,
):
    """
    Test that /team/update for a standalone team allows lowering the budget
    below the team's current value even when the new value still exceeds the
    caller's personal max_budget.

    Scenario:
    - User (team admin) has personal max_budget=$100
    - Standalone team exists with current budget=$500
    - User lowers team budget to $300 (a decrease, still above user's $100)
    - Expected: Should succeed (decrease is not an increase above team budget)
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_UserTable,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    team_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="standalone-lower-budget-admin",
        models=[],
    )

    update_request = UpdateTeamRequest(
        team_id="standalone-lower-budget-123",
        max_budget=300.0,  # Lower than current $500, still above user's $100
    )

    dummy_request = MagicMock(spec=Request)

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
    ):
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "standalone-lower-budget-123"
        mock_existing_team.organization_id = None
        mock_existing_team.max_budget = 500.0
        mock_existing_team.model_id = None
        mock_existing_team.model_dump.return_value = {
            "team_id": "standalone-lower-budget-123",
            "organization_id": None,
            "max_budget": 500.0,
            "members_with_roles": [
                {"user_id": "standalone-lower-budget-admin", "role": "admin"}
            ],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )
        mock_prisma.jsonify_team_object = lambda db_data: db_data

        mock_user_obj = LiteLLM_UserTable(
            user_id="standalone-lower-budget-admin",
            max_budget=100.0,
        )
        mock_cache.async_get_cache = AsyncMock(return_value=mock_user_obj)
        mock_cache.async_set_cache = AsyncMock()

        mock_updated_team = MagicMock()
        mock_updated_team.team_id = "standalone-lower-budget-123"
        mock_updated_team.organization_id = None
        mock_updated_team.max_budget = 300.0
        mock_updated_team.litellm_model_table = None
        mock_updated_team.model_dump.return_value = {
            "team_id": "standalone-lower-budget-123",
            "organization_id": None,
            "max_budget": 300.0,
        }
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )

        result = await update_team(
            data=update_request,
            http_request=dummy_request,
            user_api_key_dict=team_admin_user,
        )

        assert result is not None
        assert result["data"].max_budget == 300.0


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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
            new=AsyncMock(return_value=mock_org),
        ) as mock_get_org,
    ):
        # Mock existing org-scoped team
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "org-team-456"
        mock_existing_team.organization_id = "test-org-update"
        mock_existing_team.max_budget = 80.0
        mock_existing_team.model_dump.return_value = {
            "team_id": "org-team-456",
            "organization_id": "test-org-update",
            "max_budget": 80.0,
            "members_with_roles": [
                {"user_id": "org-admin-update-test", "role": "admin"}
            ],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )

        # Should raise ProxyException because new budget exceeds org's max_budget
        with pytest.raises(ProxyException) as exc_info:
            await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=org_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == "400"
        assert (
            "organization" in str(exc_info.value.message).lower()
            or "budget" in str(exc_info.value.message).lower()
        )


@pytest.mark.asyncio
async def test_update_team_standalone_models_not_gated_by_user_limit(
    disable_audit_logging_for_mocked_team,
):
    """
    Test that /team/update for a standalone team does NOT gate the team's models
    by the caller's personal allowed models.

    A team admin authorized via _verify_team_access() may set the team's models
    independently of their own personal model list on update.

    Scenario:
    - Team admin has personal models=['gpt-3.5-turbo']
    - Standalone team exists (no organization_id)
    - Admin updates team models to ['gpt-4'] (not in their personal list)
    - Expected: Should succeed (personal models are irrelevant on /team/update)
    """
    from fastapi import Request

    from litellm.proxy._types import UpdateTeamRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    team_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="non-admin-update-models-test",
        models=["gpt-3.5-turbo"],  # Restrictive personal model list
    )

    update_request = UpdateTeamRequest(
        team_id="standalone-team-models-123",
        models=["gpt-4"],  # Not in the admin's personal allowed models
    )

    dummy_request = MagicMock(spec=Request)

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
    ):
        # Mock existing standalone team (no organization_id)
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "standalone-team-models-123"
        mock_existing_team.organization_id = None  # Standalone team
        mock_existing_team.models = ["gpt-3.5-turbo"]
        mock_existing_team.model_id = None
        mock_existing_team.model_dump.return_value = {
            "team_id": "standalone-team-models-123",
            "organization_id": None,
            "models": ["gpt-3.5-turbo"],
            "members_with_roles": [
                {"user_id": "non-admin-update-models-test", "role": "admin"}
            ],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )
        mock_prisma.jsonify_team_object = lambda db_data: db_data
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_cache.async_set_cache = AsyncMock()

        mock_updated_team = MagicMock()
        mock_updated_team.team_id = "standalone-team-models-123"
        mock_updated_team.organization_id = None
        mock_updated_team.litellm_model_table = None
        mock_updated_team.model_dump.return_value = {
            "team_id": "standalone-team-models-123",
            "organization_id": None,
            "models": ["gpt-4"],
        }
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )

        result = await update_team(
            data=update_request,
            http_request=dummy_request,
            user_api_key_dict=team_admin_user,
        )

        assert result is not None


@pytest.mark.asyncio
async def test_update_team_org_scoped_budget_bypasses_user_limit(
    disable_audit_logging_for_mocked_team,
):
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
            new=AsyncMock(return_value=mock_org),
        ) as mock_get_org,
    ):
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
            "members_with_roles": [
                {"user_id": "org-admin-update-budget-test", "role": "admin"}
            ],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )
        mock_prisma.jsonify_team_object = lambda db_data: db_data

        # Mock user cache to return user with restrictive budget
        mock_user_obj = LiteLLM_UserTable(
            user_id="org-admin-update-budget-test",
            max_budget=3.0,  # Restrictive personal budget
        )
        mock_cache.async_get_cache = AsyncMock(return_value=mock_user_obj)
        mock_cache.async_set_cache = (
            AsyncMock()
        )  # Mock cache set for _cache_team_object

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
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )

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
async def test_update_team_org_scoped_models_bypasses_user_limit(
    disable_audit_logging_for_mocked_team,
):
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
            new=AsyncMock(return_value=mock_org),
        ) as mock_get_org,
    ):
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
            "members_with_roles": [
                {"user_id": "org-admin-update-models-test", "role": "admin"}
            ],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )
        mock_prisma.jsonify_team_object = lambda db_data: db_data
        mock_cache.async_set_cache = (
            AsyncMock()
        )  # Mock cache set for _cache_team_object

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
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )

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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
            new=AsyncMock(return_value=mock_org),
        ) as mock_get_org,
    ):
        # Mock existing org-scoped team
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "org-team-update-models-fail-123"
        mock_existing_team.organization_id = "test-org-update-models-fail"
        mock_existing_team.models = ["gpt-4"]
        mock_existing_team.model_dump.return_value = {
            "team_id": "org-team-update-models-fail-123",
            "organization_id": "test-org-update-models-fail",
            "models": ["gpt-4"],
            "members_with_roles": [
                {"user_id": "org-admin-update-models-fail-test", "role": "admin"}
            ],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )

        # Should raise ProxyException because claude-3-opus is not in org's allowed models
        with pytest.raises(ProxyException) as exc_info:
            await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=org_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == "400"
        assert (
            "claude-3-opus" in str(exc_info.value.message)
            or "organization" in str(exc_info.value.message).lower()
        )


@pytest.mark.asyncio
async def test_update_team_org_scoped_models_with_all_proxy_models(
    disable_audit_logging_for_mocked_team,
):
    """
    Test that /team/update for an org-scoped team succeeds when organization has 'all-proxy-models'.

    Scenario:
    - Organization has models=['all-proxy-models'] (catch-all for all models)
    - Org-scoped team exists
    - User tries to update team models to ['rerank-english-v3.0', 'text-embedding-3-small', 'gpt-4o-mini-test']
    - Expected: Should succeed because 'all-proxy-models' allows all models
    """
    from fastapi import Request

    from litellm.proxy._types import (
        SpecialModelNames,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Create user (org admin)
    org_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="org-admin-all-proxy-models-test",
        models=[],
    )

    # Create update request with models that aren't explicitly in org's models list
    # but should be allowed because org has 'all-proxy-models'
    update_request = UpdateTeamRequest(
        team_id="org-team-all-proxy-models-123",
        models=["rerank-english-v3.0", "text-embedding-3-small", "gpt-4o-mini-test"],
    )

    dummy_request = MagicMock(spec=Request)

    # Mock organization with 'all-proxy-models' (catch-all)
    mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
    mock_org.organization_id = "test-org-all-proxy-models"
    mock_org.models = [SpecialModelNames.all_proxy_models.value]  # Allows all models
    mock_org.litellm_budget_table = None

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
            new=AsyncMock(return_value=mock_org),
        ) as mock_get_org,
    ):
        # Mock existing org-scoped team
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "org-team-all-proxy-models-123"
        mock_existing_team.organization_id = "test-org-all-proxy-models"
        mock_existing_team.models = ["gpt-4"]
        mock_existing_team.model_id = None
        mock_existing_team.model_dump.return_value = {
            "team_id": "org-team-all-proxy-models-123",
            "organization_id": "test-org-all-proxy-models",
            "models": ["gpt-4"],
            "members_with_roles": [
                {"user_id": "org-admin-all-proxy-models-test", "role": "admin"}
            ],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )
        mock_prisma.jsonify_team_object = lambda db_data: db_data
        mock_cache.async_set_cache = (
            AsyncMock()
        )  # Mock cache set for _cache_team_object

        # Mock team update
        mock_updated_team = MagicMock()
        mock_updated_team.team_id = "org-team-all-proxy-models-123"
        mock_updated_team.organization_id = "test-org-all-proxy-models"
        mock_updated_team.models = [
            "rerank-english-v3.0",
            "text-embedding-3-small",
            "gpt-4o-mini-test",
        ]
        mock_updated_team.litellm_model_table = None
        mock_updated_team.model_dump.return_value = {
            "team_id": "org-team-all-proxy-models-123",
            "organization_id": "test-org-all-proxy-models",
            "models": [
                "rerank-english-v3.0",
                "text-embedding-3-small",
                "gpt-4o-mini-test",
            ],
        }
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )

        # Should NOT raise an exception - 'all-proxy-models' allows all models
        result = await update_team(
            data=update_request,
            http_request=dummy_request,
            user_api_key_dict=org_admin_user,
        )

        # Verify the team was updated successfully with the new models
        assert result is not None
        assert result["data"].models == [
            "rerank-english-v3.0",
            "text-embedding-3-small",
            "gpt-4o-mini-test",
        ]


@pytest.mark.asyncio
async def test_update_team_tpm_limit_not_gated_by_user_limit(
    disable_audit_logging_for_mocked_team,
):
    """
    Test that /team/update does NOT gate the team's tpm_limit by the caller's
    personal tpm_limit.

    A team admin authorized via _verify_team_access() may raise the team's
    tpm_limit above their own personal tpm_limit on update.

    Scenario:
    - Team admin has personal tpm_limit=1000
    - Standalone team exists with tpm_limit=500
    - Admin updates team tpm_limit to 5000 (above their personal 1000)
    - Expected: Should succeed (personal tpm is irrelevant on /team/update)
    """
    from fastapi import Request

    from litellm.proxy._types import UpdateTeamRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    team_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="tpm-limit-user",
        models=[],
        tpm_limit=1000,  # Restrictive personal TPM limit
    )

    update_request = UpdateTeamRequest(
        team_id="team-tpm-test-123",
        tpm_limit=5000,  # Above the admin's personal 1000
    )

    dummy_request = MagicMock(spec=Request)

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ),
    ):
        # Mock existing standalone team
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "team-tpm-test-123"
        mock_existing_team.organization_id = None
        mock_existing_team.tpm_limit = 500
        mock_existing_team.model_id = None
        mock_existing_team.model_dump.return_value = {
            "team_id": "team-tpm-test-123",
            "organization_id": None,
            "tpm_limit": 500,
            "members_with_roles": [{"user_id": "tpm-limit-user", "role": "admin"}],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )
        mock_prisma.jsonify_team_object = lambda db_data: db_data
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_cache.async_set_cache = AsyncMock()

        mock_updated_team = MagicMock()
        mock_updated_team.team_id = "team-tpm-test-123"
        mock_updated_team.organization_id = None
        mock_updated_team.litellm_model_table = None
        mock_updated_team.model_dump.return_value = {
            "team_id": "team-tpm-test-123",
            "organization_id": None,
            "tpm_limit": 5000,
        }
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )

        result = await update_team(
            data=update_request,
            http_request=dummy_request,
            user_api_key_dict=team_admin_user,
        )

        assert result is not None


@pytest.mark.asyncio
async def test_update_team_rpm_limit_not_gated_by_user_limit(
    disable_audit_logging_for_mocked_team,
):
    """
    Test that /team/update does NOT gate the team's rpm_limit by the caller's
    personal rpm_limit.

    Scenario:
    - Team admin has personal rpm_limit=100
    - Standalone team exists with rpm_limit=50
    - Admin updates team rpm_limit to 500 (above their personal 100)
    - Expected: Should succeed (personal rpm is irrelevant on /team/update)
    """
    from fastapi import Request

    from litellm.proxy._types import UpdateTeamRequest, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    team_admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="rpm-limit-user",
        models=[],
        rpm_limit=100,  # Restrictive personal RPM limit
    )

    update_request = UpdateTeamRequest(
        team_id="team-rpm-test-123",
        rpm_limit=500,  # Above the admin's personal 100
    )

    dummy_request = MagicMock(spec=Request)

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ),
    ):
        # Mock existing standalone team
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "team-rpm-test-123"
        mock_existing_team.organization_id = None
        mock_existing_team.rpm_limit = 50
        mock_existing_team.model_id = None
        mock_existing_team.model_dump.return_value = {
            "team_id": "team-rpm-test-123",
            "organization_id": None,
            "rpm_limit": 50,
            "members_with_roles": [{"user_id": "rpm-limit-user", "role": "admin"}],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )
        mock_prisma.jsonify_team_object = lambda db_data: db_data
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_cache.async_set_cache = AsyncMock()

        mock_updated_team = MagicMock()
        mock_updated_team.team_id = "team-rpm-test-123"
        mock_updated_team.organization_id = None
        mock_updated_team.litellm_model_table = None
        mock_updated_team.model_dump.return_value = {
            "team_id": "team-rpm-test-123",
            "organization_id": None,
            "rpm_limit": 500,
        }
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )

        result = await update_team(
            data=update_request,
            http_request=dummy_request,
            user_api_key_dict=team_admin_user,
        )

        assert result is not None


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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch("litellm.proxy.proxy_server._license_check") as mock_license,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
            new=AsyncMock(return_value=mock_org),
        ),
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
        assert exc_info.value.code == "400"
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch("litellm.proxy.proxy_server._license_check") as mock_license,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
            new=AsyncMock(return_value=mock_org),
        ),
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
        assert exc_info.value.code == "400"
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
        rpm_limit=100,  # Restrictive user RPM limit
    )

    # Create team request exceeding user limits but within org limits
    team_request = NewTeamRequest(
        team_alias="org-bypass-test-team",
        organization_id="test-org-bypass",
        tpm_limit=10000,  # Exceeds user's 1000 but within org's 50000
        rpm_limit=1000,  # Exceeds user's 100 but within org's 5000
    )

    dummy_request = MagicMock(spec=Request)

    # Mock organization with generous limits
    mock_budget_table = MagicMock(spec=LiteLLM_BudgetTable)
    mock_budget_table.tpm_limit = 50000  # Generous org TPM limit
    mock_budget_table.rpm_limit = 5000  # Generous org RPM limit
    mock_budget_table.max_budget = None

    mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
    mock_org.organization_id = "test-org-bypass"
    mock_org.models = ["gpt-4"]
    mock_org.litellm_budget_table = mock_budget_table

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch("litellm.proxy.proxy_server._license_check") as mock_license,
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
            new=AsyncMock(return_value=mock_org),
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._add_team_members_to_team",
            new=AsyncMock(),
        ),
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
        mock_created_team.access_group_ids = None
        mock_created_team.model_dump.return_value = {
            "team_id": "new-bypass-team-id",
            "team_alias": "org-bypass-test-team",
            "tpm_limit": 10000,
            "rpm_limit": 1000,
            "metadata": None,
            "members_with_roles": [],
        }
        mock_prisma.db.litellm_teamtable.create = AsyncMock(
            return_value=mock_created_team
        )
        _wire_team_create_tx(mock_prisma)
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_created_team
        )
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
            new=AsyncMock(return_value=mock_org),
        ),
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
            "members_with_roles": [
                {"user_id": "org-admin-update-tpm-test", "role": "admin"}
            ],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )

        # Should raise ProxyException because TPM exceeds org limit
        with pytest.raises(ProxyException) as exc_info:
            await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=org_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == "400"
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

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
            new=AsyncMock(return_value=mock_org),
        ),
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
            "members_with_roles": [
                {"user_id": "org-admin-update-rpm-test", "role": "admin"}
            ],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )

        # Should raise ProxyException because RPM exceeds org limit
        with pytest.raises(ProxyException) as exc_info:
            await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=org_admin_user,
            )

        # Verify exception details
        assert exc_info.value.code == "400"
        assert "rpm" in str(exc_info.value.message).lower()


@pytest.mark.asyncio
async def test_update_team_org_scoped_tpm_rpm_bypasses_user_limit(
    disable_audit_logging_for_mocked_team,
):
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
        rpm_limit=100,  # Restrictive user RPM limit
    )

    # Create update request exceeding user limits but within org limits
    update_request = UpdateTeamRequest(
        team_id="org-team-update-bypass-123",
        tpm_limit=10000,  # Exceeds user's 1000 but within org's 50000
        rpm_limit=1000,  # Exceeds user's 100 but within org's 5000
    )

    dummy_request = MagicMock(spec=Request)

    # Mock organization with generous limits
    mock_budget_table = MagicMock(spec=LiteLLM_BudgetTable)
    mock_budget_table.tpm_limit = 50000  # Generous org TPM limit
    mock_budget_table.rpm_limit = 5000  # Generous org RPM limit
    mock_budget_table.max_budget = None

    mock_org = MagicMock(spec=LiteLLM_OrganizationTable)
    mock_org.organization_id = "test-org-update-bypass"
    mock_org.models = ["gpt-4"]
    mock_org.litellm_budget_table = mock_budget_table

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_org_object",
            new=AsyncMock(return_value=mock_org),
        ),
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
            "members_with_roles": [
                {"user_id": "org-admin-update-bypass-test", "role": "admin"}
            ],
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )
        mock_cache.async_set_cache = AsyncMock()
        mock_logging.internal_usage_cache.dual_cache.async_delete_cache = AsyncMock()

        # Mock team update
        mock_updated_team = MagicMock(spec=LiteLLM_TeamTable)
        mock_updated_team.team_id = "org-team-update-bypass-123"
        mock_updated_team.tpm_limit = 10000
        mock_updated_team.rpm_limit = 1000
        mock_updated_team.access_group_ids = None
        mock_updated_team.model_dump.return_value = {
            "team_id": "org-team-update-bypass-123",
            "tpm_limit": 10000,
            "rpm_limit": 1000,
        }
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )
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
async def test_update_team_guardrails_with_org_id(
    disable_audit_logging_for_mocked_team,
):
    """
    Test that updating team guardrails works when team has an organization_id.
    The fix ensures 'teams' field is included when fetching organization data.
    """
    from fastapi import Request

    from litellm.proxy._types import (
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
    mock_org_member = MagicMock()
    mock_org_member.user_id = "org-admin-guardrails-test"
    mock_org.members = [mock_org_member]
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
        "members": [
            {
                "user_id": "org-admin-guardrails-test",
                "organization_id": "test-org-guardrails",
                "created_at": datetime(2024, 1, 1),
                "updated_at": datetime(2024, 1, 1),
            }
        ],
        "teams": [],
    }

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ),
        patch(
            "litellm.proxy.proxy_server.premium_user",
            True,  # Required for guardrails feature
        ),
        patch("litellm.proxy.proxy_server.llm_router", MagicMock()),
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
            "members_with_roles": [
                {"user_id": "org-admin-guardrails-test", "role": "admin"}
            ],
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

        # Destination-org guard in update_team queries for the caller's
        # ORG_ADMIN membership on the destination org. Return a match so
        # the guardrails-update path (the subject under test) proceeds.
        mock_org_admin_membership = MagicMock()
        mock_org_admin_membership.user_id = "org-admin-guardrails-test"
        mock_org_admin_membership.organization_id = "test-org-guardrails"
        mock_prisma.db.litellm_organizationmembership.find_many = AsyncMock(
            return_value=[mock_org_admin_membership]
        )

        # Mock team update
        mock_updated_team = MagicMock(spec=LiteLLM_TeamTable)
        mock_updated_team.team_id = "team-guardrails-123"
        mock_updated_team.organization_id = "test-org-guardrails"
        mock_updated_team.metadata = {
            "guardrails": ["aporia-pre-call", "aporia-post-call"]
        }
        mock_updated_team.litellm_model_table = None
        mock_updated_team.access_group_ids = None
        mock_updated_team.model_dump.return_value = {
            "team_id": "team-guardrails-123",
            "organization_id": "test-org-guardrails",
            "metadata": {"guardrails": ["aporia-pre-call", "aporia-post-call"]},
        }
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )
        mock_prisma.jsonify_team_object = MagicMock(side_effect=lambda db_data: db_data)
        # async_get_cache must be an AsyncMock so `await` in get_org_object works
        mock_cache.async_get_cache = AsyncMock(return_value=None)
        mock_cache.async_set_cache = AsyncMock()

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
            assert result["data"].metadata["guardrails"] == [
                "aporia-pre-call",
                "aporia-post-call",
            ]

            # Verify that organization fetch was called with proper include clause
            # The function is called twice: once by fetch_and_validate_organization (with include)
            # and once by get_org_object (without include). We verify the first call has 'teams'.
            assert mock_prisma.db.litellm_organizationtable.find_unique.call_count >= 1

            # Get the first call (from fetch_and_validate_organization)
            first_call_kwargs = (
                mock_prisma.db.litellm_organizationtable.find_unique.call_args_list[
                    0
                ].kwargs
            )

            # Verify that 'teams' is included in the fetch
            assert "include" in first_call_kwargs
            assert "teams" in first_call_kwargs["include"]
            assert first_call_kwargs["include"]["teams"] is True


def test_transform_teams_to_deleted_records():
    from datetime import datetime, timezone

    user_api_key_dict = UserAPIKeyAuth(
        user_id="user-123",
        api_key="sk-test",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
    )

    team1 = LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="test-team-1",
        members_with_roles=[
            Member(user_id="user-1", role="admin"),
            Member(user_id="user-2", role="user"),
        ],
        metadata={"test": "value"},
        model_max_budget={},
        model_spend={},
    )

    team2 = LiteLLM_TeamTable(
        team_id="team-2",
        team_alias="test-team-2",
        members_with_roles=[],
        metadata=None,
        model_max_budget={"gpt-4": {"budget_limit": 100.0}},
        model_spend={},
    )

    records = _transform_teams_to_deleted_records(
        teams=[team1, team2],
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by="admin-user",
    )

    assert len(records) == 2
    assert all("deleted_at" in record for record in records)
    assert all("deleted_by" in record for record in records)
    assert all("deleted_by_api_key" in record for record in records)
    assert all("litellm_changed_by" in record for record in records)
    assert all(record["deleted_by"] == "user-123" for record in records)
    # UserAPIKeyAuth hashes the api_key, so we check against the hashed value
    assert all(
        record["deleted_by_api_key"] == user_api_key_dict.api_key for record in records
    )
    assert all(record["litellm_changed_by"] == "admin-user" for record in records)

    record1 = records[0]
    assert record1["team_id"] == "team-1"
    assert isinstance(record1["members_with_roles"], str)
    assert isinstance(record1["metadata"], str)
    assert "litellm_model_table" not in record1
    assert "object_permission" not in record1
    assert "id" not in record1

    record2 = records[1]
    assert record2["team_id"] == "team-2"
    # model_max_budget should be converted to JSON string if it exists
    if "model_max_budget" in record2:
        assert isinstance(record2["model_max_budget"], str)


def test_transform_teams_to_deleted_records_empty_list():
    user_api_key_dict = UserAPIKeyAuth(
        user_id="user-123",
        api_key="sk-test",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
    )

    records = _transform_teams_to_deleted_records(
        teams=[],
        user_api_key_dict=user_api_key_dict,
    )

    assert records == []


@pytest.mark.asyncio
async def test_save_deleted_team_records():
    mock_prisma_client = AsyncMock()
    mock_create_many = AsyncMock()
    mock_prisma_client.db.litellm_deletedteamtable.create_many = mock_create_many

    records = [
        {
            "team_id": "team-1",
            "team_alias": "test-team-1",
            "deleted_at": "2024-01-01T00:00:00Z",
            "deleted_by": "admin",
        },
        {
            "team_id": "team-2",
            "team_alias": "test-team-2",
            "deleted_at": "2024-01-01T00:00:00Z",
            "deleted_by": "admin",
        },
    ]

    await _save_deleted_team_records(records=records, prisma_client=mock_prisma_client)

    mock_create_many.assert_called_once_with(data=records)


@pytest.mark.asyncio
async def test_save_deleted_team_records_empty_list():
    mock_prisma_client = AsyncMock()
    mock_create_many = AsyncMock()
    mock_prisma_client.db.litellm_deletedteamtable.create_many = mock_create_many

    await _save_deleted_team_records(records=[], prisma_client=mock_prisma_client)

    mock_create_many.assert_not_called()


@pytest.mark.asyncio
async def test_persist_deleted_team_records():
    mock_prisma_client = AsyncMock()
    mock_create_many = AsyncMock()
    mock_prisma_client.db.litellm_deletedteamtable.create_many = mock_create_many

    user_api_key_dict = UserAPIKeyAuth(
        user_id="user-123",
        api_key="sk-test",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
    )

    team = LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="test-team",
        members_with_roles=[
            Member(user_id="user-1", role="admin"),
        ],
        metadata={},
        model_max_budget={},
        model_spend={},
    )

    await _persist_deleted_team_records(
        teams=[team],
        prisma_client=mock_prisma_client,
        user_api_key_dict=user_api_key_dict,
        litellm_changed_by="admin-user",
    )

    mock_create_many.assert_called_once()
    call_args = mock_create_many.call_args
    assert "data" in call_args.kwargs
    records = call_args.kwargs["data"]
    assert len(records) == 1
    assert records[0]["team_id"] == "team-1"
    assert records[0]["deleted_by"] == "user-123"
    assert records[0]["litellm_changed_by"] == "admin-user"


@pytest.mark.asyncio
async def test_delete_team_persists_deleted_teams(
    monkeypatch,
    disable_audit_logging_for_mocked_team,
):
    from litellm.proxy._types import DeleteTeamRequest

    mock_prisma_client = AsyncMock()
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_id="admin-user",
        api_key="sk-admin",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
    )

    team1 = LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="test-team-1",
        members_with_roles=[
            Member(user_id="user-1", role="admin"),
        ],
        metadata={},
        model_max_budget={},
        model_spend={},
    )

    mock_find_unique = AsyncMock(return_value=team1)
    mock_prisma_client.db.litellm_teamtable.find_unique = mock_find_unique

    mock_delete_data = AsyncMock(return_value={"deleted_teams": ["team-1"]})
    mock_prisma_client.delete_data = mock_delete_data

    mock_create_many_teams = AsyncMock()
    mock_prisma_client.db.litellm_deletedteamtable.create_many = mock_create_many_teams

    mock_create_many_keys = AsyncMock()
    mock_prisma_client.db.litellm_deletedverificationtoken.create_many = (
        mock_create_many_keys
    )

    mock_find_many_keys = AsyncMock(return_value=[])
    mock_prisma_client.db.litellm_verificationtoken.find_many = mock_find_many_keys

    # delete_team now deletes team BYOK models inside a transaction; this team has none.
    mock_tx = AsyncMock()
    mock_tx.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
    mock_tx_cm = MagicMock()
    mock_tx_cm.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_tx_cm.__aexit__ = AsyncMock(return_value=False)
    mock_prisma_client.db.tx = MagicMock(return_value=mock_tx_cm)
    _wire_team_delete_tx(mock_prisma_client)

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma_client,
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.create_audit_log_for_update",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.litellm_proxy_admin_name",
        "admin",
    )
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.team_endpoints.team_member_delete",
        AsyncMock(return_value=team1),
    )

    data = DeleteTeamRequest(team_ids=["team-1"])

    result = await delete_team(
        data=data,
        http_request=MagicMock(),
        user_api_key_dict=mock_user_api_key_dict,
        litellm_changed_by="admin-user",
    )

    mock_create_many_teams.assert_called_once()
    call_args = mock_create_many_teams.call_args
    assert "data" in call_args.kwargs
    records = call_args.kwargs["data"]
    assert len(records) == 1
    assert records[0]["team_id"] == "team-1"
    assert records[0]["deleted_by"] == "admin-user"
    assert records[0]["litellm_changed_by"] == "admin-user"


@pytest.mark.asyncio
async def test_delete_team_sweeps_references_outside_members_with_roles(
    monkeypatch,
    disable_audit_logging_for_mocked_team,
):
    """
    Regression pin for LIT-5511: a deleted team stayed visible on user records.

    `delete_team` drove all of its cleanup off `team.members_with_roles`, so a user row that
    referenced the team by any other route (`/user/update`, SSO sync, a membership row written
    without a matching roster entry) kept the dangling team id forever and `/user/info` kept
    listing the deleted team. The roster here is deliberately EMPTY, so nothing the per-member
    `team_member_delete` path does can make this test pass.

    Both cache keys `_cache_team_object` writes are asserted in the same delete: the id key feeds
    `get_team_object` and the alias key feeds the JWT `team_alias_jwt_field` path, so either one
    surviving keeps the deleted team resolvable for auth until its TTL expires.
    """
    from litellm.proxy._types import DeleteTeamRequest
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    doomed_team = LiteLLM_TeamTable(
        team_id="team-doomed",
        team_alias="doomed-team",
        members_with_roles=[],
        metadata={},
        model_max_budget={},
        model_spend={},
    )

    cache_state_when_rows_deleted = {}

    async def record_cache_state_then_delete(*args, **kwargs):
        cache_state_when_rows_deleted["doomed_still_cached"] = (
            fresh_cache.get_cache(key="team_id:team-doomed") is not None
        )
        return 1

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=doomed_team)
    mock_prisma_client.delete_data = AsyncMock(return_value={"deleted_keys": 0})
    mock_prisma_client.db.litellm_deletedteamtable.create_many = AsyncMock()
    mock_prisma_client.db.litellm_deletedverificationtoken.create_many = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])

    mock_execute_raw = AsyncMock()
    mock_prisma_client.db.execute_raw = mock_execute_raw
    mock_membership_delete_many = AsyncMock()
    mock_prisma_client.db.litellm_teammembership.delete_many = mock_membership_delete_many
    mock_prisma_client.db.litellm_teamtable.delete_many = AsyncMock(side_effect=record_cache_state_then_delete)

    mock_tx = AsyncMock()
    mock_tx.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
    mock_tx_cm = MagicMock()
    mock_tx_cm.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_tx_cm.__aexit__ = AsyncMock(return_value=False)
    mock_prisma_client.db.tx = MagicMock(return_value=mock_tx_cm)

    # The locked delete-and-sweep transaction /team/member_add serializes against, kept
    # separate from mock_tx above (the BYOK-model-cleanup transaction, unrelated to this lock).
    _wire_team_delete_tx(mock_prisma_client)
    mock_lock_tx = mock_prisma_client.tx.return_value.__aenter__.return_value

    fresh_cache = UserApiKeyCache()
    for cached_team_id, cached_alias in (
        ("team-doomed", "doomed-team"),
        ("team-kept", "kept-team"),
    ):
        cached_obj = LiteLLM_TeamTableCachedObj(
            team_id=cached_team_id, team_alias=cached_alias
        )
        fresh_cache.set_cache(key=f"team_id:{cached_team_id}", value=cached_obj)
        fresh_cache.set_cache(key=f"team_alias:{cached_alias}", value=cached_obj)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", fresh_cache)
    monkeypatch.setattr("litellm.proxy.proxy_server.create_audit_log_for_update", AsyncMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin")

    await delete_team(
        data=DeleteTeamRequest(team_ids=["team-doomed"]),
        http_request=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(
            user_id="admin-user",
            api_key="sk-admin",
            user_role=LitellmUserRoles.PROXY_ADMIN.value,
        ),
        litellm_changed_by="admin-user",
    )

    # array_remove strips just the deleted id in one statement; a read-filter-write of the whole
    # array would drop any team a concurrent /team/member_add appended between read and write
    assert "array_remove" in _STRIP_DELETED_TEAM_FROM_USERS_SQL
    assert mock_execute_raw.await_args_list == [
        call(_STRIP_DELETED_TEAM_FROM_USERS_SQL, "team-doomed"),
        call(_STRIP_DELETED_TEAM_FROM_USERS_SQL, "team-doomed"),
    ], (
        "the unlocked sweep must run once to catch pre-existing drift, and the locked sweep "
        "(alongside the delete, under the same advisory lock member_add takes) must run again "
        "so a member_add that wrote its reference just before losing the lock is still reaped"
    )

    # same two passes for the membership rows, the second under the lock alongside the delete
    assert mock_membership_delete_many.await_args_list == [
        call(where={"team_id": {"in": ("team-doomed",)}}),
        call(where={"team_id": {"in": ("team-doomed",)}}),
    ]

    assert mock_lock_tx.query_raw.await_args_list == [call(TEAM_ADVISORY_LOCK_SQL, "team-doomed")], (
        "the advisory lock must be acquired before the team row is deleted"
    )

    assert fresh_cache.get_cache(key="team_id:team-doomed") is None
    assert fresh_cache.get_cache(key="team_alias:doomed-team") is None
    assert fresh_cache.get_cache(key="team_id:team-kept") is not None
    assert fresh_cache.get_cache(key="team_alias:kept-team") is not None

    # Eviction must run AFTER the rows are gone: both writers of these keys hydrate from the db,
    # so evicting first lets a concurrent auth lookup re-cache the still-present team.
    assert cache_state_when_rows_deleted["doomed_still_cached"] is True


@pytest.mark.asyncio
async def test_delete_team_evicts_the_auth_cache_of_the_keys_it_deletes(
    monkeypatch,
    disable_audit_logging_for_mocked_team,
):
    """
    A virtual key scoped to the team is deleted from the db with the team, but auth resolves a
    cached key object without re-reading the team, so leaving the cache entry behind lets that key
    keep buying access until its TTL expires. Verified live: without this eviction the same key
    still returns HTTP 200 on /v1/chat/completions right after /team/delete.
    """
    from litellm.proxy._types import DeleteTeamRequest, LiteLLM_VerificationToken
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    team = LiteLLM_TeamTable(
        team_id="team-doomed",
        team_alias="doomed-team",
        members_with_roles=[],
        metadata={},
        model_max_budget={},
        model_spend={},
    )
    team_key = LiteLLM_VerificationToken(token="hashed-doomed-key", team_id="team-doomed")

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=team)
    mock_prisma_client.delete_data = AsyncMock(return_value={"deleted_teams": ["team-doomed"]})
    mock_prisma_client.db.litellm_deletedteamtable.create_many = AsyncMock()
    mock_prisma_client.db.litellm_deletedverificationtoken.create_many = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[team_key])
    mock_prisma_client.db.execute_raw = AsyncMock()
    mock_prisma_client.db.litellm_teammembership.delete_many = AsyncMock()

    mock_tx = AsyncMock()
    mock_tx.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
    mock_tx_cm = MagicMock()
    mock_tx_cm.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_tx_cm.__aexit__ = AsyncMock(return_value=False)
    mock_prisma_client.db.tx = MagicMock(return_value=mock_tx_cm)
    _wire_team_delete_tx(mock_prisma_client)

    fresh_cache = UserApiKeyCache()
    fresh_cache.set_cache(key="hashed-doomed-key", value=UserAPIKeyAuth(token="hashed-doomed-key", team_id="team-doomed"))
    fresh_cache.set_cache(key="hashed-unrelated-key", value=UserAPIKeyAuth(token="hashed-unrelated-key"))

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", fresh_cache)
    monkeypatch.setattr("litellm.proxy.proxy_server.create_audit_log_for_update", AsyncMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin")

    await delete_team(
        data=DeleteTeamRequest(team_ids=["team-doomed"]),
        http_request=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(
            user_id="admin-user",
            api_key="sk-admin",
            user_role=LitellmUserRoles.PROXY_ADMIN.value,
        ),
        litellm_changed_by="admin-user",
    )

    assert fresh_cache.get_cache(key="hashed-doomed-key") is None
    # a key that had nothing to do with the deleted team must survive
    assert fresh_cache.get_cache(key="hashed-unrelated-key") is not None


@pytest.mark.asyncio
async def test_delete_team_failing_locked_sweep_rolls_back_the_delete_and_leaves_the_cache_alone(
    monkeypatch,
    disable_audit_logging_for_mocked_team,
):
    """
    The team delete and its post-delete reconcile sweep run inside one transaction, under the
    team's advisory lock, so a sweep failure rolls the delete back with it rather than leaving
    the row gone with the sweep half done. Cache eviction only runs after that transaction
    commits, so a failure here must leave the team exactly as it was: still in the db, and
    still cached. Evicting a cache entry for a delete that never actually committed would be
    the same class of bug this PR exists to fix, just on the other side of the transaction.
    """
    from litellm.proxy._types import DeleteTeamRequest
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    team = LiteLLM_TeamTable(
        team_id="team-doomed",
        team_alias="doomed-team",
        members_with_roles=[],
        metadata={},
        model_max_budget={},
        model_spend={},
    )

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=team)
    mock_prisma_client.delete_data = AsyncMock(return_value={"deleted_teams": ["team-doomed"]})
    mock_prisma_client.db.litellm_deletedteamtable.create_many = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    mock_prisma_client.db.litellm_teammembership.delete_many = AsyncMock()
    # the unlocked pre-delete sweep succeeds, the locked post-delete sweep blows up
    mock_prisma_client.db.execute_raw = AsyncMock(side_effect=[None, ConnectionError("db went away")])

    mock_tx = AsyncMock()
    mock_tx.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
    mock_tx_cm = MagicMock()
    mock_tx_cm.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_tx_cm.__aexit__ = AsyncMock(return_value=False)
    mock_prisma_client.db.tx = MagicMock(return_value=mock_tx_cm)
    _wire_team_delete_tx(mock_prisma_client)

    fresh_cache = UserApiKeyCache()
    cached_obj = LiteLLM_TeamTableCachedObj(team_id="team-doomed", team_alias="doomed-team")
    fresh_cache.set_cache(key="team_id:team-doomed", value=cached_obj)
    fresh_cache.set_cache(key="team_alias:doomed-team", value=cached_obj)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", fresh_cache)
    monkeypatch.setattr("litellm.proxy.proxy_server.create_audit_log_for_update", AsyncMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin")

    with pytest.raises(ConnectionError):
        await delete_team(
            data=DeleteTeamRequest(team_ids=["team-doomed"]),
            http_request=MagicMock(),
            user_api_key_dict=UserAPIKeyAuth(
                user_id="admin-user",
                api_key="sk-admin",
                user_role=LitellmUserRoles.PROXY_ADMIN.value,
            ),
            litellm_changed_by="admin-user",
        )

    # the transaction that deletes the row and runs the locked sweep never committed, so
    # cache eviction (which only runs after that commit) must never have been reached
    assert fresh_cache.get_cache(key="team_id:team-doomed") is not None
    assert fresh_cache.get_cache(key="team_alias:doomed-team") is not None


@pytest.mark.asyncio
async def test_delete_team_broadcasts_cache_invalidation_to_other_workers(
    monkeypatch,
    disable_audit_logging_for_mocked_team,
):
    """
    Evicting locally only reaches the worker that handled the delete. Without the broadcast, every
    other worker keeps serving the deleted team, and the deleted team's keys, out of its own
    in-memory cache until the TTL, so both stay usable for auth cluster-wide.
    """
    from litellm.proxy._types import DeleteTeamRequest, LiteLLM_VerificationToken
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    team = LiteLLM_TeamTable(
        team_id="team-doomed",
        team_alias="doomed-team",
        members_with_roles=[],
        metadata={},
        model_max_budget={},
        model_spend={},
    )

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=team)
    mock_prisma_client.delete_data = AsyncMock(return_value={"deleted_teams": ["team-doomed"]})
    mock_prisma_client.db.litellm_deletedteamtable.create_many = AsyncMock()
    mock_prisma_client.db.litellm_deletedverificationtoken.create_many = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[LiteLLM_VerificationToken(token="hashed-doomed-key", team_id="team-doomed")]
    )
    mock_prisma_client.db.execute_raw = AsyncMock()
    mock_prisma_client.db.litellm_teammembership.delete_many = AsyncMock()

    mock_tx = AsyncMock()
    mock_tx.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
    mock_tx_cm = MagicMock()
    mock_tx_cm.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_tx_cm.__aexit__ = AsyncMock(return_value=False)
    mock_prisma_client.db.tx = MagicMock(return_value=mock_tx_cm)
    _wire_team_delete_tx(mock_prisma_client)

    published = []

    async def record_publish(cache_key):
        published.append(cache_key)

    monkeypatch.setattr("litellm.proxy.auth.auth_checks.publish_auth_cache_invalidation", record_publish)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr("litellm.proxy.proxy_server.create_audit_log_for_update", AsyncMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin")

    await delete_team(
        data=DeleteTeamRequest(team_ids=["team-doomed"]),
        http_request=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(
            user_id="admin-user",
            api_key="sk-admin",
            user_role=LitellmUserRoles.PROXY_ADMIN.value,
        ),
        litellm_changed_by="admin-user",
    )

    # the deleted key first, then both keys `_cache_team_object` writes: miss the alias one and the
    # JWT-by-alias path keeps resolving the team, miss the token and the key still authenticates
    assert published == ["hashed-doomed-key", "team_id:team-doomed", "team_alias:doomed-team"]


@pytest.mark.asyncio
async def test_delete_team_survives_a_failing_cache_backend(
    monkeypatch,
    disable_audit_logging_for_mocked_team,
):
    """
    Cache eviction runs after the reference sweep has already committed, so a cache backend that
    is unreachable must not abort the delete. If it did, `/team/delete` would fail with the team
    row still present but its user references and membership rows already gone.
    """
    from litellm.proxy._types import DeleteTeamRequest, LiteLLM_VerificationToken
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    team = LiteLLM_TeamTable(
        team_id="team-doomed",
        team_alias="doomed-team",
        members_with_roles=[],
        metadata={},
        model_max_budget={},
        model_spend={},
    )

    mock_prisma_client = AsyncMock()
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=team)
    mock_delete_data = AsyncMock(return_value={"deleted_teams": ["team-doomed"]})
    mock_prisma_client.delete_data = mock_delete_data
    mock_prisma_client.db.litellm_deletedteamtable.create_many = AsyncMock()
    mock_prisma_client.db.litellm_deletedverificationtoken.create_many = AsyncMock()
    # a key to evict: its eviction runs after the key rows are already deleted, so it must not
    # raise either
    mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[LiteLLM_VerificationToken(token="hashed-doomed-key", team_id="team-doomed")]
    )
    mock_prisma_client.db.execute_raw = AsyncMock()
    mock_prisma_client.db.litellm_teammembership.delete_many = AsyncMock()

    mock_tx = AsyncMock()
    mock_tx.litellm_proxymodeltable.find_many = AsyncMock(return_value=[])
    mock_tx_cm = MagicMock()
    mock_tx_cm.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_tx_cm.__aexit__ = AsyncMock(return_value=False)
    mock_prisma_client.db.tx = MagicMock(return_value=mock_tx_cm)
    _wire_team_delete_tx(mock_prisma_client)

    exploding_logging_obj = MagicMock()
    exploding_logging_obj.internal_usage_cache.dual_cache.async_delete_cache = AsyncMock(
        side_effect=ConnectionError("redis is down")
    )

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", UserApiKeyCache())
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", exploding_logging_obj)
    monkeypatch.setattr("litellm.proxy.proxy_server.create_audit_log_for_update", AsyncMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin")

    result = await delete_team(
        data=DeleteTeamRequest(team_ids=["team-doomed"]),
        http_request=MagicMock(),
        user_api_key_dict=UserAPIKeyAuth(
            user_id="admin-user",
            api_key="sk-admin",
            user_role=LitellmUserRoles.PROXY_ADMIN.value,
        ),
        litellm_changed_by="admin-user",
    )

    assert result == {"deleted_teams": ["team-doomed"]}
    mock_prisma_client.db.litellm_teamtable.delete_many.assert_any_await(where={"team_id": {"in": ["team-doomed"]}})
    assert exploding_logging_obj.internal_usage_cache.dual_cache.async_delete_cache.await_count > 0


@pytest.mark.asyncio
async def test_team_member_delete_persists_deleted_keys(monkeypatch):
    from litellm.proxy._types import TeamMemberDeleteRequest
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        LiteLLM_VerificationToken,
    )

    mock_prisma_client = AsyncMock()
    mock_user_api_key_dict = UserAPIKeyAuth(
        user_id="admin-user",
        api_key="sk-admin",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
    )

    team = LiteLLM_TeamTable(
        team_id="team-1",
        team_alias="test-team",
        members_with_roles=[
            Member(user_id="user-123", role="admin"),
        ],
        metadata={},
        model_max_budget={},
        model_spend={},
    )

    key1 = LiteLLM_VerificationToken(
        token="hashed-token-1",
        user_id="user-123",
        team_id="team-1",
        key_alias="test-key-1",
        spend=100.0,
        max_budget=1000.0,
        models=["gpt-4"],
        aliases={},
        config={},
        permissions={},
        metadata={},
        model_max_budget={},
    )

    key2 = LiteLLM_VerificationToken(
        token="hashed-token-2",
        user_id="user-123",
        team_id="team-1",
        key_alias="test-key-2",
        spend=50.0,
        max_budget=500.0,
        models=["gpt-3.5-turbo"],
        aliases={},
        config={},
        permissions={},
        metadata={},
        model_max_budget={},
    )

    mock_find_unique_team = AsyncMock(return_value=team)
    mock_prisma_client.db.litellm_teamtable.find_unique = mock_find_unique_team

    mock_find_many_user = AsyncMock(
        return_value=[
            MagicMock(
                user_id="user-123",
                teams=["team-1"],
                model_dump=lambda: {"user_id": "user-123", "teams": ["team-1"]},
            )
        ]
    )
    mock_prisma_client.db.litellm_usertable.find_many = mock_find_many_user

    mock_update_team = AsyncMock()
    mock_prisma_client.db.litellm_teamtable.update = mock_update_team

    mock_update_user = AsyncMock()
    mock_prisma_client.db.litellm_usertable.update = mock_update_user

    mock_delete_membership = AsyncMock()
    mock_prisma_client.db.litellm_teammembership.delete_many = mock_delete_membership

    mock_find_many_keys = AsyncMock(return_value=[key1, key2])
    mock_prisma_client.db.litellm_verificationtoken.find_many = mock_find_many_keys

    mock_delete_keys = AsyncMock()
    mock_prisma_client.db.litellm_verificationtoken.delete_many = mock_delete_keys

    mock_create_many_keys = AsyncMock()
    mock_prisma_client.db.litellm_deletedverificationtoken.create_many = (
        mock_create_many_keys
    )

    _wire_member_delete_tx(mock_prisma_client)

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.prisma_client",
        mock_prisma_client,
    )
    monkeypatch.setattr(
        "litellm.proxy.management_endpoints.team_endpoints._is_user_team_admin",
        lambda **kwargs: True,
    )

    data = TeamMemberDeleteRequest(team_id="team-1", user_id="user-123")

    result = await team_member_delete(
        data=data,
        user_api_key_dict=mock_user_api_key_dict,
    )

    mock_create_many_keys.assert_called_once()
    call_args = mock_create_many_keys.call_args
    assert "data" in call_args.kwargs
    records = call_args.kwargs["data"]
    assert len(records) == 2
    assert all(record["deleted_by"] == "admin-user" for record in records)
    assert all(record["team_id"] == "team-1" for record in records)
    assert all(record["user_id"] == "user-123" for record in records)
    mock_delete_keys.assert_called_once()


@pytest.mark.asyncio
async def test_new_team_negative_max_budget():
    """
    Test that NewTeamRequest model allows negative max_budget values.
    Validation is done at API level, not model level.

    This prevents GET requests from breaking when they receive data with negative budgets.
    """
    from litellm.proxy._types import NewTeamRequest

    # Should not raise any errors at model level
    request = NewTeamRequest(team_alias="test-team", max_budget=-7.0)
    assert request.max_budget == -7.0


@pytest.mark.asyncio
async def test_new_team_negative_team_member_budget():
    """
    Test that NewTeamRequest model allows negative team_member_budget values.
    Validation is done at API level, not model level.
    """
    from litellm.proxy._types import NewTeamRequest

    # Should not raise any errors at model level
    request = NewTeamRequest(team_alias="test-team", team_member_budget=-10.0)
    assert request.team_member_budget == -10.0


@pytest.mark.asyncio
async def test_update_team_negative_max_budget():
    """
    Test that UpdateTeamRequest model allows negative max_budget values.
    Validation is done at API level, not model level.
    """
    from litellm.proxy._types import UpdateTeamRequest

    # Should not raise any errors at model level
    request = UpdateTeamRequest(team_id="test-team-id", max_budget=-5.0)
    assert request.max_budget == -5.0


@pytest.mark.asyncio
async def test_update_team_negative_team_member_budget():
    """
    Test that UpdateTeamRequest model allows negative team_member_budget values.
    Validation is done at API level, not model level.
    """
    from litellm.proxy._types import UpdateTeamRequest

    # Should not raise any errors at model level
    request = UpdateTeamRequest(team_id="test-team-id", team_member_budget=-15.0)
    assert request.team_member_budget == -15.0


# Parametrized tests for soft_budget in create endpoint
@pytest.mark.parametrize(
    "soft_budget,max_budget,should_succeed,expected_soft_budget,expected_max_budget,error_message",
    [
        # Test 1: Soft budget only - success + soft budget set
        (50.0, None, True, 50.0, None, None),
        # Test 2: Soft budget with higher max budget, success with both set
        (50.0, 100.0, True, 50.0, 100.0, None),
        # Test 3: Soft budget with lower max budget, fail
        (
            100.0,
            50.0,
            False,
            None,
            None,
            "soft_budget (100.0) must be strictly lower than max_budget (50.0)",
        ),
        # Test 4: Soft budget equal to max budget, fail
        (
            100.0,
            100.0,
            False,
            None,
            None,
            "soft_budget (100.0) must be strictly lower than max_budget (100.0)",
        ),
    ],
)
@pytest.mark.asyncio
async def test_new_team_soft_budget_validation(
    soft_budget,
    max_budget,
    should_succeed,
    expected_soft_budget,
    expected_max_budget,
    error_message,
):
    """
    Test soft_budget validation in /team/new endpoint.

    Covers:
    - Soft budget only - success + soft budget set
    - Soft budget with higher max budget, success with both set
    - Soft budget with lower max budget, fail
    """
    from fastapi import Request

    from litellm.proxy._types import NewTeamRequest, ProxyException, UserAPIKeyAuth
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Create admin user to bypass user budget checks
    admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin-user",
        models=[],
    )

    # Create team request with soft_budget and optionally max_budget
    team_request = NewTeamRequest(
        team_alias="test-soft-budget-team",
        soft_budget=soft_budget,
        max_budget=max_budget,
    )

    dummy_request = MagicMock(spec=Request)

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server._license_check") as mock_license,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
    ):
        # Setup mocks
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_license.is_team_count_over_limit.return_value = False
        mock_prisma.jsonify_team_object = lambda db_data: db_data
        mock_prisma.get_data = AsyncMock(return_value=None)
        mock_prisma.update_data = AsyncMock()

        # Mock user cache
        from litellm.proxy._types import LiteLLM_UserTable

        mock_user_obj = LiteLLM_UserTable(
            user_id="admin-user",
            max_budget=None,  # Admin has no budget limit
        )
        mock_cache.async_get_cache = AsyncMock(return_value=mock_user_obj)

        # Mock team creation
        mock_created_team = MagicMock()
        mock_created_team.team_id = "test-team-123"
        mock_created_team.team_alias = "test-soft-budget-team"
        mock_created_team.soft_budget = expected_soft_budget
        mock_created_team.max_budget = expected_max_budget
        mock_created_team.members_with_roles = []
        mock_created_team.metadata = None
        mock_created_team.default_team_member_models = None
        mock_created_team.model_dump.return_value = {
            "team_id": "test-team-123",
            "team_alias": "test-soft-budget-team",
            "soft_budget": expected_soft_budget,
            "max_budget": expected_max_budget,
            "members_with_roles": [],
        }
        mock_prisma.db.litellm_teamtable.create = AsyncMock(
            return_value=mock_created_team
        )
        _wire_team_create_tx(mock_prisma)
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_created_team
        )

        # Mock model table
        mock_prisma.db.litellm_modeltable = MagicMock()
        mock_prisma.db.litellm_modeltable.create = AsyncMock(
            return_value=MagicMock(id="model123")
        )

        # Mock user table operations
        mock_user = MagicMock()
        mock_user.user_id = "admin-user"
        mock_user.model_dump.return_value = {
            "user_id": "admin-user",
            "teams": ["test-team-123"],
        }
        mock_prisma.db.litellm_usertable = MagicMock()
        mock_prisma.db.litellm_usertable.upsert = AsyncMock(return_value=mock_user)
        mock_prisma.db.litellm_usertable.update_many = AsyncMock()
        mock_prisma.db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)
        mock_prisma.db.litellm_usertable.update = AsyncMock(return_value=mock_user)

        # Mock team membership table
        mock_membership = MagicMock()
        mock_membership.model_dump.return_value = {
            "team_id": "test-team-123",
            "user_id": "admin-user",
            "budget_id": None,
        }
        mock_prisma.db.litellm_teammembership = MagicMock()
        mock_prisma.db.litellm_teammembership.create = AsyncMock(
            return_value=mock_membership
        )

        if should_succeed:
            # Should NOT raise an exception
            result = await new_team(
                data=team_request,
                http_request=dummy_request,
                user_api_key_dict=admin_user,
            )

            # Verify the team was created successfully with correct values
            assert result is not None
            assert result["team_id"] == "test-team-123"
            if expected_soft_budget is not None:
                assert result["soft_budget"] == expected_soft_budget
            if expected_max_budget is not None:
                assert result["max_budget"] == expected_max_budget
        else:
            # Should raise ProxyException
            with pytest.raises(ProxyException) as exc_info:
                await new_team(
                    data=team_request,
                    http_request=dummy_request,
                    user_api_key_dict=admin_user,
                )

            # Verify exception details
            assert exc_info.value.code == "400"
            if error_message:
                assert error_message in str(exc_info.value.message)


# Parametrized tests for soft_budget in update endpoint
@pytest.mark.parametrize(
    "existing_soft_budget,existing_max_budget,update_soft_budget,update_max_budget,should_succeed,expected_soft_budget,expected_max_budget,error_message",
    [
        # Test 1: Soft budget only (no previous max_budget) - success with soft budget set
        (None, None, 50.0, None, True, 50.0, None, None),
        # Test 2: Soft budget with max budget - success if soft budget is strictly lower than max budget
        (None, None, 50.0, 100.0, True, 50.0, 100.0, None),
        # Test 3: Soft budget with max budget - fail if soft budget >= max budget
        (
            None,
            None,
            100.0,
            50.0,
            False,
            None,
            None,
            "soft_budget (100.0) must be strictly lower than max_budget (50.0)",
        ),
        # Test 4: Only max budget with existing soft_budget, success with max_budget strictly greater
        (50.0, None, None, 100.0, True, 50.0, 100.0, None),
        # Test 5: Only max budget with existing soft_budget, fail if max_budget <= soft_budget
        (
            50.0,
            None,
            None,
            50.0,
            False,
            None,
            None,
            "max_budget (50.0) must be strictly greater than soft_budget (50.0)",
        ),
        # Test 6: Update both soft_budget and max_budget - success if soft < max
        (30.0, 100.0, 40.0, 80.0, True, 40.0, 80.0, None),
        # Test 7: Update both soft_budget and max_budget - fail if soft >= max
        (
            30.0,
            100.0,
            80.0,
            40.0,
            False,
            None,
            None,
            "soft_budget (80.0) must be strictly lower than max_budget (40.0)",
        ),
    ],
)
@pytest.mark.asyncio
async def test_update_team_soft_budget_validation(
    existing_soft_budget,
    existing_max_budget,
    update_soft_budget,
    update_max_budget,
    should_succeed,
    expected_soft_budget,
    expected_max_budget,
    error_message,
    disable_audit_logging_for_mocked_team,
):
    """
    Test soft_budget validation in /team/update endpoint.

    Covers:
    - Soft budget only (no previous max_budget) - success with soft budget set
    - Soft budget with max budget - success if soft budget is strictly lower than max budget, fail otherwise
    - Only max budget with existing soft_budget, success with max_budget strictly greater, fail otherwise
    """
    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_UserTable,
        ProxyException,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Create admin user to bypass user budget checks
    admin_user = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin-user",
        models=[],
    )

    # Create update request
    update_request = UpdateTeamRequest(
        team_id="test-team-123",
        soft_budget=update_soft_budget,
        max_budget=update_max_budget,
    )

    dummy_request = MagicMock(spec=Request)

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server.user_api_key_cache") as mock_cache,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.proxy_server.create_audit_log_for_update", new=AsyncMock()
        ) as mock_audit,
    ):
        # Mock existing team with existing budgets
        mock_existing_team = MagicMock()
        mock_existing_team.team_id = "test-team-123"
        mock_existing_team.organization_id = None
        mock_existing_team.soft_budget = existing_soft_budget
        mock_existing_team.max_budget = existing_max_budget
        mock_existing_team.model_dump.return_value = {
            "team_id": "test-team-123",
            "organization_id": None,
            "soft_budget": existing_soft_budget,
            "max_budget": existing_max_budget,
        }
        mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )

        # Mock user cache
        mock_user_obj = LiteLLM_UserTable(
            user_id="admin-user",
            max_budget=None,  # Admin has no budget limit
        )
        mock_cache.async_get_cache = AsyncMock(return_value=mock_user_obj)

        # Mock updated team - preserve existing values if not being updated
        final_soft_budget = (
            update_soft_budget
            if update_soft_budget is not None
            else existing_soft_budget
        )
        final_max_budget = (
            update_max_budget if update_max_budget is not None else existing_max_budget
        )

        mock_updated_team = MagicMock()
        mock_updated_team.team_id = "test-team-123"
        mock_updated_team.organization_id = None
        mock_updated_team.soft_budget = final_soft_budget
        mock_updated_team.max_budget = final_max_budget
        mock_updated_team.model_dump.return_value = {
            "team_id": "test-team-123",
            "organization_id": None,
            "soft_budget": final_soft_budget,
            "max_budget": final_max_budget,
        }
        mock_prisma.db.litellm_teamtable.update = AsyncMock(
            return_value=mock_updated_team
        )
        mock_prisma.jsonify_team_object = lambda db_data: db_data
        mock_cache.async_set_cache = (
            AsyncMock()
        )  # Mock cache set for _cache_team_object

        if should_succeed:
            # Should NOT raise an exception
            result = await update_team(
                data=update_request,
                http_request=dummy_request,
                user_api_key_dict=admin_user,
            )

            # Verify the team was updated successfully with correct values
            assert result is not None
            assert result["data"].team_id == "test-team-123"
            # Verify soft_budget matches expected value (or final computed value if expected is None)
            if expected_soft_budget is not None:
                assert result["data"].soft_budget == expected_soft_budget
            else:
                assert result["data"].soft_budget == final_soft_budget
            # Verify max_budget matches expected value (or final computed value if expected is None)
            if expected_max_budget is not None:
                assert result["data"].max_budget == expected_max_budget
            else:
                assert result["data"].max_budget == final_max_budget
        else:
            # Should raise ProxyException
            with pytest.raises(ProxyException) as exc_info:
                await update_team(
                    data=update_request,
                    http_request=dummy_request,
                    user_api_key_dict=admin_user,
                )

            # Verify exception details
            assert exc_info.value.code == "400"
            if error_message:
                assert error_message in str(exc_info.value.message)


@pytest.mark.asyncio
async def test_new_team_positive_budgets_accepted():
    """
    Test that NewTeamRequest accepts positive budget values.
    """
    from litellm.proxy._types import NewTeamRequest

    # Should not raise any errors
    request = NewTeamRequest(
        team_alias="test-team", max_budget=100.0, team_member_budget=50.0
    )
    assert request.max_budget == 100.0
    assert request.team_member_budget == 50.0


@pytest.mark.asyncio
async def test_new_team_with_router_settings(mock_db_client, mock_admin_auth):
    """
    Test that /team/new correctly handles router_settings by:
    1. Accepting router_settings as a dict parameter
    2. Serializing router_settings to JSON when saving to database
    3. Storing router_settings in the team record
    """
    # Configure mocked prisma client
    mock_db_client.jsonify_team_object = lambda db_data: db_data
    mock_db_client.get_data = AsyncMock(return_value=None)
    mock_db_client.update_data = AsyncMock(return_value=MagicMock())
    mock_db_client.db = MagicMock()

    # Mock model table creation
    mock_db_client.db.litellm_modeltable = MagicMock()
    mock_db_client.db.litellm_modeltable.create = AsyncMock(
        return_value=MagicMock(id="model123")
    )

    # Capture team table creation
    team_create_result = MagicMock(
        team_id="team-router-456",
    )
    team_create_result.model_dump.return_value = {
        "team_id": "team-router-456",
    }
    mock_team_create = AsyncMock(return_value=team_create_result)
    mock_team_count = AsyncMock(return_value=0)
    mock_db_client.db.litellm_teamtable = MagicMock()
    mock_db_client.db.litellm_teamtable.create = mock_team_create
    _wire_team_create_tx(mock_db_client)
    mock_db_client.db.litellm_teamtable.count = mock_team_count
    mock_db_client.db.litellm_teamtable.update = AsyncMock(
        return_value=team_create_result
    )

    # Mock user table
    mock_db_client.db.litellm_usertable = MagicMock()
    mock_db_client.db.litellm_usertable.update = AsyncMock(return_value=MagicMock())

    from fastapi import Request

    from litellm.proxy._types import NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    # Test router_settings with sample data
    router_settings_data = {
        "routing_strategy": "usage-based",
        "num_retries": 3,
        "retry_policy": {"max_retries": 5},
    }

    # Build request with router_settings
    team_request = NewTeamRequest(
        team_alias="my-team-router",
        router_settings=router_settings_data,
    )

    dummy_request = MagicMock(spec=Request)

    # Execute the endpoint function
    await new_team(
        data=team_request,
        http_request=dummy_request,
        user_api_key_dict=mock_admin_auth,
    )

    # Verify team creation was called
    assert mock_team_create.call_count == 1
    created_team_kwargs = mock_team_create.call_args.kwargs
    team_data = created_team_kwargs["data"]

    # Verify router_settings is serialized to JSON string
    assert "router_settings" in team_data
    assert isinstance(team_data["router_settings"], str)

    # Verify router_settings can be deserialized and matches input
    deserialized_settings = json.loads(team_data["router_settings"])
    assert deserialized_settings == router_settings_data


@pytest.mark.asyncio
async def test_get_team_daily_activity_member_with_permission_sees_all_spend(
    mock_db_client,
):
    """
    Test that non-admin team members with /team/daily/activity permission
    can see all team spend (no API key filtering), same as team admins.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        get_team_daily_activity,
    )

    # Create a non-admin user
    user_id = "test_user_with_perm_123"
    team_id = "test_team_789"
    user_api_key_dict = UserAPIKeyAuth(
        user_id=user_id, user_role=LitellmUserRoles.INTERNAL_USER
    )

    # Mock user info
    mock_user_info = LiteLLM_UserTable(
        user_id=user_id,
        teams=[team_id],
        max_budget=1000.0,
        spend=0.0,
        user_email="member@example.com",
        user_role="internal_user",
    )

    # Mock team with user as non-admin member AND /team/daily/activity permission
    mock_team_member = Member(user_id=user_id, role="user")
    mock_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_team.team_id = team_id
    mock_team.team_alias = "Test Team"
    mock_team.members_with_roles = [mock_team_member]
    mock_team.team_member_permissions = ["/team/daily/activity"]
    mock_team.model_dump.return_value = {
        "team_id": team_id,
        "team_alias": "Test Team",
        "members_with_roles": [{"user_id": user_id, "role": "user"}],
        "team_member_permissions": ["/team/daily/activity"],
    }

    # Setup mocks
    mock_db_client.db.litellm_teamtable.find_many = AsyncMock(return_value=[mock_team])

    # Mock get_user_object
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
        new_callable=AsyncMock,
    ) as mock_get_user_object:
        mock_get_user_object.return_value = mock_user_info

        # Mock get_daily_activity to capture the api_key parameter
        with patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_daily_activity",
            new_callable=AsyncMock,
        ) as mock_get_daily_activity:
            mock_get_daily_activity.return_value = MagicMock()

            # Call the endpoint
            await get_team_daily_activity(
                team_ids=team_id,
                start_date="2024-01-01",
                end_date="2024-01-02",
                model=None,
                api_key=None,
                page=1,
                page_size=10,
                exclude_team_ids=None,
                user_api_key_dict=user_api_key_dict,
            )

            # Verify get_daily_activity was called WITHOUT API key filtering
            mock_get_daily_activity.assert_called_once()
            call_kwargs = mock_get_daily_activity.call_args[1]
            assert call_kwargs["api_key"] is None
            assert call_kwargs["entity_id"] == [team_id]

            # Verify user's API keys were NOT fetched
            if (
                hasattr(mock_db_client.db.litellm_verificationtoken, "find_many")
                and mock_db_client.db.litellm_verificationtoken.find_many.called
            ):
                pytest.fail("API keys should not be fetched for members with /team/daily/activity permission")


@pytest.mark.asyncio
async def test_get_team_daily_activity_member_without_permission_filters_by_keys(
    mock_db_client,
):
    """
    Test that non-admin team members WITHOUT /team/daily/activity permission
    still have their results filtered by their own API keys.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        get_team_daily_activity,
    )

    # Create a non-admin user
    user_id = "test_user_no_perm_123"
    team_id = "test_team_789"
    user_api_key_dict = UserAPIKeyAuth(
        user_id=user_id, user_role=LitellmUserRoles.INTERNAL_USER
    )

    # Mock user info
    mock_user_info = LiteLLM_UserTable(
        user_id=user_id,
        teams=[team_id],
        max_budget=1000.0,
        spend=0.0,
        user_email="member@example.com",
        user_role="internal_user",
    )

    # Mock team with user as non-admin member and NO usage permission
    mock_team_member = Member(user_id=user_id, role="user")
    mock_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_team.team_id = team_id
    mock_team.team_alias = "Test Team"
    mock_team.members_with_roles = [mock_team_member]
    mock_team.team_member_permissions = ["/key/info"]
    mock_team.model_dump.return_value = {
        "team_id": team_id,
        "team_alias": "Test Team",
        "members_with_roles": [{"user_id": user_id, "role": "user"}],
        "team_member_permissions": ["/key/info"],
    }

    # Mock user's API keys
    user_api_key_1 = MagicMock()
    user_api_key_1.token = "user_key_abc"
    user_api_key_2 = MagicMock()
    user_api_key_2.token = "user_key_def"

    # Setup mocks
    mock_db_client.db.litellm_teamtable.find_many = AsyncMock(return_value=[mock_team])
    mock_db_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[user_api_key_1, user_api_key_2]
    )

    # Mock get_user_object
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
        new_callable=AsyncMock,
    ) as mock_get_user_object:
        mock_get_user_object.return_value = mock_user_info

        # Mock get_daily_activity to capture the api_key parameter
        with patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_daily_activity",
            new_callable=AsyncMock,
        ) as mock_get_daily_activity:
            mock_get_daily_activity.return_value = MagicMock()

            # Call the endpoint
            await get_team_daily_activity(
                team_ids=team_id,
                start_date="2024-01-01",
                end_date="2024-01-02",
                model=None,
                api_key=None,
                page=1,
                page_size=10,
                exclude_team_ids=None,
                user_api_key_dict=user_api_key_dict,
            )

            # Verify get_daily_activity was called WITH API key filtering
            mock_get_daily_activity.assert_called_once()
            call_kwargs = mock_get_daily_activity.call_args[1]
            assert call_kwargs["api_key"] == ["user_key_abc", "user_key_def"]
            assert call_kwargs["entity_id"] == [team_id]

            # Verify user's API keys were fetched
            mock_db_client.db.litellm_verificationtoken.find_many.assert_called_once()


@pytest.mark.asyncio
async def test_update_team_with_router_settings(
    mock_db_client,
    mock_admin_auth,
    disable_audit_logging_for_mocked_team,
):
    """
    Test that /team/update correctly handles router_settings by:
    1. Accepting router_settings as a dict parameter
    2. Serializing router_settings to JSON when updating database
    3. Updating router_settings in the team record
    """
    # Configure mocked prisma client
    mock_db_client.jsonify_team_object = lambda db_data: db_data
    mock_db_client.db = MagicMock()

    # Mock existing team row
    existing_team_mock = MagicMock()
    existing_team_mock.team_id = "team-router-update-789"
    existing_team_mock.organization_id = None
    existing_team_mock.models = []
    existing_team_mock.members_with_roles = []
    existing_team_mock.model_dump.return_value = {
        "team_id": "team-router-update-789",
        "organization_id": None,
        "models": [],
        "members_with_roles": [],
    }

    # Mock team table find_unique and update
    updated_team_result = MagicMock(
        team_id="team-router-update-789",
    )
    updated_team_result.model_dump.return_value = {
        "team_id": "team-router-update-789",
    }
    mock_team_find_unique = AsyncMock(return_value=existing_team_mock)
    mock_team_update = AsyncMock(return_value=updated_team_result)
    mock_db_client.db.litellm_teamtable = MagicMock()
    mock_db_client.db.litellm_teamtable.find_unique = mock_team_find_unique
    mock_db_client.db.litellm_teamtable.update = mock_team_update

    from fastapi import Request

    from litellm.proxy._types import UpdateTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    # Test router_settings with updated data
    router_settings_data = {
        "routing_strategy": "latency-based",
        "num_retries": 2,
    }

    # Build update request with router_settings
    team_update_request = UpdateTeamRequest(
        team_id="team-router-update-789",
        router_settings=router_settings_data,
    )

    dummy_request = MagicMock(spec=Request)

    # Execute the endpoint function
    await update_team(
        data=team_update_request,
        http_request=dummy_request,
        user_api_key_dict=mock_admin_auth,
    )

    # Verify team update was called
    assert mock_team_update.call_count == 1
    updated_team_kwargs = mock_team_update.call_args.kwargs
    team_data = updated_team_kwargs["data"]

    # Verify router_settings is serialized to JSON string
    assert "router_settings" in team_data
    assert isinstance(team_data["router_settings"], str)

    # Verify router_settings can be deserialized and matches input
    deserialized_settings = json.loads(team_data["router_settings"])
    assert deserialized_settings == router_settings_data


@pytest.mark.asyncio
async def test_get_team_daily_activity_non_admin_filters_by_user_api_keys(
    mock_db_client,
):
    """
    Test that non-team-admin users only see their own spend (filtered by their API keys)
    when calling /team/daily/activity endpoint.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        get_team_daily_activity,
    )

    # Create a non-admin user
    user_id = "test_user_123"
    team_id = "test_team_456"
    user_api_key_dict = UserAPIKeyAuth(
        user_id=user_id, user_role=LitellmUserRoles.INTERNAL_USER
    )

    # Mock user info
    mock_user_info = LiteLLM_UserTable(
        user_id=user_id,
        teams=[team_id],
        max_budget=1000.0,
        spend=0.0,
        user_email="test@example.com",
        user_role="internal_user",
    )

    # Mock team with user as non-admin member
    mock_team_member = Member(user_id=user_id, role="user")
    mock_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_team.team_id = team_id
    mock_team.team_alias = "Test Team"
    mock_team.members_with_roles = [mock_team_member]
    mock_team.model_dump.return_value = {
        "team_id": team_id,
        "team_alias": "Test Team",
        "members_with_roles": [{"user_id": user_id, "role": "user"}],
    }

    # Mock user's API keys
    user_api_key_1 = MagicMock()
    user_api_key_1.token = "user_key_1"
    user_api_key_2 = MagicMock()
    user_api_key_2.token = "user_key_2"

    # Setup mocks
    mock_db_client.db.litellm_teamtable.find_many = AsyncMock(return_value=[mock_team])
    mock_db_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[user_api_key_1, user_api_key_2]
    )

    # Mock get_user_object
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
        new_callable=AsyncMock,
    ) as mock_get_user_object:
        mock_get_user_object.return_value = mock_user_info

        # Mock get_daily_activity to capture the api_key parameter
        with patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_daily_activity",
            new_callable=AsyncMock,
        ) as mock_get_daily_activity:
            mock_get_daily_activity.return_value = MagicMock()

            # Call the endpoint
            await get_team_daily_activity(
                team_ids=team_id,
                start_date="2024-01-01",
                end_date="2024-01-02",
                model=None,
                api_key=None,
                page=1,
                page_size=10,
                exclude_team_ids=None,
                user_api_key_dict=user_api_key_dict,
            )

            # Verify get_daily_activity was called with user's API keys as filter
            mock_get_daily_activity.assert_called_once()
            call_kwargs = mock_get_daily_activity.call_args[1]
            assert call_kwargs["api_key"] == ["user_key_1", "user_key_2"]
            assert call_kwargs["entity_id"] == [team_id]

            # Verify user's API keys were fetched
            mock_db_client.db.litellm_verificationtoken.find_many.assert_called_once()
            api_key_call_kwargs = (
                mock_db_client.db.litellm_verificationtoken.find_many.call_args[1]
            )
            assert api_key_call_kwargs["where"] == {"user_id": user_id}


@pytest.mark.asyncio
async def test_get_team_daily_activity_team_admin_sees_all_spend(mock_db_client):
    """
    Test that team admin users see all team spend (no API key filtering)
    when calling /team/daily/activity endpoint.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        get_team_daily_activity,
    )

    # Create a team admin user
    user_id = "test_admin_123"
    team_id = "test_team_456"
    user_api_key_dict = UserAPIKeyAuth(
        user_id=user_id, user_role=LitellmUserRoles.INTERNAL_USER
    )

    # Mock user info
    mock_user_info = LiteLLM_UserTable(
        user_id=user_id,
        teams=[team_id],
        max_budget=1000.0,
        spend=0.0,
        user_email="admin@example.com",
        user_role="internal_user",
    )

    # Mock team with user as admin member
    mock_team_member = Member(user_id=user_id, role="admin")
    mock_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_team.team_id = team_id
    mock_team.team_alias = "Test Team"
    mock_team.members_with_roles = [mock_team_member]
    mock_team.model_dump.return_value = {
        "team_id": team_id,
        "team_alias": "Test Team",
        "members_with_roles": [{"user_id": user_id, "role": "admin"}],
    }

    # Setup mocks
    mock_db_client.db.litellm_teamtable.find_many = AsyncMock(return_value=[mock_team])

    # Mock get_user_object
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
        new_callable=AsyncMock,
    ) as mock_get_user_object:
        mock_get_user_object.return_value = mock_user_info

        # Mock get_daily_activity to capture the api_key parameter
        with patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_daily_activity",
            new_callable=AsyncMock,
        ) as mock_get_daily_activity:
            mock_get_daily_activity.return_value = MagicMock()

            # Call the endpoint
            await get_team_daily_activity(
                team_ids=team_id,
                start_date="2024-01-01",
                end_date="2024-01-02",
                model=None,
                api_key=None,
                page=1,
                page_size=10,
                exclude_team_ids=None,
                user_api_key_dict=user_api_key_dict,
            )

            # Verify get_daily_activity was called WITHOUT API key filtering
            mock_get_daily_activity.assert_called_once()
            call_kwargs = mock_get_daily_activity.call_args[1]
            assert call_kwargs["api_key"] is None
            assert call_kwargs["entity_id"] == [team_id]

            # Verify user's API keys were NOT fetched (since they're admin)
            if (
                hasattr(mock_db_client.db.litellm_verificationtoken, "find_many")
                and mock_db_client.db.litellm_verificationtoken.find_many.called
            ):
                # If it was called, that's unexpected for admin users
                pytest.fail("API keys should not be fetched for team admin users")


@pytest.mark.asyncio
async def test_validate_and_populate_member_user_info_both_provided_match():
    """
    Test _validate_and_populate_member_user_info when both user_email and user_id
    are provided and they match the same user in the database.
    """
    # Create member with both user_email and user_id
    member = Member(user_email="test@example.com", user_id="user-123", role="user")

    # Mock prisma client
    mock_prisma_client = MagicMock()

    # Mock user object that matches both email and user_id
    mock_user = MagicMock()
    mock_user.user_id = "user-123"
    mock_user.user_email = "test@example.com"

    # Mock get_data to return single user matching email
    mock_prisma_client.get_data = AsyncMock(return_value=[mock_user])

    # Call the function
    result = await _validate_and_populate_member_user_info(
        member=member,
        prisma_client=mock_prisma_client,
    )

    # Verify result matches input (both already provided and match)
    assert result.user_email == "test@example.com"
    assert result.user_id == "user-123"

    # Verify get_data was called with correct parameters
    mock_prisma_client.get_data.assert_called_once_with(
        key_val={"user_email": "test@example.com"},
        table_name="user",
        query_type="find_all",
    )


@pytest.mark.asyncio
async def test_validate_and_populate_member_user_info_only_email_provided():
    """
    Test _validate_and_populate_member_user_info when only user_email is provided.
    Should populate user_id from database.
    """
    # Create member with only user_email
    member = Member(user_email="test@example.com", user_id=None, role="user")

    # Mock prisma client
    mock_prisma_client = MagicMock()

    # Mock user object from find_first
    mock_user_find_first = MagicMock()
    mock_user_find_first.user_id = "user-456"
    mock_user_find_first.user_email = "test@example.com"

    # Mock find_first to return the user
    mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(
        return_value=mock_user_find_first
    )

    # Mock get_data to return single user (no duplicates)
    mock_prisma_client.get_data = AsyncMock(return_value=[mock_user_find_first])

    # Call the function
    result = await _validate_and_populate_member_user_info(
        member=member,
        prisma_client=mock_prisma_client,
    )

    # Verify user_id was populated
    assert result.user_email == "test@example.com"
    assert result.user_id == "user-456"

    # Verify find_first was called with correct parameters
    mock_prisma_client.db.litellm_usertable.find_first.assert_called_once_with(
        where={"user_email": {"equals": "test@example.com", "mode": "insensitive"}}
    )

    # Verify get_data was called to check for duplicates
    mock_prisma_client.get_data.assert_called_once_with(
        key_val={"user_email": "test@example.com"},
        table_name="user",
        query_type="find_all",
    )


@pytest.mark.asyncio
async def test_validate_and_populate_member_user_info_only_user_id_not_found():
    """
    Test _validate_and_populate_member_user_info when only user_id is provided
    but the user doesn't exist in the database. Should allow it to pass with
    user_email as None (will be upserted later).
    """
    # Create member with only user_id
    member = Member(user_email=None, user_id="nonexistent-user", role="user")

    # Mock prisma client
    mock_prisma_client = MagicMock()

    # Mock find_unique to return None (user not found)
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=None)

    # Call the function - should NOT raise an exception
    result = await _validate_and_populate_member_user_info(
        member=member,
        prisma_client=mock_prisma_client,
    )

    # Verify the result - should return member with user_id set and user_email as None
    assert result.user_id == "nonexistent-user"
    assert result.user_email is None
    assert result.role == "user"

    # Verify find_unique was called with correct parameters
    mock_prisma_client.db.litellm_usertable.find_unique.assert_called_once_with(
        where={"user_id": "nonexistent-user"}
    )


@pytest.mark.asyncio
async def test_list_available_teams_returns_empty_list_when_none_configured():
    """
    Test that /team/available returns an empty list when no available teams
    are configured, instead of raising an exception.
    """
    import litellm

    mock_request = MagicMock()
    mock_user_key = UserAPIKeyAuth(user_id="test-user", token="fake-token")

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
        # Case 1: default_internal_user_params is None
        original = litellm.default_internal_user_params
        litellm.default_internal_user_params = None
        result = await list_available_teams(
            http_request=mock_request,
            user_api_key_dict=mock_user_key,
        )
        assert result == []

        # Case 2: default_internal_user_params exists but has no "available_teams" key
        litellm.default_internal_user_params = {"some_other_param": "value"}
        result = await list_available_teams(
            http_request=mock_request,
            user_api_key_dict=mock_user_key,
        )
        assert result == []

        litellm.default_internal_user_params = original


@pytest.mark.asyncio
async def test_list_team_v1_batches_key_queries():
    """
    Test that list_team fetches all keys in a single batched query
    instead of issuing one query per team (N+1).
    """
    from unittest.mock import AsyncMock, MagicMock, Mock, patch

    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_TeamMembership,
        LiteLLM_TeamTable,
        LitellmUserRoles,
        TeamListResponseObject,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import list_team

    mock_request = Mock(spec=Request)

    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        user_id="admin_user",
    )

    # Two teams
    team1 = LiteLLM_TeamTable(team_id="team-1", team_alias="Team One")
    team2 = LiteLLM_TeamTable(team_id="team-2", team_alias="Team Two")

    # Mock keys belonging to different teams
    key1 = MagicMock()
    key1.team_id = "team-1"
    key2 = MagicMock()
    key2.team_id = "team-1"
    key3 = MagicMock()
    key3.team_id = "team-2"

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client,
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._authorize_and_filter_teams",
            new_callable=AsyncMock,
            return_value=[team1, team2],
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_all_team_memberships",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):

        async def filtered_find_many(**kwargs):
            where = kwargs.get("where", {})
            tid = where.get("team_id")
            if tid == "team-1":
                return [key1, key2]
            elif tid == "team-2":
                return [key3]
            return [key1, key2, key3]

        mock_prisma_client.db.litellm_verificationtoken.find_many = AsyncMock(
            side_effect=filtered_find_many
        )

        result = await list_team(
            http_request=mock_request,
            user_api_key_dict=mock_user_api_key_dict,
        )

        # Verify keys are correctly distributed
        assert len(result) == 2
        # Results are sorted by team_alias
        assert result[0].team_id == "team-1"
        assert result[0].keys == [key1, key2]
        assert result[1].team_id == "team-2"
        assert result[1].keys == [key3]


def test_new_team_request_accepts_team_member_budget_duration():
    """Test that NewTeamRequest does not silently drop team_member_budget_duration."""
    from litellm.proxy._types import NewTeamRequest

    request = NewTeamRequest(
        team_member_budget=20.0,
        team_member_budget_duration="30d",
    )
    assert request.team_member_budget == 20.0
    assert request.team_member_budget_duration == "30d"


@pytest.mark.asyncio
async def test_create_team_member_budget_table_with_duration():
    """Verify that create_team_member_budget_table passes budget_duration
    through to the new_budget call when team_member_budget_duration is provided."""
    from litellm.proxy._types import NewTeamRequest, UserAPIKeyAuth, LitellmUserRoles
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    mock_budget_response = MagicMock(budget_id="budget-abc")
    mock_admin = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)

    data = NewTeamRequest(
        team_alias="test-team",
        team_member_budget=20.0,
        team_member_budget_duration="30d",
    )

    with patch(
        "litellm.proxy.management_endpoints.budget_management_endpoints.new_budget",
        new_callable=AsyncMock,
        return_value=mock_budget_response,
    ) as mock_new_budget:
        result = await TeamMemberBudgetHandler.create_team_member_budget_table(
            data=data,
            new_team_data_json={"metadata": None},
            user_api_key_dict=mock_admin,
            team_member_budget=20.0,
            team_member_budget_duration="30d",
        )

        mock_new_budget.assert_awaited_once()
        budget_request = mock_new_budget.call_args.kwargs["budget_obj"]
        assert budget_request.budget_duration == "30d"
        assert budget_request.max_budget == 20.0
        assert result["metadata"]["team_member_budget_id"] == "budget-abc"


# ---------------------------------------------------------------------------
# Tests for _batch_resolve_access_group_resources
# ---------------------------------------------------------------------------


class TestBatchResolveAccessGroupResources:
    """Tests for the batch access group resource resolution helper."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_ids(self):
        """Empty list should return empty dict."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            _batch_resolve_access_group_resources,
        )

        assert await _batch_resolve_access_group_resources([]) == {}

    @pytest.mark.asyncio
    async def test_single_access_group(self):
        """Single access group should return its resources."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            _batch_resolve_access_group_resources,
        )

        fake_row = MagicMock()
        fake_row.access_group_id = "ag-1"
        fake_row.access_model_names = ["gpt-4", "claude-3"]
        fake_row.access_mcp_server_ids = ["mcp-1"]
        fake_row.access_agent_ids = ["agent-1", "agent-2"]

        fake_prisma = MagicMock()
        fake_prisma.db.litellm_accessgrouptable.find_many = AsyncMock(
            return_value=[fake_row]
        )

        with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma):
            result = await _batch_resolve_access_group_resources(["ag-1"])

        assert sorted(result["ag-1"].access_model_names) == ["claude-3", "gpt-4"]
        assert result["ag-1"].access_mcp_server_ids == ["mcp-1"]
        assert sorted(result["ag-1"].access_agent_ids) == ["agent-1", "agent-2"]

    @pytest.mark.asyncio
    async def test_multiple_access_groups(self):
        """Multiple access groups returned in a single query."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            _batch_resolve_access_group_resources,
        )

        row1 = MagicMock()
        row1.access_group_id = "ag-1"
        row1.access_model_names = ["gpt-4"]
        row1.access_mcp_server_ids = ["mcp-1"]
        row1.access_agent_ids = ["agent-1"]

        row2 = MagicMock()
        row2.access_group_id = "ag-2"
        row2.access_model_names = ["gemini"]
        row2.access_mcp_server_ids = ["mcp-2"]
        row2.access_agent_ids = ["agent-2"]

        fake_prisma = MagicMock()
        fake_prisma.db.litellm_accessgrouptable.find_many = AsyncMock(
            return_value=[row1, row2]
        )

        with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma):
            result = await _batch_resolve_access_group_resources(["ag-1", "ag-2"])

        assert result["ag-1"].access_model_names == ["gpt-4"]
        assert result["ag-2"].access_model_names == ["gemini"]

    @pytest.mark.asyncio
    async def test_missing_access_group_omitted(self):
        """If an access group doesn't exist in DB, it's simply not in the result."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            _batch_resolve_access_group_resources,
        )

        row1 = MagicMock()
        row1.access_group_id = "ag-1"
        row1.access_model_names = ["gpt-4"]
        row1.access_mcp_server_ids = []
        row1.access_agent_ids = []

        fake_prisma = MagicMock()
        fake_prisma.db.litellm_accessgrouptable.find_many = AsyncMock(
            return_value=[row1]
        )

        with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma):
            result = await _batch_resolve_access_group_resources(["ag-1", "ag-missing"])

        assert "ag-1" in result
        assert "ag-missing" not in result

    @pytest.mark.asyncio
    async def test_returns_empty_when_prisma_unavailable(self):
        """If prisma_client is None, should return empty dict."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            _batch_resolve_access_group_resources,
        )

        with patch("litellm.proxy.proxy_server.prisma_client", None):
            result = await _batch_resolve_access_group_resources(["ag-1"])

        assert result == {}

    @pytest.mark.asyncio
    async def test_deduplicates_input_ids(self):
        """Duplicate IDs in input should result in a single DB lookup."""
        from litellm.proxy.management_endpoints.team_endpoints import (
            _batch_resolve_access_group_resources,
        )

        row1 = MagicMock()
        row1.access_group_id = "ag-1"
        row1.access_model_names = ["gpt-4"]
        row1.access_mcp_server_ids = []
        row1.access_agent_ids = []

        fake_find_many = AsyncMock(return_value=[row1])
        fake_prisma = MagicMock()
        fake_prisma.db.litellm_accessgrouptable.find_many = fake_find_many

        with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma):
            result = await _batch_resolve_access_group_resources(
                ["ag-1", "ag-1", "ag-1"]
            )

        # Should have been called with deduplicated list
        call_args = fake_find_many.call_args
        assert len(call_args.kwargs["where"]["access_group_id"]["in"]) == 1
        assert "ag-1" in result


class TestResolveTeamAccessGroupResources:
    """Tests for the per-team access group resolution on /team/info."""

    @pytest.mark.asyncio
    async def test_populates_flat_lists_and_per_group_details(self):
        """access_group_details must attribute each model to the group granting it,
        so the UI can show provenance on hover; flat lists stay for back-compat.
        Duplicated ids must collapse to one entry (response amplification), and the
        input object must stay untouched (resolution returns a copy)."""
        from litellm.proxy._types import TeamInfoResponseObjectTeamTable
        from litellm.proxy.management_endpoints.team_endpoints import (
            _resolve_team_access_group_resources,
        )

        row1 = MagicMock()
        row1.access_group_id = "ag-1"
        row1.access_group_name = "shared-models"
        row1.access_model_names = ["gpt-4", "claude-3"]
        row1.access_mcp_server_ids = ["mcp-1"]
        row1.access_agent_ids = []

        row2 = MagicMock()
        row2.access_group_id = "ag-2"
        row2.access_group_name = "extra-models"
        row2.access_model_names = ["claude-3", "gemini"]
        row2.access_mcp_server_ids = []
        row2.access_agent_ids = ["agent-1"]

        fake_prisma = MagicMock()
        fake_prisma.db.litellm_accessgrouptable.find_many = AsyncMock(
            return_value=[row1, row2]
        )

        team_info = TeamInfoResponseObjectTeamTable(
            team_id="team-1", access_group_ids=["ag-1", "ag-2", "ag-1", "ag-missing"]
        )
        with patch("litellm.proxy.proxy_server.prisma_client", fake_prisma):
            resolved = await _resolve_team_access_group_resources(team_info)

        assert team_info.access_group_details is None
        assert sorted(resolved.access_group_models or []) == [
            "claude-3",
            "gemini",
            "gpt-4",
        ]
        assert resolved.access_group_mcp_server_ids == ["mcp-1"]
        assert resolved.access_group_agent_ids == ["agent-1"]
        assert [
            (d.access_group_id, d.access_group_name, d.models)
            for d in (resolved.access_group_details or [])
        ] == [
            ("ag-1", "shared-models", ("gpt-4", "claude-3")),
            ("ag-2", "extra-models", ("claude-3", "gemini")),
        ]

    @pytest.mark.asyncio
    async def test_no_access_groups_leaves_details_unset(self):
        from litellm.proxy._types import TeamInfoResponseObjectTeamTable
        from litellm.proxy.management_endpoints.team_endpoints import (
            _resolve_team_access_group_resources,
        )

        team_info = TeamInfoResponseObjectTeamTable(team_id="team-1", access_group_ids=[])
        resolved = await _resolve_team_access_group_resources(team_info)

        assert resolved.access_group_details is None
        assert resolved.access_group_models is None


@pytest.mark.asyncio
async def test_verify_team_access_denies_unauthorized_user():
    """
    Test that _verify_team_access raises 403 when the caller is not a proxy admin,
    not a team admin, and not an org admin for the team's organization.
    """
    team_obj = LiteLLM_TeamTable(
        team_id="team-123",
        team_alias="test-team",
        members_with_roles=[
            Member(role="admin", user_id="other_admin_user"),
        ],
        organization_id="org-456",
    )

    # Caller is an internal user with no admin role and not in the team
    caller = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="unauthorized_user",
    )

    with patch(
        "litellm.proxy.management_endpoints.team_endpoints._is_user_org_admin_for_team",
        new_callable=AsyncMock,
        return_value=False,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await _verify_team_access(
                team_obj=team_obj,
                user_api_key_dict=caller,
            )
        assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_update_team_rejects_unauthorized_caller():
    """
    Test that /team/update returns 403 when the caller is not a proxy admin,
    not a team admin, and not an org admin — exercising the _verify_team_access
    guard added to the update_team endpoint.
    """
    from unittest.mock import Mock

    from fastapi import Request

    mock_request = Mock(spec=Request)
    caller = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="unauthorized_user",
    )

    from litellm.proxy._types import UpdateTeamRequest

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma_client,
        patch("litellm.proxy.proxy_server.llm_router"),
        patch("litellm.proxy.proxy_server.user_api_key_cache"),
        patch("litellm.proxy.proxy_server.proxy_logging_obj"),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._is_user_org_admin_for_team",
            new_callable=AsyncMock,
            return_value=False,
        ),
    ):
        mock_existing_team = MagicMock()
        mock_existing_team.model_dump.return_value = {
            "team_id": "team-123",
            "team_alias": "test-team",
            "members_with_roles": [
                {"role": "admin", "user_id": "other_admin_user"},
            ],
            "organization_id": "org-456",
        }
        mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=mock_existing_team
        )

        update_request = UpdateTeamRequest(
            team_id="team-123",
            team_alias="updated-alias",
        )

        with pytest.raises(ProxyException) as exc_info:
            await update_team(
                data=update_request,
                http_request=mock_request,
                user_api_key_dict=caller,
            )
        assert exc_info.value.code == "403"


# ----- /team/{team_id}/members/me -----


def _build_team_for_me(team_id, members):
    """Real LiteLLM_TeamTableCachedObj as get_team_object would return."""
    return LiteLLM_TeamTableCachedObj(
        team_id=team_id,
        team_alias="team-vec",
        members_with_roles=[Member(**m) for m in members],
        metadata={},
        models=[],
        spend=0.0,
    )


def _build_membership_for_me(user_id, team_id, *, spend=12.34, max_budget=100.0):
    """Real LiteLLM_TeamMembership as get_team_membership would return."""
    return LiteLLM_TeamMembership(
        user_id=user_id,
        team_id=team_id,
        budget_id="b-1",
        spend=spend,
        total_spend=spend,
        litellm_budget_table=LiteLLM_BudgetTableFull(
            budget_id="b-1",
            max_budget=max_budget,
            soft_budget=None,
            tpm_limit=500,
            rpm_limit=50,
            model_max_budget=None,
            budget_duration="30d",
            budget_reset_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            allowed_models=None,
            created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        ),
    )


def _patch_member_me_helpers(*, team, membership=None, user=None):
    """Patch the three auth helpers used by team_member_me with AsyncMocks."""
    return (
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
            AsyncMock(return_value=team),
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_team_membership",
            AsyncMock(return_value=membership),
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
            AsyncMock(return_value=user),
        ),
    )


@pytest.mark.asyncio
async def test_team_member_me_returns_caller_membership(mock_db_client):
    """A team member receives their own membership row, not other members'."""
    from fastapi import Request

    from litellm.proxy.management_endpoints.team_endpoints import team_member_me

    team_id = "team-me-1"
    caller_id = "alice@example.com"
    other_id = "bob@example.com"
    caller_auth = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id=caller_id
    )

    team = _build_team_for_me(
        team_id,
        [
            {"user_id": caller_id, "user_email": None, "role": "user"},
            {"user_id": other_id, "user_email": None, "role": "admin"},
        ],
    )
    membership = _build_membership_for_me(caller_id, team_id, spend=42.0)
    user = LiteLLM_UserTable(user_id=caller_id, user_email=caller_id, max_budget=None)

    p_team, p_membership, p_user = _patch_member_me_helpers(
        team=team, membership=membership, user=user
    )
    with p_team, p_membership as mock_get_membership, p_user:
        response = await team_member_me(
            http_request=MagicMock(spec=Request),
            team_id=team_id,
            user_api_key_dict=caller_auth,
        )

    assert response.user_id == caller_id
    assert response.team_id == team_id
    assert response.role == "user"
    assert response.spend == 42.0
    assert response.team_alias == "team-vec"
    assert response.litellm_budget_table is not None
    assert response.litellm_budget_table.max_budget == 100.0
    # budget_reset_at must survive end-to-end — proves the BudgetTableFull
    # variant of the Union is selected (created_at is present), not the base
    # LiteLLM_BudgetTable which would silently strip this field.
    assert response.litellm_budget_table.budget_reset_at == datetime(
        2026, 5, 1, tzinfo=timezone.utc
    )

    # Membership lookup must scope to caller_id, not just team_id — proves the
    # endpoint cannot return another member's row.
    call_kwargs = mock_get_membership.call_args.kwargs
    assert call_kwargs["user_id"] == caller_id
    assert call_kwargs["team_id"] == team_id


@pytest.mark.asyncio
async def test_team_member_me_matches_email_only_member(mock_db_client):
    """
    Members onboarded by email may have user_id=None on the stored entry —
    the lookup must fall back to email matching against the caller, otherwise
    a valid team member gets a false 404.
    """
    from fastapi import Request

    from litellm.proxy.management_endpoints.team_endpoints import team_member_me

    team_id = "team-me-email"
    caller_id = "u-123"
    caller_email = "alice@example.com"
    caller_auth = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id=caller_id,
        user_email=caller_email,
    )

    # Member entry with user_id=None and email matching the caller's email.
    team = _build_team_for_me(
        team_id,
        [{"user_id": None, "user_email": caller_email, "role": "user"}],
    )
    membership = _build_membership_for_me(caller_id, team_id, spend=7.0)
    user = LiteLLM_UserTable(
        user_id=caller_id, user_email=caller_email, max_budget=None
    )

    p_team, p_membership, p_user = _patch_member_me_helpers(
        team=team, membership=membership, user=user
    )
    with p_team, p_membership, p_user:
        response = await team_member_me(
            http_request=MagicMock(spec=Request),
            team_id=team_id,
            user_api_key_dict=caller_auth,
        )

    assert response.user_id == caller_id
    assert response.role == "user"
    assert response.spend == 7.0


@pytest.mark.asyncio
async def test_team_member_me_returns_404_for_non_member(mock_db_client):
    """A user who is not a member of the team gets 404, regardless of role."""
    from fastapi import Request, HTTPException

    from litellm.proxy.management_endpoints.team_endpoints import team_member_me

    team_id = "team-me-2"
    caller_id = "outsider@example.com"
    caller_auth = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id=caller_id
    )

    team = _build_team_for_me(
        team_id,
        [{"user_id": "someone_else", "user_email": None, "role": "user"}],
    )

    p_team, p_membership, p_user = _patch_member_me_helpers(team=team)
    with p_team, p_membership, p_user:
        with pytest.raises(HTTPException) as exc_info:
            await team_member_me(
                http_request=MagicMock(spec=Request),
                team_id=team_id,
                user_api_key_dict=caller_auth,
            )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_team_member_me_returns_404_for_proxy_admin_not_in_team(
    mock_db_client, mock_admin_auth
):
    """
    Proxy admins get 404 if they are not actually a member of the team.
    `me` only resolves for actual team members; admins use /team/info instead.
    """
    from fastapi import Request, HTTPException

    from litellm.proxy.management_endpoints.team_endpoints import team_member_me

    team_id = "team-me-3"
    mock_admin_auth.user_id = "admin_user_999"

    team = _build_team_for_me(
        team_id,
        [{"user_id": "someone_else", "user_email": None, "role": "user"}],
    )

    p_team, p_membership, p_user = _patch_member_me_helpers(team=team)
    with p_team, p_membership, p_user:
        with pytest.raises(HTTPException) as exc_info:
            await team_member_me(
                http_request=MagicMock(spec=Request),
                team_id=team_id,
                user_api_key_dict=mock_admin_auth,
            )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_team_member_me_returns_defaults_when_no_membership_row(mock_db_client):
    """
    Caller is in members_with_roles but has no LiteLLM_TeamMembership row yet
    (no per-member budget configured) — return defaults rather than 404.
    """
    from fastapi import Request

    from litellm.proxy.management_endpoints.team_endpoints import team_member_me

    team_id = "team-me-4"
    caller_id = "newmember@example.com"
    caller_auth = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id=caller_id
    )

    team = _build_team_for_me(
        team_id,
        [{"user_id": caller_id, "user_email": None, "role": "user"}],
    )

    p_team, p_membership, p_user = _patch_member_me_helpers(team=team)
    with p_team, p_membership, p_user:
        response = await team_member_me(
            http_request=MagicMock(spec=Request),
            team_id=team_id,
            user_api_key_dict=caller_auth,
        )

    assert response.user_id == caller_id
    assert response.role == "user"
    assert response.spend == 0.0
    assert response.litellm_budget_table is None


@pytest.mark.asyncio
async def test_team_member_me_rejects_team_key_without_user_id(mock_db_client):
    """A team key with no user_id can't resolve 'me' — must return 400."""
    from fastapi import Request, HTTPException

    from litellm.proxy.management_endpoints.team_endpoints import team_member_me

    team_key_auth = UserAPIKeyAuth(team_id="team-me-5", user_id=None)

    with pytest.raises(HTTPException) as exc_info:
        await team_member_me(
            http_request=MagicMock(spec=Request),
            team_id="team-me-5",
            user_api_key_dict=team_key_auth,
        )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_team_member_me_returns_404_for_unknown_team(mock_db_client):
    """Unknown team_id returns 404 — propagated from get_team_object."""
    from fastapi import Request, HTTPException

    from litellm.proxy.management_endpoints.team_endpoints import team_member_me

    caller_auth = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER, user_id="alice@example.com"
    )

    # get_team_object raises 404 directly when the team is missing.
    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        AsyncMock(
            side_effect=HTTPException(
                status_code=404, detail={"error": "Team doesn't exist in db."}
            )
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await team_member_me(
                http_request=MagicMock(spec=Request),
                team_id="does-not-exist",
                user_api_key_dict=caller_auth,
            )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_new_team_encrypts_callback_vars(
    mock_db_client, mock_admin_auth, monkeypatch
):
    """/team/new must encrypt callback_vars values before they reach the DB."""
    from fastapi import Request

    from litellm.proxy._types import NewTeamRequest
    from litellm.proxy.common_utils.callback_utils import decrypt_callback_vars
    from litellm.proxy.management_endpoints.team_endpoints import new_team
    from litellm.proxy.utils import PrismaClient

    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-32-bytes-aaaaaaaaaaaaaa")

    # Use the real jsonify helpers so the encrypted dict goes through the
    # actual JSON serialization production uses (catches non-serializable
    # ciphertext, missing fields, etc.).
    mock_db_client.jsonify_object = PrismaClient.jsonify_object.__get__(mock_db_client)
    mock_db_client.jsonify_team_object = PrismaClient.jsonify_team_object.__get__(
        mock_db_client
    )
    mock_db_client.get_data = AsyncMock(return_value=None)
    mock_db_client.db = MagicMock()
    mock_db_client.db.litellm_teamtable = MagicMock()
    team_create_result = MagicMock(team_id="team-456", object_permission_id=None)
    team_create_result.model_dump.return_value = {"team_id": "team-456"}
    mock_team_create = AsyncMock(return_value=team_create_result)
    mock_db_client.db.litellm_teamtable.create = mock_team_create
    _wire_team_create_tx(mock_db_client)
    mock_db_client.db.litellm_teamtable.count = AsyncMock(return_value=0)
    mock_db_client.db.litellm_teamtable.update = AsyncMock(
        return_value=team_create_result
    )
    mock_db_client.db.litellm_usertable = MagicMock()
    mock_db_client.db.litellm_usertable.update = AsyncMock(return_value=MagicMock())

    team_request = NewTeamRequest(
        team_alias="my-team",
        metadata={
            "logging": [
                {
                    "callback_name": "langfuse",
                    "callback_type": "success",
                    "callback_vars": {
                        "langfuse_public_key": "pk-real",
                        "langfuse_secret_key": "sk-real",
                    },
                }
            ]
        },
    )

    await new_team(
        data=team_request,
        http_request=MagicMock(spec=Request),
        user_api_key_dict=mock_admin_auth,
    )

    written = mock_team_create.call_args.kwargs["data"]
    # jsonify_team_object serializes the metadata dict to a JSON string before
    # the DB write, so we round-trip through json.loads to inspect it.
    metadata = json.loads(written["metadata"])
    cv = metadata["logging"][0]["callback_vars"]
    assert cv["langfuse_secret_key"] != "sk-real"
    recovered = decrypt_callback_vars(metadata)["logging"][0]["callback_vars"]
    assert recovered["langfuse_secret_key"] == "sk-real"


def _non_admin_auth():
    return UserAPIKeyAuth(
        user_id="u-team-admin", user_role=LitellmUserRoles.INTERNAL_USER
    )


def test_check_passthrough_routes_caller_permission_team():
    from litellm.proxy._types import NewTeamRequest
    from litellm.proxy.management_endpoints.common_utils import (
        _check_passthrough_routes_caller_permission,
    )

    admin = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    non_admin = _non_admin_auth()

    _check_passthrough_routes_caller_permission(
        NewTeamRequest(allowed_passthrough_routes=["/foo/*"]), admin, entity="team"
    )

    _check_passthrough_routes_caller_permission(
        NewTeamRequest(), non_admin, entity="team"
    )
    _check_passthrough_routes_caller_permission(
        NewTeamRequest(allowed_passthrough_routes=[]), non_admin, entity="team"
    )

    with pytest.raises(HTTPException) as exc:
        _check_passthrough_routes_caller_permission(
            NewTeamRequest(allowed_passthrough_routes=["/admin/*"]),
            non_admin,
            entity="team",
        )
    assert exc.value.status_code == 403
    assert "allowed_passthrough_routes" in str(exc.value.detail)
    assert "team" in str(exc.value.detail)

    with pytest.raises(HTTPException) as exc:
        _check_passthrough_routes_caller_permission(
            NewTeamRequest(metadata={"allowed_passthrough_routes": ["/admin/*"]}),
            non_admin,
            entity="team",
        )
    assert exc.value.status_code == 403
    assert "metadata.allowed_passthrough_routes" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_new_team_blocks_non_admin_passthrough_routes(mock_db_client):
    """A non-proxy-admin cannot self-grant pass-through routes via /team/new."""
    mock_db_client.db.litellm_teamtable.count = AsyncMock(return_value=0)
    from fastapi import Request

    from litellm.proxy._types import NewTeamRequest, ProxyException
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    with patch(
        "litellm.proxy.management_endpoints.team_endpoints._check_user_team_limits",
        AsyncMock(return_value=None),
    ):
        with pytest.raises(ProxyException) as exc:
            await new_team(
                data=NewTeamRequest(
                    team_alias="t", allowed_passthrough_routes=["/admin/*"]
                ),
                http_request=MagicMock(spec=Request),
                user_api_key_dict=_non_admin_auth(),
            )
    assert str(exc.value.code) == "403"
    assert "allowed_passthrough_routes" in str(exc.value.message)


@pytest.mark.asyncio
async def test_update_team_blocks_non_admin_passthrough_routes(mock_db_client):
    """Even a team manager (non-proxy-admin) cannot set pass-through routes via
    /team/update — the gate runs after _verify_team_access."""
    from fastapi import Request

    from litellm.proxy._types import ProxyException, UpdateTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    existing = MagicMock()
    existing.model_dump.return_value = {"team_id": "t1"}
    mock_db_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=existing)

    with patch(
        "litellm.proxy.management_endpoints.team_endpoints._verify_team_access",
        AsyncMock(return_value=None),
    ):
        with pytest.raises(ProxyException) as exc:
            await update_team(
                data=UpdateTeamRequest(
                    team_id="t1", allowed_passthrough_routes=["/admin/*"]
                ),
                http_request=MagicMock(spec=Request),
                user_api_key_dict=_non_admin_auth(),
            )
    assert str(exc.value.code) == "403"
    assert "allowed_passthrough_routes" in str(exc.value.message)


def test_set_budget_reset_at_clears_when_budget_duration_null():
    """
    When budget_duration is explicitly set to null, _set_budget_reset_at
    should set budget_reset_at=None in updated_kv so Prisma clears it in the DB.
    """
    from litellm.proxy._types import UpdateTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import _set_budget_reset_at

    data = UpdateTeamRequest(team_id="test-team", budget_duration=None)
    updated_kv = {"team_id": "test-team", "budget_duration": None}

    _set_budget_reset_at(data, updated_kv)

    assert "budget_reset_at" in updated_kv
    assert updated_kv["budget_reset_at"] is None


def test_set_budget_reset_at_noop_when_budget_duration_not_sent():
    """
    When budget_duration is NOT sent (unset), _set_budget_reset_at should
    not add budget_reset_at to updated_kv.
    """
    from litellm.proxy._types import UpdateTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import _set_budget_reset_at

    data = UpdateTeamRequest(team_id="test-team")
    updated_kv = {"team_id": "test-team"}

    _set_budget_reset_at(data, updated_kv)

    assert "budget_reset_at" not in updated_kv


def test_set_budget_reset_at_sets_value_when_budget_duration_provided():
    """
    When budget_duration is set to a valid string, _set_budget_reset_at
    should compute and set budget_reset_at.
    """
    from litellm.proxy._types import UpdateTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import _set_budget_reset_at

    data = UpdateTeamRequest(team_id="test-team", budget_duration="30d")
    updated_kv = {"team_id": "test-team", "budget_duration": "30d"}

    _set_budget_reset_at(data, updated_kv)

    assert "budget_reset_at" in updated_kv
    assert updated_kv["budget_reset_at"] is not None


@pytest.mark.asyncio
async def test_clear_team_member_budget_duration_calls_update_budget():
    """
    When team_member_budget_duration is explicitly null and a budget row
    exists, clear_team_member_budget_fields should call update_budget
    with budget_duration=None and budget_reset_at=None.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key="sk-1234",
        user_id="admin-user",
    )

    team_table = LiteLLM_TeamTable(
        team_id="test-team",
        metadata={"team_member_budget_id": "budget-123"},
        members_with_roles=[],
    )

    updated_kv = {
        "team_id": "test-team",
        "team_member_budget_duration": None,
    }

    with patch(
        "litellm.proxy.management_endpoints.budget_management_endpoints.update_budget",
        new_callable=AsyncMock,
    ) as mock_update_budget:
        result = await TeamMemberBudgetHandler.clear_team_member_budget_fields(
            team_table=team_table,
            user_api_key_dict=mock_user_api_key_dict,
            updated_kv=updated_kv,
            explicitly_set_fields={"team_member_budget_duration"},
        )

    mock_update_budget.assert_awaited_once()
    budget_request = mock_update_budget.call_args.kwargs["budget_obj"]
    assert budget_request.budget_id == "budget-123"
    assert "budget_duration" in budget_request.model_fields_set
    assert budget_request.budget_duration is None
    assert "budget_reset_at" in budget_request.model_fields_set
    assert budget_request.budget_reset_at is None
    assert "team_member_budget_duration" not in result


@pytest.mark.asyncio
async def test_clear_team_member_budget_clears_max_budget():
    """
    When team_member_budget is explicitly null, clear_team_member_budget_fields
    should call update_budget with max_budget=None.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key="sk-1234",
        user_id="admin-user",
    )

    team_table = LiteLLM_TeamTable(
        team_id="test-team",
        metadata={"team_member_budget_id": "budget-456"},
        members_with_roles=[],
    )

    updated_kv = {
        "team_id": "test-team",
        "team_member_budget": None,
    }

    with patch(
        "litellm.proxy.management_endpoints.budget_management_endpoints.update_budget",
        new_callable=AsyncMock,
    ) as mock_update_budget:
        result = await TeamMemberBudgetHandler.clear_team_member_budget_fields(
            team_table=team_table,
            user_api_key_dict=mock_user_api_key_dict,
            updated_kv=updated_kv,
            explicitly_set_fields={"team_member_budget"},
        )

    mock_update_budget.assert_awaited_once()
    budget_request = mock_update_budget.call_args.kwargs["budget_obj"]
    assert budget_request.budget_id == "budget-456"
    assert "max_budget" in budget_request.model_fields_set
    assert budget_request.max_budget is None
    assert "team_member_budget" not in result


@pytest.mark.asyncio
async def test_clear_team_member_rpm_tpm_limits():
    """
    When team_member_rpm_limit and team_member_tpm_limit are explicitly null,
    clear_team_member_budget_fields should clear both on the budget row.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key="sk-1234",
        user_id="admin-user",
    )

    team_table = LiteLLM_TeamTable(
        team_id="test-team",
        metadata={"team_member_budget_id": "budget-789"},
        members_with_roles=[],
    )

    updated_kv = {
        "team_id": "test-team",
        "team_member_rpm_limit": None,
        "team_member_tpm_limit": None,
    }

    with patch(
        "litellm.proxy.management_endpoints.budget_management_endpoints.update_budget",
        new_callable=AsyncMock,
    ) as mock_update_budget:
        result = await TeamMemberBudgetHandler.clear_team_member_budget_fields(
            team_table=team_table,
            user_api_key_dict=mock_user_api_key_dict,
            updated_kv=updated_kv,
            explicitly_set_fields={"team_member_rpm_limit", "team_member_tpm_limit"},
        )

    mock_update_budget.assert_awaited_once()
    budget_request = mock_update_budget.call_args.kwargs["budget_obj"]
    assert budget_request.budget_id == "budget-789"
    assert "rpm_limit" in budget_request.model_fields_set
    assert budget_request.rpm_limit is None
    assert "tpm_limit" in budget_request.model_fields_set
    assert budget_request.tpm_limit is None
    assert "team_member_rpm_limit" not in result
    assert "team_member_tpm_limit" not in result


@pytest.mark.asyncio
async def test_clear_all_team_member_fields_at_once():
    """
    When all team_member fields are explicitly null, all corresponding
    budget row fields should be cleared in a single update.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key="sk-1234",
        user_id="admin-user",
    )

    team_table = LiteLLM_TeamTable(
        team_id="test-team",
        metadata={"team_member_budget_id": "budget-all"},
        members_with_roles=[],
    )

    updated_kv = {
        "team_id": "test-team",
        "team_member_budget": None,
        "team_member_budget_duration": None,
        "team_member_rpm_limit": None,
        "team_member_tpm_limit": None,
    }

    all_fields = {
        "team_member_budget",
        "team_member_budget_duration",
        "team_member_rpm_limit",
        "team_member_tpm_limit",
    }

    with patch(
        "litellm.proxy.management_endpoints.budget_management_endpoints.update_budget",
        new_callable=AsyncMock,
    ) as mock_update_budget:
        result = await TeamMemberBudgetHandler.clear_team_member_budget_fields(
            team_table=team_table,
            user_api_key_dict=mock_user_api_key_dict,
            updated_kv=updated_kv,
            explicitly_set_fields=all_fields,
        )

    mock_update_budget.assert_awaited_once()
    budget_request = mock_update_budget.call_args.kwargs["budget_obj"]
    assert budget_request.budget_id == "budget-all"
    assert budget_request.max_budget is None
    assert budget_request.budget_duration is None
    assert budget_request.budget_reset_at is None
    assert budget_request.rpm_limit is None
    assert budget_request.tpm_limit is None
    for field in all_fields:
        assert field not in result


@pytest.mark.asyncio
async def test_team_member_budget_duration_not_sent_does_not_update():
    """
    When team_member_budget_duration is NOT sent in the request, no budget
    update should occur and the field should not appear in updated_kv.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    updated_kv = {"team_id": "test-team", "max_budget": 200}

    _team_member_fields_in_request = {
        field
        for field in [
            "team_member_budget",
            "team_member_rpm_limit",
            "team_member_tpm_limit",
            "team_member_budget_duration",
        ]
        if field in updated_kv
    }

    assert len(_team_member_fields_in_request) == 0

    TeamMemberBudgetHandler._clean_team_member_fields(updated_kv)

    assert "team_member_budget_duration" not in updated_kv
    assert "team_member_budget" not in updated_kv


@pytest.mark.asyncio
async def test_clear_team_member_budget_fields_no_budget_row_skips_update():
    from litellm.proxy.management_endpoints.team_endpoints import (
        TeamMemberBudgetHandler,
    )

    mock_user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key="sk-1234",
        user_id="admin-user",
    )

    team_table = LiteLLM_TeamTable(
        team_id="test-team",
        metadata=None,
        members_with_roles=[],
    )

    updated_kv = {
        "team_id": "test-team",
        "team_member_budget": None,
        "team_member_rpm_limit": None,
    }

    with patch(
        "litellm.proxy.management_endpoints.budget_management_endpoints.update_budget",
        new_callable=AsyncMock,
    ) as mock_update_budget:
        result = await TeamMemberBudgetHandler.clear_team_member_budget_fields(
            team_table=team_table,
            user_api_key_dict=mock_user_api_key_dict,
            updated_kv=updated_kv,
            explicitly_set_fields={"team_member_budget", "team_member_rpm_limit"},
        )

    mock_update_budget.assert_not_awaited()
    assert "team_member_budget" not in result
    assert "team_member_rpm_limit" not in result


@pytest.mark.asyncio
async def test_team_info_forwards_key_limit_to_get_data():
    """/team/info must thread its ``key_limit`` query param into the key
    lookup so the database caps how many keys are returned for the team.
    """
    from fastapi import Request

    from litellm.proxy.management_endpoints import team_endpoints

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=LiteLLM_TeamTable(team_id="team-1")
    )
    mock_prisma.get_data = AsyncMock(return_value=[])

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch.object(
            team_endpoints, "get_all_team_memberships", AsyncMock(return_value=[])
        ),
    ):
        await team_endpoints.team_info(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            key_limit=7,
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    assert mock_prisma.get_data.await_args.kwargs["limit"] == 7


@pytest.mark.asyncio
async def test_team_info_returns_model_aliases():
    """/team/info must join LiteLLM_ModelTable so the response exposes the team's
    current model aliases; without the ``litellm_model_table`` include the field
    comes back null and the Admin UI can never display them.
    """
    from fastapi import Request

    from litellm.proxy.management_endpoints import team_endpoints

    team_row = LiteLLM_TeamTable(
        team_id="team-1",
        litellm_model_table=LiteLLM_ModelTable(
            id=1,
            model_aliases={"gpt-4o": "gpt-4o-team-1"},
            created_by="admin",
            updated_by="admin",
        ),
    )

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)
    mock_prisma.get_data = AsyncMock(return_value=[])

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch.object(
            team_endpoints, "get_all_team_memberships", AsyncMock(return_value=[])
        ),
    ):
        response = await team_endpoints.team_info(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    include = mock_prisma.db.litellm_teamtable.find_unique.await_args.kwargs["include"]
    assert include["litellm_model_table"] is True

    litellm_model_table = response["team_info"].litellm_model_table
    assert litellm_model_table is not None
    assert litellm_model_table.model_aliases == {"gpt-4o": "gpt-4o-team-1"}


@pytest.mark.asyncio
async def test_team_info_hydrates_member_emails_from_the_user_table():
    """/team/info must fill in emails missing from the members_with_roles snapshot.

    members_with_roles is written at add-time, so a member added by user_id alone
    carries user_email=None forever. Without this join the Admin UI's member table
    shows "-" for a user that has an email on their user row. A stored email is left
    exactly as-is.
    """
    from fastapi import Request

    from litellm.proxy.management_endpoints import team_endpoints

    team_row = LiteLLM_TeamTable(
        team_id="team-1",
        members_with_roles=[
            Member(user_id="no-email-on-roster", role="admin"),
            Member(user_id="already-stored", user_email="stored@example.com", role="user"),
        ],
    )

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)
    mock_prisma.get_data = AsyncMock(return_value=[])

    find_many = AsyncMock(
        return_value=[
            LiteLLM_UserTable(
                user_id="no-email-on-roster",
                user_email="real@example.com",
                max_budget=None,
                spend=0.0,
                models=[],
            )
        ]
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch.object(team_endpoints, "get_all_team_memberships", AsyncMock(return_value=[])),
        patch.object(team_endpoints, "UserRepository") as repo,
    ):
        repo.return_value.table.find_many = find_many

        response = await team_endpoints.team_info(
            http_request=MagicMock(spec=Request),
            team_id="team-1",
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN),
        )

    members = response["team_info"].members_with_roles
    assert [(m.user_id, m.user_email) for m in members] == [
        ("no-email-on-roster", "real@example.com"),
        ("already-stored", "stored@example.com"),
    ]
    # only the member actually missing an email is looked up
    assert find_many.await_args.kwargs["where"] == {"user_id": {"in": ["no-email-on-roster"]}}


@pytest.mark.asyncio
async def test_update_model_table_clears_aliases_with_empty_map():
    """``model_aliases={}`` on /team/update must persist an empty map (json.dumps({}))
    so existing aliases are cleared, while ``model_aliases=None`` must be a no-op that
    leaves the model table untouched.
    """
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_modeltable.create = AsyncMock()
    mock_prisma.db.litellm_modeltable.upsert = AsyncMock(
        return_value=MagicMock(id="model-123")
    )
    user_api_key_dict = UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin"
    )

    returned_model_id = await _update_model_table(
        data=UpdateTeamRequest(team_id="team-1", model_aliases={}),
        model_id="model-123",
        prisma_client=mock_prisma,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name="default_user_id",
    )

    mock_prisma.db.litellm_modeltable.upsert.assert_awaited_once()
    upsert_kwargs = mock_prisma.db.litellm_modeltable.upsert.await_args.kwargs
    assert upsert_kwargs["where"] == {"id": "model-123"}
    assert upsert_kwargs["data"]["update"]["model_aliases"] == json.dumps({})
    assert upsert_kwargs["data"]["create"]["model_aliases"] == json.dumps({})
    assert returned_model_id == "model-123"

    mock_prisma.db.litellm_modeltable.create.reset_mock()
    mock_prisma.db.litellm_modeltable.upsert.reset_mock()

    noop_model_id = await _update_model_table(
        data=UpdateTeamRequest(team_id="team-1", model_aliases=None),
        model_id="model-123",
        prisma_client=mock_prisma,
        user_api_key_dict=user_api_key_dict,
        litellm_proxy_admin_name="default_user_id",
    )

    mock_prisma.db.litellm_modeltable.create.assert_not_called()
    mock_prisma.db.litellm_modeltable.upsert.assert_not_called()
    assert noop_model_id == "model-123"


class TestEmitTeamMembersMetric:
    """The _emit_team_members_metric seam between the team handlers and Prometheus."""

    @pytest.fixture
    def restore_callbacks(self):
        import litellm

        original = litellm.callbacks
        yield
        litellm.callbacks = original

    def _team(self, member_count):
        return LiteLLM_TeamTable(
            team_id="team-x",
            team_alias="X",
            members_with_roles=[
                Member(user_id=f"u{i}", role="user") for i in range(member_count)
            ],
        )

    def test_emits_with_team_when_logger_registered(self, restore_callbacks):
        import litellm
        from litellm.integrations.prometheus import PrometheusLogger
        from litellm.proxy.management_endpoints.team_endpoints import (
            _emit_team_members_metric,
        )

        fake_logger = MagicMock(spec=PrometheusLogger)
        litellm.callbacks = [fake_logger]

        team = self._team(3)
        _emit_team_members_metric(team)

        fake_logger.set_team_members_metric.assert_called_once_with(team)

    def test_noop_when_no_logger_registered(self, restore_callbacks):
        import litellm
        from litellm.proxy.management_endpoints.team_endpoints import (
            _emit_team_members_metric,
        )

        litellm.callbacks = []
        # Must not raise when Prometheus is not enabled.
        _emit_team_members_metric(self._team(2))

    def test_metric_failure_does_not_break_request(self, restore_callbacks):
        import litellm
        from litellm.integrations.prometheus import PrometheusLogger
        from litellm.proxy.management_endpoints.team_endpoints import (
            _emit_team_members_metric,
        )

        fake_logger = MagicMock(spec=PrometheusLogger)
        fake_logger.set_team_members_metric.side_effect = Exception("boom")
        litellm.callbacks = [fake_logger]

        # A metric failure must be swallowed, not propagated to the handler.
        _emit_team_members_metric(self._team(1))
        fake_logger.set_team_members_metric.assert_called_once()


@pytest.mark.asyncio
async def test_new_team_rejects_reserved_ui_session_team_id():
    """
    /team/new must reject team_id "litellm-dashboard" (UI_TEAM_ID): it is the
    virtual team stamped on every UI dashboard session token, so a real DB row
    with that id would bind its budget and permissions to every UI session.
    """
    from fastapi import Request

    from litellm.proxy._types import UI_TEAM_ID, NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    team_request = NewTeamRequest(
        team_alias="dashboard-clone",
        team_id=UI_TEAM_ID,
    )
    dummy_request = MagicMock(spec=Request)

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server._license_check") as mock_license,
    ):
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_license.is_team_count_over_limit.return_value = False
        mock_prisma.get_data = AsyncMock(return_value=None)

        with pytest.raises(ProxyException) as exc_info:
            await new_team(
                data=team_request,
                http_request=dummy_request,
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN
                ),
            )

        assert exc_info.value.code == "400"
        assert "reserved" in str(exc_info.value.message)
        mock_prisma.get_data.assert_not_called()


# ---------------------------------------------------------------------------
# PATCH /team/{team_id} — RFC 7386 JSON Merge Patch
#
# The new PATCH endpoint delegates to the same write path as POST /team/update;
# the single intended divergence is metadata. POST replaces the metadata column
# wholesale, PATCH merges it per RFC 7386 (omit preserves, null deletes, value
# overwrites, recursing into nested objects). Every other field must behave
# identically. Each test drives BOTH endpoints against an identical mocked team
# and asserts on the exact dict handed to litellm_teamtable.update.
# ---------------------------------------------------------------------------

_PATCH_TEAM_ID = "team-merge-patch-test"
_ABSENT = object()


async def _drive_team_write(
    kind,
    *,
    existing_metadata=None,
    existing_kwargs=None,
    payload=None,
    raw_body=None,
    user=None,
    find_returns_none=False,
    mock_sink=None,
):
    """Drive POST ``update_team`` or PATCH ``patch_team`` against a mocked team.

    Returns ``(endpoint_result, update_mock)``; propagates whatever the endpoint
    raises. Inspect ``update_mock.call_args.kwargs["data"]`` for the DB write.
    Pass a dict as ``mock_sink`` to receive the update mock even when the
    endpoint raises.
    """
    from unittest.mock import AsyncMock, MagicMock, Mock
    from unittest.mock import patch as _patch

    from fastapi import Request

    from litellm.proxy._types import (
        LiteLLM_TeamTable,
        LitellmUserRoles,
        PatchTeamRequest,
        UpdateTeamRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.management_endpoints.team_endpoints import (
        patch_team,
        update_team,
    )

    existing = LiteLLM_TeamTable(
        team_id=_PATCH_TEAM_ID,
        team_alias="t",
        metadata=existing_metadata,
        organization_id=None,
        **(existing_kwargs or {}),
    )
    auth = user or UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="u")

    with (
        _patch("litellm.proxy.proxy_server.prisma_client") as pc,
        _patch("litellm.proxy.proxy_server.llm_router", None),
        _patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        _patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
        _patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        _patch(
            "litellm.proxy.management_endpoints.team_endpoints._refresh_cached_team",
            new=AsyncMock(),
        ),
    ):
        pc.db.litellm_teamtable.find_unique = AsyncMock(
            return_value=None if find_returns_none else existing
        )
        pc.db.litellm_teamtable.update = AsyncMock(
            return_value=LiteLLM_TeamTable(team_id=_PATCH_TEAM_ID, team_alias="t")
        )
        pc.jsonify_team_object = MagicMock(side_effect=lambda db_data: db_data)
        if mock_sink is not None:
            mock_sink["update"] = pc.db.litellm_teamtable.update

        req = Mock(spec=Request)
        if kind == "post":
            result = await update_team(
                data=UpdateTeamRequest(team_id=_PATCH_TEAM_ID, **(payload or {})),
                http_request=req,
                user_api_key_dict=auth,
                litellm_changed_by=None,
            )
        else:
            body = raw_body if raw_body is not None else dict(payload or {})
            result = await patch_team(
                team_id=_PATCH_TEAM_ID,
                data=PatchTeamRequest.model_validate(body),
                http_request=req,
                user_api_key_dict=auth,
                litellm_changed_by=None,
            )
        return result, pc.db.litellm_teamtable.update


async def _written_metadata(kind, existing_metadata, body):
    _, update_mock = await _drive_team_write(kind, existing_metadata=existing_metadata, payload=body)
    written = update_mock.call_args.kwargs["data"]
    return written["metadata"] if "metadata" in written else _ABSENT


# (label, existing_metadata, merge_patch_body, expected_POST_metadata, expected_PATCH_metadata)
_METADATA_MAPPING = [
    (
        "omit-metadata-preserves-in-both",
        {"cost_center": "1234"},
        {"tpm_limit": 5},
        _ABSENT,  # POST: metadata column left untouched
        _ABSENT,  # PATCH: metadata column left untouched
    ),
    (
        "add-key-POST-wipes-others-PATCH-preserves",
        {"cost_center": "1234", "foo": "bar"},
        {"metadata": {"foo": "baz"}},
        {"foo": "baz"},  # POST replaces wholesale -> cost_center wiped
        {"cost_center": "1234", "foo": "baz"},  # PATCH merges -> cost_center kept
    ),
    (
        "overwrite-plus-null-delete-plus-add",
        {"cost_center": "1234", "foo": "bar"},
        {"metadata": {"cost_center": "9999", "foo": None, "new": "x"}},
        {"cost_center": "9999", "foo": None, "new": "x"},  # POST stores the literal null
        {"cost_center": "9999", "new": "x"},  # PATCH deletes foo via null
    ),
    (
        "null-delete-one-key",
        {"a": 1, "b": 2},
        {"metadata": {"b": None}},
        {"b": None},  # POST wholesale replace -> only b:null survives
        {"a": 1},  # PATCH deletes b, preserves a
    ),
    (
        "nested-object-deep-merge",
        {"settings": {"x": 1, "y": 2}},
        {"metadata": {"settings": {"y": 3, "z": 4}}},
        {"settings": {"y": 3, "z": 4}},  # POST replaces the nested object wholesale
        {"settings": {"x": 1, "y": 3, "z": 4}},  # PATCH deep-merges the nested object
    ),
    (
        "nested-object-null-delete",
        {"settings": {"x": 1, "y": 2}},
        {"metadata": {"settings": {"x": None}}},
        {"settings": {"x": None}},  # POST wholesale
        {"settings": {"y": 2}},  # PATCH deletes nested key, keeps sibling
    ),
    (
        "empty-object-POST-clears-PATCH-noops",
        {"a": 1},
        {"metadata": {}},
        {},  # POST replaces with an empty object
        {"a": 1},  # PATCH: an empty patch is a no-op
    ),
    (
        "metadata-null-clears-in-both",
        {"a": 1},
        {"metadata": None},
        None,  # POST clears the column
        None,  # PATCH: an RFC 7386 null patch clears the column too (parity)
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label, existing_metadata, body, expected_post, expected_patch",
    _METADATA_MAPPING,
    ids=[row[0] for row in _METADATA_MAPPING],
)
async def test_post_vs_patch_metadata_write_mapping(
    label, existing_metadata, body, expected_post, expected_patch
):
    """Exhaustive map: POST replaces metadata wholesale, PATCH merges per RFC 7386."""
    post_meta = await _written_metadata("post", existing_metadata, body)
    patch_meta = await _written_metadata("patch", existing_metadata, body)

    assert post_meta == expected_post, f"POST metadata mismatch for '{label}'"
    assert patch_meta == expected_patch, f"PATCH metadata mismatch for '{label}'"


@pytest.mark.asyncio
async def test_patch_preserves_required_metadata_key_that_post_would_wipe():
    """The reason PATCH exists: editing one metadata key must not silently drop
    the others, which POST /team/update does because it replaces wholesale."""
    existing = {"cost_center": "FINOPS-1", "team_notes": "keep me"}
    body = {"metadata": {"team_notes": "edited"}}

    post_meta = await _written_metadata("post", existing, body)
    patch_meta = await _written_metadata("patch", existing, body)

    assert post_meta == {"team_notes": "edited"}
    assert "cost_center" not in post_meta  # wiped by POST
    assert patch_meta == {"cost_center": "FINOPS-1", "team_notes": "edited"}  # preserved by PATCH


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, field, expected",
    [
        ({"tpm_limit": 50}, "tpm_limit", 50),
        ({"tpm_limit": None}, "tpm_limit", None),
        ({"models": ["gpt-4", "claude-3"]}, "models", ["gpt-4", "claude-3"]),
        ({"blocked": True}, "blocked", True),
        ({"max_budget": 10.0}, "max_budget", 10.0),
    ],
)
async def test_top_level_fields_identical_post_and_patch(body, field, expected):
    """Non-metadata fields are unaffected by merge semantics: value overwrites in both,
    and neither touches metadata when the patch omits it."""
    _, post_update = await _drive_team_write("post", existing_metadata={"k": "v"}, payload=body)
    _, patch_update = await _drive_team_write("patch", existing_metadata={"k": "v"}, payload=body)
    post_written = post_update.call_args.kwargs["data"]
    patch_written = patch_update.call_args.kwargs["data"]

    assert post_written[field] == expected
    assert patch_written[field] == expected
    assert "metadata" not in post_written
    assert "metadata" not in patch_written


@pytest.mark.asyncio
async def test_patch_strips_system_managed_metadata_key_like_post():
    """A caller cannot inject/overwrite server-owned keys via PATCH any more than
    via POST: team_member_budget_id is stripped from the write in both."""
    existing = {"team_member_budget_id": "budget-123", "cost_center": "1234"}
    body = {"metadata": {"team_member_budget_id": "HACKED", "cost_center": "9999"}}

    post_meta = await _written_metadata("post", existing, body)
    patch_meta = await _written_metadata("patch", existing, body)

    assert "team_member_budget_id" not in post_meta
    assert "team_member_budget_id" not in patch_meta
    assert post_meta == {"cost_center": "9999"}
    assert patch_meta == {"cost_center": "9999"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"json": ["not", "an", "object"]},
        {"json": "a-string"},
        {"json": 42},
        {"content": b"{not json"},
        {"json": {"tpm_limit": "not-an-int"}},
    ],
    ids=["list", "string", "number", "malformed-json", "wrong-field-type"],
)
def test_patch_rejects_a_malformed_body_with_422(kwargs):
    """The body is a declared parameter, so FastAPI rejects a malformed one before the
    handler runs. This is the same 422 POST /team/update already returns; the route
    previously answered 400 here and 500 for a wrongly typed field, reporting a caller
    mistake as a server fault."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm.proxy._types import PatchTeamRequest

    app = FastAPI()

    @app.patch("/team/{team_id}")
    async def _route(team_id: str, data: PatchTeamRequest):  # pragma: no cover - schema only
        return {}

    response = TestClient(app).patch("/team/abc", **kwargs)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_patch_rejects_team_id_mismatch_between_path_and_body():
    from litellm.proxy._types import ProxyException

    with pytest.raises(ProxyException) as exc:
        await _drive_team_write(
            "patch",
            existing_metadata={"a": 1},
            raw_body={"team_id": "some-other-team", "tpm_limit": 5},
        )
    assert exc.value.code == "400" or exc.value.code == 400


@pytest.mark.asyncio
async def test_patch_accepts_matching_team_id_in_body():
    """A body team_id equal to the path is tolerated and does not leak into the write."""
    _, update_mock = await _drive_team_write(
        "patch",
        existing_metadata={"a": 1},
        raw_body={"team_id": _PATCH_TEAM_ID, "tpm_limit": 7},
    )
    written = update_mock.call_args.kwargs["data"]
    assert written["tpm_limit"] == 7


@pytest.mark.asyncio
async def test_patch_team_not_found_returns_404():
    from litellm.proxy._types import ProxyException

    # metadata present -> patch_team does its own existence check
    with pytest.raises(ProxyException) as exc:
        await _drive_team_write(
            "patch", raw_body={"metadata": {"cost_center": "1"}}, find_returns_none=True
        )
    assert exc.value.code == "404" or exc.value.code == 404

    # metadata absent -> existence check happens in the delegated update_team
    with pytest.raises(ProxyException) as exc2:
        await _drive_team_write("patch", raw_body={"tpm_limit": 5}, find_returns_none=True)
    assert exc2.value.code == "404" or exc2.value.code == 404


@pytest.mark.asyncio
async def test_patch_enforces_team_access_via_delegation():
    """PATCH inherits POST's team-level RBAC: a caller who is neither proxy admin,
    team admin, nor org admin of the team is rejected."""
    from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth

    outsider = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, user_id="outsider")
    with pytest.raises(ProxyException) as exc:
        await _drive_team_write(
            "patch", raw_body={"tpm_limit": 5}, user=outsider
        )
    assert exc.value.code == "403" or exc.value.code == 403


@pytest.mark.asyncio
async def test_patch_returns_full_team_object_not_wrapper():
    """Per REST convention the PATCH response is the full team, not POST's
    {"team_id", "data"} envelope."""
    from litellm.proxy._types import LiteLLM_TeamTable

    result, _ = await _drive_team_write(
        "patch", existing_metadata={"a": 1}, raw_body={"metadata": {"b": 2}}
    )
    assert isinstance(result, LiteLLM_TeamTable)
    assert result.team_id == _PATCH_TEAM_ID


# ---------------------------------------------------------------------------
# custom_team_metadata_validate wiring: the configured validator must gate
# every team write path (POST /team/new, POST /team/update, PATCH /team/{id})
# and must see the metadata that will actually be written.
# ---------------------------------------------------------------------------

from contextlib import contextmanager

from litellm.proxy.management_helpers.team_metadata_validation import (
    TEAM_METADATA_VALIDATOR_REGISTRY,
    TeamMetadataValidationResult,
)


@contextmanager
def _configured_team_metadata_validator(validator):
    TEAM_METADATA_VALIDATOR_REGISTRY.set(validator)
    try:
        with (
            patch("litellm.proxy.proxy_server.premium_user", True),
            patch("litellm.proxy.proxy_server.general_settings", {}),
        ):
            yield
    finally:
        TEAM_METADATA_VALIDATOR_REGISTRY.set(None)


def _recording_validator(recorded, valid=True, error_message=None):
    async def validator(payload):
        recorded.append(payload)
        return TeamMetadataValidationResult(valid=valid, error_message=error_message)

    return validator


@pytest.mark.asyncio
async def test_update_validator_sees_replacement_on_post_and_merged_on_patch():
    """POST hands the validator the wholesale replacement; PATCH hands it the
    RFC 7386 merged result including preserved keys."""
    existing = {"cost_center": "OLD", "keep": 1}
    body = {"metadata": {"cost_center": "NEW"}}

    recorded_post = []
    with _configured_team_metadata_validator(_recording_validator(recorded_post)):
        await _drive_team_write("post", existing_metadata=existing, payload=body)

    recorded_patch = []
    with _configured_team_metadata_validator(_recording_validator(recorded_patch)):
        await _drive_team_write("patch", existing_metadata=existing, payload=body)

    assert len(recorded_post) == 1
    assert recorded_post[0].operation == "update"
    assert recorded_post[0].metadata == {"cost_center": "NEW"}
    assert recorded_post[0].existing_metadata == existing

    assert len(recorded_patch) == 1
    assert recorded_patch[0].operation == "update"
    assert recorded_patch[0].metadata == {"cost_center": "NEW", "keep": 1}
    assert recorded_patch[0].existing_metadata == existing


@pytest.mark.asyncio
async def test_patch_null_delete_removes_key_from_validated_metadata():
    """Deleting a key via PATCH null must be visible to the validator as the
    key's absence in the resulting metadata, so a required key cannot be
    silently dropped."""
    recorded = []
    with _configured_team_metadata_validator(_recording_validator(recorded)):
        await _drive_team_write(
            "patch",
            existing_metadata={"cost_center": "OLD", "keep": 1},
            payload={"metadata": {"cost_center": None}},
        )

    assert len(recorded) == 1
    assert recorded[0].metadata == {"keep": 1}
    assert recorded[0].existing_metadata == {"cost_center": "OLD", "keep": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["post", "patch"])
async def test_update_without_metadata_skips_validator(kind):
    recorded = []
    with _configured_team_metadata_validator(_recording_validator(recorded)):
        await _drive_team_write(kind, existing_metadata={"k": "v"}, payload={"tpm_limit": 5})

    assert recorded == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["post", "patch"])
async def test_update_validator_rejection_blocks_db_write(kind):
    recorded = []
    validator = _recording_validator(recorded, valid=False, error_message="cost center rejected, contact FinOps")
    sink = {}

    with _configured_team_metadata_validator(validator):
        with pytest.raises(ProxyException) as exc_info:
            await _drive_team_write(
                kind,
                existing_metadata={"cost_center": "OLD"},
                payload={"metadata": {"cost_center": "BAD"}},
                mock_sink=sink,
            )

    assert str(exc_info.value.code) == "400"
    assert "cost center rejected, contact FinOps" in str(exc_info.value.message)
    assert len(recorded) == 1
    sink["update"].assert_not_awaited()


@pytest.mark.asyncio
async def test_new_team_validator_runs_without_metadata_and_rejection_blocks_create():
    """Create always validates, even when the request carries no metadata, so a
    required-key policy can reject a team created without one."""
    from fastapi import Request

    from litellm.proxy._types import NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    recorded = []
    validator = _recording_validator(recorded, valid=False, error_message="cost_center is required")

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server._license_check") as mock_license,
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        _configured_team_metadata_validator(validator),
    ):
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_prisma.db.litellm_teamtable.create = AsyncMock()
        _wire_team_create_tx(mock_prisma)
        mock_license.is_team_count_over_limit.return_value = False

        with pytest.raises(ProxyException) as exc_info:
            await new_team(
                data=NewTeamRequest(team_alias="no-metadata-team"),
                http_request=MagicMock(spec=Request),
                user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1"),
            )

        assert str(exc_info.value.code) == "400"
        assert "cost_center is required" in str(exc_info.value.message)
        assert len(recorded) == 1
        assert recorded[0].operation == "create"
        assert recorded[0].metadata == {}
        assert recorded[0].existing_metadata is None
        mock_prisma.db.litellm_teamtable.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_team_validator_accept_proceeds_to_create(mock_db_client, mock_admin_auth):
    from fastapi import Request

    from litellm.proxy._types import NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    mock_db_client.jsonify_team_object = lambda db_data: db_data
    mock_db_client.get_data = AsyncMock(return_value=None)
    mock_db_client.update_data = AsyncMock(return_value=MagicMock())
    mock_db_client.db = MagicMock()

    team_create_result = MagicMock(team_id="team-accept-1")
    team_create_result.model_dump.return_value = {"team_id": "team-accept-1"}
    mock_db_client.db.litellm_teamtable = MagicMock()
    mock_db_client.db.litellm_teamtable.create = AsyncMock(return_value=team_create_result)
    _wire_team_create_tx(mock_db_client)
    mock_db_client.db.litellm_teamtable.count = AsyncMock(return_value=0)
    mock_db_client.db.litellm_teamtable.update = AsyncMock(return_value=team_create_result)
    mock_db_client.db.litellm_usertable = MagicMock()
    mock_db_client.db.litellm_usertable.update = AsyncMock(return_value=MagicMock())

    recorded = []
    with _configured_team_metadata_validator(_recording_validator(recorded)):
        await new_team(
            data=NewTeamRequest(team_alias="accepted-team", metadata={"cost_center": "CC-1001"}),
            http_request=MagicMock(spec=Request),
            user_api_key_dict=mock_admin_auth,
        )

    assert len(recorded) == 1
    assert recorded[0].operation == "create"
    assert recorded[0].metadata == {"cost_center": "CC-1001"}
    mock_db_client.db.litellm_teamtable.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_new_team_rejection_precedes_model_alias_write():
    """A rejected create must not leave an orphaned LiteLLM_ModelTable row:
    validation runs before the model_aliases insert."""
    from fastapi import Request

    from litellm.proxy._types import NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    validator = _recording_validator([], valid=False, error_message="cost_center is required")

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma,
        patch("litellm.proxy.proxy_server._license_check") as mock_license,
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        _configured_team_metadata_validator(validator),
    ):
        mock_prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        mock_prisma.db.litellm_teamtable.create = AsyncMock()
        _wire_team_create_tx(mock_prisma)
        mock_prisma.db.litellm_modeltable.create = AsyncMock(return_value=MagicMock(id="model-1"))
        mock_license.is_team_count_over_limit.return_value = False

        with pytest.raises(ProxyException):
            await new_team(
                data=NewTeamRequest(
                    team_alias="alias-orphan-check",
                    model_aliases={"alias-model": "gpt-4o"},
                ),
                http_request=MagicMock(spec=Request),
                user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1"),
            )

        mock_prisma.db.litellm_modeltable.create.assert_not_awaited()
        mock_prisma.db.litellm_teamtable.create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["post", "patch"])
async def test_update_existing_metadata_excludes_system_managed_keys(kind):
    """The validator's existing_metadata must be symmetric with metadata:
    server-owned keys (team_member_budget_id) are stripped from both, so a
    key-preservation validator never sees them 'disappear'."""
    recorded = []
    stored = {"cost_center": "CC-1001", "team_member_budget_id": "budget-abc"}

    with _configured_team_metadata_validator(_recording_validator(recorded)):
        await _drive_team_write(
            kind,
            existing_metadata=dict(stored),
            payload={"metadata": {"notes": "x"}},
        )

    assert len(recorded) == 1
    assert recorded[0].existing_metadata == {"cost_center": "CC-1001"}
    assert "team_member_budget_id" not in recorded[0].metadata


# ---------------------------------------------------------------------------
# PATCH body is validated through PatchTeamRequest before it is handed to
# update_team. The write below must stay byte-identical to what the untyped
# **body construction produced, or a partial update starts writing columns the
# caller never mentioned.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_patch_writes_only_the_keys_the_caller_sent():
    """An omitted field must not reach the DB write at all. If validation ever
    materialises defaults, every unmentioned column gets overwritten with null."""
    _, update_mock = await _drive_team_write("patch", raw_body={"tpm_limit": 5})
    written = update_mock.call_args.kwargs["data"]

    assert written["tpm_limit"] == 5
    for untouched in ("rpm_limit", "max_budget", "models", "blocked", "budget_duration"):
        assert untouched not in written, f"{untouched} was written despite not being sent"


@pytest.mark.asyncio
async def test_patch_preserves_explicit_null_as_a_clear():
    """null is a clear, not an omission: it has to survive validation and reach the write."""
    _, update_mock = await _drive_team_write("patch", raw_body={"max_budget": None})
    written = update_mock.call_args.kwargs["data"]

    assert "max_budget" in written
    assert written["max_budget"] is None


def _patch_body_to_update_request(body: dict):
    """The exact reshaping patch_team performs between the raw body and update_team."""
    from litellm.proxy._types import PatchTeamRequest, UpdateTeamRequest

    parsed = PatchTeamRequest.model_validate(body)
    return UpdateTeamRequest(
        team_id=_PATCH_TEAM_ID,
        **parsed.model_dump(exclude_unset=True, exclude={"team_id"}),
    )


@pytest.mark.parametrize(
    "body",
    [
        {"tpm_limit": 5},
        {"max_budget": None},
        {"object_permission": {"vector_stores": []}},
        {"metadata": {"a": 1, "b": None}},
        {"models": ["gpt-4"], "blocked": False},
    ],
    ids=["scalar", "explicit-null", "partial-nested", "metadata-with-null", "list-and-false"],
)
def test_patch_body_reshaping_adds_no_keys_the_caller_did_not_send(body):
    """Validating through PatchTeamRequest must be shape-preserving. If it ever
    materialises defaults, a partial update silently overwrites untouched columns,
    and for the merge-only object_permission it would wipe sibling sub-keys."""
    reshaped = _patch_body_to_update_request(body)
    dumped = reshaped.model_dump(exclude_unset=True, exclude={"team_id"})

    assert dumped == body
    assert reshaped.model_fields_set == set(body) | {"team_id"}


@pytest.mark.asyncio
async def test_patch_ignores_unknown_body_keys():
    """Unknown keys were silently dropped by the previous construction; keep that."""
    _, update_mock = await _drive_team_write(
        "patch", raw_body={"tpm_limit": 5, "not_a_team_field": "x"}
    )
    written = update_mock.call_args.kwargs["data"]

    assert written["tpm_limit"] == 5
    assert "not_a_team_field" not in written


def test_patch_team_request_makes_team_id_optional():
    """PATCH takes team_id from the path, so the body model must not require it,
    while still inheriting every UpdateTeamRequest field."""
    from litellm.proxy._types import PatchTeamRequest, UpdateTeamRequest

    parsed = PatchTeamRequest.model_validate({"tpm_limit": 5})

    assert parsed.team_id is None
    assert parsed.model_fields_set == {"tpm_limit"}
    assert set(UpdateTeamRequest.model_fields).issubset(set(PatchTeamRequest.model_fields))


def test_patch_team_route_publishes_its_request_body_schema():
    """The dashboard's generated client types this call off the OpenAPI spec, which
    FastAPI can only emit because the body is a declared parameter."""
    from litellm.proxy.proxy_server import app

    operation = app.openapi()["paths"]["/team/{team_id}"]["patch"]
    schema = operation["requestBody"]["content"]["application/json"]["schema"]

    assert schema == {"$ref": "#/components/schemas/PatchTeamRequest"}
    properties = app.openapi()["components"]["schemas"]["PatchTeamRequest"]["properties"]
    assert "tpm_limit" in properties and "metadata" in properties


@pytest.mark.asyncio
async def test_get_all_team_memberships_validates_rows():
    from litellm.proxy._types import LiteLLM_TeamMembership
    from litellm.proxy.management_endpoints.team_endpoints import (
        get_all_team_memberships,
    )

    membership_row = MagicMock()
    membership_row.model_dump = lambda: {
        "user_id": "member-1",
        "team_id": "team-1",
        "spend": 2.5,
    }

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_teammembership.find_many = AsyncMock(return_value=[membership_row])

    result = await get_all_team_memberships(mock_prisma_client, ["team-1"], user_id="member-1")

    assert len(result) == 1
    assert isinstance(result[0], LiteLLM_TeamMembership)
    assert result[0].user_id == "member-1"
    assert result[0].team_id == "team-1"
    assert result[0].spend == 2.5
    find_many_kwargs = mock_prisma_client.db.litellm_teammembership.find_many.call_args.kwargs
    assert find_many_kwargs["where"] == {"team_id": {"in": ["team-1"]}, "user_id": {"in": ["member-1"]}}


@pytest.mark.asyncio
async def test_list_available_teams_filters_joined_and_validates_rows(monkeypatch):
    from fastapi import Request

    import litellm
    from litellm.proxy.management_endpoints.team_endpoints import list_available_teams

    monkeypatch.setattr(
        litellm,
        "default_internal_user_params",
        {"available_teams": ["team-open", "team-joined"]},
    )

    user_row = MagicMock()
    user_row.model_dump = lambda: {"user_id": "u-1", "teams": ["team-joined"]}

    open_team_row = MagicMock()
    open_team_row.model_dump = lambda: {"team_id": "team-open", "team_alias": "open team"}

    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_usertable.find_unique = AsyncMock(return_value=user_row)
    mock_prisma_client.db.litellm_teamtable.find_many = AsyncMock(return_value=[open_team_row])

    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client):
        result = await list_available_teams(
            http_request=MagicMock(spec=Request),
            user_api_key_dict=UserAPIKeyAuth(user_id="u-1"),
        )

    assert len(result) == 1
    assert isinstance(result[0], LiteLLM_TeamTable)
    assert result[0].team_id == "team-open"
    assert result[0].team_alias == "open team"
    find_many_kwargs = mock_prisma_client.db.litellm_teamtable.find_many.call_args.kwargs
    assert find_many_kwargs["where"] == {"team_id": {"in": ["team-open"]}}


@pytest.mark.asyncio
async def test_get_team_metadata_schema_returns_configured_fields():
    from litellm.proxy.management_endpoints.team_endpoints import get_team_metadata_schema
    from litellm.proxy.management_helpers.team_metadata_validation import (
        TEAM_METADATA_SCHEMA_REGISTRY,
        parse_team_metadata_schema,
    )

    TEAM_METADATA_SCHEMA_REGISTRY.set(
        parse_team_metadata_schema(
            [
                {"key": "cost_center", "label": "Cost Center"},
                {"key": "app_name", "label": "Application Name"},
            ]
        )
    )
    try:
        result = await get_team_metadata_schema()
    finally:
        TEAM_METADATA_SCHEMA_REGISTRY.set(())

    assert [field.key for field in result.fields] == ["cost_center", "app_name"]
    assert result.fields[0].label == "Cost Center"
    assert result.fields[1].label == "Application Name"


@pytest.mark.asyncio
async def test_get_team_metadata_schema_empty_when_unconfigured():
    from litellm.proxy.management_endpoints.team_endpoints import get_team_metadata_schema
    from litellm.proxy.management_helpers.team_metadata_validation import (
        TEAM_METADATA_SCHEMA_REGISTRY,
    )

    TEAM_METADATA_SCHEMA_REGISTRY.set(())
    result = await get_team_metadata_schema()

    assert result.fields == ()


def test_get_team_metadata_schema_route_requires_auth():
    from litellm.proxy.management_helpers.team_metadata_validation import (
        TEAM_METADATA_SCHEMA_REGISTRY,
        parse_team_metadata_schema,
    )

    with patch("litellm.proxy.proxy_server.master_key", "sk-1234"):
        response = client.get("/team/metadata_schema")
    assert response.status_code == 401

    TEAM_METADATA_SCHEMA_REGISTRY.set(parse_team_metadata_schema([{"key": "cost_center", "label": "Cost Center"}]))
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1"
    )
    try:
        authed = client.get("/team/metadata_schema")
    finally:
        app.dependency_overrides.pop(user_api_key_auth, None)
        TEAM_METADATA_SCHEMA_REGISTRY.set(())

    assert authed.status_code == 200
    assert authed.json() == {"fields": [{"key": "cost_center", "label": "Cost Center"}]}


def test_team_metadata_schema_route_is_readable_by_non_admins():
    from litellm.proxy._types import LiteLLMRoutes

    assert "/team/metadata_schema" in LiteLLMRoutes.info_routes.value
    assert "/team/metadata_schema" in LiteLLMRoutes.management_routes.value


def _provisioning_caller(role: LitellmUserRoles) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_id="caller-1", user_role=role)


def test_validate_member_user_id_provisioning_allows_proxy_admin():
    """Proxy admins may add a user_id that has no user row yet."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_member_user_id_provisioning,
    )

    _validate_member_user_id_provisioning(
        members=[Member(user_id="brand-new", role="user")],
        existing_user_ids=frozenset(),
        user_api_key_dict=_provisioning_caller(LitellmUserRoles.PROXY_ADMIN),
    )


def test_validate_member_user_id_provisioning_rejects_unknown_user_id_for_non_proxy_admin():
    """A non-proxy-admin cannot add a user_id that has no user row yet."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_member_user_id_provisioning,
    )

    with pytest.raises(HTTPException) as exc_info:
        _validate_member_user_id_provisioning(
            members=[Member(user_id="brand-new", role="user")],
            existing_user_ids=frozenset(),
            user_api_key_dict=_provisioning_caller(LitellmUserRoles.INTERNAL_USER),
        )

    assert exc_info.value.status_code == 403
    assert "brand-new" in str(exc_info.value.detail)


def test_validate_member_user_id_provisioning_allows_existing_user_id_for_non_proxy_admin():
    """A non-proxy-admin may still add a user that already exists."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_member_user_id_provisioning,
    )

    _validate_member_user_id_provisioning(
        members=[Member(user_id="already-here", role="user")],
        existing_user_ids=frozenset({"already-here"}),
        user_api_key_dict=_provisioning_caller(LitellmUserRoles.INTERNAL_USER),
    )


def test_validate_member_user_id_provisioning_allows_email_only_member_for_non_proxy_admin():
    """Inviting by user_email stays open to non-proxy-admins; the user_id is server-allocated."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_member_user_id_provisioning,
    )

    _validate_member_user_id_provisioning(
        members=[Member(user_email="invitee@example.com", role="user")],
        existing_user_ids=frozenset(),
        user_api_key_dict=_provisioning_caller(LitellmUserRoles.INTERNAL_USER),
    )


def test_validate_member_user_id_provisioning_rejects_unknown_user_id_paired_with_email():
    """Supplying a user_email alongside an unknown user_id does not lift the restriction."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_member_user_id_provisioning,
    )

    with pytest.raises(HTTPException) as exc_info:
        _validate_member_user_id_provisioning(
            members=[Member(user_id="chosen-id", user_email="invitee@example.com", role="user")],
            existing_user_ids=frozenset(),
            user_api_key_dict=_provisioning_caller(LitellmUserRoles.INTERNAL_USER),
        )

    assert exc_info.value.status_code == 403


def test_validate_member_user_id_provisioning_reports_every_unknown_member():
    """A bulk add names each unknown user_id rather than only the first."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        _validate_member_user_id_provisioning,
    )

    with pytest.raises(HTTPException) as exc_info:
        _validate_member_user_id_provisioning(
            members=[
                Member(user_id="known", role="user"),
                Member(user_id="unknown-a", role="user"),
                Member(user_id="unknown-b", role="user"),
            ],
            existing_user_ids=frozenset({"known"}),
            user_api_key_dict=_provisioning_caller(LitellmUserRoles.INTERNAL_USER),
        )

    detail = str(exc_info.value.detail)
    assert "unknown-a" in detail
    assert "unknown-b" in detail


@pytest.mark.asyncio
async def test_resolve_existing_member_user_ids_matches_caller_supplied_user_ids():
    """Caller-supplied user_ids resolve in one query; unknown ones resolve to nothing."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        _resolve_existing_member_user_ids,
    )

    prisma_client = MagicMock()
    find_many = AsyncMock(
        return_value=[LiteLLM_UserTable(user_id="by-id", max_budget=None, spend=0.0, user_email=None, models=[])]
    )

    with patch("litellm.proxy.management_endpoints.team_endpoints.UserRepository") as repo:
        repo.return_value.table.find_many = find_many

        resolved = await _resolve_existing_member_user_ids(
            members=[
                Member(user_id="by-id", role="user"),
                Member(user_id="missing", role="user"),
                Member(user_email="someone@example.com", role="user"),
            ],
            prisma_client=prisma_client,
        )

    assert resolved == frozenset({"by-id"})
    # one round-trip, and email-only members contribute no id to look up
    find_many.assert_awaited_once()
    assert find_many.await_args.kwargs["where"] == {"user_id": {"in": ["by-id", "missing"]}}


@pytest.mark.asyncio
async def test_resolve_existing_member_user_ids_skips_the_query_when_no_user_ids():
    """An all-email payload must not hit the database at all."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        _resolve_existing_member_user_ids,
    )

    with patch("litellm.proxy.management_endpoints.team_endpoints.UserRepository") as repo:
        repo.return_value.table.find_many = AsyncMock()

        resolved = await _resolve_existing_member_user_ids(
            members=[Member(user_email="a@example.com", role="user")],
            prisma_client=MagicMock(),
        )

    assert resolved == frozenset()
    repo.return_value.table.find_many.assert_not_awaited()


def _user_row(user_id: str, user_email: str | None) -> LiteLLM_UserTable:
    return LiteLLM_UserTable(
        user_id=user_id, user_email=user_email, max_budget=None, spend=0.0, models=[]
    )


@pytest.mark.asyncio
async def test_hydrate_member_emails_fills_in_emails_the_roster_snapshot_never_captured():
    """A member added by user_id alone has user_email=None on the stored roster entry.

    /team/info has to fill it in from the user row, or the UI renders "-" for a user
    that plainly has an email.
    """
    from litellm.proxy.management_endpoints.team_endpoints import _hydrate_member_emails

    find_many = AsyncMock(return_value=[_user_row("by-id", "found@example.com")])

    with patch("litellm.proxy.management_endpoints.team_endpoints.UserRepository") as repo:
        repo.return_value.table.find_many = find_many

        hydrated = await _hydrate_member_emails(
            prisma_client=MagicMock(),
            members=[Member(user_id="by-id", role="admin")],
        )

    assert [(m.user_id, m.user_email, m.role) for m in hydrated] == [("by-id", "found@example.com", "admin")]
    find_many.assert_awaited_once()
    assert find_many.await_args.kwargs["where"] == {"user_id": {"in": ["by-id"]}}


@pytest.mark.asyncio
async def test_hydrate_member_emails_never_overwrites_a_stored_email():
    """The snapshot wins wherever it has a value - hydration only fills blanks.

    Overwriting would be a real behavior change to /team/info; filling a null is not.
    """
    from litellm.proxy.management_endpoints.team_endpoints import _hydrate_member_emails

    find_many = AsyncMock(return_value=[_user_row("has-email", "current@example.com")])

    with patch("litellm.proxy.management_endpoints.team_endpoints.UserRepository") as repo:
        repo.return_value.table.find_many = find_many

        hydrated = await _hydrate_member_emails(
            prisma_client=MagicMock(),
            members=[Member(user_id="has-email", user_email="stored@example.com", role="user")],
        )

    assert hydrated[0].user_email == "stored@example.com"
    # nothing was missing, so no round-trip either
    find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_hydrate_member_emails_leaves_members_alone_when_the_user_row_has_no_email():
    """A user row with no email leaves the member as-is rather than inventing one."""
    from litellm.proxy.management_endpoints.team_endpoints import _hydrate_member_emails

    with patch("litellm.proxy.management_endpoints.team_endpoints.UserRepository") as repo:
        repo.return_value.table.find_many = AsyncMock(return_value=[_user_row("no-email", None)])

        hydrated = await _hydrate_member_emails(
            prisma_client=MagicMock(),
            members=[Member(user_id="no-email", role="user"), Member(user_email="e@example.com", role="user")],
        )

    assert [m.user_email for m in hydrated] == [None, "e@example.com"]


@pytest.mark.asyncio
async def test_hydrate_member_emails_skips_the_query_when_every_member_has_one():
    """No blanks means /team/info pays for no extra query."""
    from litellm.proxy.management_endpoints.team_endpoints import _hydrate_member_emails

    with patch("litellm.proxy.management_endpoints.team_endpoints.UserRepository") as repo:
        repo.return_value.table.find_many = AsyncMock()

        hydrated = await _hydrate_member_emails(
            prisma_client=MagicMock(),
            members=[Member(user_id="a", user_email="a@example.com", role="user")],
        )

    assert hydrated[0].user_email == "a@example.com"
    repo.return_value.table.find_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_team_members_list_stamps_email_for_a_member_added_by_user_id():
    """Identity resolution runs both ways, so new roster entries stop being born blank.

    Previously only user_id was backfilled (from email); a member added by user_id
    was written with user_email=None forever.
    """
    from litellm.proxy.management_endpoints.team_endpoints import (
        _update_team_members_list,
    )

    mock_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_team.members_with_roles = []

    await _update_team_members_list(
        data=TeamMemberAddRequest(team_id="test-team-123", member=Member(user_id="new-user-123", role="user")),
        complete_team_data=mock_team,
        updated_users=[_user_row("new-user-123", "new@example.com")],
    )

    assert len(mock_team.members_with_roles) == 1
    assert mock_team.members_with_roles[0].user_email == "new@example.com"


@pytest.mark.asyncio
async def test_update_team_members_list_stamps_email_for_each_member_in_a_bulk_add():
    """Same both-ways resolution for the list branch."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        _update_team_members_list,
    )

    mock_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_team.members_with_roles = []

    await _update_team_members_list(
        data=TeamMemberAddRequest(
            team_id="test-team-123",
            member=[Member(user_id="u1", role="user"), Member(user_email="u2@example.com", role="admin")],
        ),
        complete_team_data=mock_team,
        updated_users=[_user_row("u1", "u1@example.com"), _user_row("u2", "u2@example.com")],
    )

    assert [(m.user_id, m.user_email) for m in mock_team.members_with_roles] == [
        ("u1", "u1@example.com"),
        ("u2", "u2@example.com"),
    ]


def test_pre_existing_user_ids_counts_ids_filled_in_by_member_resolution():
    """An id the member-resolution step filled in came from a matched row, so it pre-existed.

    This is what keeps a case-variant email invite of an existing user from being
    recorded as a newly created user.
    """
    from litellm.proxy.management_endpoints.team_endpoints import _pre_existing_user_ids

    # member arrived email-only; resolution matched an existing row and filled in the id
    resolved_member = Member(user_id="matched-existing", user_email="Someone@Example.com", role="user")

    assert _pre_existing_user_ids(
        members=[resolved_member],
        caller_supplied_user_ids=frozenset(),
        existing_user_ids=frozenset(),
    ) == frozenset({"matched-existing"})


def test_pre_existing_user_ids_excludes_caller_supplied_ids_that_do_not_exist():
    """A caller-supplied id that resolved to nothing is genuinely new, so it stays out."""
    from litellm.proxy.management_endpoints.team_endpoints import _pre_existing_user_ids

    assert _pre_existing_user_ids(
        members=[Member(user_id="brand-new", role="user"), Member(user_id="already-here", role="user")],
        caller_supplied_user_ids=frozenset({"brand-new", "already-here"}),
        existing_user_ids=frozenset({"already-here"}),
    ) == frozenset({"already-here"})


def test_members_audit_value_serializes_to_a_json_object():
    """The audit-log columns hold a JSON object; a top-level array is rejected by the DB."""
    from litellm.proxy.management_endpoints.team_endpoints import _members_audit_value

    payload = json.loads(_members_audit_value([Member(user_id="u1", role="admin"), Member(user_id="u2", role="user")]))

    assert isinstance(payload, dict)
    assert [m["user_id"] for m in payload["members_with_roles"]] == ["u1", "u2"]


@pytest.mark.asyncio
async def test_team_member_add_audits_a_user_created_from_a_list_payload(monkeypatch):
    """A user created by a list payload must still be reported as newly created.

    For a list payload the member-list reconciliation back-fills the caller's own
    Member objects with the ids of users this request just created. The set of
    pre-existing ids therefore has to be captured before that runs, otherwise a
    freshly created user looks like it was already there and no creation is recorded.
    """
    from litellm.proxy._types import TeamMemberAddRequest
    from litellm.proxy.management_endpoints.team_endpoints import team_member_add

    team_id = "team-list-audit"
    created_user_id = "generated-uuid-for-new-invitee"
    member = Member(user_email="invitee@example.com", role="user")

    mock_prisma_client = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    monkeypatch.setattr("litellm.proxy.proxy_server.litellm_proxy_admin_name", "default_user_id")

    team_row = LiteLLM_TeamTable(team_id=team_id, members_with_roles=[])
    created_user = LiteLLM_UserTable(
        user_id=created_user_id, user_email="invitee@example.com", max_budget=None, spend=0.0, models=[]
    )
    updated_team = MagicMock()
    updated_team.model_dump.return_value = {"team_id": team_id, "members_with_roles": []}

    async def fake_add_team_members_to_team(**kwargs):
        # mirrors _update_team_members_list: the list branch mutates the caller's Member in place
        member.user_id = created_user_id
        return updated_team, [created_user], []

    with (
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
            new_callable=AsyncMock,
            return_value=team_row,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._validate_team_member_add_permissions",
            new_callable=AsyncMock,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._validate_and_populate_member_user_info",
            new_callable=AsyncMock,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._resolve_existing_member_user_ids",
            new_callable=AsyncMock,
            return_value=frozenset(),
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._add_team_members_to_team",
            side_effect=fake_add_team_members_to_team,
        ),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints._create_team_member_add_audit_logs",
            new_callable=AsyncMock,
        ) as mock_audit,
    ):
        await team_member_add(
            data=TeamMemberAddRequest(team_id=team_id, member=[member]),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin-1"),
        )

    mock_audit.assert_called_once()
    assert created_user_id not in mock_audit.call_args.kwargs["existing_user_ids"]


def test_validate_member_user_id_provisioning_caps_the_ids_it_echoes_back():
    """A large member list must not echo every id back in the error body."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        _MAX_REPORTED_UNKNOWN_USER_IDS,
        _validate_member_user_id_provisioning,
    )

    members = [Member(user_id=f"u{i}", role="user") for i in range(500)]

    with pytest.raises(HTTPException) as exc_info:
        _validate_member_user_id_provisioning(
            members=members,
            existing_user_ids=frozenset(),
            user_api_key_dict=_provisioning_caller(LitellmUserRoles.INTERNAL_USER),
        )

    detail = str(exc_info.value.detail)
    assert "u0" in detail
    assert f"u{_MAX_REPORTED_UNKNOWN_USER_IDS}" not in detail
    assert f"and {500 - _MAX_REPORTED_UNKNOWN_USER_IDS} more" in detail
    assert len(detail) < 1000


_TEAM_ESTIMATE = "default_estimated_output_tokens"
_TEAM_ESTIMATE_PER_MODEL = "default_estimated_output_tokens_per_model"


@pytest.mark.parametrize(
    "label, request_body, existing_metadata, allowed",
    [
        ("nothing declared", {}, None, True),
        ("declared top-level with none stored", {_TEAM_ESTIMATE: 1}, None, False),
        ("declared inside metadata with none stored", {"metadata": {_TEAM_ESTIMATE: 1}}, None, False),
        (
            "per-model map declared inside metadata",
            {"metadata": {_TEAM_ESTIMATE_PER_MODEL: {"gpt-4": 1}}},
            None,
            False,
        ),
        ("unrelated edit, metadata omitted", {"tpm_limit": 99}, {_TEAM_ESTIMATE: 2000}, True),
        ("stored value resent unchanged", {_TEAM_ESTIMATE: 2000}, {_TEAM_ESTIMATE: 2000}, True),
        ("stored value lowered", {_TEAM_ESTIMATE: 1}, {_TEAM_ESTIMATE: 2000}, False),
        ("stored value raised", {_TEAM_ESTIMATE: 9000}, {_TEAM_ESTIMATE: 2000}, False),
        (
            "stored value cleared by sending a metadata blob without it",
            {"metadata": {"other": "keep"}},
            {_TEAM_ESTIMATE: 2000, "other": "keep"},
            False,
        ),
        (
            "stored value resent inside the metadata blob",
            {"metadata": {_TEAM_ESTIMATE: 2000, "other": "keep"}},
            {_TEAM_ESTIMATE: 2000, "other": "keep"},
            True,
        ),
        (
            "per-model map resent unchanged",
            {_TEAM_ESTIMATE_PER_MODEL: {"gpt-4": 4096}},
            {_TEAM_ESTIMATE_PER_MODEL: {"gpt-4": 4096}},
            True,
        ),
        (
            "one model in the per-model map lowered",
            {_TEAM_ESTIMATE_PER_MODEL: {"gpt-4": 1}},
            {_TEAM_ESTIMATE_PER_MODEL: {"gpt-4": 4096}},
            False,
        ),
    ],
)
def test_team_output_token_estimate_admin_gate_matrix(label, request_body, existing_metadata, allowed):
    """A team admin may only leave a team's stored output-token estimate exactly as it is.

    A team admin can write team metadata, and every key on the team inherits the
    team declaration, so without this a team admin could shrink the reservation
    for the whole team and under-reserve against an organization TPM window the
    organization set above them. Same value-transition rule as the key gate,
    including the raw-metadata route and clearing by omission.
    """
    from litellm.proxy._types import UpdateTeamRequest
    from litellm.proxy.auth.auth_utils import (
        enforce_output_token_estimates_are_admin_only,
    )

    def _call(caller):
        enforce_output_token_estimates_are_admin_only(
            data=UpdateTeamRequest(team_id="t", **request_body),
            existing_metadata=existing_metadata,
            user_api_key_dict=caller,
            entity="team",
        )

    team_admin = UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        api_key="sk-team-admin",
        user_id="team-admin",
    )
    if allowed:
        _call(team_admin)
    else:
        with pytest.raises(HTTPException) as exc:
            _call(team_admin)
        assert exc.value.status_code == 403
        assert "on a team" in str(exc.value.detail)

    _call(
        UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN,
            api_key="sk-admin",
            user_id="admin",
        )
    )


def _wire_update_team(stack, existing_metadata):
    """Mock just enough of update_team to reach (or pass) the estimate gate."""
    from unittest.mock import AsyncMock, MagicMock, patch

    mock_prisma_client = stack.enter_context(patch("litellm.proxy.proxy_server.prisma_client"))
    stack.enter_context(patch("litellm.proxy.proxy_server.llm_router"))
    stack.enter_context(patch("litellm.proxy.proxy_server.user_api_key_cache"))
    stack.enter_context(patch("litellm.proxy.proxy_server.proxy_logging_obj"))
    stack.enter_context(patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"))
    stack.enter_context(patch("litellm.proxy.management_endpoints.team_endpoints._cache_team_object"))

    existing_team = MagicMock()
    existing_team.metadata = existing_metadata
    existing_team.model_dump.return_value = {
        "team_id": "test_team_id",
        "team_alias": "test_team",
        "metadata": existing_metadata,
        "members_with_roles": [{"user_id": "team-admin", "role": "admin"}],
    }
    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=existing_team)

    updated_team = MagicMock()
    updated_team.team_id = "test_team_id"
    updated_team.model_dump.return_value = {"team_id": "test_team_id"}
    mock_prisma_client.db.litellm_teamtable.update = AsyncMock(return_value=updated_team)
    mock_prisma_client.jsonify_team_object = MagicMock(side_effect=lambda db_data: db_data)
    return mock_prisma_client


@pytest.mark.asyncio
async def test_update_team_output_token_estimate_lowered_rejected_for_team_admin():
    """End-to-end wiring: _verify_team_access admits a team admin, so the gate
    has to fire inside update_team itself."""
    import contextlib
    from unittest.mock import Mock

    from fastapi import Request

    from litellm.proxy._types import UpdateTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    with contextlib.ExitStack() as stack:
        _wire_update_team(stack, {_TEAM_ESTIMATE: 4000})
        with pytest.raises(ProxyException) as exc:
            await update_team(
                data=UpdateTeamRequest(team_id="test_team_id", default_estimated_output_tokens=1),
                http_request=Mock(spec=Request),
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.INTERNAL_USER,
                    api_key="sk-team-admin",
                    user_id="team-admin",
                ),
            )

    assert str(exc.value.code) == "403"
    assert "on a team" in str(exc.value.message)


@pytest.mark.asyncio
async def test_update_team_output_token_estimate_unchanged_allows_team_admin_edit(
    disable_audit_logging_for_mocked_team,
):
    """The team settings form resends every field it renders, so gating on
    presence would break a team admin editing an unrelated setting."""
    import contextlib
    from unittest.mock import Mock

    from fastapi import Request

    from litellm.proxy._types import UpdateTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    with contextlib.ExitStack() as stack:
        prisma = _wire_update_team(stack, {_TEAM_ESTIMATE: 4000})
        await update_team(
            data=UpdateTeamRequest(
                team_id="test_team_id",
                team_alias="renamed",
                default_estimated_output_tokens=4000,
            ),
            http_request=Mock(spec=Request),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.INTERNAL_USER,
                api_key="sk-team-admin",
                user_id="team-admin",
            ),
        )

    assert prisma.db.litellm_teamtable.update.called


@pytest.mark.asyncio
async def test_new_team_output_token_estimate_rejected_for_non_admin():
    """/team/new is the other write path into the same stored declaration."""
    from unittest.mock import Mock

    from fastapi import Request

    from litellm.proxy._types import NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    with pytest.raises(ProxyException) as exc:
        await new_team(
            data=NewTeamRequest(team_alias="t", default_estimated_output_tokens=1),
            http_request=Mock(spec=Request),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.INTERNAL_USER,
                api_key="sk-alice",
                user_id="alice",
            ),
        )

    assert str(exc.value.code) == "403"
    assert "on a team" in str(exc.value.message)


_TEAM_BATCH_LIMIT = "batch_enqueued_token_limit"


@pytest.mark.asyncio
async def test_update_team_batch_enqueued_token_limit_raised_rejected_for_team_admin():
    """_verify_team_access admits a team admin, so the gate has to fire inside
    update_team itself to keep the team's batch quota admin-owned."""
    import contextlib
    from unittest.mock import Mock

    from fastapi import Request

    from litellm.proxy._types import UpdateTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import update_team

    with contextlib.ExitStack() as stack:
        _wire_update_team(stack, {_TEAM_BATCH_LIMIT: 100000})
        with pytest.raises(ProxyException) as exc:
            await update_team(
                data=UpdateTeamRequest(team_id="test_team_id", metadata={_TEAM_BATCH_LIMIT: 10**12}),
                http_request=Mock(spec=Request),
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.INTERNAL_USER,
                    api_key="sk-team-admin",
                    user_id="team-admin",
                ),
            )

    assert str(exc.value.code) == "403"
    assert "on a team" in str(exc.value.message)


@pytest.mark.asyncio
async def test_new_team_batch_enqueued_token_limit_rejected_for_non_admin():
    """/team/new is the other write path into the same stored metadata."""
    from unittest.mock import Mock

    from fastapi import Request

    from litellm.proxy._types import NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    with pytest.raises(ProxyException) as exc:
        await new_team(
            data=NewTeamRequest(team_alias="t", metadata={_TEAM_BATCH_LIMIT: 100000}),
            http_request=Mock(spec=Request),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.INTERNAL_USER,
                api_key="sk-alice",
                user_id="alice",
            ),
        )

    assert str(exc.value.code) == "403"
    assert "on a team" in str(exc.value.message)


@pytest.mark.asyncio
async def test_get_team_daily_activity_aggregated_scopes_and_flags(mock_db_client):
    """The aggregated endpoint must apply the same non-admin key scoping as the
    paginated one and request the per-team entity breakdown with the caller's
    timezone, so the Team Usage UI gets every day in one response."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        get_team_daily_activity_aggregated,
    )

    user_id = "test_user_123"
    team_id = "test_team_456"
    user_api_key_dict = UserAPIKeyAuth(
        user_id=user_id, user_role=LitellmUserRoles.INTERNAL_USER
    )

    mock_user_info = LiteLLM_UserTable(
        user_id=user_id,
        teams=[team_id],
        max_budget=1000.0,
        spend=0.0,
        user_email="test@example.com",
        user_role="internal_user",
    )

    mock_team_member = Member(user_id=user_id, role="user")
    mock_team = MagicMock(spec=LiteLLM_TeamTable)
    mock_team.team_id = team_id
    mock_team.team_alias = "Test Team"
    mock_team.members_with_roles = [mock_team_member]
    mock_team.model_dump.return_value = {
        "team_id": team_id,
        "team_alias": "Test Team",
        "members_with_roles": [{"user_id": user_id, "role": "user"}],
    }

    user_api_key_1 = MagicMock()
    user_api_key_1.token = "user_key_1"

    mock_db_client.db.litellm_teamtable.find_many = AsyncMock(return_value=[mock_team])
    mock_db_client.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[user_api_key_1]
    )

    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_user_object",
        new_callable=AsyncMock,
    ) as mock_get_user_object:
        mock_get_user_object.return_value = mock_user_info

        with patch(
            "litellm.proxy.management_endpoints.team_endpoints.get_daily_activity_aggregated",
            new_callable=AsyncMock,
        ) as mock_aggregated:
            mock_aggregated.return_value = MagicMock()

            await get_team_daily_activity_aggregated(
                team_ids=team_id,
                start_date="2024-01-01",
                end_date="2024-01-31",
                model=None,
                api_key=None,
                exclude_team_ids=None,
                timezone=480,
                user_api_key_dict=user_api_key_dict,
            )

            mock_aggregated.assert_called_once()
            call_kwargs = mock_aggregated.call_args[1]
            assert call_kwargs["api_key"] == ["user_key_1"]
            assert call_kwargs["entity_id"] == [team_id]
            assert call_kwargs["entity_metadata_field"] == {
                team_id: {"team_alias": "Test Team"}
            }
            assert call_kwargs["include_entity_breakdown"] is True
            assert call_kwargs["timezone_offset_minutes"] == 480
            assert call_kwargs["table_name"] == "litellm_dailyteamspend"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "start_date,end_date,expected_error",
    [
        ("2020-01-01", "2026-12-31", "at most 400 days"),
        ("0000-01-01", "9999-12-31", "valid YYYY-MM-DD"),
        ("2024-06-01", "2024-01-01", "on or after"),
        ("not-a-date", "2024-01-31", "valid YYYY-MM-DD"),
        (None, "2024-01-31", "start_date and end_date"),
    ],
)
async def test_get_team_daily_activity_aggregated_rejects_bad_ranges(
    mock_db_client, start_date, end_date, expected_error
):
    """The aggregated endpoint has no pagination bounding its work, so an
    unbounded or malformed range must 400 before any query runs."""
    from litellm.proxy.management_endpoints.team_endpoints import (
        get_team_daily_activity_aggregated,
    )

    with patch(
        "litellm.proxy.management_endpoints.team_endpoints.get_daily_activity_aggregated",
        new_callable=AsyncMock,
    ) as mock_aggregated:
        with pytest.raises(HTTPException) as exc_info:
            await get_team_daily_activity_aggregated(
                team_ids=None,
                start_date=start_date,
                end_date=end_date,
                model=None,
                api_key=None,
                exclude_team_ids=None,
                timezone=None,
                user_api_key_dict=UserAPIKeyAuth(
                    user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN
                ),
            )

        assert exc_info.value.status_code == 400
        assert expected_error in str(exc_info.value.detail)
        mock_aggregated.assert_not_called()


def _wire_new_team_prisma(mock_db_client):
    mock_db_client.jsonify_team_object = lambda db_data: db_data
    mock_db_client.get_data = AsyncMock(return_value=None)
    mock_db_client.db = MagicMock()

    created_team = MagicMock(team_id="team-defaults")
    created_team.model_dump.return_value = {"team_id": "team-defaults"}

    mock_db_client.db.litellm_teamtable = MagicMock()
    mock_db_client.db.litellm_teamtable.count = AsyncMock(return_value=0)
    mock_db_client.db.litellm_teamtable.create = AsyncMock(return_value=created_team)
    _wire_team_create_tx(mock_db_client)
    mock_db_client.db.litellm_teamtable.update = AsyncMock(return_value=created_team)
    mock_db_client.db.litellm_usertable = MagicMock()
    mock_db_client.db.litellm_usertable.update = AsyncMock(return_value=MagicMock())

    return mock_db_client.db.litellm_teamtable.create


@pytest.mark.asyncio
async def test_new_team_explicit_null_budget_duration_beats_configured_default(
    mock_db_client, mock_admin_auth, monkeypatch
):
    """An explicit `"budget_duration": null` asks for a lifetime budget that never resets.

    Gating on the value alone made that indistinguishable from omitting the field,
    so the default overrode the opt-out and budget_reset_at got stamped.
    """
    from fastapi import Request

    import litellm
    from litellm.proxy._types import NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    monkeypatch.setattr(litellm, "default_team_settings", None)
    monkeypatch.setattr(litellm, "default_team_params", {"budget_duration": "30d"})
    mock_team_create = _wire_new_team_prisma(mock_db_client)

    await new_team(
        data=NewTeamRequest(team_alias="lifetime-budget-team", budget_duration=None),
        http_request=MagicMock(spec=Request),
        user_api_key_dict=mock_admin_auth,
    )

    team_data = mock_team_create.call_args.kwargs["data"]
    assert team_data.get("budget_duration") is None
    assert team_data.get("budget_reset_at") is None


@pytest.mark.asyncio
async def test_new_team_omitted_budget_duration_still_takes_configured_default(
    mock_db_client, mock_admin_auth, monkeypatch
):
    """Omitting the field keeps applying the default, the behavior the explicit-null fix must not break."""
    from fastapi import Request

    import litellm
    from litellm.proxy._types import NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    monkeypatch.setattr(litellm, "default_team_settings", None)
    monkeypatch.setattr(litellm, "default_team_params", {"budget_duration": "30d"})
    mock_team_create = _wire_new_team_prisma(mock_db_client)

    await new_team(
        data=NewTeamRequest(team_alias="default-budget-team"),
        http_request=MagicMock(spec=Request),
        user_api_key_dict=mock_admin_auth,
    )

    team_data = mock_team_create.call_args.kwargs["data"]
    assert team_data.get("budget_duration") == "30d"
    assert team_data.get("budget_reset_at") is not None


@pytest.mark.asyncio
async def test_new_team_explicit_null_max_budget_still_takes_configured_default(
    mock_db_client, mock_admin_auth, monkeypatch
):
    """The explicit-null opt-out is budget_duration-only: nulling limit fields
    (max_budget, tpm/rpm) must not skip configured defaults, or any team creator
    could mint uncapped teams (veria finding on PR #36699)."""
    from fastapi import Request

    import litellm
    from litellm.proxy._types import NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import new_team

    monkeypatch.setattr(litellm, "default_team_settings", None)
    monkeypatch.setattr(litellm, "default_team_params", {"max_budget": 100.0})
    mock_team_create = _wire_new_team_prisma(mock_db_client)

    await new_team(
        data=NewTeamRequest(team_alias="unlimited-budget-team", max_budget=None),
        http_request=MagicMock(spec=Request),
        user_api_key_dict=mock_admin_auth,
    )

    team_data = mock_team_create.call_args.kwargs["data"]
    assert team_data.get("max_budget") == 100.0


class _FakeMirrorDb:
    """Stands in for prisma inside the access-group mirror.

    Dispatches on the statement so a change to the SQL's shape is visible here, but it
    cannot validate the SQL itself: it reimplements the array semantics in Python, so it
    passes whatever the statement says. Correctness of the SQL is pinned against a real
    Postgres in tests/proxy_admin_ui_tests/test_access_group_team_sync.py.
    """

    def __init__(self, access_groups, teams, plain_lists=False):
        self._access_groups = access_groups
        self._teams = teams
        self._plain_lists = plain_lists
        self.transactions = []

    def _team_ids(self, group_id):
        stored = self._access_groups[group_id]
        return stored if self._plain_lists else stored["assigned_team_ids"]

    async def _query_raw(self, sql, *args):
        assert self._open, "mirror statement ran outside a transaction"
        if "pg_advisory_xact_lock" in sql:
            self.transactions[-1].append("lock")
            return [{"locked": False}]
        if "LiteLLM_TeamTable" in sql:
            self.transactions[-1].append("read")
            team_id = args[0]
            if team_id not in self._teams:
                return []
            return [{"access_group_ids": list(self._teams[team_id])}]

        team_id, desired = args
        if sql.lstrip().startswith("SELECT"):
            self.transactions[-1].append("affected")
            affected = [g for g in self._access_groups if g in desired or team_id in self._team_ids(g)]
            return [{"access_group_id": group_id} for group_id in affected]

        if "array_append" in sql:
            self.transactions[-1].append("attach")
            changed = [
                g for g in desired if g in self._access_groups and team_id not in self._team_ids(g)
            ]
            for group_id in changed:
                self._team_ids(group_id).append(team_id)
        else:
            self.transactions[-1].append("detach")
            changed = [
                g for g in self._access_groups if team_id in self._team_ids(g) and g not in desired
            ]
            for group_id in changed:
                self._team_ids(group_id).remove(team_id)
        return [{"access_group_id": group_id} for group_id in changed]

    async def _create_team(self, data, include=None):
        self.transactions[-1].append("create")
        team_id = data["team_id"]
        self._teams[team_id] = list(data.get("access_group_ids") or ())
        return SimpleNamespace(
            team_id=team_id,
            access_group_ids=list(self._teams[team_id]),
            model_dump=lambda: {"team_id": team_id},
        )

    def tx(self, *_args, **_kwargs):
        outer = self

        class _Tx:
            async def __aenter__(self):
                outer.transactions.append([])
                outer._open = True
                return SimpleNamespace(
                    query_raw=outer._query_raw,
                    litellm_teamtable=SimpleNamespace(create=outer._create_team),
                )

            async def __aexit__(self, *_exc_info):
                outer._open = False
                return None

        return _Tx()

    _open = False


@pytest.mark.asyncio
async def test_update_team_syncs_access_group_assigned_team_ids_in_both_directions(
    disable_audit_logging_for_mocked_team,
):
    """
    A team-side edit of `access_group_ids` must be mirrored onto every affected access
    group's `assigned_team_ids`, in one transaction, in both directions.

    `assigned_team_ids` is not display-only. `get_authorized_resources_from_key_access_groups`
    reads it as an authorization input, so a group the team dropped must stop granting its
    resources to keys on that team, and a group the team added must start granting them.
    A single-direction assertion would pass against a fix that only ever removes (or only
    ever adds), so this covers add, remove, untouched, and the authorization consequence.
    """
    from unittest.mock import Mock

    from fastapi import Request

    from litellm.proxy._types import LiteLLM_AccessGroupTable
    from litellm.proxy.auth.auth_checks import (
        get_authorized_resources_from_key_access_groups,
    )

    access_groups = {
        "ag-drop": {"assigned_team_ids": ["team-a"], "access_model_names": ["dropped-model"]},
        "ag-keep": {"assigned_team_ids": ["team-a"], "access_model_names": ["kept-model"]},
        "ag-add": {"assigned_team_ids": [], "access_model_names": ["added-model"]},
        "ag-other-team": {"assigned_team_ids": ["team-b"], "access_model_names": ["other-model"]},
    }
    committed_team_groups = ["ag-keep", "ag-add"]
    fake_db = _FakeMirrorDb(access_groups, {"team-a": committed_team_groups})

    existing_team = MagicMock()
    existing_team.access_group_ids = ["ag-drop", "ag-keep"]
    existing_team.metadata = {}
    existing_team.max_budget = None
    existing_team.organization_id = None
    existing_team.team_alias = "team-a"
    existing_team.model_dump.return_value = {"team_id": "team-a", "team_alias": "team-a"}

    updated_team = MagicMock()
    updated_team.team_id = "team-a"
    updated_team.access_group_ids = committed_team_groups
    updated_team.model_dump.return_value = {"team_id": "team-a"}

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as prisma,
        patch("litellm.proxy.proxy_server.llm_router"),
        patch("litellm.proxy.proxy_server.user_api_key_cache"),
        patch("litellm.proxy.proxy_server.proxy_logging_obj"),
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch("litellm.proxy.management_endpoints.team_endpoints._refresh_cached_team"),
        patch(
            "litellm.proxy.management_helpers.access_group_team_sync.invalidate_access_group_cache",
            new_callable=AsyncMock,
        ) as invalidate_cache,
    ):
        prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=existing_team)
        prisma.db.litellm_teamtable.update = AsyncMock(return_value=updated_team)
        prisma.db.tx = fake_db.tx
        prisma.jsonify_team_object = MagicMock(side_effect=lambda db_data: db_data)

        await update_team(
            data=UpdateTeamRequest(team_id="team-a", access_group_ids=committed_team_groups),
            http_request=Mock(spec=Request),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin"),
        )

    assert access_groups["ag-drop"]["assigned_team_ids"] == []
    assert access_groups["ag-add"]["assigned_team_ids"] == ["team-a"]
    assert access_groups["ag-keep"]["assigned_team_ids"] == ["team-a"]
    assert access_groups["ag-other-team"]["assigned_team_ids"] == ["team-b"]

    assert fake_db.transactions == [["lock", "read", "affected", "attach", "detach"]]
    assert {call.args[0] for call in invalidate_cache.call_args_list} == {"ag-drop", "ag-keep", "ag-add"}

    async def _get_access_object(*, access_group_id, **_kwargs):
        stored = access_groups[access_group_id]
        return LiteLLM_AccessGroupTable(
            access_group_id=access_group_id,
            access_group_name=access_group_id,
            access_model_names=list(stored["access_model_names"]),
            assigned_team_ids=list(stored["assigned_team_ids"]),
            assigned_key_ids=[],
        )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
        patch(
            "litellm.proxy.auth.auth_checks.get_access_object",
            new_callable=AsyncMock,
            side_effect=_get_access_object,
        ),
    ):
        authorized_models = await get_authorized_resources_from_key_access_groups(
            valid_token=UserAPIKeyAuth(
                token="sk-hash",
                models=[],
                team_id="team-a",
                access_group_ids=["ag-drop", "ag-keep", "ag-add"],
            ),
            team_object=LiteLLM_TeamTable(team_id="team-a", models=[]),
            resource_field="access_model_names",
        )

    assert sorted(authorized_models) == ["added-model", "kept-model"]


@pytest.mark.asyncio
async def test_sync_reads_the_committed_team_row_rather_than_the_callers_snapshot():
    """
    The mirror takes no desired-state argument on purpose. It locks the team and reads
    the row as committed, so two concurrent writers for one team converge on the row the
    last one committed instead of each replaying its own stale snapshot. Reconciling also
    means a retry heals a half-applied sync, where a before/after delta computes nothing.

    The same holds for the cache step: the groups to drop come from the reconciled set,
    not from the rows this attempt happened to change, so a retry after an unreachable
    cache still drops the entries even though its statements are now no-ops.

    A team with no row at all is deletion, and must detach from every group.
    """
    from litellm.proxy.management_helpers.access_group_team_sync import (
        sync_team_access_group_membership,
    )

    access_groups = {"ag-1": ["team-a", "team-b"], "ag-2": ["team-a"], "ag-3": []}
    teams = {"team-a": ["ag-2", "ag-3"]}
    fake_db = _FakeMirrorDb(access_groups, teams, plain_lists=True)
    prisma_client = SimpleNamespace(db=SimpleNamespace(tx=fake_db.tx))

    with patch(
        "litellm.proxy.management_helpers.access_group_team_sync.invalidate_access_group_cache",
        new_callable=AsyncMock,
        side_effect=[ConnectionError("redis unreachable"), None, None],
    ) as invalidate_cache:
        with pytest.raises(ConnectionError):
            await sync_team_access_group_membership(prisma_client=prisma_client, team_id="team-a")
        assert access_groups == {"ag-1": ["team-b"], "ag-2": ["team-a"], "ag-3": ["team-a"]}
        assert {call.args[0] for call in invalidate_cache.call_args_list} == {"ag-1", "ag-2", "ag-3"}

        invalidate_cache.reset_mock()
        invalidate_cache.side_effect = None
        await sync_team_access_group_membership(prisma_client=prisma_client, team_id="team-a")
        assert access_groups == {"ag-1": ["team-b"], "ag-2": ["team-a"], "ag-3": ["team-a"]}
        assert {call.args[0] for call in invalidate_cache.call_args_list} == {"ag-2", "ag-3"}

        invalidate_cache.reset_mock()
        del teams["team-a"]
        await sync_team_access_group_membership(prisma_client=prisma_client, team_id="team-a")
        assert access_groups == {"ag-1": ["team-b"], "ag-2": [], "ag-3": []}
        assert {call.args[0] for call in invalidate_cache.call_args_list} == {"ag-2", "ag-3"}

    assert fake_db.transactions == [["lock", "read", "affected", "attach", "detach"]] * 3


@pytest.mark.asyncio
async def test_new_team_and_delete_team_both_drive_the_mirror(
    disable_audit_logging_for_mocked_team,
):
    """Every writer of `team.access_group_ids` has to reach the mirror, not just update.
    These pin the wiring on the other two paths; the mirror's own behavior is covered above.

    Creation has to insert the team row and mirror it in one transaction. With the mirror
    in a transaction of its own, a sync that fails leaves a committed team whose groups
    never learned about it, and the retry is rejected as a duplicate team id."""
    from unittest.mock import Mock

    from fastapi import Request

    from litellm.proxy._types import DeleteTeamRequest, NewTeamRequest
    from litellm.proxy.management_endpoints.team_endpoints import delete_team, new_team

    access_groups = {"ag-1": [], "ag-2": []}
    fake_db = _FakeMirrorDb(access_groups, {}, plain_lists=True)

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as prisma,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch("litellm.proxy.proxy_server.user_api_key_cache"),
        patch("litellm.proxy.proxy_server.proxy_logging_obj"),
        patch("litellm.proxy.management_endpoints.team_endpoints._add_team_members_to_team", new_callable=AsyncMock),
        patch(
            "litellm.proxy.management_helpers.access_group_team_sync.invalidate_access_group_cache",
            new_callable=AsyncMock,
        ) as invalidate_cache,
    ):
        prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=None)
        prisma.db.litellm_teamtable.count = AsyncMock(return_value=0)
        prisma.db.tx = fake_db.tx
        prisma.jsonify_team_object = MagicMock(side_effect=lambda db_data: db_data)
        prisma.get_data = AsyncMock(return_value=None)

        await new_team(
            data=NewTeamRequest(team_id="team-new", team_alias="new", access_group_ids=["ag-1"]),
            http_request=Mock(spec=Request),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin"),
        )

    assert access_groups == {"ag-1": ["team-new"], "ag-2": []}
    assert fake_db.transactions == [["create", "lock", "read", "affected", "attach", "detach"]]
    assert {call.args[0] for call in invalidate_cache.call_args_list} == {"ag-1"}

    team_row = LiteLLM_TeamTable(team_id="team-gone", models=[], access_group_ids=["ag-1"])

    with (
        patch("litellm.proxy.proxy_server.prisma_client") as prisma,
        patch("litellm.proxy.proxy_server.litellm_proxy_admin_name", "admin"),
        patch("litellm.proxy.proxy_server.llm_router", None),
        patch("litellm.proxy.management_endpoints.team_endpoints._persist_deleted_team_records", new_callable=AsyncMock),
        patch("litellm.proxy.management_endpoints.team_endpoints._verify_team_access", new_callable=AsyncMock),
        patch(
            "litellm.proxy.management_endpoints.team_endpoints.sync_team_access_group_membership",
            new_callable=AsyncMock,
        ) as sync,
    ):
        prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)
        prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
        prisma.delete_data = AsyncMock(return_value=[team_row])
        prisma.db.execute_raw = AsyncMock(return_value=0)
        prisma.db.litellm_teammembership.delete_many = AsyncMock(return_value=0)

        await delete_team(
            data=DeleteTeamRequest(team_ids=["team-gone"]),
            http_request=Mock(spec=Request),
            user_api_key_dict=UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin"),
        )

    assert sync.await_args_list[0].kwargs["team_id"] == "team-gone"


@pytest.mark.asyncio
async def test_invalidate_access_group_cache_deletes_the_cached_object():
    """The mirror's cache step is what stops a revoked group granting from cache until TTL,
    so pin that it actually reaches the delete rather than only being called."""
    from litellm.proxy.management_helpers.access_group_team_sync import (
        invalidate_access_group_cache,
    )

    cache, logging_obj = MagicMock(), MagicMock()
    with (
        patch("litellm.proxy.proxy_server.user_api_key_cache", cache),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", logging_obj),
        patch(
            "litellm.proxy.management_helpers.access_group_team_sync._delete_cache_access_object",
            new_callable=AsyncMock,
        ) as delete_cached,
    ):
        await invalidate_access_group_cache("ag-1")

    assert delete_cached.await_args.kwargs == {
        "access_group_id": "ag-1",
        "user_api_key_cache": cache,
        "proxy_logging_obj": logging_obj,
    }


def test_validate_team_member_reset_spend_value_rejects_non_numeric():
    with pytest.raises(HTTPException) as exc:
        _validate_team_member_reset_spend_value(
            reset_to="not-a-number",
            membership=LiteLLM_TeamMembership(user_id="u1", team_id="t1", spend=10.0),
        )
    assert exc.value.status_code == 400


def test_validate_team_member_reset_spend_value_rejects_negative():
    with pytest.raises(HTTPException) as exc:
        _validate_team_member_reset_spend_value(
            reset_to=-1.0,
            membership=LiteLLM_TeamMembership(user_id="u1", team_id="t1", spend=10.0),
        )
    assert exc.value.status_code == 400


@pytest.mark.parametrize("reset_to", [float("nan"), float("inf"), float("-inf")])
def test_validate_team_member_reset_spend_value_rejects_non_finite(reset_to):
    """NaN and +/-inf are instances of float and compare False against every bound
    below (`nan < 0`, `nan > current_spend` are both False), so an isinstance-and-range
    check alone lets them through to persist as the member's spend and silently
    disable every later budget comparison against it."""
    with pytest.raises(HTTPException) as exc:
        _validate_team_member_reset_spend_value(
            reset_to=reset_to,
            membership=LiteLLM_TeamMembership(user_id="u1", team_id="t1", spend=10.0),
        )
    assert exc.value.status_code == 400


@pytest.mark.parametrize("reset_to", [True, False])
def test_reset_spend_request_rejects_bool_reset_to(reset_to):
    """bool is a subclass of int, so pydantic silently coerces True/False into 1.0/0.0 for a
    ``float`` field: {"reset_to": true} would otherwise reach _validate_team_member_reset_spend_value
    as an indistinguishable 1.0 and reset the member's spend instead of failing the request."""
    with pytest.raises(ValidationError):
        ResetSpendRequest(reset_to=reset_to)


def test_validate_team_member_reset_spend_value_rejects_above_current_spend():
    with pytest.raises(HTTPException) as exc:
        _validate_team_member_reset_spend_value(
            reset_to=20.0,
            membership=LiteLLM_TeamMembership(user_id="u1", team_id="t1", spend=10.0),
        )
    assert exc.value.status_code == 400


def test_validate_team_member_reset_spend_value_rejects_above_max_budget():
    with pytest.raises(HTTPException) as exc:
        _validate_team_member_reset_spend_value(
            reset_to=10.0,
            membership=LiteLLM_TeamMembership(
                user_id="u1",
                team_id="t1",
                spend=10.0,
                litellm_budget_table=LiteLLM_BudgetTable(budget_id="b1", max_budget=5.0),
            ),
        )
    assert exc.value.status_code == 400


def test_validate_team_member_reset_spend_value_accepts_valid_reset():
    result = _validate_team_member_reset_spend_value(
        reset_to=0.0,
        membership=LiteLLM_TeamMembership(user_id="u1", team_id="t1", spend=10.0),
    )
    assert result == 0.0


@pytest.mark.asyncio
async def test_reset_team_member_spend_fn_success(monkeypatch):
    """A proxy admin resetting a stuck team member's spend must write the DB
    row to reset_to AND invalidate the cached spend/membership state, or the
    429 the endpoint exists to clear keeps firing off the stale cache.
    Asserted against real cache reads, not mock call args, so a change that
    keeps the call but drops its effect still fails."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    mock_prisma_client = MagicMock()
    mock_proxy_logging_obj = MagicMock()
    real_cache = UserApiKeyCache()
    await real_cache.async_set_cache(key="team-1_member-1", value="stale-membership")
    await real_cache.async_set_cache(key="team_membership:member-1:team-1", value="stale-membership")
    real_spend_counter_cache = DualCache()
    real_spend_counter_cache.in_memory_cache.set_cache(key="spend:team_member:member-1:team-1", value=999.0)

    membership_row = LiteLLM_TeamMembership(
        user_id="member-1",
        team_id="team-1",
        spend=10.0,
        litellm_budget_table=LiteLLM_BudgetTable(budget_id="b1", max_budget=50.0),
    )
    mock_prisma_client.db.litellm_teammembership.find_unique = AsyncMock(return_value=membership_row)
    mock_prisma_client.db.litellm_teammembership.update = AsyncMock(return_value=membership_row)

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", real_cache)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", mock_proxy_logging_obj)
    monkeypatch.setattr("litellm.proxy.proxy_server.spend_counter_cache", real_spend_counter_cache)

    with patch(  # test-quality-ok: no live DB here; matches this file's established convention for endpoint-logic unit tests
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        AsyncMock(return_value=LiteLLM_TeamTable(team_id="team-1")),
    ):
        response = await reset_team_member_spend_fn(
            team_id="team-1",
            user_id="member-1",
            data=ResetSpendRequest(reset_to=0.0),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin", user_id="admin-user"
            ),
        )

    assert response["spend"] == 0.0
    assert response["previous_spend"] == 10.0
    assert response["max_budget"] == 50.0
    mock_prisma_client.db.litellm_teammembership.update.assert_awaited_once_with(
        where={"user_id_team_id": {"user_id": "member-1", "team_id": "team-1"}},
        data={"spend": 0.0},
    )
    assert await real_cache.async_get_cache(key="team-1_member-1") is None
    assert await real_cache.async_get_cache(key="team_membership:member-1:team-1") is None
    assert real_spend_counter_cache.in_memory_cache.get_cache(key="spend:team_member:member-1:team-1") == 0.0


@pytest.mark.asyncio
async def test_reset_team_member_spend_fn_membership_not_found(monkeypatch):
    mock_prisma_client = MagicMock()
    mock_prisma_client.db.litellm_teammembership.find_unique = AsyncMock(return_value=None)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", MagicMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock())

    with patch(  # test-quality-ok: no live DB here; matches this file's established convention for endpoint-logic unit tests
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        AsyncMock(return_value=LiteLLM_TeamTable(team_id="team-1")),
    ):
        with pytest.raises(HTTPException) as exc:
            await reset_team_member_spend_fn(
                team_id="team-1",
                user_id="ghost-user",
                data=ResetSpendRequest(reset_to=0.0),
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin", user_id="admin-user"
                ),
            )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_reset_team_member_spend_fn_team_not_found(monkeypatch):
    mock_prisma_client = MagicMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", MagicMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock())

    with patch(  # test-quality-ok: no live DB here; matches this file's established convention for endpoint-logic unit tests
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        AsyncMock(side_effect=HTTPException(status_code=404, detail={"error": "Team doesn't exist in db."})),
    ):
        with pytest.raises(HTTPException) as exc:
            await reset_team_member_spend_fn(
                team_id="ghost-team",
                user_id="member-1",
                data=ResetSpendRequest(reset_to=0.0),
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin", user_id="admin-user"
                ),
            )
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_reset_team_member_spend_fn_forbidden_for_non_admin(monkeypatch):
    """A caller who is neither proxy admin, org admin, nor this team's admin must be refused,
    matching every other team-mutating endpoint's authorization."""
    mock_prisma_client = MagicMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", MagicMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock())

    with patch(  # test-quality-ok: no live DB here; matches this file's established convention for endpoint-logic unit tests
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        AsyncMock(return_value=LiteLLM_TeamTable(team_id="team-1", members_with_roles=[])),
    ):
        with pytest.raises(HTTPException) as exc:
            await reset_team_member_spend_fn(
                team_id="team-1",
                user_id="member-1",
                data=ResetSpendRequest(reset_to=0.0),
                user_api_key_dict=UserAPIKeyAuth(
                    user_role=LitellmUserRoles.INTERNAL_USER, api_key="sk-user", user_id="plain-user"
                ),
            )
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_reset_team_member_spend_fn_team_admin_cannot_reset_own_spend(monkeypatch):
    """_verify_team_access authorizes a team admin over their own team with no check that the
    target differs from the caller. Unchecked, that admin could target their own membership row
    and repeatedly zero it right before it crosses their per-member cap, consuming the shared
    team budget without the configured limit ever binding (Veria finding on PR #37971)."""
    mock_prisma_client = MagicMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", MagicMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock())

    team_admin = UserAPIKeyAuth(user_role=LitellmUserRoles.INTERNAL_USER, api_key="sk-admin", user_id="team-admin-1")
    with patch(  # test-quality-ok: no live DB here; matches this file's established convention for endpoint-logic unit tests
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        AsyncMock(
            return_value=LiteLLM_TeamTable(
                team_id="team-1",
                members_with_roles=[Member(user_id="team-admin-1", role="admin")],
            )
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            await reset_team_member_spend_fn(
                team_id="team-1",
                user_id="team-admin-1",
                data=ResetSpendRequest(reset_to=0.0),
                user_api_key_dict=team_admin,
            )
    assert exc.value.status_code == 403
    mock_prisma_client.db.litellm_teammembership.update.assert_not_called()


@pytest.mark.asyncio
async def test_reset_team_member_spend_fn_proxy_admin_can_reset_own_spend(monkeypatch):
    """The self-reset guard is scoped to non-proxy-admin roles: a proxy admin resetting their
    own membership spend is the platform-wide trust boundary, not a team-scoped one."""
    mock_prisma_client = MagicMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", MagicMock())
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock())

    membership_row = LiteLLM_TeamMembership(user_id="admin-user", team_id="team-1", spend=10.0)
    mock_prisma_client.db.litellm_teammembership.find_unique = AsyncMock(return_value=membership_row)
    mock_prisma_client.db.litellm_teammembership.update = AsyncMock(return_value=membership_row)

    with patch(  # test-quality-ok: no live DB here; matches this file's established convention for endpoint-logic unit tests
        "litellm.proxy.management_endpoints.team_endpoints.get_team_object",
        AsyncMock(return_value=LiteLLM_TeamTable(team_id="team-1")),
    ):
        response = await reset_team_member_spend_fn(
            team_id="team-1",
            user_id="admin-user",
            data=ResetSpendRequest(reset_to=0.0),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin", user_id="admin-user"
            ),
        )
    assert response["spend"] == 0.0


@pytest.mark.asyncio
async def test_team_member_update_invalidates_team_member_spend_state_when_budget_patch_applied(monkeypatch):
    """Raising a stuck member's max_budget_in_team via the documented /team/member_update
    endpoint must invalidate the cached membership state, or the raised cap never reaches the
    admission check and the member stays 429ing. The live spend counter itself must be left
    untouched: only the cap changed, and deleting the counter would force a reseed from the
    DB's own spend column, which lags the live counter via periodic batch writes, briefly
    UNDER-enforcing the raised cap against a spend value lower than what was actually tracked.
    Asserted against real cache reads, not mock call args, so a change that keeps the call but
    drops its effect still fails."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    mock_prisma_client = MagicMock()
    real_cache = UserApiKeyCache()
    await real_cache.async_set_cache(key="team-1_member-1", value="stale-membership")
    await real_cache.async_set_cache(key="team_membership:member-1:team-1", value="stale-membership")
    real_spend_counter_cache = DualCache()
    real_spend_counter_cache.in_memory_cache.set_cache(key="spend:team_member:member-1:team-1", value=999.0)

    team_row = LiteLLM_TeamTable(team_id="team-1", metadata={}, members_with_roles=[])
    team_info_response = {
        "team_info": team_row,
        "team_memberships": [LiteLLM_TeamMembership(user_id="member-1", team_id="team-1", budget_id=None)],
    }

    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)

    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", real_cache)
    monkeypatch.setattr("litellm.proxy.proxy_server.spend_counter_cache", real_spend_counter_cache)

    mock_tx = AsyncMock()
    mock_prisma_client.tx.return_value.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_prisma_client.tx.return_value.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(  # test-quality-ok: no live DB here; matches this file's established convention for endpoint-logic unit tests
            "litellm.proxy.management_endpoints.team_endpoints.team_info",
            AsyncMock(return_value=team_info_response),
        ),
        patch(  # test-quality-ok: no live DB here; matches this file's established convention for endpoint-logic unit tests
            "litellm.proxy.management_endpoints.team_endpoints._upsert_budget_and_membership",
            AsyncMock(),
        ),
    ):
        await team_member_update(
            data=TeamMemberUpdateRequest(team_id="team-1", user_id="member-1", max_budget_in_team=999999.0),
            http_request=MagicMock(),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin", user_id="admin-user"
            ),
        )

    assert await real_cache.async_get_cache(key="team-1_member-1") is None
    assert await real_cache.async_get_cache(key="team_membership:member-1:team-1") is None
    assert real_spend_counter_cache.in_memory_cache.get_cache(key="spend:team_member:member-1:team-1") == 999.0


@pytest.mark.asyncio
async def test_team_member_update_skips_invalidation_when_no_budget_fields_sent(monkeypatch):
    """A role-only update carries an empty budget_patch and touches no budget state,
    so the member's cached spend/membership state must be left untouched."""
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

    mock_prisma_client = MagicMock()
    real_cache = UserApiKeyCache()
    await real_cache.async_set_cache(key="team-1_member-1", value="still-fresh-membership")
    real_spend_counter_cache = DualCache()
    real_spend_counter_cache.in_memory_cache.set_cache(key="spend:team_member:member-1:team-1", value=1.5)

    team_row = LiteLLM_TeamTable(team_id="team-1", metadata={}, members_with_roles=[])
    team_info_response = {
        "team_info": team_row,
        "team_memberships": [LiteLLM_TeamMembership(user_id="member-1", team_id="team-1", budget_id=None)],
    }

    mock_prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)

    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma_client)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", real_cache)
    monkeypatch.setattr("litellm.proxy.proxy_server.spend_counter_cache", real_spend_counter_cache)

    mock_tx = AsyncMock()
    mock_prisma_client.tx.return_value.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_prisma_client.tx.return_value.__aexit__ = AsyncMock(return_value=None)

    with (
        patch(  # test-quality-ok: no live DB here; matches this file's established convention for endpoint-logic unit tests
            "litellm.proxy.management_endpoints.team_endpoints.team_info",
            AsyncMock(return_value=team_info_response),
        ),
        patch(  # test-quality-ok: no live DB here; matches this file's established convention for endpoint-logic unit tests
            "litellm.proxy.management_endpoints.team_endpoints._upsert_budget_and_membership",
            AsyncMock(),
        ),
    ):
        await team_member_update(
            data=TeamMemberUpdateRequest(team_id="team-1", user_id="member-1"),
            http_request=MagicMock(),
            user_api_key_dict=UserAPIKeyAuth(
                user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin", user_id="admin-user"
            ),
        )

    assert await real_cache.async_get_cache(key="team-1_member-1") == "still-fresh-membership"
    assert real_spend_counter_cache.in_memory_cache.get_cache(key="spend:team_member:member-1:team-1") == 1.5
