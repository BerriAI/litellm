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
from litellm.proxy._types import LiteLLM_TeamTable, LitellmUserRoles
from litellm.proxy.management_endpoints.team_endpoints import (
    user_api_key_auth,  # Assuming this dependency is needed
)
from litellm.proxy.management_endpoints.team_endpoints import (
    GetTeamMemberPermissionsResponse,
    UpdateTeamMemberPermissionsRequest,
    router,
)
from litellm.proxy.management_helpers.team_member_permission_checks import (
    TeamMemberPermissionChecks,
)
from litellm.proxy.proxy_server import app

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


# Test for /team/permissions_list endpoint (GET)
@pytest.mark.asyncio
async def test_get_team_permissions_list_success(mock_db_client, mock_admin_auth):
    """
    Test successful retrieval of team member permissions.
    """
    test_team_id = "test-team-123"
    mock_team_data = {
        "team_id": test_team_id,
        "team_alias": "Test Team",
        "team_member_permissions": ["/key/generate", "/key/update"],
        "spend": 0.0,
    }
    mock_team_row = MagicMock()
    mock_team_row.model_dump.return_value = mock_team_data
    mock_db_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_team_row
    )

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
    mock_db_client.db.litellm_teamtable.find_unique.assert_awaited_once_with(
        where={"team_id": test_team_id}
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
    update_payload = {
        "team_id": test_team_id,
        "team_member_permissions": ["/key/generate", "/key/update"],
    }

    mock_existing_team_data = {
        "team_id": test_team_id,
        "team_alias": "Existing Team",
        "team_member_permissions": ["/key/list"],
        "spend": 0.0,
        "models": [],
    }
    mock_updated_team_data = {
        **mock_existing_team_data,
        "team_member_permissions": update_payload["team_member_permissions"],
    }

    mock_existing_team_row = MagicMock(spec=LiteLLM_TeamTable)
    mock_existing_team_row.model_dump.return_value = mock_existing_team_data
    # Set attributes directly if model_dump isn't enough for LiteLLM_TeamTable usage
    for key, value in mock_existing_team_data.items():
        setattr(mock_existing_team_row, key, value)

    mock_updated_team_row = MagicMock(spec=LiteLLM_TeamTable)
    mock_updated_team_row.model_dump.return_value = mock_updated_team_data
    # Set attributes directly if model_dump isn't enough for LiteLLM_TeamTable usage
    for key, value in mock_updated_team_data.items():
        setattr(mock_updated_team_row, key, value)

    mock_db_client.db.litellm_teamtable.find_unique = AsyncMock(
        return_value=mock_existing_team_row
    )
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

    mock_db_client.db.litellm_teamtable.find_unique.assert_awaited_once_with(
        where={"team_id": test_team_id}
    )
    mock_db_client.db.litellm_teamtable.update.assert_awaited_once_with(
        where={"team_id": test_team_id},
        data={"team_member_permissions": update_payload["team_member_permissions"]},
    )

    # Clean up dependency override
    app.dependency_overrides = {}
