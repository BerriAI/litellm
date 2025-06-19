from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import LiteLLM_UserTable
from litellm.proxy.management_endpoints.scim.scim_v2 import patch_user
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIMPatchOp,
    SCIMPatchOperation,
)


@pytest.mark.asyncio
async def test_patch_user_updates_fields():
    mock_user = LiteLLM_UserTable(
        user_id="user-1",
        user_email="test@example.com",
        user_alias="Old",
        teams=[],
        metadata={},
    )

    async def mock_update(*, where, data):
        if "user_alias" in data:
            mock_user.user_alias = data["user_alias"]
        if "metadata" in data:
            mock_user.metadata = data["metadata"]
        if "teams" in data:
            mock_user.teams = data["teams"]
        if "sso_user_id" in data:
            mock_user.sso_user_id = data["sso_user_id"]
        return mock_user

    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.db = mock_db
    mock_db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)
    mock_db.litellm_usertable.update = AsyncMock(side_effect=mock_update)
    mock_db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    patch_ops = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(op="replace", path="displayName", value="New Name"),
            SCIMPatchOperation(op="replace", path="active", value="False"),
        ]
    )

    with patch("litellm.proxy.proxy_server.prisma_client", mock_client):
        result = await patch_user(user_id="user-1", patch_ops=patch_ops)

    mock_db.litellm_usertable.update.assert_called_once()
    assert result.displayName == "New Name"
    assert mock_user.metadata.get("scim_active") is False


@pytest.mark.asyncio
async def test_patch_user_manages_group_memberships():
    mock_user = LiteLLM_UserTable(
        user_id="user-2",
        user_email="test@example.com",
        user_alias="Old",
        teams=["old-team"],
        metadata={},
    )

    async def mock_update(*, where, data):
        if "teams" in data:
            mock_user.teams = data["teams"]
        if "metadata" in data:
            mock_user.metadata = data["metadata"]
        return mock_user

    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.db = mock_db
    mock_db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)
    mock_db.litellm_usertable.update = AsyncMock(side_effect=mock_update)

    async def mock_add(data, user_api_key_dict):
        mock_user.teams.append(data.team_id)

    async def mock_delete(data, user_api_key_dict):
        if data.team_id in mock_user.teams:
            mock_user.teams.remove(data.team_id)

    patch_ops = SCIMPatchOp(
        Operations=[
            SCIMPatchOperation(op="add", path="groups", value=[{"value": "new-team"}]),
            SCIMPatchOperation(op="remove", path="groups", value=[{"value": "old-team"}]),
        ]
    )

    with patch("litellm.proxy.proxy_server.prisma_client", mock_client), patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_add",
        AsyncMock(side_effect=mock_add),
    ) as mock_add_fn, patch(
        "litellm.proxy.management_endpoints.scim.scim_v2.team_member_delete",
        AsyncMock(side_effect=mock_delete),
    ) as mock_del_fn:
        await patch_user(user_id="user-2", patch_ops=patch_ops)

    assert mock_add_fn.called
    assert mock_del_fn.called
    assert mock_user.teams == ["new-team"]

