"""Tests for SCIM-driven virtual key deactivation.

When a SCIM provider deprovisions a user (DELETE) or marks them inactive
(PATCH/PUT with active=False), virtual keys owned by that user must stop
working immediately. Reactivating (active=True) must un-block them.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import LiteLLM_UserTable
from litellm.proxy.management_endpoints.scim.scim_v2 import (
    _set_user_keys_blocked,
    delete_user,
    patch_user,
)
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIMPatchOp,
    SCIMPatchOperation,
    SCIMUser,
    SCIMUserEmail,
    SCIMUserName,
)


def _build_token_row(token: str, user_id: str, blocked: bool):
    row = MagicMock()
    row.token = token
    row.user_id = user_id
    row.blocked = blocked
    return row


def _build_prisma_with_keys(user_keys, mock_user=None, updated_user=None):
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_client.db = mock_db
    if mock_user is not None:
        mock_db.litellm_usertable.find_unique = AsyncMock(return_value=mock_user)
    if updated_user is not None:
        mock_db.litellm_usertable.update = AsyncMock(return_value=updated_user)
    mock_db.litellm_usertable.delete = AsyncMock(return_value=None)
    mock_db.litellm_teamtable.find_unique = AsyncMock(return_value=None)

    mock_db.litellm_verificationtoken.find_many = AsyncMock(return_value=user_keys)
    mock_db.litellm_verificationtoken.update_many = AsyncMock(return_value=None)
    return mock_client, mock_db


@pytest.mark.asyncio
async def test_set_user_keys_blocked_flips_state_and_invalidates_cache():
    """_set_user_keys_blocked must update_many AND invalidate each token in the cache."""
    keys = [
        _build_token_row("hash-1", "user-x", blocked=False),
        _build_token_row("hash-2", "user-x", blocked=False),
    ]
    mock_client, mock_db = _build_prisma_with_keys(keys)

    cache_deletions = []

    async def fake_delete(hashed_token, user_api_key_cache, proxy_logging_obj):
        cache_deletions.append(hashed_token)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_client),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2._delete_cache_key_object",
            AsyncMock(side_effect=fake_delete),
        ),
    ):
        flipped = await _set_user_keys_blocked(user_id="user-x", blocked=True)

    assert flipped == 2
    mock_db.litellm_verificationtoken.update_many.assert_awaited_once_with(
        where={
            "user_id": "user-x",
            "OR": [{"blocked": False}, {"blocked": None}],
        },
        data={"blocked": True},
    )
    assert sorted(cache_deletions) == ["hash-1", "hash-2"]


@pytest.mark.asyncio
async def test_set_user_keys_blocked_noop_when_no_matching_keys():
    """If no keys match the desired flip, neither update_many nor cache delete runs."""
    mock_client, mock_db = _build_prisma_with_keys(user_keys=[])

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_client),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2._delete_cache_key_object",
            AsyncMock(),
        ) as mocked_delete,
    ):
        flipped = await _set_user_keys_blocked(user_id="user-x", blocked=True)

    assert flipped == 0
    mock_db.litellm_verificationtoken.update_many.assert_not_called()
    mocked_delete.assert_not_called()


@pytest.mark.asyncio
async def test_scim_delete_user_blocks_keys_before_deleting_user():
    """SCIM DELETE /Users/{id} must block the user's keys before removing the row."""
    user_id = "user-to-delete"
    mock_user = LiteLLM_UserTable(
        user_id=user_id,
        user_email="x@example.com",
        user_alias=None,
        teams=[],
        metadata={},
    )
    keys = [_build_token_row("hash-a", user_id, blocked=False)]
    mock_client, mock_db = _build_prisma_with_keys(keys, mock_user=mock_user)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_client),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2._delete_cache_key_object",
            AsyncMock(),
        ),
    ):
        response = await delete_user(user_id=user_id)

    assert response.status_code == 204
    mock_db.litellm_verificationtoken.update_many.assert_awaited_once_with(
        where={
            "user_id": user_id,
            "OR": [{"blocked": False}, {"blocked": None}],
        },
        data={"blocked": True},
    )
    mock_db.litellm_usertable.delete.assert_awaited_once_with(
        where={"user_id": user_id}
    )


@pytest.mark.asyncio
async def test_scim_patch_user_active_false_blocks_keys():
    user_id = "scim-user"
    mock_user = LiteLLM_UserTable(
        user_id=user_id,
        user_email="x@example.com",
        user_alias=None,
        teams=[],
        metadata={"scim_active": True},
    )
    updated_user = LiteLLM_UserTable(
        user_id=user_id,
        user_email="x@example.com",
        user_alias=None,
        teams=[],
        metadata={"scim_active": False, "scim_metadata": {}},
    )
    keys = [_build_token_row("hash-z", user_id, blocked=False)]
    mock_client, mock_db = _build_prisma_with_keys(
        keys, mock_user=mock_user, updated_user=updated_user
    )

    patch_ops = SCIMPatchOp(
        Operations=[SCIMPatchOperation(op="replace", path="active", value="False")]
    )
    mock_scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        id=user_id,
        userName=user_id,
        name=SCIMUserName(familyName="X", givenName="Y"),
        emails=[SCIMUserEmail(value="x@example.com")],
        active=False,
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_client),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
            AsyncMock(return_value=mock_scim_user),
        ),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2._delete_cache_key_object",
            AsyncMock(),
        ),
    ):
        await patch_user(user_id=user_id, patch_ops=patch_ops)

    mock_db.litellm_verificationtoken.update_many.assert_awaited_once_with(
        where={
            "user_id": user_id,
            "OR": [{"blocked": False}, {"blocked": None}],
        },
        data={"blocked": True},
    )


@pytest.mark.asyncio
async def test_scim_patch_user_active_true_unblocks_keys():
    user_id = "scim-user"
    mock_user = LiteLLM_UserTable(
        user_id=user_id,
        user_email="x@example.com",
        user_alias=None,
        teams=[],
        metadata={"scim_active": False},
    )
    updated_user = LiteLLM_UserTable(
        user_id=user_id,
        user_email="x@example.com",
        user_alias=None,
        teams=[],
        metadata={"scim_active": True, "scim_metadata": {}},
    )
    keys = [_build_token_row("hash-r", user_id, blocked=True)]
    mock_client, mock_db = _build_prisma_with_keys(
        keys, mock_user=mock_user, updated_user=updated_user
    )

    patch_ops = SCIMPatchOp(
        Operations=[SCIMPatchOperation(op="replace", path="active", value="True")]
    )
    mock_scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        id=user_id,
        userName=user_id,
        name=SCIMUserName(familyName="X", givenName="Y"),
        emails=[SCIMUserEmail(value="x@example.com")],
        active=True,
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_client),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
            AsyncMock(return_value=mock_scim_user),
        ),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2._delete_cache_key_object",
            AsyncMock(),
        ),
    ):
        await patch_user(user_id=user_id, patch_ops=patch_ops)

    mock_db.litellm_verificationtoken.update_many.assert_awaited_once_with(
        where={"user_id": user_id, "blocked": True},
        data={"blocked": False},
    )


@pytest.mark.asyncio
async def test_scim_patch_user_no_active_change_does_not_touch_keys():
    """A patch that doesn't flip active must not call update_many on tokens."""
    user_id = "scim-user"
    mock_user = LiteLLM_UserTable(
        user_id=user_id,
        user_email="x@example.com",
        user_alias="Old",
        teams=[],
        metadata={"scim_active": True},
    )
    updated_user = LiteLLM_UserTable(
        user_id=user_id,
        user_email="x@example.com",
        user_alias="New",
        teams=[],
        metadata={"scim_active": True, "scim_metadata": {}},
    )
    mock_client, mock_db = _build_prisma_with_keys(
        user_keys=[], mock_user=mock_user, updated_user=updated_user
    )

    patch_ops = SCIMPatchOp(
        Operations=[SCIMPatchOperation(op="replace", path="displayName", value="New")]
    )
    mock_scim_user = SCIMUser(
        schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
        id=user_id,
        userName=user_id,
        name=SCIMUserName(familyName="X", givenName="Y"),
        emails=[SCIMUserEmail(value="x@example.com")],
        active=True,
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_client),
        patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", MagicMock()),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2.ScimTransformations.transform_litellm_user_to_scim_user",
            AsyncMock(return_value=mock_scim_user),
        ),
        patch(
            "litellm.proxy.management_endpoints.scim.scim_v2._delete_cache_key_object",
            AsyncMock(),
        ),
    ):
        await patch_user(user_id=user_id, patch_ops=patch_ops)

    mock_db.litellm_verificationtoken.find_many.assert_not_called()
    mock_db.litellm_verificationtoken.update_many.assert_not_called()
