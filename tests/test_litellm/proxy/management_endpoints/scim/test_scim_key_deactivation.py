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
    update_user,
)
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIMPatchOp,
    SCIMPatchOperation,
    SCIMUser,
    SCIMUserEmail,
    SCIMUserName,
)


def _build_token_row(token: str, user_id: str, blocked: bool, metadata=None):
    row = MagicMock()
    row.token = token
    row.user_id = user_id
    row.blocked = blocked
    row.metadata = metadata if metadata is not None else {}
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
    mock_db.litellm_verificationtoken.update = AsyncMock(return_value=None)
    mock_db.litellm_invitationlink.delete_many = AsyncMock(return_value=None)
    mock_db.litellm_organizationmembership.delete_many = AsyncMock(return_value=None)
    mock_db.litellm_teammembership.delete_many = AsyncMock(return_value=None)
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
    # Each key gets a per-row update that flips `blocked` and stamps the
    # SCIM-block marker into metadata.
    assert mock_db.litellm_verificationtoken.update.await_count == 2
    update_calls = mock_db.litellm_verificationtoken.update.await_args_list
    seen_tokens = set()
    for call in update_calls:
        kwargs = call.kwargs or call[1]
        seen_tokens.add(kwargs["where"]["token"])
        assert kwargs["data"]["blocked"] is True
        assert '"scim_blocked": true' in kwargs["data"]["metadata"]
    assert seen_tokens == {"hash-1", "hash-2"}
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
    mock_db.litellm_verificationtoken.update.assert_not_called()
    mock_db.litellm_verificationtoken.update_many.assert_not_called()
    mocked_delete.assert_not_called()


@pytest.mark.asyncio
async def test_set_user_keys_unblocked_skips_admin_blocked_keys():
    """Reactivation must leave keys an admin blocked (no scim_blocked marker) alone."""
    keys = [
        # SCIM-blocked: should be unblocked.
        _build_token_row(
            "hash-scim", "user-x", blocked=True, metadata={"scim_blocked": True}
        ),
        # Admin-blocked for unrelated reasons: must remain blocked.
        _build_token_row("hash-admin", "user-x", blocked=True, metadata={}),
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
        flipped = await _set_user_keys_blocked(user_id="user-x", blocked=False)

    assert flipped == 1
    mock_db.litellm_verificationtoken.update.assert_awaited_once()
    update_kwargs = mock_db.litellm_verificationtoken.update.await_args.kwargs
    assert update_kwargs["where"] == {"token": "hash-scim"}
    assert update_kwargs["data"]["blocked"] is False
    assert cache_deletions == ["hash-scim"]


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
    mock_db.litellm_verificationtoken.update.assert_awaited_once()
    update_kwargs = mock_db.litellm_verificationtoken.update.await_args.kwargs
    assert update_kwargs["where"] == {"token": "hash-a"}
    assert update_kwargs["data"]["blocked"] is True
    assert '"scim_blocked": true' in update_kwargs["data"]["metadata"]
    mock_db.litellm_usertable.delete.assert_awaited_once_with(
        where={"user_id": user_id}
    )


@pytest.mark.asyncio
async def test_scim_delete_user_clears_fk_referenced_rows_before_user_delete():
    user_id = "user-with-invite"
    mock_user = LiteLLM_UserTable(
        user_id=user_id,
        user_email="x@example.com",
        user_alias=None,
        teams=[],
        metadata={},
    )
    mock_client, mock_db = _build_prisma_with_keys(user_keys=[], mock_user=mock_user)

    call_order: list = []
    mock_db.litellm_invitationlink.delete_many = AsyncMock(
        side_effect=lambda **kw: call_order.append(("invitation", kw)) or None
    )
    mock_db.litellm_organizationmembership.delete_many = AsyncMock(
        side_effect=lambda **kw: call_order.append(("orgmembership", kw)) or None
    )
    mock_db.litellm_teammembership.delete_many = AsyncMock(
        side_effect=lambda **kw: call_order.append(("teammembership", kw)) or None
    )
    mock_db.litellm_usertable.delete = AsyncMock(
        side_effect=lambda **kw: call_order.append(("user", kw)) or None
    )

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

    mock_db.litellm_invitationlink.delete_many.assert_awaited_once()
    inv_kwargs = mock_db.litellm_invitationlink.delete_many.await_args.kwargs
    assert inv_kwargs == {
        "where": {
            "OR": [
                {"user_id": user_id},
                {"created_by": user_id},
                {"updated_by": user_id},
            ]
        }
    }
    mock_db.litellm_organizationmembership.delete_many.assert_awaited_once_with(
        where={"user_id": user_id}
    )
    mock_db.litellm_teammembership.delete_many.assert_awaited_once_with(
        where={"user_id": user_id}
    )

    stages = [stage for stage, _ in call_order]
    assert stages.index("user") > stages.index("invitation")
    assert stages.index("user") > stages.index("orgmembership")


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

    mock_db.litellm_verificationtoken.update.assert_awaited_once()
    update_kwargs = mock_db.litellm_verificationtoken.update.await_args.kwargs
    assert update_kwargs["where"] == {"token": "hash-z"}
    assert update_kwargs["data"]["blocked"] is True
    assert '"scim_blocked": true' in update_kwargs["data"]["metadata"]


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
    keys = [
        _build_token_row(
            "hash-r", user_id, blocked=True, metadata={"scim_blocked": True}
        )
    ]
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

    mock_db.litellm_verificationtoken.update.assert_awaited_once()
    update_kwargs = mock_db.litellm_verificationtoken.update.await_args.kwargs
    assert update_kwargs["where"] == {"token": "hash-r"}
    assert update_kwargs["data"]["blocked"] is False
    # SCIM-block marker is stripped on reactivation.
    assert "scim_blocked" not in update_kwargs["data"]["metadata"]


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


def _build_put_user_payload(user_id: str, **overrides) -> dict:
    payload = {
        "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
        "id": user_id,
        "userName": user_id,
        "name": {"givenName": "Y", "familyName": "X"},
        "emails": [{"value": "x@example.com", "primary": True}],
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_scim_put_user_omitting_active_preserves_deactivated_state():
    """PUT without `active` must not silently reactivate a SCIM-deactivated user
    nor unblock their SCIM-blocked keys."""
    user_id = "scim-user"
    deactivated = LiteLLM_UserTable(
        user_id=user_id,
        user_email="x@example.com",
        user_alias=None,
        teams=[],
        metadata={"scim_active": False},
    )
    keys = [
        _build_token_row(
            "hash-keep-blocked", user_id, blocked=True, metadata={"scim_blocked": True}
        )
    ]
    mock_client, mock_db = _build_prisma_with_keys(
        keys, mock_user=deactivated, updated_user=deactivated
    )

    put_user = SCIMUser.model_validate(_build_put_user_payload(user_id))

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
        await update_user(user_id=user_id, user=put_user)

    mock_db.litellm_verificationtoken.update.assert_not_called()
    mock_db.litellm_verificationtoken.update_many.assert_not_called()
    mock_db.litellm_usertable.update.assert_awaited_once()
    update_kwargs = mock_db.litellm_usertable.update.await_args.kwargs
    assert '"scim_active": false' in update_kwargs["data"]["metadata"]


@pytest.mark.asyncio
async def test_scim_put_user_explicit_active_false_blocks_keys():
    """PUT explicitly setting active=False on an active user must cascade to keys."""
    user_id = "scim-user"
    active = LiteLLM_UserTable(
        user_id=user_id,
        user_email="x@example.com",
        user_alias=None,
        teams=[],
        metadata={"scim_active": True},
    )
    deactivated = LiteLLM_UserTable(
        user_id=user_id,
        user_email="x@example.com",
        user_alias=None,
        teams=[],
        metadata={"scim_active": False, "scim_metadata": {}},
    )
    keys = [_build_token_row("hash-block-me", user_id, blocked=False)]
    mock_client, mock_db = _build_prisma_with_keys(
        keys, mock_user=active, updated_user=deactivated
    )

    put_user = SCIMUser.model_validate(_build_put_user_payload(user_id, active=False))

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
        await update_user(user_id=user_id, user=put_user)

    mock_db.litellm_verificationtoken.update.assert_awaited_once()
    update_kwargs = mock_db.litellm_verificationtoken.update.await_args.kwargs
    assert update_kwargs["where"] == {"token": "hash-block-me"}
    assert update_kwargs["data"]["blocked"] is True
    assert '"scim_blocked": true' in update_kwargs["data"]["metadata"]
