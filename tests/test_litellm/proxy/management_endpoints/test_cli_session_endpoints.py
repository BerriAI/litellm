import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../"))

from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.cli_session_endpoints import (
    list_cli_sessions_endpoint,
    revoke_cli_session_endpoint,
)

from tests.test_litellm.proxy.auth.test_cli_session_registry import (
    FakeCLISessionTable,
    FakePrismaClient,
)


def _row(session_id: str, *, revoked_at=None):
    now = datetime.now(timezone.utc)
    return {
        "session_id": session_id,
        "user_id": "u-1",
        "team_id": "t-1",
        "created_at": now,
        "expires_at": now + timedelta(hours=24),
        "revoked_at": revoked_at,
        "revoked_by": None,
    }


def _caller(role: LitellmUserRoles, user_id: str = "admin-1") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(token="sk-hash", user_id=user_id, user_role=role)


def _proxy(table: FakeCLISessionTable):
    return (
        patch("litellm.proxy.proxy_server.prisma_client", FakePrismaClient(table)),
        patch("litellm.proxy.proxy_server.user_api_key_cache", DualCache()),
    )


@pytest.mark.asyncio
async def test_list_returns_registered_sessions():
    table = FakeCLISessionTable({"s-1": _row("s-1")})
    prisma_patch, cache_patch = _proxy(table)

    with prisma_patch, cache_patch:
        listed = await list_cli_sessions_endpoint(user_api_key_dict=_caller(LitellmUserRoles.PROXY_ADMIN))

    assert listed.total_count == 1
    assert listed.sessions[0].session_id == "s-1"
    assert listed.sessions[0].user_id == "u-1"


@pytest.mark.asyncio
async def test_admin_viewer_can_list_but_not_revoke():
    """Read-only admins get the visibility half without the ability to cut a user off."""
    table = FakeCLISessionTable({"s-1": _row("s-1")})
    viewer = _caller(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
    prisma_patch, cache_patch = _proxy(table)

    with prisma_patch, cache_patch:
        assert (await list_cli_sessions_endpoint(user_api_key_dict=viewer)).total_count == 1

        with pytest.raises(HTTPException) as exc_info:
            await revoke_cli_session_endpoint(session_id="s-1", user_api_key_dict=viewer)

    assert exc_info.value.status_code == 403
    assert table.rows["s-1"]["revoked_at"] is None


@pytest.mark.asyncio
async def test_internal_user_cannot_see_other_peoples_sessions():
    table = FakeCLISessionTable({"s-1": _row("s-1")})
    prisma_patch, cache_patch = _proxy(table)

    with prisma_patch, cache_patch:
        with pytest.raises(HTTPException) as exc_info:
            await list_cli_sessions_endpoint(user_api_key_dict=_caller(LitellmUserRoles.INTERNAL_USER))

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_revoke_marks_the_session_and_attributes_the_operator():
    table = FakeCLISessionTable({"s-1": _row("s-1")})
    prisma_patch, cache_patch = _proxy(table)

    with prisma_patch, cache_patch:
        revoked = await revoke_cli_session_endpoint(
            session_id="s-1",
            user_api_key_dict=_caller(LitellmUserRoles.PROXY_ADMIN, user_id="admin-7"),
        )

    assert revoked.revoked_at is not None
    assert revoked.revoked_by == "admin-7"


@pytest.mark.asyncio
async def test_revoking_an_unknown_session_is_a_404():
    table = FakeCLISessionTable()
    prisma_patch, cache_patch = _proxy(table)

    with prisma_patch, cache_patch:
        with pytest.raises(HTTPException) as exc_info:
            await revoke_cli_session_endpoint(session_id="nope", user_api_key_dict=_caller(LitellmUserRoles.PROXY_ADMIN))

    assert exc_info.value.status_code == 404
