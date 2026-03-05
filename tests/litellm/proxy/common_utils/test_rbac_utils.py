"""
Tests for litellm/proxy/common_utils/rbac_utils.py

Covers check_feature_access_for_user for agents and vector_stores features.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils.rbac_utils import check_feature_access_for_user


def _make_user(role: str, user_id: str = "user-1") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(user_role=role, user_id=user_id)


# general_settings is imported from litellm.proxy.proxy_server inside the
# function, so we patch it via patch.dict on the original dict.
_GS_PATH = "litellm.proxy.proxy_server.general_settings"


# ---------------------------------------------------------------------------
# Proxy admin is always allowed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_proxy_admin_always_allowed():
    user = _make_user(LitellmUserRoles.PROXY_ADMIN.value)
    with patch.dict(_GS_PATH, {"disable_agents_for_internal_users": True}):
        await check_feature_access_for_user(user, "agents")


@pytest.mark.asyncio
async def test_proxy_admin_view_only_always_allowed():
    user = _make_user(LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value)
    with patch.dict(_GS_PATH, {"disable_agents_for_internal_users": True}):
        await check_feature_access_for_user(user, "agents")


# ---------------------------------------------------------------------------
# Feature not disabled — everyone allowed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_feature_not_disabled_allows_internal_user():
    user = _make_user(LitellmUserRoles.INTERNAL_USER.value)
    with patch.dict(_GS_PATH, {}, clear=True):
        await check_feature_access_for_user(user, "agents")


@pytest.mark.asyncio
async def test_feature_not_disabled_allows_vector_stores():
    user = _make_user(LitellmUserRoles.INTERNAL_USER.value)
    with patch.dict(_GS_PATH, {"disable_vector_stores_for_internal_users": False}, clear=True):
        await check_feature_access_for_user(user, "vector_stores")


# ---------------------------------------------------------------------------
# Feature disabled, team-admin exemption OFF — internal user blocked
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agents_disabled_blocks_internal_user():
    user = _make_user(LitellmUserRoles.INTERNAL_USER.value)
    with patch.dict(
        _GS_PATH,
        {"disable_agents_for_internal_users": True, "allow_agents_for_team_admins": False},
        clear=True,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await check_feature_access_for_user(user, "agents")
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_vector_stores_disabled_blocks_internal_user():
    user = _make_user(LitellmUserRoles.INTERNAL_USER.value)
    with patch.dict(
        _GS_PATH,
        {"disable_vector_stores_for_internal_users": True, "allow_vector_stores_for_team_admins": False},
        clear=True,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await check_feature_access_for_user(user, "vector_stores")
    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# Feature disabled, allow_team_admins ON — team admin allowed, non-admin blocked
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_agents_disabled_team_admin_allowed():
    user = _make_user(LitellmUserRoles.INTERNAL_USER.value, user_id="team-admin-user")
    with patch.dict(
        _GS_PATH,
        {"disable_agents_for_internal_users": True, "allow_agents_for_team_admins": True},
        clear=True,
    ):
        with patch(
            "litellm.proxy.management_endpoints.common_utils._user_has_admin_privileges",
            new=AsyncMock(return_value=True),
        ):
            await check_feature_access_for_user(user, "agents")


@pytest.mark.asyncio
async def test_agents_disabled_non_team_admin_blocked():
    user = _make_user(LitellmUserRoles.INTERNAL_USER.value, user_id="regular-user")
    with patch.dict(
        _GS_PATH,
        {"disable_agents_for_internal_users": True, "allow_agents_for_team_admins": True},
        clear=True,
    ):
        with patch(
            "litellm.proxy.management_endpoints.common_utils._user_has_admin_privileges",
            new=AsyncMock(return_value=False),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await check_feature_access_for_user(user, "agents")
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_vector_stores_disabled_team_admin_allowed():
    user = _make_user(LitellmUserRoles.INTERNAL_USER.value, user_id="team-admin-user")
    with patch.dict(
        _GS_PATH,
        {"disable_vector_stores_for_internal_users": True, "allow_vector_stores_for_team_admins": True},
        clear=True,
    ):
        with patch(
            "litellm.proxy.management_endpoints.common_utils._user_has_admin_privileges",
            new=AsyncMock(return_value=True),
        ):
            await check_feature_access_for_user(user, "vector_stores")


@pytest.mark.asyncio
async def test_vector_stores_disabled_non_team_admin_blocked():
    user = _make_user(LitellmUserRoles.INTERNAL_USER.value, user_id="regular-user")
    with patch.dict(
        _GS_PATH,
        {"disable_vector_stores_for_internal_users": True, "allow_vector_stores_for_team_admins": True},
        clear=True,
    ):
        with patch(
            "litellm.proxy.management_endpoints.common_utils._user_has_admin_privileges",
            new=AsyncMock(return_value=False),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await check_feature_access_for_user(user, "vector_stores")
    assert exc_info.value.status_code == 403
