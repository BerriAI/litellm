"""
Tests for litellm/proxy/common_utils/rbac_utils.py

Covers check_feature_access_for_user for agents and vector_stores features,
plus check_org_admin_can_generate_keys for key generation gating.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils.rbac_utils import (
    check_feature_access_for_user,
    check_org_admin_can_generate_keys,
)


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
    with patch.dict(
        _GS_PATH, {"disable_vector_stores_for_internal_users": False}, clear=True
    ):
        await check_feature_access_for_user(user, "vector_stores")


# ---------------------------------------------------------------------------
# Feature disabled, team-admin exemption OFF — internal user blocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agents_disabled_blocks_internal_user():
    user = _make_user(LitellmUserRoles.INTERNAL_USER.value)
    with patch.dict(
        _GS_PATH,
        {
            "disable_agents_for_internal_users": True,
            "allow_agents_for_team_admins": False,
        },
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
        {
            "disable_vector_stores_for_internal_users": True,
            "allow_vector_stores_for_team_admins": False,
        },
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
        {
            "disable_agents_for_internal_users": True,
            "allow_agents_for_team_admins": True,
        },
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
        {
            "disable_agents_for_internal_users": True,
            "allow_agents_for_team_admins": True,
        },
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
        {
            "disable_vector_stores_for_internal_users": True,
            "allow_vector_stores_for_team_admins": True,
        },
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
        {
            "disable_vector_stores_for_internal_users": True,
            "allow_vector_stores_for_team_admins": True,
        },
        clear=True,
    ):
        with patch(
            "litellm.proxy.management_endpoints.common_utils._user_has_admin_privileges",
            new=AsyncMock(return_value=False),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await check_feature_access_for_user(user, "vector_stores")
    assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# check_org_admin_can_generate_keys
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_org_admin_key_generate_not_disabled_allows_org_admin():
    user = _make_user(LitellmUserRoles.ORG_ADMIN.value)
    with patch.dict(_GS_PATH, {}, clear=True):
        await check_org_admin_can_generate_keys(user)


@pytest.mark.asyncio
async def test_org_admin_key_generate_disabled_blocks_org_admin():
    user = _make_user(LitellmUserRoles.ORG_ADMIN.value)
    with patch.dict(_GS_PATH, {"disable_key_generate_for_org_admin": True}, clear=True):
        with pytest.raises(HTTPException) as exc_info:
            await check_org_admin_can_generate_keys(user)
    assert exc_info.value.status_code == 403


@pytest.mark.parametrize(
    "role",
    [
        LitellmUserRoles.PROXY_ADMIN.value,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
        LitellmUserRoles.INTERNAL_USER.value,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value,
        LitellmUserRoles.TEAM.value,
    ],
)
@pytest.mark.asyncio
async def test_key_generate_disable_flag_does_not_affect_other_roles(role):
    """Only ORG_ADMIN is gated by this flag — other roles pass through."""
    user = _make_user(role)
    with patch.dict(
        _GS_PATH,
        {"disable_key_generate_for_org_admin": True},
        clear=True,
    ):
        # No exception expected — these roles are not org admins, so the flag
        # should be a no-op. Other auth checks (in the endpoint itself) still
        # apply.
        await check_org_admin_can_generate_keys(user)


@pytest.mark.asyncio
async def test_org_admin_role_enum_and_string_both_blocked():
    """UserAPIKeyAuth.user_role may be either the enum or the string value."""
    with patch.dict(_GS_PATH, {"disable_key_generate_for_org_admin": True}, clear=True):
        user_str = _make_user(LitellmUserRoles.ORG_ADMIN.value)
        with pytest.raises(HTTPException):
            await check_org_admin_can_generate_keys(user_str)

        user_enum = UserAPIKeyAuth(
            user_role=LitellmUserRoles.ORG_ADMIN, user_id="user-1"
        )
        with pytest.raises(HTTPException):
            await check_org_admin_can_generate_keys(user_enum)
