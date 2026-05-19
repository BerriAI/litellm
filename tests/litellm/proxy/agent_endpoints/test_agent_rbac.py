"""
Tests for RBAC enforcement on agent endpoints.

Verifies that check_feature_access_for_user is called and that a 403 is
raised when agents are disabled for internal users.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


def _make_internal_user(user_id: str = "user-1") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER.value,
        user_id=user_id,
    )


def _make_admin_user(user_id: str = "admin-1") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
        user_id=user_id,
    )


# ---------------------------------------------------------------------------
# get_agents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_agents_blocked_for_internal_user_when_disabled():
    """get_agents should raise 403 when agents are disabled for internal users."""
    from litellm.proxy.agent_endpoints.endpoints import get_agents

    user = _make_internal_user()
    gs = {
        "disable_agents_for_internal_users": True,
        "allow_agents_for_team_admins": False,
    }

    request_mock = MagicMock()
    with patch.dict("litellm.proxy.proxy_server.general_settings", gs, clear=True):
        with pytest.raises(HTTPException) as exc_info:
            await get_agents(request=request_mock, user_api_key_dict=user)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_get_agents_allowed_when_not_disabled():
    """get_agents should not raise RBAC 403 when agents are not disabled."""
    from litellm.proxy.agent_endpoints.endpoints import get_agents

    user = _make_internal_user()
    request_mock = MagicMock()

    with patch.dict("litellm.proxy.proxy_server.general_settings", {}, clear=True):
        with patch(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
            MagicMock(get_agent_list=MagicMock(return_value=[])),
        ):
            with patch(
                "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.get_allowed_agents",
                new=AsyncMock(return_value=[]),
            ):
                result = await get_agents(request=request_mock, user_api_key_dict=user)
    assert result == []


# ---------------------------------------------------------------------------
# get_agent_daily_activity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_agent_daily_activity_blocked_when_disabled():
    from litellm.proxy.agent_endpoints.endpoints import get_agent_daily_activity

    user = _make_internal_user()
    gs = {
        "disable_agents_for_internal_users": True,
        "allow_agents_for_team_admins": False,
    }

    with patch.dict("litellm.proxy.proxy_server.general_settings", gs, clear=True):
        with pytest.raises(HTTPException) as exc_info:
            await get_agent_daily_activity(user_api_key_dict=user)
    assert exc_info.value.status_code == 403
