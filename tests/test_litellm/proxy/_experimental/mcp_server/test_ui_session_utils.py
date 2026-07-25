from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import UserAPIKeyAuth

from litellm.proxy._experimental.mcp_server.ui_session_utils import (
    build_effective_auth_contexts,
    clone_user_api_key_auth_with_team,
    resolve_ui_session_team_ids,
)


def test_clone_user_api_key_auth_with_team_creates_independent_copy():
    original = UserAPIKeyAuth(team_id="team-original", user_id="user-123")

    cloned = clone_user_api_key_auth_with_team(original, "team-override")

    assert cloned is not original
    assert cloned.team_id == "team-override"
    assert original.team_id == "team-original"


@pytest.mark.asyncio
async def test_resolve_ui_session_team_ids_returns_unique_ids(monkeypatch):
    user_auth = UserAPIKeyAuth(
        team_id=UI_SESSION_TOKEN_TEAM_ID,
        user_id="user-1",
    )

    fake_user = SimpleNamespace(teams=["team-a", "team-b", "team-a", "", None, "team-c"])

    get_user_object = AsyncMock(return_value=fake_user)
    monkeypatch.setattr(
        "litellm.proxy.auth.auth_checks.get_user_object",
        get_user_object,
    )

    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", None)

    team_ids = await resolve_ui_session_team_ids(user_auth)

    assert team_ids == ["team-a", "team-b", "team-c"]
    get_user_object.assert_awaited_once()
    assert get_user_object.await_args.kwargs["check_db_only"] is True


@pytest.mark.asyncio
async def test_resolve_ui_session_team_ids_observes_next_call_revocation(monkeypatch):
    user_auth = UserAPIKeyAuth(
        team_id=UI_SESSION_TOKEN_TEAM_ID,
        user_id="user-revoked",
    )

    get_user_object = AsyncMock(
        side_effect=[
            SimpleNamespace(teams=["team-a"]),
            SimpleNamespace(teams=[]),
        ]
    )
    monkeypatch.setattr(
        "litellm.proxy.auth.auth_checks.get_user_object",
        get_user_object,
    )

    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", None)

    assert await resolve_ui_session_team_ids(user_auth) == ["team-a"]
    assert await resolve_ui_session_team_ids(user_auth) == []
    assert get_user_object.await_count == 2
    assert all(call.kwargs["check_db_only"] is True for call in get_user_object.await_args_list)


@pytest.mark.asyncio
async def test_resolve_ui_session_team_ids_short_circuits_when_not_ui_session():
    normal_user = UserAPIKeyAuth(team_id="regular-team", user_id="user-1")

    result = await resolve_ui_session_team_ids(normal_user)

    assert result == []


@pytest.mark.asyncio
async def test_build_effective_auth_contexts_reloads_ui_session_as_admitted_user(
    monkeypatch,
):
    user_auth = UserAPIKeyAuth(team_id=UI_SESSION_TOKEN_TEAM_ID, user_id="user-42")
    admitted = UserAPIKeyAuth(user_id="user-42")
    admitted.mcp_admitted_user_subject = True

    reload_user = AsyncMock(return_value=admitted)
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._reload_admitted_user",
        reload_user,
    )

    contexts = await build_effective_auth_contexts(user_auth)

    assert contexts == [admitted]
    assert contexts[0].mcp_admitted_user_subject is True
    reload_user.assert_awaited_once_with("user-42")


@pytest.mark.asyncio
async def test_build_effective_auth_contexts_returns_non_ui_context_unchanged(
    monkeypatch,
):
    user_auth = UserAPIKeyAuth(team_id="existing-team", user_id="user-7")

    contexts = await build_effective_auth_contexts(user_auth)

    assert contexts == [user_auth]


@pytest.mark.asyncio
async def test_build_effective_auth_contexts_propagates_reload_failure_closed(
    monkeypatch,
):
    user_auth = UserAPIKeyAuth(
        team_id=UI_SESSION_TOKEN_TEAM_ID,
        user_id="offboarded-user",
    )

    reload_user = AsyncMock(side_effect=RuntimeError("live user unavailable"))
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._reload_admitted_user",
        reload_user,
    )

    with pytest.raises(RuntimeError, match="live user unavailable"):
        await build_effective_auth_contexts(user_auth)
