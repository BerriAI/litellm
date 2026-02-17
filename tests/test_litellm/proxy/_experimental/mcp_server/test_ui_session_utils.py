import threading
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

    fake_user = SimpleNamespace(
        teams=["team-a", "team-b", "team-a", "", None, "team-c"]
    )

    monkeypatch.setattr(
        "litellm.proxy.auth.auth_checks.get_user_object",
        AsyncMock(return_value=fake_user),
    )

    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", object())
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", None)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", None)

    team_ids = await resolve_ui_session_team_ids(user_auth)

    assert team_ids == ["team-a", "team-b", "team-c"]


@pytest.mark.asyncio
async def test_resolve_ui_session_team_ids_short_circuits_when_not_ui_session():
    normal_user = UserAPIKeyAuth(team_id="regular-team", user_id="user-1")

    result = await resolve_ui_session_team_ids(normal_user)

    assert result == []


@pytest.mark.asyncio
async def test_build_effective_auth_contexts_returns_cloned_contexts(monkeypatch):
    user_auth = UserAPIKeyAuth(team_id=UI_SESSION_TOKEN_TEAM_ID, user_id="user-42")

    mock_resolve = AsyncMock(return_value=["team-one", "team-two"])
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.ui_session_utils.resolve_ui_session_team_ids",
        mock_resolve,
    )

    contexts = await build_effective_auth_contexts(user_auth)

    assert [ctx.team_id for ctx in contexts] == ["team-one", "team-two"]
    assert all(ctx is not user_auth for ctx in contexts)
    mock_resolve.assert_awaited_once_with(user_auth)


@pytest.mark.asyncio
async def test_build_effective_auth_contexts_returns_original_when_no_resolution(monkeypatch):
    user_auth = UserAPIKeyAuth(team_id="existing-team", user_id="user-7")

    mock_resolve = AsyncMock(return_value=[])
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.ui_session_utils.resolve_ui_session_team_ids",
        mock_resolve,
    )

    contexts = await build_effective_auth_contexts(user_auth)

    assert contexts == [user_auth]
    mock_resolve.assert_awaited_once_with(user_auth)


@pytest.mark.asyncio
async def test_build_effective_auth_contexts_handles_unpicklable_parent_span(monkeypatch):
    class DummySpan:
        def __init__(self) -> None:
            self._lock = threading.RLock()

    parent_span = DummySpan()
    user_auth = UserAPIKeyAuth(
        team_id=UI_SESSION_TOKEN_TEAM_ID,
        user_id="user-span",
        parent_otel_span=parent_span,
    )

    mock_resolve = AsyncMock(return_value=["team-span"])
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.ui_session_utils.resolve_ui_session_team_ids",
        mock_resolve,
    )

    contexts = await build_effective_auth_contexts(user_auth)

    assert contexts[0].team_id == "team-span"
    assert contexts[0].parent_otel_span is parent_span
