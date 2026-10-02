from typing import Final

import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.authorization import resolve_trace_read_scope
from litellm.proxy.tracing_endpoints import _trace_scope
from litellm.tracing.types import TraceScope


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auth", "expected"),
    (
        (UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN), TraceScope(all_teams=1, user_id="", team_ids=(), api_key_hash="")),
        (UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY), TraceScope(all_teams=1, user_id="", team_ids=(), api_key_hash="")),
        (
            UserAPIKeyAuth(user_id="user", token="key", team_id="unpermitted"),
            TraceScope(all_teams=0, user_id="user", team_ids=("permitted",), api_key_hash="key"),
        ),
        (UserAPIKeyAuth(token="key", team_id="unpermitted"), TraceScope(all_teams=0, user_id="", team_ids=(), api_key_hash="key")),
    ),
)
async def test_log_visibility_uses_user_and_permitted_teams_instead_of_key_team_membership(
    auth: UserAPIKeyAuth,
    expected: TraceScope,
) -> None:
    async def permitted_teams(caller: UserAPIKeyAuth) -> tuple[str, ...]:
        assert caller is auth
        return ("permitted",)

    scope: Final = await resolve_trace_read_scope(auth, lambda: permitted_teams(auth))
    assert scope is not None
    assert _trace_scope(scope) == expected


@pytest.mark.asyncio
async def test_missing_team_permissions_preserve_authenticated_user_and_key_visibility() -> None:
    async def no_teams(auth: UserAPIKeyAuth) -> tuple[str, ...]:
        return ()

    auth: Final = UserAPIKeyAuth(user_id="user", token="key", team_id="team")
    scope: Final = await resolve_trace_read_scope(auth, lambda: no_teams(auth))
    assert scope is not None
    assert _trace_scope(scope) == TraceScope(all_teams=0, user_id="user", team_ids=(), api_key_hash="key")


@pytest.mark.asyncio
async def test_team_membership_without_authenticated_identity_does_not_grant_log_access() -> None:
    async def unexpected_lookup() -> tuple[str, ...]:
        pytest.fail("Identity-less callers cannot consult team permissions")

    assert await resolve_trace_read_scope(UserAPIKeyAuth(team_id="team"), unexpected_lookup) is None
