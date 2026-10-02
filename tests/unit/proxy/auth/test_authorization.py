import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.authorization import resolve_trace_read_scope


@pytest.mark.asyncio
async def test_team_membership_without_authenticated_identity_does_not_grant_log_access() -> None:
    async def unexpected_lookup() -> tuple[str, ...]:
        pytest.fail("Identity-less callers cannot consult team permissions")

    assert await resolve_trace_read_scope(UserAPIKeyAuth(team_id="team"), unexpected_lookup) is None
