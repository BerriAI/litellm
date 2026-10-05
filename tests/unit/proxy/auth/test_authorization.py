from typing import Final

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.authorization import OwnedRows, resolve_owned_read_scope, resolve_trace_read_scope


@pytest.mark.asyncio
@pytest.mark.parametrize("token", (None, "key"))
async def test_team_membership_or_key_without_user_does_not_grant_log_access(token: str | None) -> None:
    async def unexpected_lookup() -> tuple[str, ...]:
        pytest.fail("Identity-less callers cannot consult team permissions")

    assert await resolve_trace_read_scope(UserAPIKeyAuth(team_id="team", token=token), unexpected_lookup) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("token", (None, "key"))
@pytest.mark.parametrize("lookup_fails", (False, True))
async def test_trace_reads_share_user_and_team_scope_regardless_of_key(token: str | None, lookup_fails: bool) -> None:
    async def lookup() -> tuple[str, ...]:
        if lookup_fails:
            raise RuntimeError("team lookup failed")
        return ("permitted",)

    expected: Final = OwnedRows("caller", () if lookup_fails else ("permitted",))
    assert await resolve_owned_read_scope("caller", lookup) == expected
    assert await resolve_trace_read_scope(UserAPIKeyAuth(user_id="caller", token=token), lookup) == expected
