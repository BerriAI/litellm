from typing import Final

import pytest

from litellm.proxy.auth.authorization import OwnedRows, resolve_owned_read_scope


@pytest.mark.asyncio
async def test_missing_user_does_not_grant_log_access() -> None:
    async def unexpected_lookup() -> tuple[str, ...]:
        pytest.fail("Identity-less callers cannot consult team permissions")

    assert await resolve_owned_read_scope(None, unexpected_lookup) == OwnedRows(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("lookup_fails", (False, True))
async def test_log_reads_share_user_and_team_scope(lookup_fails: bool) -> None:
    async def lookup() -> tuple[str, ...]:
        if lookup_fails:
            raise RuntimeError("team lookup failed")
        return ("permitted",)

    expected: Final = OwnedRows("caller", () if lookup_fails else ("permitted",))
    assert await resolve_owned_read_scope("caller", lookup) == expected
