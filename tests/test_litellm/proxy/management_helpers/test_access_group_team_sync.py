
import pytest


from litellm.proxy.management_helpers.access_group_team_sync import (
    invalidate_access_group_caches,
)


@pytest.mark.asyncio
async def test_one_unreachable_cache_does_not_skip_the_other_groups(monkeypatch):
    """
    `assigned_team_ids` is an authorization input, so a group whose cache still holds the
    revoked grant keeps serving it until the entry is dropped.

    A sequential loop would stop at the first failing group and leave the groups behind it
    serving stale grants, and swallowing the failure would report success to the admin for
    a revoke that never took effect. Every group has to be attempted, and the endpoint has
    to fail so the caller can retry.
    """
    attempted: list[str] = []

    async def _invalidate(access_group_id: str) -> None:
        attempted.append(access_group_id)
        if access_group_id == "ag-redis-down":
            raise ConnectionError("redis unreachable")

    monkeypatch.setattr(
        "litellm.proxy.management_helpers.access_group_team_sync.invalidate_access_group_cache",
        _invalidate,
    )

    with pytest.raises(ConnectionError):
        await invalidate_access_group_caches(("ag-redis-down", "ag-2", "ag-3"))

    assert attempted == ["ag-redis-down", "ag-2", "ag-3"]
