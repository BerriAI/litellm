from datetime import timedelta
from typing import Final

import fakeredis
import pytest
from redis.exceptions import ResponseError

from litellm.redis.stores.mcp_credential_refresh import RefreshLockTarget, TransportMcpRefreshLockStore
from tests._support.fake_redis import fake_redis_transport

TARGET: Final = RefreshLockTarget(user_id="u", server_id="s")


class _RedisWithoutEvalsha(fakeredis.FakeAsyncRedis):
    async def evalsha(self, sha: str, numkeys: int, *keys_and_args: str | bytes | float) -> object:
        raise ResponseError("NOPERM this user has no permissions to run the 'evalsha' command")


@pytest.mark.asyncio
async def test_extend_and_release_honor_the_owner_with_plain_eval_when_evalsha_is_denied() -> None:
    raw: Final = _RedisWithoutEvalsha(server=fakeredis.FakeServer())
    store: Final = TransportMcpRefreshLockStore(fake_redis_transport(raw, failure_threshold=1, namespace="ns"))
    assert await store.acquire(TARGET, "owner-a", ttl=timedelta(seconds=30)) is True
    lock_key: Final = "ns:mcp:refresh_lock:u:s"

    assert await store.extend(TARGET, "owner-b", ttl=timedelta(seconds=300)) is False
    assert await store.release(TARGET, "owner-b") is False
    assert await raw.get(lock_key) == b"owner-a"
    assert await raw.pttl(lock_key) <= 30_000

    assert await store.extend(TARGET, "owner-a", ttl=timedelta(seconds=300)) is True
    assert 30_000 < await raw.pttl(lock_key) <= 300_000
    assert await store.release(TARGET, "owner-a") is True
    assert await raw.get(lock_key) is None
