import os
import uuid
from typing import Final

import pytest
from redis import Redis

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.hooks.batch_enqueued_tokens import (
    BatchEnqueuedTokenOverLimit,
    BatchEnqueuedTokenReservation,
    BatchEnqueuedTokenScope,
    BatchEnqueuedTokenStore,
)
from litellm.proxy.utils import InternalUsageCache


@pytest.mark.asyncio
async def test_redis_lua_path_full_lifecycle() -> None:
    redis_host: Final = os.environ["REDIS_HOST"]
    redis_port: Final = int(os.environ["REDIS_PORT"])
    redis_cache: Final = RedisCache(host=redis_host, port=redis_port)
    store: Final = BatchEnqueuedTokenStore(
        internal_usage_cache=InternalUsageCache(DualCache(redis_cache=redis_cache, default_in_memory_ttl=60))
    )
    suffix: Final = uuid.uuid4().hex[:8]
    key_scope: Final = BatchEnqueuedTokenScope(key="api_key", value=f"api_key-{suffix}", limit=100)
    team_scope: Final = BatchEnqueuedTokenScope(key="team", value=f"team-{suffix}", limit=50)
    key_counter: Final = f"batch_enqueued_tokens:api_key:api_key-{suffix}"
    team_counter: Final = f"batch_enqueued_tokens:team:team-{suffix}"
    batch_id: Final = f"batch_{uuid.uuid4().hex}"
    record_key: Final = f"batch_enqueued_token_reservation:{batch_id}"

    try:
        over: Final = await store.reserve(tokens=60, scopes=(key_scope, team_scope))
        assert over == BatchEnqueuedTokenOverLimit(scope=team_scope, enqueued=0)

        reservation: Final = await store.reserve(tokens=50, scopes=(key_scope, team_scope))
        assert isinstance(reservation, BatchEnqueuedTokenReservation)
        assert reservation.backend == "redis"
        with Redis(host=redis_host, port=redis_port) as raw:
            assert int(raw.get(key_counter) or 0) == 50
            assert int(raw.get(team_counter) or 0) == 50

        assert isinstance(await store.reserve(tokens=1, scopes=(key_scope, team_scope)), BatchEnqueuedTokenOverLimit)

        await store.save_reservation(batch_id, reservation)
        popped: Final = await store.pop_reservation(batch_id)
        assert popped == reservation
        assert await store.pop_reservation(batch_id) is None

        await store.refund(popped)
        with Redis(host=redis_host, port=redis_port) as raw:
            assert int(raw.get(key_counter) or 0) == 0
            assert int(raw.get(team_counter) or 0) == 0

        refill: Final = await store.reserve(tokens=50, scopes=(key_scope, team_scope))
        assert isinstance(refill, BatchEnqueuedTokenReservation)
        await store.refund(refill)
    finally:
        with Redis(host=redis_host, port=redis_port) as raw:
            raw.delete(key_counter, team_counter, record_key)
