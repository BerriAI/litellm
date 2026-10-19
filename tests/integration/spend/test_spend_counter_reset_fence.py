"""The reset fence: a spend reset SETs it in the same pipeline as the zeroed counter, and the stale-floor
repair's Lua script leaves a fenced counter alone, so a pod still holding the pre-reset floor cannot raise
the counter the reset zeroed for the whole counter TTL."""

import os
import uuid
from typing import Final

from integration._support.spend_counters import rig_cache

from litellm.caching.redis_cache import RedisCache


async def test_set_max_holds_while_the_reset_fence_exists() -> None:
    cache: Final = RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), namespace="fence")
    counter: Final = f"spend:key:{uuid.uuid4().hex}"
    fence: Final = cache.reset_fence_key(counter)
    assert await cache.async_set_cache_pipeline_with_ttls(((fence, 1, 10), (counter, 0.0, 3600))) is None

    assert await cache.async_set_max(key=counter, value=12.0, ttl=3600, fence_key=fence) == 0.0

    with rig_cache() as raw:
        assert raw.delete(cache.check_and_fix_namespace(key=fence)) == 1
    assert await cache.async_set_max(key=counter, value=12.0, ttl=3600, fence_key=fence) == 12.0
    with rig_cache() as raw:
        raw.delete(cache.check_and_fix_namespace(key=counter))
