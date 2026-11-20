from typing import Final

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.redis.stores.auth_cache_invalidation import PythonCaches
from litellm.rust_bridge.host import auth_cache_invalidation


def test_a_broadcast_value_reaches_the_given_copies_only() -> None:
    copy: Final = InMemoryCache()
    evicted: Final = InMemoryCache()
    evicted.set_cache("spend:key:k-1", "hook state")

    auth_cache_invalidation.apply(PythonCaches(copies=(copy,), evicted=(evicted,)), "spend:key:k-1", 2.5, 60.0)

    assert copy.get_cache("spend:key:k-1") == 2.5
    assert evicted.get_cache("spend:key:k-1") == "hook state"


def test_a_broadcast_eviction_drops_the_key_from_every_given_cache() -> None:
    copy: Final = InMemoryCache()
    evicted: Final = InMemoryCache()
    for cache in (copy, evicted):
        cache.set_cache("team_id:t-1", "stale")

    auth_cache_invalidation.apply(PythonCaches(copies=(copy,), evicted=(evicted,)), "team_id:t-1", None, None)

    assert [copy.get_cache("team_id:t-1"), evicted.get_cache("team_id:t-1")] == [None, None]
