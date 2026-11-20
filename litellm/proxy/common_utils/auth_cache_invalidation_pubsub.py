import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.user_api_key_cache import AUTH_OBJECTS_TARGET

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.redis.stores.auth_cache_invalidation import AuthCacheInvalidation

_BULK_EVICTION_WAIT_SECONDS: Final = 2.0


def _channel() -> "AuthCacheInvalidation | None":
    from litellm.proxy.proxy_server import proxy_config

    return proxy_config.auth_cache_invalidation


async def publish_auth_cache_invalidation(
    cache_key: str, new_value: float | None = None, ttl: float | None = None
) -> None:
    """
    Best-effort broadcast so every worker drops its local in-memory copy of a
    mutated management object; without this, only the handling worker and Redis
    are evicted and other workers keep serving the stale object until its TTL.

    Passing ``new_value`` broadcasts a SET instead of a delete: every subscriber
    (including the publishing worker's own, which receives its own message)
    writes the value instead of deleting the key. A spend reset uses this so the
    handler's self-delivered message cannot erase the freshly-written post-reset
    counter or floor marker.

    The publish runs natively in the background, so a Redis that accepts
    connections but never replies costs the caller nothing.
    """
    channel: Final = _channel()
    if channel is not None:
        _ = channel.publish(cache_key, new_value, ttl)


async def broadcast_evictions(cache_keys: Sequence[str]) -> None:
    """
    Tell every other worker to drop these keys, in one pipelined Redis round trip.

    For bulk callers that already dropped the keys here and from Redis (``evict_local`` and
    ``evict_shared``). The caller waits for the round trip at most two seconds, so a Redis that
    never answers cannot hang the request.
    """
    channel: Final = _channel()
    if channel is not None:
        await channel.broadcast_evictions(cache_keys)


def evict_local(cache_keys: Sequence[str], user_api_key_cache: "UserApiKeyCache") -> None:
    """
    Drop cached objects from this worker's memory alone, before any Redis round trip, so a deleted
    key stops authenticating on the handling worker the instant its rows are gone, whatever Redis
    is doing.
    """
    for cache_key in cache_keys:
        user_api_key_cache.delete_local(cache_key)


@with_service_target(AUTH_OBJECTS_TARGET)
async def evict_shared(cache_keys: Sequence[str], cache: "UserApiKeyCache | DualCache") -> None:
    """
    Drop cached objects from the cache's memory and, when it has one, its Redis layer, in one
    chunked round trip.

    Runs right after ``evict_local``: with ``enable_redis_auth_cache`` on, a miss on this worker
    refills from Redis, so an entry dropped from memory alone comes straight back. Best-effort and
    bounded like ``broadcast_evictions``: the rows are already gone, so a Redis that errors or never
    answers must not fail the caller.
    """
    if not cache_keys:
        return
    try:
        await asyncio.wait_for(cache.async_delete_cache_keys(cache_keys), timeout=_BULK_EVICTION_WAIT_SECONDS)
    except Exception as e:  # noqa: BLE001  # best-effort: a Redis error must not fail a committed write
        verbose_proxy_logger.warning(
            "Failed to drop %d cached entries from Redis; stale objects may be served until their TTL expires: %s",
            len(cache_keys),
            e,
        )


async def evict_and_broadcast(cache_keys: Sequence[str], user_api_key_cache: "UserApiKeyCache") -> None:
    """
    Drop cached management objects here and on every other worker.

    Every endpoint that mutates a cached object must call this: auth serves those objects
    cache-first with no freshness check, so a mutation that leaves the entry in place keeps the
    stale object enforced until its TTL expires (LIT-3803). Best-effort: the DB write has already
    committed, so a cache backend error must not fail the endpoint.
    """
    await _evict_and_broadcast(cache_keys, user_api_key_cache)


async def evict_state_and_broadcast(cache_keys: Sequence[str], cache: "DualCache") -> None:
    await _evict_and_broadcast(cache_keys, cache)


@with_service_target(AUTH_OBJECTS_TARGET)
async def _evict_and_broadcast(cache_keys: Sequence[str], cache: "UserApiKeyCache | DualCache") -> None:
    for cache_key in cache_keys:
        try:
            await cache.async_delete_cache(key=cache_key)
        except Exception as e:  # noqa: BLE001  # best-effort eviction: any cache backend error must not fail the mutation
            verbose_proxy_logger.warning(
                "Failed to evict cached entry %s; a stale object may be served until its TTL expires: %s",
                cache_key,
                e,
            )
        await publish_auth_cache_invalidation(cache_key=cache_key)
