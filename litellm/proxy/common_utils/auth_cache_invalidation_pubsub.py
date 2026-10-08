import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.config_sync_pubsub import (  # noqa: F401  # legacy module exports
    ConfigSyncPubSub,
    _ConfigSyncPubSub,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    _pubsub_capable_client,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    coordination_redis_cache,
    pubsub_capable_client,
)
from litellm.proxy.common_utils.user_api_key_cache import AUTH_OBJECTS_TARGET

if TYPE_CHECKING:
    from litellm.caching.in_memory_cache import InMemoryCache
    from litellm.caching.redis_cache import RedisCache
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

AUTH_CACHE_INVALIDATION_CHANNEL: Final = "litellm_proxy.auth_cache_invalidation"
_POLL_TIMEOUT_SECONDS: Final = 1.0
_MAX_PENDING_PUBLISHES: Final = 1024
_MAX_IN_FLIGHT_PUBLISHES: Final = 16
PUBLISH_BACKLOG_SLICE: Final = 256
_PUBLISH_BACKLOG_WAIT_SECONDS: Final = 2.0
_pending_publishes: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: strong refs keep background publishes alive
_in_flight_publishes: Final = asyncio.Semaphore(_MAX_IN_FLIGHT_PUBLISHES)
_BACKOFF_INITIAL_SECONDS: Final = 5.0
_BACKOFF_MAX_SECONDS: Final = 60.0


def auth_cache_invalidation_channel(redis_cache: "RedisCache") -> str:
    if redis_cache.namespace is None:
        return AUTH_CACHE_INVALIDATION_CHANNEL
    return f"{redis_cache.namespace}:{AUTH_CACHE_INVALIDATION_CHANNEL}"


@dataclass(frozen=True, slots=True)
class _CacheInvalidationMessage:
    cache_key: str
    new_value: float | None = None
    ttl: float | None = None


def _cache_invalidation_message_json(cache_key: str, new_value: float | None = None, ttl: float | None = None) -> str:
    message: Final = asdict(_CacheInvalidationMessage(cache_key=cache_key, new_value=new_value, ttl=ttl))
    return json.dumps({field: value for field, value in message.items() if value is not None})


def _finite_number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _message_from_data(data: object) -> _CacheInvalidationMessage | None:
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="replace")  # rebind-ok: normalizing the wire payload to str
    if not isinstance(data, str):
        return None
    try:
        parsed: Final = json.loads(data)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    cache_key: Final = parsed.get("cache_key")
    if not isinstance(cache_key, str):
        return None
    return _CacheInvalidationMessage(
        cache_key=cache_key,
        new_value=_finite_number_or_none(parsed.get("new_value")),
        ttl=_finite_number_or_none(parsed.get("ttl")),
    )


async def _publish_to_redis(redis_cache: "RedisCache", cache_key: str, message: str) -> None:
    try:
        client: Final = pubsub_capable_client(redis_cache)
        async with _in_flight_publishes:
            await client.publish(auth_cache_invalidation_channel(redis_cache), message)
    except Exception as e:  # noqa: BLE001  # best-effort publish; mutations must never fail on redis errors
        verbose_proxy_logger.warning("auth cache invalidation publish for %s failed: %s", cache_key, e)


async def publish_auth_cache_invalidation(
    cache_key: str, new_value: float | None = None, ttl: float | None = None
) -> None:
    """
    Best-effort broadcast so every worker drops its local in-memory copy of a
    mutated management object; without this, only the handling worker and Redis
    are evicted and other workers keep serving the stale object until its TTL.

    Passing ``new_value`` broadcasts a SET instead of a delete: every subscriber
    (including the publishing worker's own, which receives its own message)
    writes the value into its additional in-memory caches rather than deleting
    the key. A spend reset uses this so the handler's self-delivered message
    cannot erase the freshly-written post-reset counter or floor marker.

    The Redis round trip runs as a background task: this call returns once the
    publish has been handed to the event loop, so a Redis that accepts
    connections but never replies costs the caller nothing. The DB write has
    already committed and the local eviction already happened, so the caller
    has nothing to do with the publish result. At most 16 publishes hold a
    Redis connection at once; the rest wait in the task set, so a wedge cannot
    drain the shared connection pool.
    """
    redis_cache: Final = coordination_redis_cache()
    if redis_cache is None:
        return
    _pending_publishes.difference_update({task for task in _pending_publishes if task.done()})
    if len(_pending_publishes) >= _MAX_PENDING_PUBLISHES:
        verbose_proxy_logger.warning(
            "auth cache invalidation publish for %s dropped: %d publishes already waiting on redis; "
            "other workers keep their cached copy until its TTL expires",
            cache_key,
            len(_pending_publishes),
        )
        return
    task: Final = asyncio.create_task(
        _publish_to_redis(
            redis_cache, cache_key, _cache_invalidation_message_json(cache_key, new_value=new_value, ttl=ttl)
        )
    )
    _pending_publishes.add(task)
    await asyncio.sleep(0)


async def await_publish_backlog() -> None:
    """
    Wait, bounded, for the publishes already handed to the event loop to reach Redis.

    A caller broadcasting more than ``_MAX_PENDING_PUBLISHES`` evictions in one burst would see the
    excess dropped above, so a bulk eviction publishes in slices of ``PUBLISH_BACKLOG_SLICE`` and
    waits here before each slice. The wait is capped so a Redis that never answers costs a bulk
    caller at most ``_PUBLISH_BACKLOG_WAIT_SECONDS`` per slice instead of hanging the request.
    """
    pending: Final = frozenset(task for task in _pending_publishes if not task.done())
    if not pending:
        return
    await asyncio.wait(pending, timeout=_PUBLISH_BACKLOG_WAIT_SECONDS)


@with_service_target(AUTH_OBJECTS_TARGET)
def evict_local(cache_keys: Sequence[str], user_api_key_cache: "UserApiKeyCache") -> None:
    """
    Drop cached objects from this worker's memory alone, ahead of a paced broadcast.

    A bulk eviction waits for the publish backlog between its slices, so every local copy is
    dropped up front: a deleted key must stop authenticating on the handling worker the instant
    its rows are gone, whatever Redis is doing.
    """
    for cache_key in cache_keys:
        user_api_key_cache.in_memory_cache_for(cache_key).delete_cache(cache_key)


@with_service_target(AUTH_OBJECTS_TARGET)
async def evict_shared(cache_keys: Sequence[str], user_api_key_cache: "UserApiKeyCache") -> None:
    """
    Drop cached objects from the cache's Redis layer, when it has one, in one chunked round trip.

    Runs right after ``evict_local``: with ``enable_redis_auth_cache`` on, a miss on this worker
    refills from Redis, so an entry dropped from memory alone comes straight back until its own
    slice of the paced broadcast reaches Redis. Best-effort and bounded like the backlog wait: the
    rows are already gone, so a Redis that errors or never answers must not fail the caller, and
    every slice still deletes its own keys behind this.
    """
    if user_api_key_cache.redis_cache is None or not cache_keys:
        return
    try:
        await asyncio.wait_for(
            user_api_key_cache.async_delete_cache_keys(cache_keys), timeout=_PUBLISH_BACKLOG_WAIT_SECONDS
        )
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
    for cache_key in cache_keys:
        try:
            await user_api_key_cache.async_delete_cache(key=cache_key)
        except Exception as e:  # noqa: BLE001  # best-effort eviction: any cache backend error must not fail the mutation
            verbose_proxy_logger.warning(
                "Failed to evict cached entry %s; a stale object may be served until its TTL expires: %s",
                cache_key,
                e,
            )
        await publish_auth_cache_invalidation(cache_key=cache_key)


class AuthCacheInvalidationSubscriber:
    __slots__ = ("_additional_in_memory_caches", "_redis_cache", "_task", "_user_api_key_cache")

    def __init__(
        self,
        redis_cache: "RedisCache",
        user_api_key_cache: "UserApiKeyCache",
        additional_in_memory_caches: Sequence["InMemoryCache"] = (),
    ) -> None:
        self._redis_cache = redis_cache
        self._user_api_key_cache = user_api_key_cache
        self._additional_in_memory_caches = tuple(additional_in_memory_caches)
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        task: Final = self._task
        if task is None:
            return
        self._task = None
        _ = task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        backoff_seconds = _BACKOFF_INITIAL_SECONDS  # rebind-ok: exponential backoff accumulator across reconnects
        while True:
            try:
                client = pubsub_capable_client(self._redis_cache)
                pubsub = client.pubsub()
                try:
                    await pubsub.subscribe(auth_cache_invalidation_channel(self._redis_cache))
                    backoff_seconds = _BACKOFF_INITIAL_SECONDS
                    await self._consume(pubsub)
                finally:
                    await self._close_pubsub(pubsub)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001  # any redis failure falls through to backoff and reconnect
                verbose_proxy_logger.warning(
                    "auth cache invalidation subscriber redis error: %s; reconnecting in %.0fs",
                    e,
                    backoff_seconds,
                )
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, _BACKOFF_MAX_SECONDS)

    async def _consume(self, pubsub: ConfigSyncPubSub) -> None:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=_POLL_TIMEOUT_SECONDS)
            if message is None:
                continue
            self._apply_message(message)

    @with_service_target(AUTH_OBJECTS_TARGET)
    def _apply_message(self, message: object) -> None:
        data: Final = message.get("data") if isinstance(message, dict) else None
        parsed: Final = _message_from_data(data)
        if parsed is None:
            return
        if parsed.new_value is not None:
            for additional_cache in self._additional_in_memory_caches:
                additional_cache.set_cache(parsed.cache_key, parsed.new_value, ttl=parsed.ttl)
            return
        self._user_api_key_cache.in_memory_cache_for(parsed.cache_key).delete_cache(parsed.cache_key)
        for additional_cache in self._additional_in_memory_caches:
            additional_cache.delete_cache(parsed.cache_key)

    @staticmethod
    async def _close_pubsub(pubsub: ConfigSyncPubSub) -> None:
        try:
            await pubsub.aclose()
        except Exception as e:  # noqa: BLE001  # best-effort close of a possibly-broken connection
            verbose_proxy_logger.debug("auth cache invalidation pubsub close failed: %s", e)
