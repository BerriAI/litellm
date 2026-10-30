import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, TypeAlias

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils import config_sync_pubsub
from litellm.proxy.common_utils.config_sync_pubsub import coordination_redis_cache
from litellm.proxy.common_utils.user_api_key_cache import AUTH_OBJECTS_TARGET
from litellm.redis.stores import auth_invalidation
from litellm.redis.stores.auth_invalidation import (
    AuthInvalidation,
    AuthInvalidationStore,
    AuthInvalidationSubscription,
    TransportAuthInvalidationStore,
    auth_invalidation_channel,
    decode_auth_message,
    encode_auth_invalidation,
    finite_number_or_none,
)
from litellm.redis.transport.contract import RedisFailure

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.caching.in_memory_cache import InMemoryCache
    from litellm.caching.redis_cache import RedisCache
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

AUTH_CACHE_INVALIDATION_CHANNEL: Final = auth_invalidation.AUTH_CACHE_INVALIDATION_CHANNEL
_POLL_TIMEOUT_SECONDS: Final = 1.0
_MAX_PENDING_PUBLISHES: Final = 1024
_MAX_IN_FLIGHT_PUBLISHES: Final = 16
_BULK_EVICTION_WAIT_SECONDS: Final = 2.0
_pending_publishes: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: strong refs keep background publishes alive
_in_flight_publishes: Final = asyncio.Semaphore(_MAX_IN_FLIGHT_PUBLISHES)
_BACKOFF_INITIAL_SECONDS: Final = 5.0
_BACKOFF_MAX_SECONDS: Final = 60.0


def auth_cache_invalidation_channel(redis_cache: "RedisCache") -> str:
    return auth_invalidation_channel(redis_cache.namespace)


_CacheInvalidationMessage: TypeAlias = AuthInvalidation
ConfigSyncPubSub: TypeAlias = config_sync_pubsub.ConfigSyncPubSub
_ConfigSyncPubSub: TypeAlias = ConfigSyncPubSub
pubsub_capable_client: Final = config_sync_pubsub.pubsub_capable_client
_pubsub_capable_client: Final = pubsub_capable_client
_cache_invalidation_message_json: Final = encode_auth_invalidation
_finite_number_or_none: Final = finite_number_or_none


def _invalidation_store(redis_cache: "RedisCache") -> AuthInvalidationStore:
    return TransportAuthInvalidationStore(redis_cache.transport(), empty_namespace=redis_cache.namespace == "")


async def _publish_to_redis(
    redis_cache: "RedisCache", cache_key: str, new_value: float | None, ttl: float | None
) -> None:
    try:
        async with _in_flight_publishes:
            _ = await _invalidation_store(redis_cache).publish_change(cache_key, new_value=new_value, ttl=ttl)
    except Exception as e:  # noqa: BLE001  # best-effort publish; mutations must never fail on redis errors
        verbose_proxy_logger.warning("auth cache invalidation publish for %s failed: %s", cache_key, e)


async def _publish_batch_to_redis(redis_cache: "RedisCache", cache_keys: tuple[str, ...]) -> None:
    try:
        async with _in_flight_publishes:
            _ = await _invalidation_store(redis_cache).publish_evictions(cache_keys)
    except Exception as e:  # noqa: BLE001  # best-effort publish; mutations must never fail on redis errors
        verbose_proxy_logger.warning("auth cache invalidation publish for %d keys failed: %s", len(cache_keys), e)


def _has_room_for_publish(key_count: int) -> bool:
    _pending_publishes.difference_update({task for task in _pending_publishes if task.done()})
    if len(_pending_publishes) < _MAX_PENDING_PUBLISHES:
        return True
    verbose_proxy_logger.warning(
        "auth cache invalidation publish for %d keys dropped: %d publishes already waiting on redis; "
        "other workers keep their cached copy until its TTL expires",
        key_count,
        len(_pending_publishes),
    )
    return False


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
    if redis_cache is None or not _has_room_for_publish(1):
        return
    task: Final = asyncio.create_task(_publish_to_redis(redis_cache, cache_key, new_value=new_value, ttl=ttl))
    _pending_publishes.add(task)
    await asyncio.sleep(0)


@with_service_target(AUTH_OBJECTS_TARGET)
async def broadcast_evictions(cache_keys: Sequence[str]) -> None:
    """
    Tell every other worker to drop these keys, in one pipelined Redis round trip.

    For bulk callers that already dropped the keys here and from Redis (``evict_local`` and
    ``evict_shared``). A publish task per key would pass the pending cap above on a roster of a few
    hundred members, and every dropped key would stay cached on the other workers until its TTL.
    The caller waits for the round trip at most ``_BULK_EVICTION_WAIT_SECONDS``, so a Redis that
    never answers cannot hang the request.
    """
    redis_cache: Final = coordination_redis_cache()
    if redis_cache is None or not cache_keys or not _has_room_for_publish(len(cache_keys)):
        return
    task: Final = asyncio.create_task(_publish_batch_to_redis(redis_cache, tuple(cache_keys)))
    _pending_publishes.add(task)
    _ = await asyncio.wait((task,), timeout=_BULK_EVICTION_WAIT_SECONDS)


def evict_local(cache_keys: Sequence[str], user_api_key_cache: "UserApiKeyCache") -> None:
    """
    Drop cached objects from this worker's memory alone, before any Redis round trip, so a deleted
    key stops authenticating on the handling worker the instant its rows are gone, whatever Redis
    is doing.
    """
    for cache_key in cache_keys:
        _drop_from_memory(cache_key, user_api_key_cache)


def _drop_from_memory(cache_key: str, user_api_key_cache: "UserApiKeyCache") -> None:
    user_api_key_cache.in_memory_cache_for(cache_key).delete_cache(cache_key)


@with_service_target(AUTH_OBJECTS_TARGET)
async def evict_shared(cache_keys: Sequence[str], cache: "DualCache") -> None:
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


@with_service_target(AUTH_OBJECTS_TARGET)
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
                pubsub = await _invalidation_store(self._redis_cache).subscribe()
                if isinstance(pubsub, RedisFailure):
                    raise ConnectionError(pubsub.message)
                try:
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

    async def _consume(self, pubsub: AuthInvalidationSubscription) -> None:
        while True:
            message = await pubsub.receive(timeout=_POLL_TIMEOUT_SECONDS)
            if message is None:
                continue
            if isinstance(message, RedisFailure):
                if message.kind == "decode":
                    continue
                raise ConnectionError(message.message)
            self._apply_message(message)

    @with_service_target(AUTH_OBJECTS_TARGET)
    def _apply_message(self, message: object) -> None:
        parsed: Final = decode_auth_message(message)
        if isinstance(parsed, RedisFailure):
            return
        if parsed.new_value is not None:
            for additional_cache in self._additional_in_memory_caches:
                additional_cache.set_cache(parsed.cache_key, parsed.new_value, ttl=parsed.ttl)
            return
        _drop_from_memory(parsed.cache_key, self._user_api_key_cache)
        for additional_cache in self._additional_in_memory_caches:
            additional_cache.delete_cache(parsed.cache_key)

    @staticmethod
    async def _close_pubsub(pubsub: AuthInvalidationSubscription) -> None:
        try:
            _ = await pubsub.close()
        except Exception as e:  # noqa: BLE001  # best-effort close of a possibly-broken connection
            verbose_proxy_logger.debug("auth cache invalidation pubsub close failed: %s", e)
