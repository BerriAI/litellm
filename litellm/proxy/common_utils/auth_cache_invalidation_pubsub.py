import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_cache import (
    pubsub_capable_client as _pubsub_capable_client,  # noqa: F401  # pyright: ignore[reportDeprecated]  # compatibility-only re-export
)
from litellm.caching.redis_cache import (
    pubsub_capable_client as pubsub_capable_client,  # noqa: PLC0414  # pyright: ignore[reportDeprecated]  # compatibility-only re-export
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.proxy.common_utils.config_sync_pubsub import (
    ConfigSyncPubSub as ConfigSyncPubSub,  # noqa: PLC0414  # backwards-compatible package export
)
from litellm.proxy.common_utils.config_sync_pubsub import (
    _ConfigSyncPubSub as _ConfigSyncPubSub,  # noqa: PLC0414  # pyright: ignore[reportPrivateUsage]  # backwards-compatible package export
)
from litellm.proxy.common_utils.config_sync_pubsub import coordination_redis_cache
from litellm.proxy.common_utils.user_api_key_cache import AUTH_OBJECTS_TARGET
from litellm.types.caching import CacheFailure, CachePublish, CacheSubscription

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.caching.in_memory_cache import InMemoryCache
    from litellm.caching.redis_cache import RedisCache
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

AUTH_CACHE_INVALIDATION_CHANNEL: Final = "litellm_proxy.auth_cache_invalidation"
_POLL_TIMEOUT_SECONDS: Final = 1.0
_MAX_PENDING_PUBLISHES: Final = 1024
_MAX_IN_FLIGHT_PUBLISHES: Final = 16
_BULK_EVICTION_WAIT_SECONDS: Final = 2.0
_pending_publishes: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: strong refs keep background publishes alive
_in_flight_publishes: Final = asyncio.Semaphore(_MAX_IN_FLIGHT_PUBLISHES)
_BACKOFF_INITIAL_SECONDS: Final = 5.0
_BACKOFF_MAX_SECONDS: Final = 60.0
_DECODED_JSON: Final = TypeAdapter(object)


def auth_cache_invalidation_channel(redis_cache: "RedisCache") -> str:
    return AUTH_CACHE_INVALIDATION_CHANNEL if redis_cache.namespace is None else f"{redis_cache.namespace}:{AUTH_CACHE_INVALIDATION_CHANNEL}"


@dataclass(frozen=True, slots=True)
class _CacheInvalidationMessage:
    cache_key: str
    new_value: float | None = None
    ttl: float | None = None


def _cache_invalidation_message_json(cache_key: str, new_value: float | None = None, ttl: float | None = None) -> str:
    message: Final[Mapping[str, object]] = MappingProxyType(
        asdict(_CacheInvalidationMessage(cache_key=cache_key, new_value=new_value, ttl=ttl))
    )
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
        parsed: Final = as_str_mapping(_DECODED_JSON.validate_python(json.loads(data)))
    except json.JSONDecodeError:
        return None
    if parsed is None:
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
        async with _in_flight_publishes:
            result: Final = await redis_cache.async_publish(channel=AUTH_CACHE_INVALIDATION_CHANNEL, message=message)
        if isinstance(result, CacheFailure):
            verbose_proxy_logger.warning("auth cache invalidation publish for %s failed: %s", cache_key, result.message)
    except Exception as e:  # noqa: BLE001  # best-effort publish; mutations must never fail on redis errors
        verbose_proxy_logger.warning("auth cache invalidation publish for %s failed: %s", cache_key, e)


async def _publish_batch_to_redis(redis_cache: "RedisCache", cache_keys: tuple[str, ...]) -> None:
    try:
        messages: Final = tuple(
            CachePublish(channel=AUTH_CACHE_INVALIDATION_CHANNEL, message=_cache_invalidation_message_json(cache_key))
            for cache_key in cache_keys
        )
        async with _in_flight_publishes:
            result: Final = await redis_cache.async_publish_many(messages=messages)
        if isinstance(result, CacheFailure):
            verbose_proxy_logger.warning(
                "auth cache invalidation publish for %d keys failed: %s", len(cache_keys), result.message
            )
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
    task: Final = asyncio.create_task(
        _publish_to_redis(
            redis_cache, cache_key, _cache_invalidation_message_json(cache_key, new_value=new_value, ttl=ttl)
        )
    )
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
    __slots__ = ("_additional_in_memory_caches", "_backoff_seconds", "_redis_cache", "_task", "_user_api_key_cache")

    def __init__(
        self,
        redis_cache: "RedisCache",
        user_api_key_cache: "UserApiKeyCache",
        additional_in_memory_caches: Sequence["InMemoryCache"] = (),
    ) -> None:
        self._redis_cache = redis_cache
        self._user_api_key_cache = user_api_key_cache
        self._additional_in_memory_caches = tuple(additional_in_memory_caches)
        self._backoff_seconds = _BACKOFF_INITIAL_SECONDS
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
        self._backoff_seconds = _BACKOFF_INITIAL_SECONDS
        while failure := await self._subscribe_and_consume():
            verbose_proxy_logger.warning(
                "auth cache invalidation subscriber redis error: %s; reconnecting in %.0fs",
                failure.message,
                self._backoff_seconds,
            )
            await asyncio.sleep(self._backoff_seconds)
            self._backoff_seconds = min(self._backoff_seconds * 2, _BACKOFF_MAX_SECONDS)

    async def _subscribe_and_consume(self) -> CacheFailure:
        try:
            subscription: Final = await self._redis_cache.async_subscribe(channel=AUTH_CACHE_INVALIDATION_CHANNEL)
            if isinstance(subscription, CacheFailure):
                return subscription
            try:
                self._backoff_seconds = _BACKOFF_INITIAL_SECONDS
                return await self._consume(subscription)
            finally:
                await self._close_pubsub(subscription)
        except Exception as error:  # noqa: BLE001  # custom cache subclasses and eviction errors must not stop reconnects
            return CacheFailure("unavailable", str(error))

    async def _consume(self, subscription: CacheSubscription) -> CacheFailure:
        while not isinstance(message := await subscription.receive(timeout_seconds=_POLL_TIMEOUT_SECONDS), CacheFailure):
            if message is None:
                continue
            self._apply_message(message)
        return message

    @with_service_target(AUTH_OBJECTS_TARGET)
    def _apply_message(self, message: str | bytes) -> None:
        parsed: Final = _message_from_data(message)
        if parsed is None:
            return
        if parsed.new_value is not None:
            for additional_cache in self._additional_in_memory_caches:
                additional_cache.set_cache(parsed.cache_key, parsed.new_value, ttl=parsed.ttl)
            return
        _drop_from_memory(parsed.cache_key, self._user_api_key_cache)
        for additional_cache in self._additional_in_memory_caches:
            additional_cache.delete_cache(parsed.cache_key)

    @staticmethod
    async def _close_pubsub(subscription: CacheSubscription) -> None:
        try:
            failure: Final = await subscription.close()
            if failure is not None:
                verbose_proxy_logger.debug("auth cache invalidation pubsub close failed: %s", failure.message)
        except Exception as e:  # noqa: BLE001  # custom subscriptions must not interrupt reconnect or cancellation
            verbose_proxy_logger.debug("auth cache invalidation pubsub close failed: %s", e)
