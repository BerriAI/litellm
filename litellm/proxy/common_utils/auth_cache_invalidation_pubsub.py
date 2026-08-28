import asyncio
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.config_sync_pubsub import (
    _ConfigSyncPubSub,
    _pubsub_capable_client,
    coordination_redis_cache,
)

if TYPE_CHECKING:
    from litellm.caching.in_memory_cache import InMemoryCache
    from litellm.caching.redis_cache import RedisCache
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

AUTH_CACHE_INVALIDATION_CHANNEL: Final = "litellm_proxy.auth_cache_invalidation"
_POLL_TIMEOUT_SECONDS: Final = 1.0
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
    """
    redis_cache: Final = coordination_redis_cache()
    if redis_cache is None:
        return
    try:
        client: Final = _pubsub_capable_client(redis_cache)
        if client is None:
            verbose_proxy_logger.debug(
                "auth cache invalidation publish for %s skipped: cluster redis client has no pub/sub support",
                cache_key,
            )
            return
        await client.publish(
            auth_cache_invalidation_channel(redis_cache),
            _cache_invalidation_message_json(cache_key, new_value=new_value, ttl=ttl),
        )
    except Exception as e:  # noqa: BLE001  # best-effort publish; mutations must never fail on redis errors
        verbose_proxy_logger.warning("auth cache invalidation publish for %s failed: %s", cache_key, e)


async def evict_and_broadcast(cache_keys: Sequence[str], user_api_key_cache: "UserApiKeyCache") -> None:
    """
    Drop cached management objects here and on every other worker.

    Every endpoint that mutates a cached object must call this: auth serves those objects
    cache-first with no freshness check, so a mutation that leaves the entry in place keeps the
    stale object enforced until its TTL expires (LIT-3803). Best-effort on both steps: the DB write
    has already committed, so a cache backend error must not fail the endpoint.
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
                client = _pubsub_capable_client(self._redis_cache)  # rebind-ok: re-resolved on every reconnect
                if client is None:
                    verbose_proxy_logger.warning(
                        "auth cache invalidation subscriber disabled: cluster redis client has no pub/sub support; "
                        "cross-worker eviction falls back to the local cache TTL"
                    )
                    return
                pubsub = client.pubsub()  # rebind-ok: fresh pubsub per reconnect
                try:
                    await pubsub.subscribe(auth_cache_invalidation_channel(self._redis_cache))
                    backoff_seconds = _BACKOFF_INITIAL_SECONDS  # rebind-ok: reset after successful subscribe
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
                backoff_seconds = min(backoff_seconds * 2, _BACKOFF_MAX_SECONDS)  # rebind-ok: backoff accumulator

    async def _consume(self, pubsub: _ConfigSyncPubSub) -> None:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=_POLL_TIMEOUT_SECONDS)
            if message is None:
                continue
            self._apply_message(message)

    def _apply_message(self, message: object) -> None:
        data: Final = message.get("data") if isinstance(message, dict) else None
        parsed: Final = _message_from_data(data)
        if parsed is None:
            return
        if parsed.new_value is not None:
            for additional_cache in self._additional_in_memory_caches:
                additional_cache.set_cache(parsed.cache_key, parsed.new_value, ttl=parsed.ttl)
            return
        in_memory_cache: Final = self._user_api_key_cache.in_memory_cache
        if in_memory_cache is not None:
            in_memory_cache.delete_cache(parsed.cache_key)
        for additional_cache in self._additional_in_memory_caches:
            additional_cache.delete_cache(parsed.cache_key)

    @staticmethod
    async def _close_pubsub(pubsub: _ConfigSyncPubSub) -> None:
        try:
            await pubsub.aclose()
        except Exception as e:  # noqa: BLE001  # best-effort close of a possibly-broken connection
            verbose_proxy_logger.debug("auth cache invalidation pubsub close failed: %s", e)
