import asyncio
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import (  # noqa: TID251  # cast narrows a json str to the Literal
    TYPE_CHECKING,
    Final,
    Literal,
    TypeAlias,
    cast,
    get_args,
)

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.config_sync_pubsub import (
    _ConfigSyncPubSub,
    _pubsub_capable_client,
    coordination_redis_cache,
)

if TYPE_CHECKING:
    from litellm.caching.in_memory_cache import InMemoryCache
    from litellm.caching.redis_cache import RedisCache

InvalidationTarget: TypeAlias = Literal["user_api_key", "config_param"]

DEFAULT_INVALIDATION_TARGET: Final[InvalidationTarget] = "user_api_key"
_INVALIDATION_TARGETS: Final[frozenset[str]] = frozenset(get_args(InvalidationTarget))

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
    cache: InvalidationTarget


def _cache_invalidation_message_json(cache_key: str, target: InvalidationTarget) -> str:
    return json.dumps(asdict(_CacheInvalidationMessage(cache_key=cache_key, cache=target)))


def _invalidation_from_message_data(data: object) -> _CacheInvalidationMessage | None:
    decoded: Final = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    if not isinstance(decoded, str):
        return None
    try:
        parsed: Final = json.loads(decoded)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    cache_key: Final = parsed.get("cache_key")
    if not isinstance(cache_key, str):
        return None
    raw_target: Final = parsed.get("cache", DEFAULT_INVALIDATION_TARGET)
    if raw_target not in _INVALIDATION_TARGETS:
        return None
    target: Final = cast(InvalidationTarget, raw_target)  # cast-ok: membership checked against the Literal's args
    return _CacheInvalidationMessage(cache_key=cache_key, cache=target)


async def publish_auth_cache_invalidation(
    cache_key: str,
    target: InvalidationTarget = DEFAULT_INVALIDATION_TARGET,
) -> None:
    """
    Best-effort broadcast so every worker drops its local in-memory copy of a
    mutated object; without this, only the handling worker and Redis are
    evicted and other workers keep serving the stale object until its TTL.
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
            _cache_invalidation_message_json(cache_key=cache_key, target=target),
        )
    except Exception as e:  # noqa: BLE001  # best-effort publish; mutations must never fail on redis errors
        verbose_proxy_logger.warning("auth cache invalidation publish for %s failed: %s", cache_key, e)


class AuthCacheInvalidationSubscriber:
    __slots__ = ("_in_memory_caches", "_redis_cache", "_task")

    def __init__(
        self,
        redis_cache: "RedisCache",
        in_memory_caches: Mapping[InvalidationTarget, "InMemoryCache"],
    ) -> None:
        self._redis_cache = redis_cache
        self._in_memory_caches = in_memory_caches
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
        invalidation: Final = _invalidation_from_message_data(data)
        if invalidation is None:
            return
        in_memory_cache: Final = self._in_memory_caches.get(invalidation.cache)
        if in_memory_cache is not None:
            in_memory_cache.delete_cache(invalidation.cache_key)

    @staticmethod
    async def _close_pubsub(pubsub: _ConfigSyncPubSub) -> None:
        try:
            await pubsub.aclose()
        except Exception as e:  # noqa: BLE001  # best-effort close of a possibly-broken connection
            verbose_proxy_logger.debug("auth cache invalidation pubsub close failed: %s", e)
