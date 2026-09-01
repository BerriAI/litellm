import asyncio
import json
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, Protocol, cast  # noqa: TID251  # untyped prisma/redis boundary needs cast

from litellm._logging import verbose_proxy_logger
from litellm.repositories.prisma_protocols import RowT_co, TableActions

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache


class _ConfigSyncPubSub(Protocol):
    def subscribe(self, *channels: str) -> Awaitable[object]: ...

    def get_message(self, *, ignore_subscribe_messages: bool, timeout: float) -> Awaitable[object]: ...

    def aclose(self) -> Awaitable[object]: ...


class _ConfigSyncPubSubClient(Protocol):
    def publish(self, channel: str, message: str) -> Awaitable[int]: ...

    def pubsub(self) -> _ConfigSyncPubSub: ...


CONFIG_SYNC_CHANNEL: Final = "litellm_proxy.config_change"
CONFIG_SYNC_DEBOUNCE_SECONDS: Final = 1.0
CONFIG_SYNC_JITTER_MAX_SECONDS: Final = 5.0
CONFIG_SYNC_MIN_RESYNC_INTERVAL_SECONDS: Final = 10.0
_POLL_TIMEOUT_SECONDS: Final = 1.0
_BACKOFF_INITIAL_SECONDS: Final = 5.0
_BACKOFF_MAX_SECONDS: Final = 60.0

_WRITE_ACTION_NAMES: Final[frozenset[str]] = frozenset(
    {"create", "create_many", "update", "update_many", "upsert", "delete", "delete_many"}
)

_CONFIG_SYNCED_TABLE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "litellm_proxymodeltable",
        "litellm_credentialstable",
        "litellm_guardrailstable",
        "litellm_policytable",
        "litellm_policyattachmenttable",
        "litellm_managedvectorstorestable",
        "litellm_managedvectorstoreindextable",
        "litellm_mcpservertable",
        "litellm_agentstable",
        "litellm_prompttable",
        "litellm_searchtoolstable",
        "litellm_ssoconfig",
        "litellm_cacheconfig",
        "litellm_configoverrides",
    }
)

_RESYNC_APPLIED_CONFIG_PARAM_NAMES: Final[frozenset[str]] = frozenset(
    {
        "general_settings",
        "router_settings",
        "litellm_settings",
        "model_cost_map_reload_config",
        "anthropic_beta_headers_reload_config",
    }
)


def coordination_redis_cache() -> "RedisCache | None":
    from litellm.proxy.proxy_server import redis_usage_cache

    return redis_usage_cache


def config_sync_channel(redis_cache: "RedisCache") -> str:
    if redis_cache.namespace is None:
        return CONFIG_SYNC_CHANNEL
    return f"{redis_cache.namespace}:{CONFIG_SYNC_CHANNEL}"


def _raw_async_client(redis_cache: "RedisCache") -> object:
    return cast(  # cast-ok: redis-py generics leave the client type partially unknown
        object,
        redis_cache.init_async_client(),  # pyright: ignore[reportUnknownMemberType]  # redis generics
    )


def _pubsub_capable_client(redis_cache: "RedisCache") -> _ConfigSyncPubSubClient | None:
    from redis.asyncio import Redis

    client: Final = _raw_async_client(redis_cache)
    if isinstance(client, Redis):
        return cast(_ConfigSyncPubSubClient, client)  # cast-ok: protocol view of the standalone redis client
    return None


@dataclass(frozen=True, slots=True)
class _ConfigChangeMessage:
    object_type: str


def _config_change_message_json(object_type: str) -> str:
    return json.dumps(asdict(_ConfigChangeMessage(object_type=object_type)))


async def publish_config_change(redis_cache: "RedisCache | None", object_type: str) -> None:
    if redis_cache is None:
        return
    try:
        client: Final = _pubsub_capable_client(redis_cache)
        if client is None:
            verbose_proxy_logger.debug(
                "config sync publish for %s skipped: cluster redis client has no pub/sub support",
                object_type,
            )
            return
        await client.publish(config_sync_channel(redis_cache), _config_change_message_json(object_type))
    except Exception as e:  # noqa: BLE001  # best-effort publish; writes must never fail on redis errors
        verbose_proxy_logger.warning("config sync publish for %s failed: %s", object_type, e)


async def publish_config_change_for_object_type(object_type: str) -> None:
    await publish_config_change(redis_cache=coordination_redis_cache(), object_type=object_type)


async def publish_config_param_change(param_name: str) -> None:
    if param_name not in _RESYNC_APPLIED_CONFIG_PARAM_NAMES:
        verbose_proxy_logger.debug(
            "config sync publish for %s skipped: no resync callback applies this param outside proxy startup",
            param_name,
        )
        return
    await publish_config_change_for_object_type(param_name)


class _PublishOnWriteActions:
    __slots__ = ("_actions", "_object_type", "_publish")

    def __init__(self, actions: object, object_type: str, publish: Callable[[str], Awaitable[None]]) -> None:
        self._actions = actions
        self._object_type = object_type
        self._publish = publish

    def __getattr__(self, name: str) -> object:
        attribute: Final = cast(object, getattr(self._actions, name))  # cast-ok: getattr on dynamic prisma actions
        if name not in _WRITE_ACTION_NAMES:
            return attribute
        write_action: Final = cast(Callable[..., Awaitable[object]], attribute)  # cast-ok: prisma actions are untyped
        object_type: Final = self._object_type
        publish: Final = self._publish

        async def _write_then_publish(
            *args: object,
            **kwargs: object,  # kwargs-ok: transparent passthrough to untyped prisma action
        ) -> object:
            result: Final = await write_action(*args, **kwargs)
            await publish(object_type)
            return result

        return _write_then_publish


def wrap_table_actions_for_config_sync(
    actions: "TableActions[RowT_co]",
    table_name: str,
    publish: Callable[[str], Awaitable[None]] = publish_config_change_for_object_type,
) -> "TableActions[RowT_co]":
    if table_name not in _CONFIG_SYNCED_TABLE_NAMES:
        return actions
    wrapped: Final = _PublishOnWriteActions(actions=actions, object_type=table_name, publish=publish)
    return cast("TableActions[RowT_co]", wrapped)  # cast-ok: dynamic write-through proxy keeps the wrapped row type


class ConfigSyncSubscriber:
    __slots__ = (
        "_backoff_initial_seconds",
        "_backoff_max_seconds",
        "_debounce_seconds",
        "_jitter_max_seconds",
        "_last_resync_at",
        "_min_resync_interval_seconds",
        "_monotonic",
        "_redis_cache",
        "_resync_callbacks",
        "_rng",
        "_sleep",
        "_task",
    )

    def __init__(
        self,
        redis_cache: "RedisCache",
        resync_callbacks: tuple[Callable[[], Awaitable[None]], ...],
        debounce_seconds: float = CONFIG_SYNC_DEBOUNCE_SECONDS,
        jitter_max_seconds: float = CONFIG_SYNC_JITTER_MAX_SECONDS,
        min_resync_interval_seconds: float = CONFIG_SYNC_MIN_RESYNC_INTERVAL_SECONDS,
        backoff_initial_seconds: float = _BACKOFF_INITIAL_SECONDS,
        backoff_max_seconds: float = _BACKOFF_MAX_SECONDS,
        rng: random.Random | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._redis_cache = redis_cache
        self._resync_callbacks = resync_callbacks
        self._debounce_seconds = debounce_seconds
        self._jitter_max_seconds = jitter_max_seconds
        self._min_resync_interval_seconds = min_resync_interval_seconds
        self._backoff_initial_seconds = backoff_initial_seconds
        self._backoff_max_seconds = backoff_max_seconds
        self._rng = rng if rng is not None else random.Random()
        self._sleep = sleep
        self._monotonic = monotonic
        self._task: asyncio.Task[None] | None = None
        self._last_resync_at: float | None = None

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
        backoff_seconds = self._backoff_initial_seconds
        while True:
            try:
                client = _pubsub_capable_client(self._redis_cache)
                if client is None:
                    verbose_proxy_logger.warning(
                        "config sync subscriber disabled: cluster redis client has no pub/sub support; "
                        "interval polling remains the only sync mechanism"
                    )
                    return
                pubsub = client.pubsub()
                try:
                    await pubsub.subscribe(config_sync_channel(self._redis_cache))
                    backoff_seconds = self._backoff_initial_seconds
                    await self._consume(pubsub)
                finally:
                    await self._close_pubsub(pubsub)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001  # any redis failure falls through to backoff and reconnect
                verbose_proxy_logger.warning(
                    "config sync subscriber redis error: %s; reconnecting in %.0fs",
                    e,
                    backoff_seconds,
                )
                await self._sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2, self._backoff_max_seconds)

    async def _consume(self, pubsub: _ConfigSyncPubSub) -> None:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=_POLL_TIMEOUT_SECONDS)
            if message is None:
                continue
            await self._sleep(self._debounce_seconds + self._rng.uniform(0.0, self._jitter_max_seconds))
            await self._wait_for_min_resync_interval()
            await self._drain_pending(pubsub)
            await self._run_resync_callbacks()
            self._last_resync_at = self._monotonic()

    async def _wait_for_min_resync_interval(self) -> None:
        if self._last_resync_at is None:
            return
        seconds_until_next_resync = self._min_resync_interval_seconds - (self._monotonic() - self._last_resync_at)
        if seconds_until_next_resync <= 0:
            return
        verbose_proxy_logger.debug(
            "config sync resync throttled for %.1fs to cap fleet-wide reload rate",
            seconds_until_next_resync,
        )
        await self._sleep(seconds_until_next_resync)

    @staticmethod
    async def _drain_pending(pubsub: _ConfigSyncPubSub) -> None:
        while await pubsub.get_message(ignore_subscribe_messages=True, timeout=0) is not None:
            pass

    async def _run_resync_callbacks(self) -> None:
        for callback in self._resync_callbacks:
            try:
                await callback()
            except Exception as e:  # noqa: BLE001  # one failing resync callback must not kill the subscriber
                verbose_proxy_logger.warning("config sync resync callback failed: %s", e)

    @staticmethod
    async def _close_pubsub(pubsub: _ConfigSyncPubSub) -> None:
        try:
            await pubsub.aclose()
        except Exception as e:  # noqa: BLE001  # best-effort close of a possibly-broken connection
            verbose_proxy_logger.debug("config sync pubsub close failed: %s", e)
