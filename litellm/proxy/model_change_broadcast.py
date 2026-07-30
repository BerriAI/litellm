"""
Cross-pod invalidation for model CRUD.

Each pod serves reads (`/model/info`, request routing) from its own in-memory
`llm_router`, reconciled with the DB every `PROXY_CONFIG_RELOAD_INTERVAL_SECONDS`.
A write handled by one pod is therefore invisible to its siblings for up to a full
interval, so a model deleted in the Admin UI keeps showing up on refreshes the load
balancer sends elsewhere.

The pod that handled the write publishes a notification on the coordination Redis;
subscribers react by re-running the same DB reconcile the timer already performs.
The notification is a trigger, never a payload: siblings always derive state from
the DB, so a dropped, duplicated or reordered message can only cost latency, and the
periodic reconcile remains the backstop.
"""

import asyncio
import contextlib
from collections.abc import Mapping
from typing import Awaitable, Callable, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import (
    MODEL_CHANGE_PUBSUB_CHANNEL,
    MODEL_CHANGE_PUBSUB_ENABLED,
    MODEL_CHANGE_PUBSUB_POLL_TIMEOUT_SECONDS,
    MODEL_CHANGE_PUBSUB_RECONNECT_SECONDS,
)

ModelChangeOperation = Literal["created", "updated", "deleted"]

PROCESS_ID: Final[str] = str(uuid.uuid4())


class ModelChangeNotification(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation: ModelChangeOperation
    model_id: str | None = None
    origin: str


class RedisPubSubConnection(Protocol):
    async def subscribe(self, *channels: str) -> None: ...

    async def get_message(
        self,
        ignore_subscribe_messages: bool = ...,
        timeout: float | None = ...,
    ) -> Mapping[str, object] | None: ...

    async def aclose(self) -> None: ...


class RedisPubSubClient(Protocol):
    async def publish(self, channel: str, message: str) -> int: ...

    def pubsub(self) -> RedisPubSubConnection: ...


class RedisPubSubBackend(Protocol):
    """The slice of `RedisCache` this module needs."""

    def check_and_fix_namespace(self, key: str) -> str: ...

    def init_async_client(self) -> RedisPubSubClient: ...


def _coordination_redis() -> RedisPubSubBackend | None:
    from litellm.proxy.proxy_server import redis_usage_cache

    return redis_usage_cache


async def broadcast_model_change(
    operation: ModelChangeOperation,
    model_id: str | None = None,
    redis_cache: RedisPubSubBackend | None = None,
) -> None:
    """
    Tell sibling pods that the model table changed. Never raises: the write it
    follows has already succeeded, and the periodic reconcile still converges.
    """
    if not MODEL_CHANGE_PUBSUB_ENABLED:
        return

    backend = redis_cache if redis_cache is not None else _coordination_redis()
    if backend is None:
        return

    notification = ModelChangeNotification(operation=operation, model_id=model_id, origin=PROCESS_ID)
    try:
        client = backend.init_async_client()
        await client.publish(
            backend.check_and_fix_namespace(MODEL_CHANGE_PUBSUB_CHANNEL),
            notification.model_dump_json(),
        )
    except Exception as e:  # noqa: BLE001  # no redis error may fail a write that already succeeded
        verbose_proxy_logger.warning(
            "Could not broadcast model change (%s, model_id=%s) to other pods: %s. "
            "They will pick it up on their next config reload.",
            operation,
            model_id,
            str(e),
        )


def _parse_notification(payload: object) -> ModelChangeNotification | None:
    raw = payload.decode() if isinstance(payload, bytes) else payload
    if not isinstance(raw, str):
        return None
    try:
        return ModelChangeNotification.model_validate_json(raw)
    except ValidationError:
        return None


class ModelChangeSubscriber:
    """
    Listens for model-change notifications and re-runs `reconcile` for each burst.

    `reconcile` is injected (the proxy passes `ProxyConfig.add_deployment`), so the
    subscriber owns no knowledge of how the router is rebuilt.
    """

    def __init__(
        self,
        redis_cache: RedisPubSubBackend,
        reconcile: Callable[[], Awaitable[None]],
        origin: str = PROCESS_ID,
        poll_timeout_seconds: float = MODEL_CHANGE_PUBSUB_POLL_TIMEOUT_SECONDS,
        reconnect_seconds: float = MODEL_CHANGE_PUBSUB_RECONNECT_SECONDS,
    ) -> None:
        self._redis_cache = redis_cache
        self._reconcile = reconcile
        self._origin = origin
        self._poll_timeout_seconds = poll_timeout_seconds
        self._reconnect_seconds = reconnect_seconds

    @property
    def channel(self) -> str:
        return self._redis_cache.check_and_fix_namespace(MODEL_CHANGE_PUBSUB_CHANNEL)

    async def listen_forever(self) -> None:
        while True:
            try:
                await self.listen_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001  # a subscriber that dies leaves this pod on the reload interval
                verbose_proxy_logger.warning(
                    "Model-change subscriber dropped off %s: %s. Reconnecting in %ss; the "
                    "periodic config reload keeps this pod converging meanwhile.",
                    self.channel,
                    str(e),
                    self._reconnect_seconds,
                )
                await asyncio.sleep(self._reconnect_seconds)

    async def listen_once(self) -> None:
        """
        One connection's lifetime. Notifications are coalesced: a burst of writes
        (deleting several models in a row) triggers a single reconcile once the channel
        goes quiet, instead of one full DB reload per message.
        """
        connection = self._redis_cache.init_async_client().pubsub()
        await connection.subscribe(self.channel)
        verbose_proxy_logger.info("Subscribed to model changes from other pods on %s", self.channel)
        try:
            pending = False
            while True:
                message = await connection.get_message(
                    ignore_subscribe_messages=True,
                    timeout=self._poll_timeout_seconds,
                )
                if message is None:
                    if pending:
                        pending = False
                        await self._reconcile()
                    continue
                pending = pending or self._is_foreign_change(message)
        finally:
            with contextlib.suppress(Exception):
                await connection.aclose()

    def _is_foreign_change(self, message: Mapping[str, object]) -> bool:
        if message.get("type") != "message":
            return False
        notification = _parse_notification(message.get("data"))
        if notification is None:
            verbose_proxy_logger.debug("Ignoring unparseable model-change notification: %s", message.get("data"))
            return False
        return notification.origin != self._origin


class ModelChangeSubscriberHandle:
    """
    Owns the subscriber task for the lifetime of the process. Holding the reference
    matters: an unreferenced asyncio task can be garbage collected mid-flight.
    """

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None

    def start(
        self,
        redis_cache: RedisPubSubBackend | None,
        reconcile: Callable[[], Awaitable[None]],
    ) -> None:
        if not MODEL_CHANGE_PUBSUB_ENABLED or redis_cache is None:
            return
        self.stop()
        subscriber = ModelChangeSubscriber(redis_cache=redis_cache, reconcile=reconcile)
        self._task = asyncio.create_task(subscriber.listen_forever())

    def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        self._task = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()


model_change_subscriber = ModelChangeSubscriberHandle()
