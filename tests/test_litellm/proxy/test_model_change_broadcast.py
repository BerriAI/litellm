import asyncio
import json

import fakeredis.aioredis
import pytest

from litellm.constants import MODEL_CHANGE_PUBSUB_CHANNEL
from litellm.proxy.model_change_broadcast import (
    ModelChangeNotification,
    ModelChangeSubscriber,
    ModelChangeSubscriberHandle,
    broadcast_model_change,
)


class FakeRedisBackend:
    """Stands in for `RedisCache`, exposing only what the broadcast module uses."""

    def __init__(self, client: fakeredis.aioredis.FakeRedis, namespace: str | None = None) -> None:
        self._client = client
        self._namespace = namespace

    def check_and_fix_namespace(self, key: str) -> str:
        if self._namespace is None:
            return key
        return f"{self._namespace}:{key}"

    def init_async_client(self) -> fakeredis.aioredis.FakeRedis:
        return self._client


class ExplodingRedisBackend:
    def check_and_fix_namespace(self, key: str) -> str:
        return key

    def init_async_client(self) -> fakeredis.aioredis.FakeRedis:
        raise ConnectionError("redis is down")


def _backend(namespace: str | None = None) -> FakeRedisBackend:
    server = fakeredis.FakeServer()
    return FakeRedisBackend(fakeredis.aioredis.FakeRedis(server=server), namespace=namespace)


async def _drain(pubsub, expected: int, timeout: float = 2.0) -> list[dict]:
    deadline = asyncio.get_event_loop().time() + timeout
    messages: list[dict] = []
    while len(messages) < expected and asyncio.get_event_loop().time() < deadline:
        message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=0.05)
        if message is not None:
            messages.append(message)
    return messages


@pytest.mark.asyncio
async def test_broadcast_publishes_notification_on_channel():
    backend = _backend()
    pubsub = backend.init_async_client().pubsub()
    await pubsub.subscribe(MODEL_CHANGE_PUBSUB_CHANNEL)

    await broadcast_model_change(operation="deleted", model_id="model-1", redis_cache=backend)

    messages = await _drain(pubsub, expected=1)
    assert len(messages) == 1
    notification = ModelChangeNotification.model_validate_json(messages[0]["data"])
    assert notification.operation == "deleted"
    assert notification.model_id == "model-1"
    assert notification.origin != ""


@pytest.mark.asyncio
async def test_broadcast_respects_redis_namespace():
    backend = _backend(namespace="tenant-a")
    pubsub = backend.init_async_client().pubsub()
    await pubsub.subscribe(f"tenant-a:{MODEL_CHANGE_PUBSUB_CHANNEL}")

    await broadcast_model_change(operation="created", model_id="model-1", redis_cache=backend)

    assert len(await _drain(pubsub, expected=1)) == 1


@pytest.mark.asyncio
async def test_broadcast_never_raises_when_redis_is_unavailable():
    await broadcast_model_change(operation="deleted", model_id="model-1", redis_cache=ExplodingRedisBackend())


@pytest.mark.asyncio
async def test_broadcast_is_a_noop_without_a_coordination_redis():
    await broadcast_model_change(operation="deleted", model_id="model-1", redis_cache=None)


class ReconcileSpy:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1


async def _run_subscriber(subscriber: ModelChangeSubscriber) -> "asyncio.Task[None]":
    task = asyncio.create_task(subscriber.listen_once())
    await asyncio.sleep(0.2)
    return task


async def _stop(task: "asyncio.Task[None]") -> None:
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def _wait_for_reconciles(spy: ReconcileSpy, expected: int, timeout: float = 3.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while spy.calls < expected and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_subscriber_reconciles_on_a_change_from_another_pod():
    """The regression: a model deleted on pod A must reach pod B's router without
    waiting for the periodic config reload."""
    backend = _backend()
    spy = ReconcileSpy()
    subscriber = ModelChangeSubscriber(
        redis_cache=backend,
        reconcile=spy,
        origin="pod-b",
        poll_timeout_seconds=0.05,
    )
    task = await _run_subscriber(subscriber)

    await broadcast_model_change(operation="deleted", model_id="model-1", redis_cache=backend)
    await _wait_for_reconciles(spy, expected=1)
    await _stop(task)

    assert spy.calls == 1


@pytest.mark.asyncio
async def test_subscriber_ignores_its_own_changes():
    backend = _backend()
    spy = ReconcileSpy()
    origin = "pod-a"
    subscriber = ModelChangeSubscriber(
        redis_cache=backend,
        reconcile=spy,
        origin=origin,
        poll_timeout_seconds=0.05,
    )
    task = await _run_subscriber(subscriber)

    own_notification = ModelChangeNotification(operation="deleted", model_id="model-1", origin=origin)
    await backend.init_async_client().publish(subscriber.channel, own_notification.model_dump_json())
    await asyncio.sleep(0.5)
    await _stop(task)

    assert spy.calls == 0


@pytest.mark.asyncio
async def test_subscriber_coalesces_a_burst_into_one_reconcile():
    backend = _backend()
    spy = ReconcileSpy()
    subscriber = ModelChangeSubscriber(
        redis_cache=backend,
        reconcile=spy,
        origin="pod-b",
        poll_timeout_seconds=0.05,
    )
    task = await _run_subscriber(subscriber)

    for model_id in ("model-1", "model-2", "model-3"):
        await broadcast_model_change(operation="deleted", model_id=model_id, redis_cache=backend)
    await _wait_for_reconciles(spy, expected=1)
    await asyncio.sleep(0.3)
    await _stop(task)

    assert spy.calls == 1


@pytest.mark.asyncio
async def test_handle_reconciles_while_running_and_stops_cleanly():
    backend = _backend()
    spy = ReconcileSpy()
    handle = ModelChangeSubscriberHandle()

    async def publish_from_another_pod(model_id: str) -> None:
        notification = ModelChangeNotification(operation="deleted", model_id=model_id, origin="pod-a")
        await backend.init_async_client().publish(
            backend.check_and_fix_namespace(MODEL_CHANGE_PUBSUB_CHANNEL),
            notification.model_dump_json(),
        )

    handle.start(redis_cache=backend, reconcile=spy)
    assert handle.is_running
    await asyncio.sleep(0.2)
    await publish_from_another_pod("model-1")
    await _wait_for_reconciles(spy, expected=1)

    handle.stop()
    await asyncio.sleep(0.1)
    assert not handle.is_running
    assert spy.calls == 1

    await publish_from_another_pod("model-2")
    await asyncio.sleep(0.5)
    assert spy.calls == 1


@pytest.mark.asyncio
async def test_handle_is_a_noop_without_a_coordination_redis():
    handle = ModelChangeSubscriberHandle()
    handle.start(redis_cache=None, reconcile=ReconcileSpy())
    assert not handle.is_running


@pytest.mark.asyncio
async def test_subscriber_ignores_unparseable_payloads():
    backend = _backend()
    spy = ReconcileSpy()
    subscriber = ModelChangeSubscriber(
        redis_cache=backend,
        reconcile=spy,
        origin="pod-b",
        poll_timeout_seconds=0.05,
    )
    task = await _run_subscriber(subscriber)
    client = backend.init_async_client()

    await client.publish(subscriber.channel, "not json")
    await client.publish(subscriber.channel, json.dumps({"operation": "exploded"}))
    await asyncio.sleep(0.5)
    await _stop(task)

    assert spy.calls == 0


@pytest.mark.asyncio
async def test_subscriber_resubscribes_after_a_failed_reconcile():
    """A reconcile that blows up (DB hiccup) must not take the subscriber down for
    the rest of the pod's life."""
    backend = _backend()
    calls: list[int] = []

    async def flaky_reconcile() -> None:
        calls.append(len(calls))
        if len(calls) == 1:
            raise RuntimeError("db unavailable")

    subscriber = ModelChangeSubscriber(
        redis_cache=backend,
        reconcile=flaky_reconcile,
        origin="pod-b",
        poll_timeout_seconds=0.05,
        reconnect_seconds=0.05,
    )
    task = asyncio.create_task(subscriber.listen_forever())
    await asyncio.sleep(0.2)

    await broadcast_model_change(operation="deleted", model_id="model-1", redis_cache=backend)
    await asyncio.sleep(0.5)
    await broadcast_model_change(operation="deleted", model_id="model-2", redis_cache=backend)
    await asyncio.sleep(0.5)
    await _stop(task)

    assert len(calls) == 2
