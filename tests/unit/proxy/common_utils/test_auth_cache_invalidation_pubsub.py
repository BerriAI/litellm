import asyncio
import hashlib
import json
import time
from collections.abc import Iterable
from unittest.mock import patch

import pytest
import fakeredis
from redis.asyncio import Redis

import litellm
import litellm.proxy.common_utils.auth_cache_invalidation_pubsub as pubsub_module
from litellm._internal_context import current_service_target
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import (
    AUTH_CACHE_INVALIDATION_CHANNEL,
    AuthCacheInvalidationSubscriber,
    broadcast_evictions,
    evict_and_broadcast,
    publish_auth_cache_invalidation,
)
from litellm.proxy.common_utils.user_api_key_cache import AUTH_OBJECTS_TARGET, UserApiKeyCache
from tests._support.fake_redis import fake_redis_cache


class _RecordingRedisClient(Redis):
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []

    async def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, message))
        return 1

    async def ping(self) -> bool:
        return True


class _WedgedPublishRedisClient(Redis):
    def __init__(self) -> None:
        self.attempted: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.release = asyncio.Event()

    async def publish(self, channel: str, message: str) -> int:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.attempted.append(message)
        await self.release.wait()
        self.in_flight -= 1
        return 1

    async def ping(self) -> bool:
        return True


class _SlowPublishRedisClient(Redis):
    def __init__(self) -> None:
        self.published: list[str] = []

    async def publish(self, channel: str, message: str) -> int:
        await asyncio.sleep(0.001)
        self.published.append(message)
        return 1

    async def ping(self) -> bool:
        return True


class _Pipeline:
    def __init__(self, client: "_PipelinedRedisClient") -> None:
        self._client = client
        self._messages: list[str] = []

    def publish(self, channel: str, message: str) -> "_Pipeline":
        self._messages.append(message)
        return self

    async def execute(self) -> tuple[int, ...]:
        await self._client.send(tuple(self._messages))
        return tuple(1 for _ in self._messages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class _PipelinedRedisClient:
    def __init__(self, wedged: bool = False) -> None:
        self.published: list[str] = []
        self.round_trips = 0
        self.transactions = []
        self.release = asyncio.Event()
        if not wedged:
            self.release.set()

    async def send(self, messages: tuple[str, ...]) -> None:
        self.round_trips += 1
        await self.release.wait()
        self.published.extend(messages)

    def pipeline(self, transaction: bool = True) -> _Pipeline:
        self.transactions.append(transaction)
        return _Pipeline(self)

    async def ping(self) -> bool:
        return True


class _FailingPublishRedisClient(Redis):
    def __init__(self) -> None:
        pass

    async def publish(self, channel: str, message: str) -> int:
        raise ConnectionError("redis down")

    async def ping(self) -> bool:
        return True


class _TargetRecordingRedisClient(Redis):
    def __init__(self) -> None:
        self.targets: list[str | None] = []

    async def publish(self, channel: str, message: str) -> int:
        self.targets.append(current_service_target())
        return 1

    async def ping(self) -> bool:
        return True


class _QueuePubSub:
    def __init__(self, initial_messages: Iterable[object] = ()) -> None:
        self.queue: asyncio.Queue[object] = asyncio.Queue()
        for message in initial_messages:
            self.queue.put_nowait(message)
        self.subscribed_channels: list[str] = []
        self.closed = False
        self.drained = asyncio.Event()

    async def subscribe(self, *channels: str) -> None:
        self.subscribed_channels.extend(channels)

    async def get_message(self, *, ignore_subscribe_messages: bool, timeout: float) -> object | None:
        if self.queue.empty():
            self.drained.set()
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    async def aclose(self) -> None:
        self.closed = True

    async def reset(self) -> None:
        await self.aclose()


class _ScriptedPubSubRedisClient(Redis):
    def __init__(self, pubsubs: Iterable[_QueuePubSub]) -> None:
        self._scripted_pubsubs = iter(pubsubs)

    def pubsub(self) -> _QueuePubSub:
        return next(self._scripted_pubsubs)

    async def ping(self) -> bool:
        return True


def _FakeRedisCache(client: object, namespace: str | None = None):
    cache = fake_redis_cache(fakeredis.FakeServer(), namespace=namespace)
    litellm.in_memory_llm_clients_cache.set_cache(key=cache._get_async_client_cache_key(), value=client)
    return cache


def _invalidation_message(cache_key: str) -> dict:
    return {
        "type": "message",
        "channel": AUTH_CACHE_INVALIDATION_CHANNEL.encode(),
        "data": json.dumps({"cache_key": cache_key}).encode(),
    }


@pytest.mark.asyncio
async def test_publish_sends_cache_key_json_on_channel() -> None:
    client = _RecordingRedisClient()
    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=_FakeRedisCache(client=client),
    ):
        await publish_auth_cache_invalidation(cache_key="project_id:p-1")

    assert client.published == [(AUTH_CACHE_INVALIDATION_CHANNEL, json.dumps({"cache_key": "project_id:p-1"}))]


@pytest.mark.asyncio
async def test_publish_uses_namespaced_channel() -> None:
    client = _RecordingRedisClient()
    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=_FakeRedisCache(client=client, namespace="ns1"),
    ):
        await publish_auth_cache_invalidation(cache_key="project_id:p-1")

    assert client.published[0][0] == f"ns1:{AUTH_CACHE_INVALIDATION_CHANNEL}"


@pytest.mark.asyncio
async def test_publish_noops_without_coordination_redis() -> None:
    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=None,
    ):
        await publish_auth_cache_invalidation(cache_key="project_id:p-1")


@pytest.mark.asyncio
async def test_publish_swallows_redis_errors() -> None:
    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=_FakeRedisCache(client=_FailingPublishRedisClient()),
    ):
        await publish_auth_cache_invalidation(cache_key="project_id:p-1")


@pytest.mark.asyncio
async def test_subscriber_deletes_local_cache_entry_on_message() -> None:
    """
    The cross-worker half of LIT-3803: a worker that did not handle the project
    mutation must drop its in-memory copy when the invalidation broadcast lands,
    instead of serving the stale object until the TTL expires.
    """
    cache = UserApiKeyCache()
    cache.in_memory_cache.set_cache("project_id:p-1", {"models": []})
    assert cache.in_memory_cache.get_cache("project_id:p-1") is not None

    pubsub = _QueuePubSub(initial_messages=[_invalidation_message("project_id:p-1")])
    subscriber = AuthCacheInvalidationSubscriber(
        redis_cache=_FakeRedisCache(client=_ScriptedPubSubRedisClient(pubsubs=[pubsub])),
        user_api_key_cache=cache,
    )
    subscriber.start()
    try:
        assert await asyncio.wait_for(pubsub.drained.wait(), timeout=5)
    finally:
        await subscriber.stop()

    assert cache.in_memory_cache.get_cache("project_id:p-1") is None
    assert pubsub.subscribed_channels == [AUTH_CACHE_INVALIDATION_CHANNEL]


@pytest.mark.asyncio
async def test_subscriber_deletes_key_object_partition_entry_on_message() -> None:
    """
    LIT-7563 moved user-key objects into their own in-memory partition; a key
    invalidation broadcast must still evict the hashed-token entry there, or a
    deleted key keeps authenticating on other workers until its TTL expires.
    """
    hashed_token = hashlib.sha256(b"sk-lit7563-hot-key").hexdigest()
    cache = UserApiKeyCache()
    cache.set_cache(hashed_token, UserAPIKeyAuth(token=hashed_token), model_type=UserAPIKeyAuth)
    assert cache.get_cache(hashed_token, model_type=UserAPIKeyAuth) is not None

    pubsub = _QueuePubSub(initial_messages=[_invalidation_message(hashed_token)])
    subscriber = AuthCacheInvalidationSubscriber(
        redis_cache=_FakeRedisCache(client=_ScriptedPubSubRedisClient(pubsubs=[pubsub])),
        user_api_key_cache=cache,
    )
    subscriber.start()
    try:
        assert await asyncio.wait_for(pubsub.drained.wait(), timeout=5)
    finally:
        await subscriber.stop()

    assert cache.get_cache(hashed_token, model_type=UserAPIKeyAuth) is None


@pytest.mark.asyncio
async def test_subscriber_deletes_additional_in_memory_cache_entry_on_message() -> None:
    """
    The spend-counter half of the same cross-worker gap: a remote worker's own
    spend counter can hold a stale value (its fallback path when that worker's
    own Redis read for the counter fails), and only clearing user_api_key_cache
    on message would leave that separate DualCache's in-memory copy untouched.
    """
    cache = UserApiKeyCache()
    spend_counter_in_memory_cache = InMemoryCache()
    spend_counter_in_memory_cache.set_cache("spend:team_member:u-1:t-1", 999.0)
    assert spend_counter_in_memory_cache.get_cache("spend:team_member:u-1:t-1") is not None

    pubsub = _QueuePubSub(initial_messages=[_invalidation_message("spend:team_member:u-1:t-1")])
    subscriber = AuthCacheInvalidationSubscriber(
        redis_cache=_FakeRedisCache(client=_ScriptedPubSubRedisClient(pubsubs=[pubsub])),
        user_api_key_cache=cache,
        additional_in_memory_caches=(spend_counter_in_memory_cache,),
    )
    subscriber.start()
    try:
        assert await asyncio.wait_for(pubsub.drained.wait(), timeout=5)
    finally:
        await subscriber.stop()

    assert spend_counter_in_memory_cache.get_cache("spend:team_member:u-1:t-1") is None


@pytest.mark.asyncio
async def test_subscriber_ignores_malformed_messages() -> None:
    cache = UserApiKeyCache()
    cache.in_memory_cache.set_cache("project_id:p-1", {"models": []})

    subscriber = AuthCacheInvalidationSubscriber(
        redis_cache=_FakeRedisCache(client=_ScriptedPubSubRedisClient(pubsubs=[_QueuePubSub()])),
        user_api_key_cache=cache,
    )
    subscriber._apply_message({"type": "message", "data": b"not json"})
    subscriber._apply_message({"type": "message", "data": json.dumps({"other": "x"}).encode()})
    subscriber._apply_message("raw string")
    subscriber._apply_message(None)

    assert cache.in_memory_cache.get_cache("project_id:p-1") is not None


@pytest.mark.asyncio
async def test_evict_and_broadcast_evicts_locally_and_returns_while_redis_publish_never_answers() -> None:
    cache = UserApiKeyCache()
    cache.set_cache("user-wedged", UserAPIKeyAuth(user_id="user-wedged"), model_type=UserAPIKeyAuth)
    client = _WedgedPublishRedisClient()

    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=_FakeRedisCache(client=client),
    ):
        started = time.monotonic()
        await evict_and_broadcast(cache_keys=("user-wedged",), user_api_key_cache=cache)
        elapsed = time.monotonic() - started

    assert elapsed < 0.1, f"handler waited {elapsed:.3f}s on a publish that never answers"
    assert cache.get_cache("user-wedged", model_type=UserAPIKeyAuth) is None
    assert client.attempted == [json.dumps({"cache_key": "user-wedged"})], "publish was not handed to redis"
    client.release.set()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_publish_holds_at_most_sixteen_redis_connections_while_redis_is_wedged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pubsub_module, "_in_flight_publishes", asyncio.Semaphore(16))
    monkeypatch.setattr(pubsub_module, "_pending_publishes", set())
    client = _WedgedPublishRedisClient()

    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=_FakeRedisCache(client=client),
    ):
        for i in range(64):
            await publish_auth_cache_invalidation(cache_key=f"user-{i}")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert client.max_in_flight == 16, f"publish tasks held {client.max_in_flight} redis connections at once"
        assert len(client.attempted) == 16, "waiters called publish before a semaphore slot freed"
        client.release.set()
        await asyncio.gather(*pubsub_module._pending_publishes)  # pyright: ignore[reportPrivateUsage]  # drain module-level tasks

    assert len(client.attempted) == 64


@pytest.mark.asyncio
async def test_broadcast_evictions_tells_redis_every_key_past_the_pending_publish_cap_in_one_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pubsub_module, "_in_flight_publishes", asyncio.Semaphore(16))
    monkeypatch.setattr(pubsub_module, "_pending_publishes", set())
    client = _PipelinedRedisClient()
    cache_keys = tuple(f"user-{i}" for i in range(3000))

    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=_FakeRedisCache(client=client),
    ):
        await broadcast_evictions(cache_keys=cache_keys)

    assert [json.loads(message)["cache_key"] for message in client.published] == list(cache_keys)
    assert client.round_trips == 1
    assert client.transactions == [False]


@pytest.mark.asyncio
async def test_broadcast_evictions_returns_within_its_bound_while_redis_never_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pubsub_module, "_in_flight_publishes", asyncio.Semaphore(16))
    monkeypatch.setattr(pubsub_module, "_pending_publishes", set())
    monkeypatch.setattr(pubsub_module, "_BULK_EVICTION_WAIT_SECONDS", 0.05)
    client = _PipelinedRedisClient(wedged=True)

    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=_FakeRedisCache(client=client),
    ):
        started = time.monotonic()
        await broadcast_evictions(cache_keys=("user-1",))
        waited = time.monotonic() - started
        assert client.published == []
        client.release.set()
        await asyncio.gather(*pubsub_module._pending_publishes)  # pyright: ignore[reportPrivateUsage]  # drain module-level tasks

    assert 0.05 <= waited < 1.0
    assert client.published == [json.dumps({"cache_key": "user-1"})]


@pytest.mark.asyncio
async def test_evict_and_broadcast_publishes_under_the_auth_objects_service_target() -> None:
    client = _TargetRecordingRedisClient()

    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=_FakeRedisCache(client=client),
    ):
        await evict_and_broadcast(cache_keys=("user-a", "user-b"), user_api_key_cache=UserApiKeyCache())
        await asyncio.gather(*pubsub_module._pending_publishes)  # pyright: ignore[reportPrivateUsage]  # drain module-level tasks

    assert client.targets == [AUTH_OBJECTS_TARGET, AUTH_OBJECTS_TARGET]


@pytest.mark.asyncio
async def test_pending_publish_bound_drops_excess_without_replaying_after_recovery(monkeypatch):
    monkeypatch.setattr(pubsub_module, "_pending_publishes", set())
    monkeypatch.setattr(pubsub_module, "_in_flight_publishes", asyncio.Semaphore(16))
    monkeypatch.setattr(pubsub_module, "_MAX_PENDING_PUBLISHES", 2)
    client = _WedgedPublishRedisClient()
    with patch.object(pubsub_module, "coordination_redis_cache", return_value=_FakeRedisCache(client)):
        await publish_auth_cache_invalidation("first")
        await publish_auth_cache_invalidation("second")
        await publish_auth_cache_invalidation("dropped")
        assert client.attempted == [json.dumps({"cache_key": key}) for key in ("first", "second")]
        client.release.set()
        await asyncio.gather(*pubsub_module._pending_publishes)
    assert client.attempted == [json.dumps({"cache_key": key}) for key in ("first", "second")]


@pytest.mark.asyncio
async def test_failed_subscription_is_closed_before_reconnect(monkeypatch):
    class RejectedPubSub(_QueuePubSub):
        async def subscribe(self, *channels):
            await super().subscribe(*channels)
            raise ConnectionError("subscribe failed")

    first = RejectedPubSub()
    second = _QueuePubSub([_invalidation_message("project_id:one")])
    cache = UserApiKeyCache()
    cache.in_memory_cache.set_cache("project_id:one", "cached")
    monkeypatch.setattr(pubsub_module, "_BACKOFF_INITIAL_SECONDS", 0)
    subscriber = AuthCacheInvalidationSubscriber(_FakeRedisCache(_ScriptedPubSubRedisClient((first, second))), cache)
    subscriber.start()
    try:
        await asyncio.wait_for(second.drained.wait(), timeout=2)
        assert first.closed
        assert cache.in_memory_cache.get_cache("project_id:one") is None
    finally:
        await subscriber.stop()
    assert second.closed


@pytest.mark.asyncio
async def test_cancelling_pending_subscription_closes_it():
    entered = asyncio.Event()

    class PendingPubSub(_QueuePubSub):
        async def subscribe(self, *channels):
            await super().subscribe(*channels)
            entered.set()
            await asyncio.Event().wait()

    pubsub = PendingPubSub()
    subscriber = AuthCacheInvalidationSubscriber(
        _FakeRedisCache(_ScriptedPubSubRedisClient((pubsub,))), UserApiKeyCache()
    )
    subscriber.start()
    await asyncio.wait_for(entered.wait(), timeout=2)
    await subscriber.stop()
    assert pubsub.closed
