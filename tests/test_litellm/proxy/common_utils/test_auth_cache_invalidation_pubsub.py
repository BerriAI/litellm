import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Iterable
from unittest.mock import patch

import pytest

import litellm.proxy.common_utils.auth_cache_invalidation_pubsub as pubsub_module
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisMessage
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import (
    AUTH_CACHE_INVALIDATION_CHANNEL,
    AuthCacheInvalidationSubscriber,
    evict_and_broadcast,
    publish_auth_cache_invalidation,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


class _WedgedPublisher:
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


class _QueuePubSub:
    """A subscription fed from a queue of messages, standing in for RedisSubscription."""

    def __init__(self, initial_messages: Iterable[RedisMessage] = ()) -> None:
        self.queue: asyncio.Queue[RedisMessage] = asyncio.Queue()
        for message in initial_messages:
            self.queue.put_nowait(message)
        self.subscribed_channels: list[str] = []
        self.closed = False

    async def get_message(self, *, timeout: float | None) -> RedisMessage | None:
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    async def aclose(self) -> None:
        self.closed = True


class _FakeRedisCache:
    """The slice of RedisCache the module uses: namespace, async_publish and async_subscribe."""

    def __init__(
        self,
        subscriptions: Iterable[_QueuePubSub] = (),
        namespace: str | None = None,
        publish: Callable[[str, str], Awaitable[int]] | None = None,
        publish_error: Exception | None = None,
    ) -> None:
        self._subscriptions = iter(subscriptions)
        self._publish = publish
        self._publish_error = publish_error
        self.namespace = namespace
        self.published: list[tuple[str, str]] = []

    async def async_publish(self, channel: str, message: str) -> int:
        if self._publish_error is not None:
            raise self._publish_error
        if self._publish is not None:
            return await self._publish(channel, message)
        self.published.append((channel, message))
        return 1

    async def async_subscribe(self, *channels: str) -> _QueuePubSub:
        subscription = next(self._subscriptions)
        subscription.subscribed_channels.extend(channels)
        return subscription


class _ClusterRedisCache(_FakeRedisCache):
    async def async_publish(self, channel: str, message: str) -> int:
        raise NotImplementedError("Redis Cluster clients have no pub/sub support")

    async def async_subscribe(self, *channels: str) -> _QueuePubSub:
        raise NotImplementedError("Redis Cluster clients have no pub/sub support")


def _invalidation_message(cache_key: str) -> RedisMessage:
    return RedisMessage(channel=AUTH_CACHE_INVALIDATION_CHANNEL, payload=json.dumps({"cache_key": cache_key}).encode())


@pytest.mark.asyncio
async def test_publish_sends_cache_key_json_on_channel() -> None:
    cache = _FakeRedisCache()
    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=cache,
    ):
        await publish_auth_cache_invalidation(cache_key="project_id:p-1")

    assert cache.published == [(AUTH_CACHE_INVALIDATION_CHANNEL, json.dumps({"cache_key": "project_id:p-1"}))]


@pytest.mark.asyncio
async def test_publish_uses_namespaced_channel() -> None:
    cache = _FakeRedisCache(namespace="ns1")
    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=cache,
    ):
        await publish_auth_cache_invalidation(cache_key="project_id:p-1")

    assert cache.published[0][0] == f"ns1:{AUTH_CACHE_INVALIDATION_CHANNEL}"


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
        return_value=_FakeRedisCache(publish_error=ConnectionError("redis down")),
    ):
        await publish_auth_cache_invalidation(cache_key="project_id:p-1")


@pytest.mark.asyncio
async def test_publish_skips_clients_without_pubsub_support() -> None:
    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=_ClusterRedisCache(),
    ):
        await publish_auth_cache_invalidation(cache_key="project_id:p-1")
        await asyncio.gather(*pubsub_module._pending_publishes)  # pyright: ignore[reportPrivateUsage]  # drain module-level tasks


@pytest.mark.asyncio
async def test_subscriber_disables_itself_without_pubsub_support() -> None:
    subscriber = AuthCacheInvalidationSubscriber(redis_cache=_ClusterRedisCache(), user_api_key_cache=UserApiKeyCache())

    subscriber.start()
    task = subscriber._task
    assert task is not None
    await asyncio.wait_for(task, timeout=5)

    assert task.done() is True


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
        redis_cache=_FakeRedisCache(subscriptions=[pubsub]),
        user_api_key_cache=cache,
    )
    subscriber.start()
    try:
        for _ in range(200):
            if cache.in_memory_cache.get_cache("project_id:p-1") is None:
                break
            await asyncio.sleep(0.01)
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
        redis_cache=_FakeRedisCache(subscriptions=[pubsub]),
        user_api_key_cache=cache,
    )
    subscriber.start()
    try:
        for _ in range(200):
            if cache.get_cache(hashed_token, model_type=UserAPIKeyAuth) is None:
                break
            await asyncio.sleep(0.01)
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
        redis_cache=_FakeRedisCache(subscriptions=[pubsub]),
        user_api_key_cache=cache,
        additional_in_memory_caches=(spend_counter_in_memory_cache,),
    )
    subscriber.start()
    try:
        for _ in range(200):
            if spend_counter_in_memory_cache.get_cache("spend:team_member:u-1:t-1") is None:
                break
            await asyncio.sleep(0.01)
    finally:
        await subscriber.stop()

    assert spend_counter_in_memory_cache.get_cache("spend:team_member:u-1:t-1") is None


@pytest.mark.asyncio
async def test_subscriber_ignores_malformed_messages() -> None:
    cache = UserApiKeyCache()
    cache.in_memory_cache.set_cache("project_id:p-1", {"models": []})

    subscriber = AuthCacheInvalidationSubscriber(
        redis_cache=_FakeRedisCache(subscriptions=[_QueuePubSub()]),
        user_api_key_cache=cache,
    )
    subscriber._apply_message(RedisMessage(channel=AUTH_CACHE_INVALIDATION_CHANNEL, payload=b"not json"))
    subscriber._apply_message(
        RedisMessage(channel=AUTH_CACHE_INVALIDATION_CHANNEL, payload=json.dumps({"other": "x"}).encode())
    )

    assert cache.in_memory_cache.get_cache("project_id:p-1") is not None


@pytest.mark.asyncio
async def test_evict_and_broadcast_evicts_locally_and_returns_while_redis_publish_never_answers() -> None:
    cache = UserApiKeyCache()
    cache.set_cache("user-wedged", UserAPIKeyAuth(user_id="user-wedged"), model_type=UserAPIKeyAuth)
    client = _WedgedPublisher()

    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=_FakeRedisCache(publish=client.publish),
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
    client = _WedgedPublisher()

    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=_FakeRedisCache(publish=client.publish),
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
