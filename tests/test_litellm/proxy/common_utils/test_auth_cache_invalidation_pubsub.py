import asyncio
import json
from typing import Iterable, List, Optional, Tuple
from unittest.mock import patch

import pytest
from redis.asyncio import Redis

from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import (
    AUTH_CACHE_INVALIDATION_CHANNEL,
    AuthCacheInvalidationSubscriber,
    publish_auth_cache_invalidation,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


class _RecordingRedisClient(Redis):
    def __init__(self) -> None:
        self.published: List[Tuple[str, str]] = []

    async def publish(self, channel: str, message: str) -> int:
        self.published.append((channel, message))
        return 1


class _FailingPublishRedisClient(Redis):
    def __init__(self) -> None:
        pass

    async def publish(self, channel: str, message: str) -> int:
        raise ConnectionError("redis down")


class _QueuePubSub:
    def __init__(self, initial_messages: Iterable[object] = ()) -> None:
        self.queue: "asyncio.Queue[object]" = asyncio.Queue()
        for message in initial_messages:
            self.queue.put_nowait(message)
        self.subscribed_channels: List[str] = []
        self.closed = False

    async def subscribe(self, *channels: str) -> None:
        self.subscribed_channels.extend(channels)

    async def get_message(self, *, ignore_subscribe_messages: bool, timeout: float) -> Optional[object]:
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    async def aclose(self) -> None:
        self.closed = True


class _ScriptedPubSubRedisClient(Redis):
    def __init__(self, pubsubs: Iterable[_QueuePubSub]) -> None:
        self._scripted_pubsubs = iter(pubsubs)

    def pubsub(self) -> _QueuePubSub:
        return next(self._scripted_pubsubs)


class _FakeRedisCache:
    def __init__(self, client: object, namespace: Optional[str] = None) -> None:
        self._client = client
        self.namespace = namespace

    def init_async_client(self) -> object:
        return self._client


def _invalidation_message(cache_key: str) -> dict:
    return {"type": "message", "data": json.dumps({"cache_key": cache_key}).encode()}


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
        for _ in range(200):
            if cache.in_memory_cache.get_cache("project_id:p-1") is None:
                break
            await asyncio.sleep(0.01)
    finally:
        await subscriber.stop()

    assert cache.in_memory_cache.get_cache("project_id:p-1") is None
    assert pubsub.subscribed_channels == [AUTH_CACHE_INVALIDATION_CHANNEL]


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
