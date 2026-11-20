from typing import Final

import pytest

from litellm._internal_context import current_service_target
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import (
    broadcast_evictions,
    evict_and_broadcast,
    publish_auth_cache_invalidation,
)
from litellm.proxy.common_utils.user_api_key_cache import AUTH_OBJECTS_TARGET, UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthObject
from litellm.redis.stores.auth_cache_invalidation import AuthCacheInvalidation
from tests.unit.redis.stores.auth_cache_invalidation_double import FakeChannel, Worker, channel


def _install(monkeypatch: pytest.MonkeyPatch, *workers: Worker) -> FakeChannel:
    invalidation, fake = channel(*workers)
    monkeypatch.setattr(proxy_server.proxy_config, "auth_cache_invalidation", invalidation)
    return fake


@pytest.mark.asyncio
async def test_publish_sends_an_eviction_or_a_value_on_the_shared_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    fake: Final = _install(monkeypatch)

    await publish_auth_cache_invalidation(cache_key="project_id:p-1")
    await publish_auth_cache_invalidation(cache_key="spend:key:k-1", new_value=0.0, ttl=60)

    assert fake.published == [("project_id:p-1", None, None), ("spend:key:k-1", 0.0, 60)]


@pytest.mark.asyncio
async def test_evict_and_broadcast_drops_the_key_here_and_on_a_peer_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """LIT-3803: a worker that did not handle the mutation must drop its copy too, or it serves the stale object
    until the TTL expires."""
    handling: Final = UserApiKeyCache()
    peer: Final = UserApiKeyCache()
    for cache in (handling, peer):
        cache.set_cache(AuthObject.user("user-1"), UserAPIKeyAuth(user_id="user-1"), model_type=UserAPIKeyAuth)
        cache.set_cache(AuthObject.user("user-2"), UserAPIKeyAuth(user_id="user-2"), model_type=UserAPIKeyAuth)
    fake: Final = _install(monkeypatch, Worker(peer))

    await evict_and_broadcast(cache_keys=(AuthObject.user("user-1").key,), user_api_key_cache=handling)

    assert fake.published == [(AuthObject.user("user-1").key, None, None)]
    for cache in (handling, peer):
        assert cache.get_cache(AuthObject.user("user-1"), model_type=UserAPIKeyAuth) is None
        assert cache.get_cache(AuthObject.user("user-2"), model_type=UserAPIKeyAuth) is not None


@pytest.mark.asyncio
async def test_broadcast_evictions_sends_every_key_in_one_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake: Final = _install(monkeypatch)
    cache_keys: Final = tuple(f"user-{i}" for i in range(3000))

    await broadcast_evictions(cache_keys=cache_keys)

    assert fake.bulk == [cache_keys]


class _TargetRecordingChannel(FakeChannel):
    targets: tuple[str | None, ...] = ()

    def publish(self, cache_key: str, new_value: float | None = None, ttl: float | None = None) -> bool:
        self.targets = (*self.targets, current_service_target())
        return super().publish(cache_key, new_value, ttl)


@pytest.mark.asyncio
async def test_evict_and_broadcast_publishes_under_the_auth_objects_service_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle: Final = _TargetRecordingChannel()
    monkeypatch.setattr(proxy_server.proxy_config, "auth_cache_invalidation", AuthCacheInvalidation(handle))

    await evict_and_broadcast(cache_keys=("user-a", "user-b"), user_api_key_cache=UserApiKeyCache())

    assert handle.targets == (AUTH_OBJECTS_TARGET, AUTH_OBJECTS_TARGET)
