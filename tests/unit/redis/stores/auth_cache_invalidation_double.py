"""A stand-in for ``NativeAuthCacheInvalidation``: every publish is recorded and delivered to each joined worker, as
the shared channel delivers it to every pod, the publisher included."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache_invalidation import AuthCacheInvalidation, PythonCaches

Message = tuple[str, float | None, float | None]


@dataclass(frozen=True, slots=True)
class Worker:
    """One pod's caches as the native subscription applies a message to them: the auth cache drops an evicted key,
    then the Python-only caches apply it."""

    auth_cache: UserApiKeyCache
    python_caches: PythonCaches = PythonCaches()

    def apply(self, message: Message) -> None:
        cache_key, new_value, ttl = message
        if new_value is None:
            self.auth_cache.delete_local(cache_key)
        self.python_caches.apply(cache_key, new_value, ttl)


@dataclass
class FakeChannel:
    """``published`` holds every message in order; ``bulk`` each ``broadcast_evictions`` batch."""

    workers: list[Worker] = field(default_factory=list)
    published: list[Message] = field(default_factory=list)
    bulk: list[tuple[str, ...]] = field(default_factory=list)
    started: bool = False
    channel: str = "litellm_proxy.auth_cache_invalidation"

    def publish(self, cache_key: str, new_value: float | None = None, ttl: float | None = None) -> bool:
        message: Message = (cache_key, new_value, ttl)
        self.published.append(message)
        for worker in self.workers:
            worker.apply(message)
        return True

    async def broadcast_evictions(self, cache_keys: Sequence[str]) -> None:
        self.bulk.append(tuple(cache_keys))
        for cache_key in cache_keys:
            _ = self.publish(cache_key)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False


def channel(*workers: Worker) -> tuple[AuthCacheInvalidation, FakeChannel]:
    fake: FakeChannel = FakeChannel(workers=list(workers))
    return AuthCacheInvalidation(fake), fake
