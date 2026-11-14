"""A stand-in for ``NativeAuthCache`` with the native store's semantics: memory in front of a Redis dict, every
direct round trip logged in ``commands`` and every pre-call declaration logged on the request's ``events``."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, TypeVar
from unittest.mock import MagicMock

from litellm.caching.redis_cache import RedisCache
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthCacheSettings, AuthCacheStore, WriteBack
from litellm.rust_bridge._native import LocalSizes

T = TypeVar("T")


@dataclass
class DeclaredOp:
    """Settles only once its request's pre-call batch was flushed, like a native declaration."""

    request: object
    answer: Callable[[], object]

    async def resolve(self) -> object:
        events: Final = getattr(self.request, "events", None)
        if events is not None and "flush_pre" not in events:
            raise ConnectionError("resolved before its batch was flushed")
        return self.answer()


def _on_request(send: object) -> bool:
    """A fake request stands in for a request's ``NativeSend``; anything else is ``NativeSend.Now()``."""
    return getattr(send, "events", None) is not None


@dataclass
class FakeAuthCache:
    """``redis`` holds what each key decodes to; ``fail`` makes every direct Redis round trip raise. Two doubles
    built on one ``redis`` and one ``generations`` are two workers on one Redis."""

    redis: dict[str, object] = field(default_factory=dict)
    generations: dict[str, str] = field(default_factory=dict)
    memory: dict[str, tuple[object, float | None]] = field(default_factory=dict)
    commands: list[tuple[object, ...]] = field(default_factory=list)
    settings: list[AuthCacheSettings] = field(default_factory=list)
    fail: BaseException | None = None

    def _round_trip(self, *command: object) -> None:
        self.commands.append(command)
        if self.fail is not None:
            raise self.fail

    @staticmethod
    def _declare(request: object, event: str) -> None:
        events: Final = getattr(request, "events", None)
        if events is not None:
            events.append(event)

    async def get(self, key: str) -> object:
        return self.get_sync(key)

    def get_sync(self, key: str) -> object:
        cached: Final = self.get_local(key)
        if cached is not None:
            return cached
        self._round_trip("GET", key)
        value: Final = self.redis.get(key)
        if value is not None:
            self.set_local(key, value)
        return value

    async def set(self, key: str, value: object, ttl_seconds: float | None = None) -> None:
        self.set_local(key, value, ttl_seconds)
        self._round_trip("SET", key, ttl_seconds)
        self.redis[key] = value

    def set_sync(self, key: str, value: object, ttl_seconds: float | None = None) -> None:
        self.set_local(key, value, ttl_seconds)
        self._round_trip("SET_STR", key, ttl_seconds)
        self.redis[key] = value

    async def lease(self, key: str) -> str | None:
        self._round_trip("GET_GENERATION", key)
        return self.generations.get(key)

    async def set_leased(self, key: str, lease: str | None, value: object, ttl_seconds: float | None = None) -> bool:
        self._round_trip("SET_LEASED", key, ttl_seconds)
        if self.generations.get(key) != lease:
            return False
        self.redis[key] = value
        return True

    def _void_leases(self, keys: Sequence[str]) -> None:
        self.generations.update((key, uuid.uuid4().hex) for key in keys)

    def declare_set(self, key: str, value: object, ttl_seconds: float | None, send: object) -> DeclaredOp:
        self.set_local(key, value, ttl_seconds)
        self._declare(send, f"declare set {key} ttl={ttl_seconds}")
        self.redis[key] = value
        return DeclaredOp(send, lambda: None)

    def declare_delete(self, key: str, send: object) -> DeclaredOp:
        self.delete_local(key)
        self._declare(send, f"declare del {key}")
        self._void_leases((key,))
        self.redis.pop(key, None)
        return DeclaredOp(send, lambda: None)

    async def set_many(self, entries: Sequence[tuple[str, object]], ttl_seconds: float | None = None) -> None:
        for key, value in entries:
            self.set_local(key, value, ttl_seconds)
        self._round_trip("SET_MANY", tuple(key for key, _ in entries), ttl_seconds)
        self.redis.update(entries)

    async def write_back(self, entries: Sequence[WriteBack], send: object) -> None:
        for key, value, ttl in entries:
            self.set_local(key, value, ttl)
            self.redis[key] = value
        written: Final = tuple((key, ttl) for key, _value, ttl in entries)
        if not _on_request(send):
            self._round_trip("WRITE_BACK", written)
            return
        self._declare(send, f"declare write_back {written}")

    async def delete(self, keys: Sequence[str]) -> None:
        self.delete_sync(keys)

    def delete_sync(self, keys: Sequence[str]) -> None:
        for key in keys:
            self.delete_local(key)
        self._round_trip("DEL", tuple(keys))
        self._void_leases(keys)
        for key in keys:
            self.redis.pop(key, None)

    def get_many(self, keys: Sequence[str], send: object) -> DeclaredOp:
        on_request: Final = _on_request(send)
        self._declare(send, f"declare auth_cache {','.join(keys)}")

        def answer() -> list[object]:
            if on_request and self.fail is not None:
                raise self.fail
            if not on_request:
                self._round_trip("MGET", tuple(keys))
            return [self.redis.get(key) for key in keys]

        return DeclaredOp(send, answer)

    async def increment_if_present(self, key: str, amount: float) -> float | None:
        self._round_trip("INCRBYFLOAT_IF_PRESENT", key, amount)
        current: Final = self.redis.get(key)
        if not isinstance(current, (int, float)):
            return None
        total: Final = float(current) + amount
        self.redis[key] = total
        return total

    def get_local(self, key: str) -> object:
        entry: Final = self.memory.get(key)
        return None if entry is None else entry[0]

    def local_ttl(self, key: str) -> float | None:
        entry: Final = self.memory.get(key)
        return None if entry is None else entry[1] if entry[1] is not None else 600

    def set_local(self, key: str, value: object, ttl_seconds: float | None = None) -> None:
        self.memory[key] = (value, ttl_seconds)

    def delete_local(self, key: str) -> None:
        self.memory.pop(key, None)

    def flush_local(self) -> None:
        self.memory.clear()

    def len_local(self) -> LocalSizes:
        return LocalSizes(objects=len(self.memory), key_objects=0)

    def reconfigure(self, *, memory_entries: int, memory_ttl_seconds: float | None = None) -> None:
        self.settings.append(AuthCacheSettings(memory_entries, memory_ttl_seconds))

    def store(self, _redis_cache: RedisCache, settings: AuthCacheSettings) -> AuthCacheStore:
        self.settings.append(settings)
        return AuthCacheStore(self)


def shared_cache(
    fake: FakeAuthCache | None = None, connection: RedisCache | None = None, **kwargs: float | None
) -> tuple[UserApiKeyCache, FakeAuthCache]:
    """A ``UserApiKeyCache`` with ``enable_redis_auth_cache`` on, its shared store backed by ``fake``."""
    double: Final = fake or FakeAuthCache()
    cache: Final = UserApiKeyCache(
        redis_cache=connection or MagicMock(spec=RedisCache),
        auth_cache=double.store,
        **kwargs,
    )
    return cache, double
