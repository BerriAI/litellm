"""A stand-in for ``NativeAuthCache`` with the native store's semantics: memory in front of a Redis dict, every
direct round trip logged in ``commands`` and every pre-call declaration logged on the request's ``events``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, TypeVar
from unittest.mock import MagicMock

from litellm.caching.redis_cache import RedisCache
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthCacheSettings, WriteBack
from litellm.rust_bridge._native import NativeRedisClient

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


@dataclass
class FakeAuthCache:
    """``redis`` holds what each key decodes to; ``fail`` makes every direct Redis round trip raise."""

    redis: dict[str, object] = field(default_factory=dict)
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

    def set_pre_call(self, request: object, key: str, value: object, ttl_seconds: float | None = None) -> DeclaredOp:
        self.set_local(key, value, ttl_seconds)
        self._declare(request, f"declare set {key} ttl={ttl_seconds}")
        self.redis[key] = value
        return DeclaredOp(request, lambda: None)

    def delete_pre_call(self, request: object, key: str) -> DeclaredOp:
        self.delete_local(key)
        self._declare(request, f"declare del {key}")
        self.redis.pop(key, None)
        return DeclaredOp(request, lambda: None)

    async def set_many(self, entries: Sequence[tuple[str, object]], ttl_seconds: float | None = None) -> None:
        for key, value in entries:
            self.set_local(key, value, ttl_seconds)
        self._round_trip("SET_MANY", tuple(key for key, _ in entries), ttl_seconds)
        self.redis.update(entries)

    async def write_back(self, request: object | None, entries: Sequence[WriteBack]) -> None:
        for key, value, ttl in entries:
            self.set_local(key, value, ttl)
            self.redis[key] = value
        written: Final = tuple((key, ttl) for key, _value, ttl in entries)
        if request is None:
            self._round_trip("WRITE_BACK", written)
            return
        self._declare(request, f"declare write_back {written}")

    async def delete(self, keys: Sequence[str]) -> None:
        self.delete_sync(keys)

    def delete_sync(self, keys: Sequence[str]) -> None:
        for key in keys:
            self.delete_local(key)
        self._round_trip("DEL", tuple(keys))
        for key in keys:
            self.redis.pop(key, None)

    def declare_get_many(self, request: object, keys: Sequence[str]) -> DeclaredOp:
        self._declare(request, f"declare auth_cache {','.join(keys)}")

        def answer() -> list[object]:
            if self.fail is not None:
                raise self.fail
            return [self.redis.get(key) for key in keys]

        return DeclaredOp(request, answer)

    async def get_many(self, keys: Sequence[str]) -> list[object]:
        self._round_trip("MGET", tuple(keys))
        return [self.redis.get(key) for key in keys]

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

    def set_local(self, key: str, value: object, ttl_seconds: float | None = None) -> None:
        self.memory[key] = (value, ttl_seconds)

    def delete_local(self, key: str) -> None:
        self.memory.pop(key, None)

    def flush_local(self) -> None:
        self.memory.clear()

    def len_local(self) -> tuple[int, int]:
        return (len(self.memory), 0)

    def factory(self, client: NativeRedisClient | None, settings: AuthCacheSettings, redis_ttl: int) -> FakeAuthCache:
        self.settings.append(settings)
        return self


def shared_cache(
    fake: FakeAuthCache | None = None, connection: RedisCache | None = None, **kwargs: float | None
) -> tuple[UserApiKeyCache, FakeAuthCache]:
    """A ``UserApiKeyCache`` with ``enable_redis_auth_cache`` on, its native store replaced by ``fake``."""
    double: Final = fake or FakeAuthCache()
    cache: Final = UserApiKeyCache(
        redis_cache=connection or MagicMock(spec=RedisCache),
        auth_cache_handle=double.factory,
        redis_client=lambda _: None,
        **kwargs,
    )
    return cache, double
