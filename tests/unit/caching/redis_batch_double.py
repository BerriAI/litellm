"""Test doubles for the request's native Redis batches and for a ``RedisCache`` that never reaches a server."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Generic, NoReturn, TypeVar

from litellm._service_logger import ServiceLogging
from litellm.caching.redis_cache import RedisCache, RedisHealth
from litellm.redis.store_settings import RedisBreakerSettings, RedisStoreSettings
from litellm.rust_bridge._native import NativeRedisBreaker

_Answer = TypeVar("_Answer")


class FakeRedisCache(RedisCache):
    """A ``RedisCache`` the native stores connect to by its settings; Python itself must never open a client on it."""

    def __init__(self, namespace: str | None = None, failure_threshold: int = 5) -> None:  # super().__init__ needs a server
        self.namespace = namespace
        self.redis_kwargs = {"host": f"batch-{uuid.uuid4().hex}.invalid", "port": 1, "socket_timeout": 0.2}
        self.service_logger_obj = ServiceLogging()
        breaker: Final = RedisBreakerSettings(
            failure_threshold=failure_threshold, recovery_timeout=30.0, timeout_min_duration=5.0
        )
        self._health = RedisHealth(  # pyright: ignore[reportGeneralTypeIssues]  # the fake skips RedisCache.__init__, which sets it
            NativeRedisBreaker.shared(self.redis_kwargs, breaker), self.service_logger_obj
        )
        self.store_settings = RedisStoreSettings(
            redis_kwargs=self.redis_kwargs, namespace=namespace, default_ttl=60, breaker=breaker
        )
        self.default_ttl = None
        self.alone: list[tuple[str, Any]] = []
        self.store: dict[str, Any] = {}

    def init_async_client(self) -> NoReturn:  # pyright: ignore[reportIncompatibleMethodOverride]  # no server behind the fake
        raise AssertionError("a Python Redis client was opened; Redis traffic goes through the native stores")

    async def async_batch_get_cache(self, key_list: Sequence[str], **kwargs: object) -> dict[str, Any]:  # pyright: ignore[reportIncompatibleMethodOverride]  # records the direct read
        self.alone.append(("MGET", tuple(key_list)))
        return {key: self.store.get(key) for key in key_list}

    async def async_increment(self, key: str, value: float, ttl: int | None = None, **kwargs: object) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]  # records the direct write
        self.alone.append(("INCRBYFLOAT", key, value))
        self.store[key] = float(self.store.get(key, 0.0)) + value
        return self.store[key]


@dataclass
class FakeNativeRequest:
    """Stands in for ``NativeRequestRedis``: records each flush in ``events`` next to what the test's pendings log."""

    post_call_deadline: float
    events: list[str] = field(default_factory=list)

    async def flush_pre(self) -> None:
        self.events.append("flush_pre")

    async def flush_post(self) -> None:
        self.events.append("flush_post")


@dataclass
class FakeNatives:
    built: list[FakeNativeRequest] = field(default_factory=list)

    def __call__(self, post_call_deadline: float) -> FakeNativeRequest:
        self.built.append(FakeNativeRequest(post_call_deadline))
        return self.built[-1]


@dataclass
class FakeNativePending(Generic[_Answer]):
    """A read declared on a ``FakeNativeRequest``; it only answers once that request's pre-call batch was flushed."""

    request: FakeNativeRequest
    answer: Callable[[], _Answer]

    async def resolve(self) -> _Answer:
        if "flush_pre" not in self.request.events:
            raise ConnectionError("resolved before its batch was flushed")
        self.request.events.append("resolve")
        return self.answer()


@dataclass
class ReadyPending(Generic[_Answer]):
    """An op declared with ``NativeSend.Now()``: it answers whenever it is resolved."""

    answer: Callable[[], _Answer]

    async def resolve(self) -> _Answer:
        return self.answer()


def fake_send(_phase: object, request: FakeNativeRequest) -> FakeNativeRequest:
    """A ``native_send_factory`` for ``FakeNatives``: the fake request stands in for its ``NativeSend``."""
    return request


def fake_pending(send: object, answer: Callable[[], _Answer]) -> FakeNativePending[_Answer] | ReadyPending[_Answer]:
    """What a fake handle declares for ``send``: on the fake request's batch, or on its own for ``Now()``."""
    return FakeNativePending(send, answer) if isinstance(send, FakeNativeRequest) else ReadyPending(answer)
