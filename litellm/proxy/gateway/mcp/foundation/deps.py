"""GatewayDeps — the leaf adapter bundle injected at the composition root.

Only LEAF adapters live here (clock, cache, http factory, plus loosely-typed
prisma/redis/settings handles). Higher layers receive this frozen bundle and
never reach for a global. ``build_test_deps()`` returns fully in-memory fakes
so the whole gateway can stand up with no network or database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

import httpx


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...


@runtime_checkable
class Cache(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...
    def delete(self, key: str) -> None: ...


@runtime_checkable
class HttpxFactory(Protocol):
    def create_client(self) -> httpx.AsyncClient: ...


@dataclass(frozen=True)
class GatewayDeps:
    prisma: object
    redis: object
    cache: Cache
    settings: object
    clock: Clock
    httpx_factory: HttpxFactory


@dataclass(frozen=True)
class FakeClock:
    _now: datetime = field(default_factory=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc))

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> "FakeClock":
        return FakeClock(_now=self._now + timedelta(seconds=seconds))


@dataclass(frozen=True)
class FakeCache:
    _store: dict[str, str] = field(default_factory=dict[str, str])

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def set(self, key: str, value: str) -> None:
        self._store[key] = value

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


def _no_network_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(599, text="no network in test deps")


@dataclass(frozen=True)
class FakeHttpxFactory:
    def create_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(_no_network_handler))


@dataclass(frozen=True)
class FakeRedis:
    pass


@dataclass(frozen=True)
class FakeSettings:
    pass


def build_test_deps() -> GatewayDeps:
    return GatewayDeps(
        prisma=object(),
        redis=FakeRedis(),
        cache=FakeCache(),
        settings=FakeSettings(),
        clock=FakeClock(),
        httpx_factory=FakeHttpxFactory(),
    )
