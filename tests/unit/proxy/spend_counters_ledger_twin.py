"""Reruns existing post-call counter scenarios with the ledger on, so ``increment_spend_counters`` runs the Rust
counter step. A scenario whose store is the Python ``FakeSpendCounters`` is redefined in its twin file on the Rust
store, which these helpers install: in memory, or over a scripted RESP server whose commands the test can read."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm.proxy.proxy_server as ps
from litellm.redis.store_settings import RedisStoreSettings
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.rust_bridge._native import NativeRedisClient, NativeSpendCounters
from litellm.rust_bridge.host import redis_events
from tests.unit.proxy.spend_tracking.native_budgets_replica import RespRedis

_INCREMENT: Final = ps.increment_spend_counters
_ADDRESS: Final = TypeAdapter(tuple[str, int])


async def _ledger_increment(*args: object, **kwargs: object) -> None:
    await _INCREMENT(*args, **{**kwargs, "windows_from_ledger": True})  # pyright: ignore[reportArgumentType]  # forwards the caller's typed arguments


def _scenario(name: str, keep: Callable[[str], bool]) -> bool:
    return not name.startswith("__") and (not name.startswith(("test_", "Test")) or keep(name))


def _namespace(module: ModuleType) -> Mapping[str, object]:
    return vars(module)


def ledger_scenarios(module: ModuleType, keep: Callable[[str], bool] = lambda _: True) -> Mapping[str, object]:
    """The module's tests and the fixtures and helpers they reach by name, with ``increment_spend_counters`` imported
    by name replaced by its ledger form."""
    return {
        name: _ledger_increment if value is _INCREMENT else value
        for name, value in _namespace(module).items()
        if _scenario(name, keep)
    }


def force_ledger_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ps, "increment_spend_counters", _ledger_increment)


def rust_memory_store(monkeypatch: pytest.MonkeyPatch, values: Mapping[str, float] | None = None) -> SpendCounterStore:
    """The proxy's spend counter store as the Rust in-memory store, holding ``values``."""
    store: Final = SpendCounterStore(connect(None).spend_counters())
    for counter, spend in (values or {}).items():
        store.set_local(counter, spend)
    monkeypatch.setattr(ps, "spend_counter_store", store)
    return store


@dataclass
class RedisCalls:
    """The Redis round trips the Rust client reports, each by its call type: ``MGET``, ``SET`` or ``pipeline``."""

    names: list[str] = field(default_factory=list)

    def record(
        self, event_loop: object, call_type: str, start_time: float, end_time: float, failure: str | None
    ) -> None:
        self.names.append(call_type)

    async def reported(self, count: int) -> tuple[str, ...]:
        """The calls reported once ``count`` have arrived; the client reports them from its own runtime."""
        for _ in range(500):
            if len(self.names) >= count:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        return tuple(self.names)


@dataclass
class SlowIncrementsRedis(RespRedis):
    """A RESP server that holds every ``INCRBYFLOAT`` reply past the client's timeout."""

    increment_delay: float = 5.0

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while header := await reader.readline():
                command = [await _bulk_string(reader) for _ in range(int(header[1:]))]
                if command[0].upper() == "INCRBYFLOAT":
                    await asyncio.sleep(self.increment_delay)
                writer.write(self.answer(tuple(command)))
                await writer.drain()
        finally:
            writer.close()


async def _bulk_string(reader: asyncio.StreamReader) -> str:
    size: Final = int((await reader.readline())[1:])
    return (await reader.readexactly(size + 2))[:-2].decode()


async def _stop_serving(server: asyncio.Server) -> None:
    server.close()
    connections: Final = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    for connection in connections:
        _ = connection.cancel()
    _ = await asyncio.gather(*connections, return_exceptions=True)


def _port(server: asyncio.Server) -> int:
    return _ADDRESS.validate_python(server.sockets[0].getsockname())[1]


@contextmanager
def rust_redis_store(
    monkeypatch: pytest.MonkeyPatch,
    redis: RespRedis,
    calls: RedisCalls | None = None,
    socket_timeout: float | None = None,
) -> Generator[SpendCounterStore]:
    """The proxy's spend counter store as the Rust store over ``redis``, its round trips reported to ``calls``."""
    if calls is not None:
        monkeypatch.setattr(redis_events, "record_redis_call", calls.record)
    loop: Final = asyncio.new_event_loop()
    thread: Final = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    server: Final = asyncio.run_coroutine_threadsafe(asyncio.start_server(redis.serve, "127.0.0.1", 0), loop).result(
        timeout=5
    )
    port: Final = _port(server)
    timeout: Final[Mapping[str, float]] = {} if socket_timeout is None else {"socket_timeout": socket_timeout}
    try:
        client: Final = NativeRedisClient.connect(
            RedisStoreSettings(
                redis_kwargs={"host": "127.0.0.1", "port": port, **timeout},
                namespace=None,
                default_ttl=60,
                breaker=None,
            )
        )
        assert client is not None
        store: Final = SpendCounterStore(NativeSpendCounters(client))
        monkeypatch.setattr(ps, "spend_counter_store", store)
        yield store
    finally:
        asyncio.run_coroutine_threadsafe(_stop_serving(server), loop).result(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()
