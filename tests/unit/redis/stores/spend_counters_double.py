"""A stand-in for ``NativeSpendCounters`` with the native store's semantics: Redis first, this worker's memory
behind it, every Redis round trip logged in ``commands``."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal, TypeVar

from litellm.redis.stores.spend_counters import SpendCounterStore, SpendIncrement
from litellm.rust_bridge._native import NativeRedisClient, NativeSpendCounterError

T = TypeVar("T")


def _done(value: T) -> asyncio.Future[T]:
    future: Final[asyncio.Future[T]] = asyncio.get_running_loop().create_future()
    future.set_result(value)
    return future


def _failed(error: BaseException) -> asyncio.Future[T]:
    future: Final[asyncio.Future[T]] = asyncio.get_running_loop().create_future()
    future.set_exception(error)
    return future


def unavailable(kind: str = "transport") -> NativeSpendCounterError:
    return NativeSpendCounterError(kind, "Redis did not answer")


def _is_number(value: object) -> bool:
    """What the native codec decodes; anything else reads as a failed Redis read."""
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))


def _check(counter: str) -> None:
    if not counter.startswith("spend:"):
        raise ValueError(f"{counter!r} is not a spend counter")


@dataclass
class DeclaredRead:
    counters: tuple[str, ...]
    owner: FakeSpendCounters
    request: object

    async def resolve(self) -> Sequence[float | None] | None:
        events: Final = getattr(self.request, "events", None)
        if events is not None and "flush_pre" not in events:
            raise ConnectionError("resolved before its batch was flushed")
        return self.owner.mget(self.counters)


@dataclass
class FakeSpendCounters:
    """``redis`` is None for a worker without Redis, and ``redis_default`` answers reads of a key it lacks. ``fail``
    makes every Redis call fail with ``failure``, ``fail_pipelines`` only the increment pipelines and ``failing`` only
    the named writes ("set", "delete")."""

    redis: dict[str, object] | None = field(default_factory=dict)
    memory: dict[str, float] = field(default_factory=dict)
    commands: list[str] = field(default_factory=list)
    fail: bool = False
    fail_pipelines: bool = False
    failing: frozenset[str] = frozenset()
    failure: str = "transport"
    redis_default: float | None = None
    redis_ttl: float = 60
    batch_reads: bool = True
    read_hook: Callable[[str], Awaitable[None]] | None = None
    on_delete: Callable[[tuple[str, ...]], None] | None = None
    deletes: list[tuple[str, ...]] = field(default_factory=list)
    post_call: list[tuple[object, tuple[SpendIncrement, ...]]] = field(default_factory=list)
    declared: list[DeclaredRead] = field(default_factory=list)

    def store(self) -> SpendCounterStore:
        client: Final = None if self.redis is None else NativeRedisClient.connect({"host": "127.0.0.1", "port": 1})
        return SpendCounterStore(client, redis_ttl=self.redis_ttl, handle=lambda _, __: self)

    def mget(self, counters: Sequence[str]) -> list[float | None] | None:
        if self.redis is None or self.fail or not self.batch_reads:
            return None
        self.commands.append(f"MGET {' '.join(counters)}")
        values: Final = [self.redis.get(counter, self.redis_default) for counter in counters]
        return values if all(_is_number(value) for value in values) else None

    async def read(self, counter: str) -> tuple[Literal["redis", "memory"], float | None]:
        _check(counter)
        if self.read_hook is not None:
            await self.read_hook(counter)
        if self.redis is not None and not self.fail:
            self.commands.append(f"GET {counter}")
            value: Final = self.redis.get(counter, self.redis_default)
            if _is_number(value):
                return "redis", value
        return "memory", self.memory.get(counter)

    def read_many(self, counters: Sequence[str]) -> asyncio.Future[Sequence[float | None] | None]:
        return _done(self.mget(counters))

    def declare_read_many(self, request: object, counters: Sequence[str]) -> DeclaredRead | None:
        if self.redis is None or not self.batch_reads:
            return None
        declared: Final = DeclaredRead(tuple(counters), self, request)
        self.declared.append(declared)
        events: Final = getattr(request, "events", None)
        if isinstance(events, list):
            events.append("declare spend_counters")
        return declared

    def increment(self, counter: str, amount: float) -> asyncio.Future[float]:
        _check(counter)
        if self.redis is None:
            return _done(self.increment_local(counter, amount))
        if self.fail:
            self.memory.pop(counter, None)
            return _failed(unavailable(self.failure))
        self.commands.append(f"INCRBYFLOAT {counter} {amount}")
        return _done(self._incr(counter, amount))

    def increment_many(self, increments: Sequence[SpendIncrement]) -> asyncio.Future[Sequence[float]]:
        if self.redis is None:
            return _done([self.increment_local(counter, amount) for counter, amount in increments])
        if self.fail or self.fail_pipelines:
            return _failed(unavailable(self.failure))
        self.commands.append(f"PIPELINE {' '.join(counter for counter, _ in increments)}")
        return _done([self._incr(counter, amount) for counter, amount in increments])

    def increment_many_post_call(self, request: object, increments: Sequence[SpendIncrement]) -> bool:
        if self.redis is None:
            return False
        self.post_call.append((request, tuple(increments)))
        return True

    def settle_post_call(self) -> None:
        """What the native task does once the post-call batch goes out."""
        pending: Final = tuple(self.post_call)
        self.post_call.clear()
        for _, increments in pending:
            if self.fail:
                for counter, _ in increments:
                    self.memory.pop(counter, None)
                continue
            self.commands.append(f"POST PIPELINE {' '.join(counter for counter, _ in increments)}")
            for counter, amount in increments:
                self._incr(counter, amount)

    def set_max(self, counter: str, spend: float) -> asyncio.Future[float | None]:
        _check(counter)
        if self.memory.get(counter) is None or self.memory[counter] < spend:
            self.memory[counter] = spend
        if self.redis is None:
            return _done(None)
        if self.fail:
            return _failed(unavailable(self.failure))
        self.commands.append(f"SETMAX {counter} {spend}")
        current: Final = self.redis.get(counter)
        if current is None or current < spend:
            self.redis[counter] = spend
        return _done(float(self.redis[counter]))

    def seed(self, counter: str, spend: float, *, increment_if_lost: bool) -> asyncio.Future[float]:
        _check(counter)
        if self.redis is None:
            cached: Final = self.memory.get(counter)
            seeded: Final = spend if cached is None or cached <= spend else float(cached)
            self.memory[counter] = seeded
            return _done(seeded)
        if self.fail:
            return _failed(unavailable(self.failure))
        self.commands.append(f"SETNX {counter} {spend}")
        current: Final = self.redis.setdefault(counter, spend)
        self.memory[counter] = float(current)
        return _done(float(current))

    def set(self, counter: str, spend: float, ttl_seconds: float | None = None) -> asyncio.Future[None]:
        _check(counter)
        self.memory[counter] = spend
        if self.redis is None:
            return _done(None)
        if self.fail or "set" in self.failing:
            return _failed(unavailable(self.failure))
        self.commands.append(f"SET {counter} {spend} {ttl_seconds}")
        self.redis[counter] = spend
        return _done(None)

    def deleted(self) -> frozenset[str]:
        """Every counter a delete reached, memory and Redis alike."""
        return frozenset(counter for batch in self.deletes for counter in batch)

    def delete(self, counters: Sequence[str]) -> asyncio.Future[None]:
        for counter in counters:
            _check(counter)
            self.memory.pop(counter, None)
        self.deletes.append(tuple(counters))
        if self.on_delete is not None:
            self.on_delete(tuple(counters))
        if self.redis is None:
            return _done(None)
        if self.fail or "delete" in self.failing:
            return _failed(unavailable(self.failure))
        self.commands.append(f"DEL {' '.join(counters)}")
        for counter in counters:
            self.redis.pop(counter, None)
        return _done(None)

    def refresh_ttl(self, counter: str) -> asyncio.Future[bool]:
        _check(counter)
        if self.redis is None:
            return _done(False)
        if self.fail:
            return _failed(unavailable(self.failure))
        self.commands.append(f"EXPIRE {counter}")
        return _done(counter in self.redis)

    def get_local(self, counter: str) -> float | None:
        _check(counter)
        return self.memory.get(counter)

    def set_local(self, counter: str, spend: float, ttl_seconds: float | None = None) -> None:
        _check(counter)
        self.memory[counter] = spend

    def delete_local(self, counter: str) -> None:
        _check(counter)
        self.memory.pop(counter, None)

    def increment_local(self, counter: str, amount: float) -> float:
        _check(counter)
        self.memory[counter] = self.memory.get(counter, 0) + amount
        return self.memory[counter]

    def flush_local(self) -> None:
        self.memory.clear()

    def snapshot(self) -> dict[str, float]:
        return dict(self.memory)

    def _incr(self, counter: str, amount: float) -> float:
        assert self.redis is not None
        self.redis[counter] = float(self.redis.get(counter, 0.0)) + amount
        self.memory[counter] = self.redis[counter]
        return self.redis[counter]
