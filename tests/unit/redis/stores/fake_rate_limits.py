"""A ``RateLimitsHandle`` whose memory tier is the real native store and whose Redis scripts are Python fakes
called the way ``main`` called its registered Lua: ``script(keys=[...], args=[...])`` with ``main``'s key strings,
answering ``main``'s raw replies. Tests keep asserting the wire shape while the limiter talks to the store."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, Generic, TypeVar

from pydantic import TypeAdapter

from litellm.redis.stores.rate_limits import (
    LocalValue,
    NativeAtomicCounter,
    NativeCounter,
    NativeEntry,
    NativeIncrement,
    NativeScope,
    RateLimitStore,
)
from litellm.rust_bridge._native import NativeRateLimits

_T = TypeVar("_T")
RawScript = Callable[..., Awaitable[object]]
RawValue = int | float | str | bytes | None
_RAW: Final = TypeAdapter(list[RawValue])


def scope_key(scope: NativeScope, suffix: str) -> str:
    """``main``'s ``f"{{{key}:{value}}}:{suffix}"``."""
    return f"{{{scope[0]}:{scope[1]}}}:{suffix}"


def counter_key(counter: NativeCounter) -> str:
    return scope_key((counter[0], counter[1]), counter[2])


def window_key(counter: NativeCounter) -> str:
    return scope_key((counter[0], counter[1]), "window")


class Resolved(Generic[_T]):
    """A script call; declared with a send that logs ``events`` (a ``FakeNativeRequest`` standing in for the
    request's ``NativeSend``), it notes the declaration there and refuses to answer before that request's pre-call
    batch was flushed."""

    def __init__(self, reply: Callable[[], Awaitable[_T]], send: object, name: str) -> None:
        self._reply: Final = reply
        self._events: Final = _events(send)
        if self._events is not None:
            self._events.append(f"declare {name}")

    def resolve(self) -> Awaitable[_T]:
        if self._events is not None and "flush_pre" not in self._events:
            raise ConnectionError("resolved before its batch was flushed")
        return asyncio.ensure_future(self._reply())


def _events(request: object) -> list[str] | None:
    events: Final = getattr(request, "events", None)
    return events if isinstance(events, list) else None  # pyright: ignore[reportUnknownVariableType]  # duck-typed test request


def _int(value: RawValue) -> int:
    assert value is not None
    return int(value)


def _text(value: RawValue) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _statused(raw: list[RawValue]) -> tuple[tuple[int, int, int] | None, list[RawValue]]:
    if _int(raw[0]) == 1:
        return (_int(raw[1]) - 1, _int(raw[2]), _int(raw[3])), []
    return None, raw[1:]


@dataclass
class FakeRateLimits:
    """Assign a fake to a ``*_script`` attribute to give the limiter a Redis that runs it; a script the test
    gave no fake for fails loudly."""

    connected: bool = True
    batch_rate_limiter_script: RawScript | None = None
    batch_counter_read_script: RawScript | None = None
    check_and_increment_by_n_script: RawScript | None = None
    window_guarded_token_increment_script: RawScript | None = None
    parallel_acquire_script: RawScript | None = None
    parallel_release_script: RawScript | None = None
    parallel_count_script: RawScript | None = None
    token_increment_script: RawScript | None = None
    redis_increments: list[tuple[str, float, int | None]] = field(default_factory=list)
    post_call: list[str] = field(default_factory=list)
    memory: NativeRateLimits = field(default_factory=lambda: NativeRateLimits(None))
    deferred: list[asyncio.Future[None]] = field(default_factory=list)

    def store(self) -> RateLimitStore:
        return RateLimitStore(None, handle=lambda _client, _ttl, _redis_ttl: self)

    async def _run(
        self, script: RawScript | None, name: str, keys: Sequence[str], args: Sequence[object]
    ) -> list[RawValue]:
        if script is None:
            raise AssertionError(f"the test gave the limiter no fake for {name}")
        return _RAW.validate_python(await script(keys=list(keys), args=list(args)))

    @property
    def has_redis(self) -> bool:
        return self.connected

    def slot_groups(self, scopes: Sequence[NativeScope], force: bool) -> list[list[int]]:
        return self.memory.slot_groups(scopes, force)

    def get_local(self, entry: NativeEntry) -> LocalValue | None:
        return self.memory.get_local(entry)

    def set_local(self, entry: NativeEntry, value: LocalValue, ttl_seconds: float | None) -> None:
        self.memory.set_local(entry, value, ttl_seconds)

    async def get(self, entry: NativeEntry, *, backfill_ttl_seconds: float | None = None) -> LocalValue | None:
        return self.memory.get_local(entry)

    def sliding_window(
        self, counters: Sequence[NativeCounter], now: int, window_size: int, send: object
    ) -> Resolved[list[tuple[str, int]]]:
        keys: Final = [key for counter in counters for key in (window_key(counter), counter_key(counter))]

        async def reply() -> list[tuple[str, int]]:
            raw: Final = await self._run(
                self.batch_rate_limiter_script, "batch_rate_limiter_script", keys, [now, window_size]
            )
            return [(_text(raw[index]) or "", _int(raw[index + 1])) for index in range(0, len(raw), 2)]

        return Resolved(reply, send, "sliding_window")

    def read_counters(self, counters: Sequence[NativeCounter], send: object) -> Resolved[list[str | None]]:
        keys: Final = [key for counter in counters for key in (window_key(counter), counter_key(counter))]

        async def reply() -> list[str | None]:
            raw: Final = await self._run(self.batch_counter_read_script, "batch_counter_read_script", keys, [])
            return [_text(value) for value in raw]

        return Resolved(reply, send, "read_counters")

    def check_and_increment(self, counters: Sequence[NativeAtomicCounter], send: object) -> Resolved[tuple[tuple[int, int, int] | None, list[tuple[int, int]]]]:
        keys: Final = [key for counter, *_ in counters for key in (window_key(counter), counter_key(counter))]
        args: Final = [arg for _counter, *numbers in counters for arg in numbers]

        async def reply() -> tuple[tuple[int, int, int] | None, list[tuple[int, int]]]:
            over_limit, rest = _statused(
                await self._run(self.check_and_increment_by_n_script, "check_and_increment_by_n_script", keys, args)
            )
            return over_limit, [(_int(rest[index]), _int(rest[index + 1])) for index in range(0, len(rest), 2)]

        return Resolved(reply, send, "check_and_increment")

    def guarded_increment(
        self,
        counter: NativeCounter,
        expected_window_start: str,
        amount: float,
        ttl_seconds: int | None,
        send: object,
    ) -> Resolved[tuple[bool, int]]:
        async def reply() -> tuple[bool, int]:
            raw: Final = await self._run(
                self.window_guarded_token_increment_script,
                "window_guarded_token_increment_script",
                [window_key(counter), counter_key(counter)],
                [expected_window_start, amount, ttl_seconds or 0],
            )
            return _int(raw[0]) == 1, _int(raw[1])

        return Resolved(reply, send, "guarded_increment")

    def parallel_acquire(
        self,
        gauges: Sequence[tuple[NativeScope, int]],
        slot_ttl_seconds: int,
        slot_id: str,
        send: object,
    ) -> Resolved[tuple[tuple[int, int, int] | None, list[int]]]:
        keys: Final = [scope_key(scope, "max_parallel_requests") for scope, _limit in gauges]
        args: Final = [arg for _scope, limit in gauges for arg in (limit, slot_ttl_seconds, slot_id)]

        async def reply() -> tuple[tuple[int, int, int] | None, list[int]]:
            over_limit, rest = _statused(
                await self._run(self.parallel_acquire_script, "parallel_acquire_script", keys, args)
            )
            return over_limit, [_int(value) for value in rest]

        return Resolved(reply, send, "parallel_acquire")

    def parallel_release(self, gauges: Sequence[NativeScope], slot_id: str, send: object) -> Resolved[list[int]]:
        keys: Final = [scope_key(scope, "max_parallel_requests") for scope in gauges]

        async def reply() -> list[int]:
            raw: Final = await self._run(
                self.parallel_release_script, "parallel_release_script", keys, [slot_id for _ in gauges]
            )
            return [_int(value) for value in raw]

        return Resolved(reply, send, "parallel_release")

    def parallel_count(
        self, gauges: Sequence[NativeScope], slot_ttl_seconds: int, send: object
    ) -> Resolved[list[int]]:
        keys: Final = [scope_key(scope, "max_parallel_requests") for scope in gauges]

        async def reply() -> list[int]:
            raw: Final = await self._run(
                self.parallel_count_script, "parallel_count_script", keys, [slot_ttl_seconds for _ in gauges]
            )
            return [_int(value) for value in raw]

        return Resolved(reply, send, "parallel_count")

    def token_increment(self, increments: Sequence[NativeIncrement], send: object) -> Resolved[list[float]]:
        keys: Final = [counter_key(counter) for counter, _amount, _ttl in increments]
        args: Final = [arg for _counter, amount, ttl in increments for arg in (amount, ttl or 0)]

        async def reply() -> list[float]:
            raw: Final = await self._run(self.token_increment_script, "token_increment_script", keys, args)
            return [float(_text(value) or 0) for value in raw]

        return Resolved(reply, send, "token_increment")

    async def settle(self) -> None:
        """Runs the post-call writes declared so far, as the request's post-call flush would."""
        await asyncio.gather(*self.deferred)

    def token_increment_post_call(self, request: object, increments: Sequence[NativeIncrement]) -> bool:
        self.post_call.append("token_increment")

        async def run() -> None:
            try:
                await self.token_increment(increments, None).resolve()
            except Exception:  # noqa: BLE001  # the store's post-call path falls back exactly like this
                await self.increment_many(increments)

        self.deferred.append(asyncio.ensure_future(run()))
        return self.connected

    def parallel_release_post_call(self, request: object, gauges: Sequence[NativeScope], slot_id: str) -> bool:
        self.post_call.append("parallel_release")

        async def run() -> None:
            try:
                await self.parallel_release(gauges, slot_id, None).resolve()
            except Exception:  # noqa: BLE001  # the store's post-call path only logs this
                return

        self.deferred.append(asyncio.ensure_future(run()))
        return self.connected

    async def increment(self, increment: NativeIncrement) -> float:
        total: Final = await self.memory.increment(increment)
        if self.connected:
            counter, amount, ttl = increment
            self.redis_increments.append((counter_key(counter), amount, ttl))
        return total

    async def increment_many(self, increments: Sequence[NativeIncrement]) -> None:
        await self.memory.increment_many(increments)
        if self.connected:
            self.redis_increments.extend((counter_key(counter), amount, ttl) for counter, amount, ttl in increments)

    def increment_many_post_call(self, request: object, increments: Sequence[NativeIncrement]) -> None:
        self.post_call.append("increment_many")
        self.deferred.append(asyncio.ensure_future(self.increment_many(increments)))
