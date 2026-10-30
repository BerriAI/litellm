import ast
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport

UsageSnapshot: TypeAlias = dict[str, int | float]


@dataclass(frozen=True, slots=True)
class InvalidSnapshot:
    empty: bool
    failure: RedisFailure | None = None


LegacySnapshot: TypeAlias = UsageSnapshot | int | float | InvalidSnapshot
_SNAPSHOT: Final = TypeAdapter[UsageSnapshot | int | float](UsageSnapshot | int | float)


def _parse(decoder: Callable[[str], object], text: str) -> object:
    return decoder(text)


@dataclass(frozen=True, slots=True)
class MinuteLimit:
    identity: str
    minute: str
    model: str | None = None


@dataclass(frozen=True, slots=True)
class GlobalLimit:
    pass


Limit: TypeAlias = MinuteLimit | GlobalLimit


def limit_key(limit: Limit) -> str:
    if isinstance(limit, GlobalLimit):
        return "global_max_parallel_requests"
    identity: Final = limit.identity if limit.model is None else f"{limit.identity}::{limit.model}"
    return f"{identity}::{limit.minute}::request_count"


def snapshot(value: object) -> LegacySnapshot | None:
    if isinstance(value, InvalidSnapshot):
        return value
    if value is None:
        return None
    try:
        return _SNAPSHOT.validate_python(value, strict=True)
    except ValidationError:
        return InvalidSnapshot(empty=not value)


def _decode(raw: bytes | None) -> LegacySnapshot | None | RedisFailure:
    if raw is None:
        return None
    try:
        decoded: Final = raw.decode("utf-8")
        try:
            return snapshot(_parse(json.loads, decoded))
        except ValueError:
            return snapshot(_parse(ast.literal_eval, decoded))
    except (ValueError, UnicodeError, SyntaxError):
        return RedisFailure(kind="decode", message="invalid legacy request limit JSON")


class LegacyRequestLimitsStore(Protocol):
    async def read(self, limit: MinuteLimit) -> LegacySnapshot | None | RedisFailure: ...

    async def read_many(
        self, limits: Sequence[Limit]
    ) -> tuple[LegacySnapshot | None | RedisFailure, ...] | RedisFailure: ...

    async def write(self, limit: MinuteLimit, value: UsageSnapshot, *, ttl: timedelta) -> bool | RedisFailure: ...

    async def write_many(
        self, values: Sequence[tuple[MinuteLimit, UsageSnapshot]], *, ttl: timedelta
    ) -> tuple[bool | RedisFailure, ...]: ...


@dataclass(frozen=True, slots=True)
class TransportLegacyRequestLimitsStore:
    transport: RedisTransport

    async def read(self, limit: MinuteLimit) -> LegacySnapshot | None | RedisFailure:
        raw: Final = await self.transport.get(limit_key(limit))
        return raw if isinstance(raw, RedisFailure) else _decode(raw)

    async def read_many(
        self, limits: Sequence[Limit]
    ) -> tuple[LegacySnapshot | None | RedisFailure, ...] | RedisFailure:
        raw: Final = await self.transport.mget(tuple(limit_key(limit) for limit in limits))
        if isinstance(raw, RedisFailure):
            return raw
        decoded: Final = tuple(_decode(value) for value in raw)
        failure: Final = next((value for value in decoded if isinstance(value, RedisFailure)), None)
        if failure is not None:
            return tuple(None if value is None else InvalidSnapshot(empty=not value, failure=failure) for value in raw)
        return decoded

    async def write(self, limit: MinuteLimit, value: UsageSnapshot, *, ttl: timedelta) -> bool | RedisFailure:
        return await self.transport.set(limit_key(limit), json.dumps(value), ttl=ttl)

    async def write_many(
        self, values: Sequence[tuple[MinuteLimit, UsageSnapshot]], *, ttl: timedelta
    ) -> tuple[bool | RedisFailure, ...]:
        batch: Final = self.transport.batch()
        replies: Final = tuple(batch.set(limit_key(limit), json.dumps(value), ttl=ttl) for limit, value in values)
        return tuple([await reply for reply in replies])
