import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Final, Literal, Protocol, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm.redis.transport.contract import EvalReply, RedisFailure, RedisTransport

ScopeKey: TypeAlias = Literal["api_key", "team"]
ReservationBackend: TypeAlias = Literal["redis", "memory"]


@dataclass(frozen=True, slots=True)
class BatchEnqueuedTokenScope:
    key: ScopeKey
    value: str
    limit: int


@dataclass(frozen=True, slots=True)
class BatchEnqueuedTokenReservation:
    tokens: int
    scopes: tuple[BatchEnqueuedTokenScope, ...]
    backend: ReservationBackend = "redis"
    owner: str = ""
    reserved_at_monotonic: float = field(default_factory=time.monotonic, compare=False)


RESERVATION_ADAPTER: Final = TypeAdapter(BatchEnqueuedTokenReservation)

_RESERVE: Final = """
local amount = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
if current + amount > limit then
    return {0, current}
end
local updated = redis.call('INCRBY', KEYS[1], amount)
redis.call('EXPIRE', KEYS[1], ttl)
return {1, updated}
"""

_REFUND: Final = """
local updated = redis.call('DECRBY', KEYS[1], tonumber(ARGV[1]))
if updated <= 0 then
    redis.call('DEL', KEYS[1])
end
return 1
"""

_SAVE_RECORD: Final = """
redis.call('SET', KEYS[1], ARGV[1], 'EX', tonumber(ARGV[2]))
return 1
"""

_POP_RECORD: Final = """
local value = redis.call('GET', KEYS[1])
if value and value ~= '' then
    redis.call('SET', KEYS[1], '', 'EX', tonumber(ARGV[1]))
end
return value
"""


def counter_key(scope_key: str, scope_value: str) -> str:
    return f"batch_enqueued_tokens:{scope_key}:{scope_value}"


def record_key(batch_id: str) -> str:
    return f"batch_enqueued_token_reservation:{batch_id}"


@dataclass(frozen=True, slots=True)
class Reserved:
    enqueued: int


@dataclass(frozen=True, slots=True)
class OverLimit:
    enqueued: int


@dataclass(frozen=True, slots=True)
class Popped:
    reservation: BatchEnqueuedTokenReservation


@dataclass(frozen=True, slots=True)
class MalformedRecord:
    pass


@dataclass(frozen=True, slots=True)
class AlreadyPopped:
    pass


def encode_reservation(reservation: BatchEnqueuedTokenReservation) -> str:
    return RESERVATION_ADAPTER.dump_json(reservation).decode("utf-8")


def decode_reservation(raw: object) -> BatchEnqueuedTokenReservation | MalformedRecord:
    try:
        if isinstance(raw, (str, bytes)):
            return RESERVATION_ADAPTER.validate_json(raw)
        return RESERVATION_ADAPTER.validate_python(raw)
    except ValidationError:
        return MalformedRecord()


class EnqueuedTokensStore(Protocol):
    async def reserve(
        self, scope_key: str, scope_value: str, tokens: int, *, limit: int, ttl: timedelta
    ) -> Reserved | OverLimit | RedisFailure: ...

    async def refund(self, scope_key: str, scope_value: str, tokens: int) -> None | RedisFailure: ...

    async def save_record(
        self, batch_id: str, reservation: BatchEnqueuedTokenReservation, *, ttl: timedelta
    ) -> None | RedisFailure: ...

    async def pop_record(
        self, batch_id: str, *, tombstone_ttl: timedelta
    ) -> Popped | AlreadyPopped | MalformedRecord | None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportEnqueuedTokensStore:
    transport: RedisTransport

    async def reserve(
        self, scope_key: str, scope_value: str, tokens: int, *, limit: int, ttl: timedelta
    ) -> Reserved | OverLimit | RedisFailure:
        reply: Final = await self.transport.eval(
            _RESERVE, (counter_key(scope_key, scope_value),), (tokens, _seconds(ttl), limit)
        )
        match reply:
            case RedisFailure():
                return reply
            case (1, int(enqueued)):
                return Reserved(enqueued)
            case (0, int(enqueued)):
                return OverLimit(enqueued)
            case _:
                return _decode_failure("enqueued-token reserve", reply)

    async def refund(self, scope_key: str, scope_value: str, tokens: int) -> None | RedisFailure:
        reply: Final = await self.transport.eval(_REFUND, (counter_key(scope_key, scope_value),), (tokens,))
        return reply if isinstance(reply, RedisFailure) else None

    async def save_record(
        self, batch_id: str, reservation: BatchEnqueuedTokenReservation, *, ttl: timedelta
    ) -> None | RedisFailure:
        reply: Final = await self.transport.eval(
            _SAVE_RECORD, (record_key(batch_id),), (encode_reservation(reservation), _seconds(ttl))
        )
        return reply if isinstance(reply, RedisFailure) else None

    async def pop_record(
        self, batch_id: str, *, tombstone_ttl: timedelta
    ) -> Popped | AlreadyPopped | MalformedRecord | None | RedisFailure:
        reply: Final = await self.transport.eval(_POP_RECORD, (record_key(batch_id),), (_seconds(tombstone_ttl),))
        match reply:
            case RedisFailure() | None:
                return reply
            case b"":
                return AlreadyPopped()
            case bytes():
                decoded: Final = decode_reservation(reply)
                return decoded if isinstance(decoded, MalformedRecord) else Popped(decoded)
            case _:
                return _decode_failure("enqueued-token reservation record", reply)


def _seconds(ttl: timedelta) -> int:
    return ttl // timedelta(seconds=1)


def _decode_failure(what: str, reply: EvalReply) -> RedisFailure:
    return RedisFailure(kind="decode", message=f"unexpected {what} reply: {reply!r}")
