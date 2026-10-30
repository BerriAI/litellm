from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeAlias

from litellm.constants import (
    REDIS_DAILY_AGENT_SPEND_UPDATE_BUFFER_KEY,
    REDIS_DAILY_END_USER_SPEND_UPDATE_BUFFER_KEY,
    REDIS_DAILY_ORG_SPEND_UPDATE_BUFFER_KEY,
    REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY,
    REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY,
    REDIS_DAILY_TEAM_SPEND_UPDATE_BUFFER_KEY,
    REDIS_UPDATE_BUFFER_KEY,
    REDIS_WINDOW_SPEND_UPDATE_BUFFER_KEY,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.redis.transport.contract import Popped, RedisFailure, RedisTransport

SpendBuffer: TypeAlias = Literal[
    "spend", "daily_user", "daily_team", "daily_org", "daily_end_user", "daily_agent", "daily_tag", "window"
]

_KEYS: Final = MappingProxyType[SpendBuffer, str](
    {
        "spend": REDIS_UPDATE_BUFFER_KEY,
        "daily_user": REDIS_DAILY_SPEND_UPDATE_BUFFER_KEY,
        "daily_team": REDIS_DAILY_TEAM_SPEND_UPDATE_BUFFER_KEY,
        "daily_org": REDIS_DAILY_ORG_SPEND_UPDATE_BUFFER_KEY,
        "daily_end_user": REDIS_DAILY_END_USER_SPEND_UPDATE_BUFFER_KEY,
        "daily_agent": REDIS_DAILY_AGENT_SPEND_UPDATE_BUFFER_KEY,
        "daily_tag": REDIS_DAILY_TAG_SPEND_UPDATE_BUFFER_KEY,
        "window": REDIS_WINDOW_SPEND_UPDATE_BUFFER_KEY,
    }
)


def spend_buffer_key(buffer: SpendBuffer) -> str:
    return _KEYS[buffer]


class SpendUpdateBufferStore(Protocol):
    async def push(self, buffer: SpendBuffer, transactions: object) -> int | RedisFailure: ...

    async def push_all(self, batches: Sequence[tuple[SpendBuffer, object]]) -> tuple[int, ...] | RedisFailure: ...

    async def pop(self, buffer: SpendBuffer, *, count: int) -> tuple[str, ...] | None | RedisFailure: ...

    async def pop_all(
        self, buffers: Sequence[SpendBuffer], *, count: int
    ) -> tuple[tuple[str, ...] | None, ...] | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportSpendUpdateBufferStore:
    transport: RedisTransport

    async def push(self, buffer: SpendBuffer, transactions: object) -> int | RedisFailure:
        return await self.transport.rpush(spend_buffer_key(buffer), safe_dumps(transactions))

    async def push_all(self, batches: Sequence[tuple[SpendBuffer, object]]) -> tuple[int, ...] | RedisFailure:
        batch: Final = self.transport.batch(name="spend_update_buffer")
        pushed: Final = tuple(
            batch.rpush(spend_buffer_key(buffer), safe_dumps(transactions)) for buffer, transactions in batches
        )
        lengths: Final = tuple([await reply for reply in pushed])
        failure: Final = next((length for length in lengths if isinstance(length, RedisFailure)), None)
        return failure if failure is not None else tuple(length for length in lengths if isinstance(length, int))

    async def pop(self, buffer: SpendBuffer, *, count: int) -> tuple[str, ...] | None | RedisFailure:
        return _decode(await self.transport.lpop(spend_buffer_key(buffer), count))

    async def pop_all(
        self, buffers: Sequence[SpendBuffer], *, count: int
    ) -> tuple[tuple[str, ...] | None, ...] | RedisFailure:
        batch: Final = self.transport.batch(name="spend_update_buffer")
        popped: Final = tuple(batch.lpop(spend_buffer_key(buffer), count) for buffer in buffers)
        decoded: Final = tuple([_decode(await reply) for reply in popped])
        failure: Final = next((entries for entries in decoded if isinstance(entries, RedisFailure)), None)
        return failure if failure is not None else tuple(e for e in decoded if not isinstance(e, RedisFailure))


def _decode(popped: Popped | RedisFailure) -> tuple[str, ...] | None | RedisFailure:
    if popped is None or isinstance(popped, RedisFailure):
        return popped
    entries: Final = (popped,) if isinstance(popped, bytes) else popped
    try:
        return tuple(entry.decode("utf-8") for entry in entries)
    except UnicodeDecodeError as error:
        return RedisFailure(kind="decode", message=f"Invalid spend update buffer entry: {error.reason}")
