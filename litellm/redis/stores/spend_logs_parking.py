import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol

from litellm.constants import REDIS_SPEND_LOGS_BUFFER_KEY
from litellm.redis.transport.contract import RedisFailure, RedisTransport


def _json_default(value: object) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)


def encode_spend_log_row(row: Mapping[str, object]) -> str:
    return json.dumps(row, default=_json_default)


class SpendLogsParkingStore(Protocol):
    async def park(self, rows: Sequence[Mapping[str, object]], *, max_rows: int) -> int | RedisFailure: ...

    async def take(self, limit: int) -> tuple[str, ...] | None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportSpendLogsParkingStore:
    transport: RedisTransport

    async def park(self, rows: Sequence[Mapping[str, object]], *, max_rows: int) -> int | RedisFailure:
        batch: Final = self.transport.batch(transaction=True, name="spend_logs_parking")
        pushed: Final = batch.rpush(REDIS_SPEND_LOGS_BUFFER_KEY, *(encode_spend_log_row(row) for row in rows))
        trimmed: Final = batch.ltrim(REDIS_SPEND_LOGS_BUFFER_KEY, -max_rows, -1)
        length: Final = await pushed
        trim: Final = await trimmed
        return trim if isinstance(trim, RedisFailure) and not isinstance(length, RedisFailure) else length

    async def take(self, limit: int) -> tuple[str, ...] | None | RedisFailure:
        popped: Final = await self.transport.lpop(REDIS_SPEND_LOGS_BUFFER_KEY, limit)
        if popped is None or isinstance(popped, RedisFailure):
            return popped
        entries: Final = (popped,) if isinstance(popped, bytes) else popped
        try:
            return tuple(entry.decode("utf-8") for entry in entries)
        except UnicodeDecodeError as error:
            return RedisFailure(kind="decode", message=f"Invalid parked spend log row: {error.reason}")
