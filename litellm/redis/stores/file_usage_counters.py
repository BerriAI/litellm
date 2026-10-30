from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Literal, Protocol, TypeAlias

from litellm.redis.transport.contract import RedisFailure, RedisTransport

FileUsageSetting: TypeAlias = Literal[
    "max_batch_file_records",
    "max_batch_file_uploads_per_day",
    "max_file_downloads_per_minute",
]
CounterScope: TypeAlias = Literal["key", "user", "team"]


@dataclass(frozen=True, slots=True)
class FileUsageCounter:
    setting: FileUsageSetting
    scope: CounterScope
    scope_id: str
    subject: str
    window_start: int


def counter_key(counter: FileUsageCounter) -> str:
    return (
        f"litellm:file_usage:{counter.setting}:{counter.scope}:{counter.scope_id}"
        f":{counter.subject}:{counter.window_start}"
    )


class FileUsageCounterStore(Protocol):
    async def increment(self, counter: FileUsageCounter, amount: float, *, ttl: timedelta) -> float | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportFileUsageCounterStore:
    transport: RedisTransport

    async def increment(self, counter: FileUsageCounter, amount: float, *, ttl: timedelta) -> float | RedisFailure:
        key: Final = counter_key(counter)
        batch: Final = self.transport.batch(transaction=False)
        incremented: Final = batch.incrbyfloat(key, amount)
        expires_in: Final = batch.ttl(key)
        count: Final = await incremented
        if isinstance(count, RedisFailure):
            return count
        remaining: Final = await expires_in
        if isinstance(remaining, RedisFailure):
            return remaining
        if remaining == -1:
            expired: Final = await self.transport.expire(key, ttl)
            if isinstance(expired, RedisFailure):
                return expired
        return count
