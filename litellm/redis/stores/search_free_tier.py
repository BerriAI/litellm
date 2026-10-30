from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from litellm.redis.transport.contract import RedisFailure, RedisTransport


def allocation_key(provider: str, shared_group: str, period_stamp: str) -> str:
    return f"litellm:search_free_tier:{provider}:{shared_group}:{period_stamp}"


class SearchFreeTierStore(Protocol):
    async def allocate_units(
        self, provider: str, shared_group: str, period_stamp: str, units: int, *, ttl: timedelta | None
    ) -> float | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportSearchFreeTierStore:
    transport: RedisTransport

    async def allocate_units(
        self, provider: str, shared_group: str, period_stamp: str, units: int, *, ttl: timedelta | None
    ) -> float | RedisFailure:
        key: Final = allocation_key(provider, shared_group, period_stamp)
        if ttl is None:
            return await self.transport.incrbyfloat(key, units)
        batch: Final = self.transport.post_call_batch() or self.transport.batch()
        increment: Final = batch.incrbyfloat(key, units)
        expiry: Final = batch.ttl(key)
        total: Final = await increment
        current_ttl: Final = await expiry
        if isinstance(total, RedisFailure):
            return total
        if isinstance(current_ttl, RedisFailure):
            return current_ttl
        if current_ttl == -1:
            expiration: Final = await self.transport.expire_seconds(key, ttl)
            if isinstance(expiration, RedisFailure):
                return expiration
        return total
