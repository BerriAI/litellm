from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from litellm.redis.transport.contract import RedisFailure, RedisTransport


def charge_marker_key(spend_key: str, batch_id: str) -> str:
    return f"{spend_key}:batch:{batch_id}"


class BatchChargeMarkerStore(Protocol):
    async def claim_batch(self, spend_key: str, batch_id: str, *, ttl: timedelta) -> bool | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportBatchChargeMarkerStore:
    transport: RedisTransport

    async def claim_batch(self, spend_key: str, batch_id: str, *, ttl: timedelta) -> bool | RedisFailure:
        key: Final = charge_marker_key(spend_key, batch_id)
        batch: Final = self.transport.batch(name="batch_charge_marker")
        polled: Final = batch.incrbyfloat(key, 1)
        expired: Final = batch.expire_seconds(key, ttl)
        polls: Final = await polled
        expiry: Final = await expired
        if isinstance(polls, RedisFailure):
            return polls
        if isinstance(expiry, RedisFailure):
            return expiry
        return polls == 1
