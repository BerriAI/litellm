import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from litellm.redis.transport.contract import RedisFailure, RedisTransport


def object_key(unified_object_id: str) -> str:
    return unified_object_id


class ManagedObjectStore(Protocol):
    async def write_object(
        self, unified_object_id: str, record: Mapping[str, object], *, ttl: timedelta | None
    ) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportManagedObjectStore:
    transport: RedisTransport

    async def write_object(
        self, unified_object_id: str, record: Mapping[str, object], *, ttl: timedelta | None
    ) -> None | RedisFailure:
        try:
            encoded: Final = json.dumps(record)
        except (TypeError, ValueError) as exc:
            return RedisFailure(kind="decode", message=str(exc))
        result: Final = await self.transport.set_seconds(object_key(unified_object_id), encoded, ttl=ttl)
        return result if isinstance(result, RedisFailure) else None
