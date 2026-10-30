import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import JsonValue, TypeAdapter

from litellm.redis.transport.contract import RedisFailure, RedisTransport

_RECORD: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def vector_resource_key(unified_resource_id: str) -> str:
    return unified_resource_id


class ManagedVectorResourceStore(Protocol):
    async def read_vector_resource(self, unified_resource_id: str) -> JsonValue | RedisFailure: ...

    async def write_vector_resource(
        self, unified_resource_id: str, record: Mapping[str, object], *, ttl: timedelta | None
    ) -> None | RedisFailure: ...

    async def invalidate_vector_resource(
        self, unified_resource_id: str, *, ttl: timedelta | None
    ) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportManagedVectorResourceStore:
    transport: RedisTransport

    async def read_vector_resource(self, unified_resource_id: str) -> JsonValue | RedisFailure:
        raw: Final = await self.transport.get(vector_resource_key(unified_resource_id))
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        try:
            return _RECORD.validate_python(json.loads(raw.decode("utf-8")))
        except (ValueError, UnicodeError):
            try:
                return _RECORD.validate_python(ast.literal_eval(raw.decode("utf-8")))
            except (ValueError, SyntaxError, UnicodeError) as exc:
                return RedisFailure(kind="decode", message=str(exc))

    async def write_vector_resource(
        self, unified_resource_id: str, record: Mapping[str, object], *, ttl: timedelta | None
    ) -> None | RedisFailure:
        try:
            encoded: Final = json.dumps(record)
        except (TypeError, ValueError) as exc:
            return RedisFailure(kind="decode", message=str(exc))
        return await self._write(unified_resource_id, encoded, ttl)

    async def invalidate_vector_resource(
        self, unified_resource_id: str, *, ttl: timedelta | None
    ) -> None | RedisFailure:
        return await self._write(unified_resource_id, "null", ttl)

    async def _write(self, unified_resource_id: str, encoded: str, ttl: timedelta | None) -> None | RedisFailure:
        if ttl is not None and ttl <= timedelta(0):
            return RedisFailure(kind="server", message="invalid expire time in set")
        result: Final = await self.transport.set(vector_resource_key(unified_resource_id), encoded, ttl=ttl)
        return result if isinstance(result, RedisFailure) else None
