import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport

_JSON: Final = TypeAdapter[JsonValue](JsonValue)
_READ: Final = """
local current = redis.call('GET', KEYS[1])
if current ~= false then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return current
"""


def affinity_key(model_name: str, caller_scope: str, session_id: str) -> str:
    return f"complexity_router_session_affinity:v1:{model_name}:{caller_scope}:{session_id}"


def pin_value(model: str, tier: str | None) -> Mapping[str, str | None]:
    return {"model": model, "tier": tier}


@dataclass(frozen=True, slots=True)
class ComplexitySessionPin:
    value: object


class ComplexitySessionAffinityStore(Protocol):
    async def read_and_refresh(
        self, model_name: str, caller_scope: str, session_id: str, *, ttl_seconds: int
    ) -> ComplexitySessionPin | RedisFailure: ...

    async def overwrite(
        self, model_name: str, caller_scope: str, session_id: str, model: str, tier: str | None, *, ttl_seconds: int
    ) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportComplexitySessionAffinityStore:
    transport: RedisTransport
    coalesce_reads: bool = False

    async def read_and_refresh(
        self, model_name: str, caller_scope: str, session_id: str, *, ttl_seconds: int
    ) -> ComplexitySessionPin | RedisFailure:
        key: Final = affinity_key(model_name, caller_scope, session_id)
        batch: Final = self.transport.request_batch() if self.coalesce_reads else None
        raw: Final = (
            await batch.eval(_READ, (key,), (int(ttl_seconds),))
            if batch is not None
            else await self.transport.eval(_READ, (key,), (int(ttl_seconds),))
        )
        if isinstance(raw, RedisFailure):
            return raw
        if not isinstance(raw, bytes):
            return ComplexitySessionPin(None)
        try:
            decoded: Final = raw.decode("utf-8")
        except UnicodeError as error:
            return RedisFailure(kind="decode", message=str(error))
        try:
            return ComplexitySessionPin(_JSON.validate_json(decoded))
        except ValidationError:
            return ComplexitySessionPin(decoded)

    async def overwrite(
        self, model_name: str, caller_scope: str, session_id: str, model: str, tier: str | None, *, ttl_seconds: int
    ) -> None | RedisFailure:
        result: Final = await self.transport.set_seconds(
            affinity_key(model_name, caller_scope, session_id),
            json.dumps(pin_value(model, tier)),
            ttl=timedelta(seconds=int(ttl_seconds)),
        )
        if isinstance(result, RedisFailure):
            return result
        return None if result else RedisFailure(kind="decode", message="Unexpected session overwrite reply")
