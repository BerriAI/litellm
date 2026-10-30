import ast
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport

_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_PREFIX: Final = "pkce_verifier:"


def pkce_verifier_key(state: str) -> str:
    return f"{_PREFIX}{state}"


def pkce_state_from_key(key: str) -> str:
    return key.removeprefix(_PREFIX)


class PKCEVerifierStore(Protocol):
    async def put(self, state: str, verifier: str, *, ttl: timedelta) -> None | RedisFailure: ...

    async def get(self, state: str) -> JsonValue | RedisFailure: ...

    async def delete(self, state: str) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportPKCEVerifierStore:
    transport: RedisTransport
    defer_writes: bool = False

    async def put(self, state: str, verifier: str, *, ttl: timedelta) -> None | RedisFailure:
        payload: Final = json.dumps({"code_verifier": verifier})
        batch: Final = self.transport.request_batch() if self.defer_writes else None
        if batch is not None:
            batch.set(pkce_verifier_key(state), payload, ttl=ttl)
            return None
        reply: Final = await self.transport.set(pkce_verifier_key(state), payload, ttl=ttl)
        return reply if isinstance(reply, RedisFailure) else None

    async def get(self, state: str) -> JsonValue | RedisFailure:
        raw: Final = await self.transport.get(pkce_verifier_key(state))
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        try:
            return _VALUE.validate_json(raw)
        except ValidationError:
            try:
                return _VALUE.validate_python(ast.literal_eval(raw.decode("utf-8")))
            except (ValueError, SyntaxError, UnicodeError):
                return RedisFailure(kind="decode", message="invalid PKCE verifier payload")

    async def delete(self, state: str) -> None | RedisFailure:
        reply: Final = await self.transport.delete(pkce_verifier_key(state))
        return reply if isinstance(reply, RedisFailure) else None
