import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport

_PAYLOAD: Final = TypeAdapter(dict[str, object])


def login_code_key(code: str) -> str:
    return f"login_code:{code}"


class LoginCodeStore(Protocol):
    async def put(self, code: str, payload: Mapping[str, object], *, ttl: timedelta) -> None | RedisFailure: ...

    async def get(self, code: str) -> Mapping[str, object] | None | RedisFailure: ...

    async def delete(self, code: str) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportLoginCodeStore:
    transport: RedisTransport
    defer_writes: bool = False

    async def put(self, code: str, payload: Mapping[str, object], *, ttl: timedelta) -> None | RedisFailure:
        batch: Final = self.transport.request_batch() if self.defer_writes else None
        if batch is not None:
            batch.set(login_code_key(code), json.dumps(payload), ttl=ttl)
            return None
        reply: Final = await self.transport.set(login_code_key(code), json.dumps(payload), ttl=ttl)
        return reply if isinstance(reply, RedisFailure) else None

    async def get(self, code: str) -> Mapping[str, object] | None | RedisFailure:
        raw: Final = await self.transport.get(login_code_key(code))
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        try:
            return _PAYLOAD.validate_json(raw)
        except ValidationError:
            try:
                return _PAYLOAD.validate_python(ast.literal_eval(raw.decode("utf-8")))
            except (ValueError, SyntaxError, UnicodeError):
                return RedisFailure(kind="decode", message="invalid login code payload")

    async def delete(self, code: str) -> None | RedisFailure:
        reply: Final = await self.transport.delete(login_code_key(code))
        return reply if isinstance(reply, RedisFailure) else None
