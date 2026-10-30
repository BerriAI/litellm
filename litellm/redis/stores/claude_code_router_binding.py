import ast
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import JsonValue, TypeAdapter

from litellm.redis.transport.contract import RedisFailure, RedisTransport

_PREFIX: Final = "claude_code_session_router:v1:"
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_OBJECT: Final = TypeAdapter(object)


@dataclass(frozen=True, slots=True)
class SessionRouterScope:
    caller: str
    session_id: str


def binding_key(scope: SessionRouterScope) -> str:
    return f"{_PREFIX}{scope.caller}:{scope.session_id}"


def binding_scope(key: str) -> SessionRouterScope | None:
    if not key.startswith(_PREFIX):
        return None
    caller, separator, session_id = key[len(_PREFIX) :].rpartition(":")
    return SessionRouterScope(caller, session_id) if separator and caller and session_id else None


def _decode(raw: bytes) -> object:
    text: Final = raw.decode("utf-8")
    try:
        return _JSON.validate_json(text)
    except ValueError:
        return _OBJECT.validate_python(ast.literal_eval(text))


class ClaudeCodeRouterBindingStore(Protocol):
    async def read_binding(self, scope: SessionRouterScope) -> str | None | RedisFailure: ...

    async def bind_router(self, scope: SessionRouterScope, model: str, *, ttl: timedelta) -> None | RedisFailure: ...

    async def remove_binding(self, scope: SessionRouterScope) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportClaudeCodeRouterBindingStore:
    transport: RedisTransport

    async def read_binding(self, scope: SessionRouterScope) -> str | None | RedisFailure:
        raw: Final = await self.transport.get(binding_key(scope))
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        try:
            value: Final = _decode(raw)
        except (ValueError, SyntaxError, UnicodeError) as error:
            return RedisFailure(kind="decode", message=str(error))
        return value if isinstance(value, str) else None

    async def bind_router(self, scope: SessionRouterScope, model: str, *, ttl: timedelta) -> None | RedisFailure:
        result: Final = await self.transport.set_seconds(binding_key(scope), json.dumps(model), ttl=ttl)
        return result if isinstance(result, RedisFailure) else None

    async def remove_binding(self, scope: SessionRouterScope) -> None | RedisFailure:
        result: Final = await self.transport.delete(binding_key(scope))
        return result if isinstance(result, RedisFailure) else None
