import ast
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import TypeAdapter

from litellm.redis.transport.contract import RedisFailure, RedisTransport

_VALUE: Final = TypeAdapter(object)


def binding_key(tenant_id: str, client_id: str) -> str:
    return f"agent_identity:{json.dumps((tenant_id, client_id))}"


def _binding_value(value: object) -> str | RedisFailure:
    if isinstance(value, str):
        return value
    return RedisFailure(kind="decode", message="Agent identity binding is not a string")


class AgentIdentityBindingsStore(Protocol):
    async def get(self, tenant_id: str, client_id: str) -> str | None | RedisFailure: ...

    async def put(self, tenant_id: str, client_id: str, agent_id: str, *, ttl: timedelta) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportAgentIdentityBindingsStore:
    transport: RedisTransport

    async def get(self, tenant_id: str, client_id: str) -> str | None | RedisFailure:
        raw: Final = await self.transport.get(binding_key(tenant_id, client_id))
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        try:
            text: Final = raw.decode("utf-8")
            try:
                return _binding_value(_VALUE.validate_python(json.loads(text)))
            except ValueError:
                return _binding_value(_VALUE.validate_python(ast.literal_eval(text)))
        except (ValueError, SyntaxError):
            pass
        return RedisFailure(kind="decode", message="Agent identity binding is not a string")

    async def put(self, tenant_id: str, client_id: str, agent_id: str, *, ttl: timedelta) -> None | RedisFailure:
        seconds: Final = int(ttl.total_seconds())
        if seconds <= 0:
            return RedisFailure(kind="server", message="invalid expire time in set")
        reply: Final = await self.transport.set(
            binding_key(tenant_id, client_id), json.dumps(agent_id), ttl=timedelta(seconds=seconds)
        )
        return reply if isinstance(reply, RedisFailure) else None
