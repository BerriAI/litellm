import ast
import json
from dataclasses import dataclass
from typing import Final, Protocol

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport

_JSON: Final = TypeAdapter[JsonValue](JsonValue)
_OBJECT: Final = TypeAdapter(object)
_CLAIM: Final = """
local current = redis.call('GET', KEYS[1])
if current == false then
  redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
  return ARGV[1]
end
if ARGV[3] then
  local decoded, stored = pcall(cjson.decode, current)
  local legacy = current
  if decoded and type(stored) == 'string' then legacy = stored end
  for _, eligible in ipairs(cjson.decode(ARGV[3])) do
    if eligible['model_id'] == legacy then
      redis.call('EXPIRE', KEYS[1], ARGV[2])
      return current
    end
  end
  if decoded and type(stored) == 'table' then
    for _, eligible in ipairs(cjson.decode(ARGV[3])) do
      local matches = true
      for key, value in pairs(eligible) do
        if stored[key] ~= value then matches = false; break end
      end
      for key, _ in pairs(stored) do
        if eligible[key] == nil then matches = false; break end
      end
      if matches then
        redis.call('EXPIRE', KEYS[1], ARGV[2])
        return current
      end
    end
  end
  redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
  return ARGV[1]
end
if current == ARGV[1] then
  redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return current
"""


def affinity_key(model_group: str, caller_scope: str, session_id: str) -> str:
    return f"deployment_affinity:v1:session:{model_group}:{caller_scope}:{session_id}"


@dataclass(frozen=True, slots=True)
class SessionPin:
    value: object


@dataclass(frozen=True, slots=True)
class UnconfirmedSessionPin:
    pass


class DeploymentSessionAffinityStore(Protocol):
    async def read(self, model_group: str, caller_scope: str, session_id: str) -> SessionPin | RedisFailure: ...

    async def claim(
        self,
        model_group: str,
        caller_scope: str,
        session_id: str,
        model_id: str,
        *,
        ttl_seconds: int,
        eligible_model_ids: frozenset[str] | None = None,
    ) -> SessionPin | UnconfirmedSessionPin | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportDeploymentSessionAffinityStore:
    transport: RedisTransport

    async def read(self, model_group: str, caller_scope: str, session_id: str) -> SessionPin | RedisFailure:
        raw: Final = await self.transport.get(affinity_key(model_group, caller_scope, session_id))
        if isinstance(raw, RedisFailure):
            return raw
        if raw is None:
            return SessionPin(None)
        try:
            decoded: Final = raw.decode("utf-8")
            try:
                return SessionPin(_JSON.validate_json(decoded))
            except ValidationError:
                return SessionPin(_OBJECT.validate_python(ast.literal_eval(decoded)))
        except (UnicodeError, ValueError, SyntaxError) as error:
            return RedisFailure(kind="decode", message=str(error))

    async def claim(
        self,
        model_group: str,
        caller_scope: str,
        session_id: str,
        model_id: str,
        *,
        ttl_seconds: int,
        eligible_model_ids: frozenset[str] | None = None,
    ) -> SessionPin | UnconfirmedSessionPin | RedisFailure:
        raw: Final = await self.transport.eval(
            _CLAIM,
            (affinity_key(model_group, caller_scope, session_id),),
            (
                json.dumps({"model_id": model_id}),
                int(ttl_seconds),
                *(
                    (json.dumps(tuple({"model_id": value} for value in eligible_model_ids)),)
                    if eligible_model_ids is not None
                    else ()
                ),
            ),
        )
        if isinstance(raw, RedisFailure):
            return raw
        if not isinstance(raw, bytes):
            return UnconfirmedSessionPin()
        try:
            decoded: Final = raw.decode("utf-8")
        except UnicodeError as error:
            return RedisFailure(kind="decode", message=str(error))
        try:
            return SessionPin(_JSON.validate_json(decoded))
        except ValidationError:
            return SessionPin(decoded)
