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
if current == ARGV[1] then
  redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return current
"""


def affinity_key(model_group: str, hashed_user_key: str) -> str:
    return f"deployment_affinity:v1:{model_group}:{hashed_user_key}"


@dataclass(frozen=True, slots=True)
class UserPin:
    value: object


@dataclass(frozen=True, slots=True)
class UnconfirmedUserPin:
    pass


class DeploymentUserAffinityStore(Protocol):
    async def read(self, model_group: str, hashed_user_key: str) -> UserPin | RedisFailure: ...

    async def claim(
        self, model_group: str, hashed_user_key: str, model_id: str, *, ttl_seconds: int
    ) -> UserPin | UnconfirmedUserPin | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportDeploymentUserAffinityStore:
    transport: RedisTransport

    async def read(self, model_group: str, hashed_user_key: str) -> UserPin | RedisFailure:
        raw: Final = await self.transport.get(affinity_key(model_group, hashed_user_key))
        if isinstance(raw, RedisFailure):
            return raw
        if raw is None:
            return UserPin(None)
        try:
            decoded: Final = raw.decode("utf-8")
            try:
                return UserPin(_JSON.validate_json(decoded))
            except ValidationError:
                return UserPin(_OBJECT.validate_python(ast.literal_eval(decoded)))
        except (UnicodeError, ValueError, SyntaxError) as error:
            return RedisFailure(kind="decode", message=str(error))

    async def claim(
        self, model_group: str, hashed_user_key: str, model_id: str, *, ttl_seconds: int
    ) -> UserPin | UnconfirmedUserPin | RedisFailure:
        raw: Final = await self.transport.eval(
            _CLAIM,
            (affinity_key(model_group, hashed_user_key),),
            (json.dumps({"model_id": model_id}), int(ttl_seconds)),
        )
        if isinstance(raw, RedisFailure):
            return raw
        if not isinstance(raw, bytes):
            return UnconfirmedUserPin()
        try:
            decoded: Final = raw.decode("utf-8")
        except UnicodeError as error:
            return RedisFailure(kind="decode", message=str(error))
        try:
            return UserPin(_JSON.validate_json(decoded))
        except ValidationError:
            return UserPin(decoded)
