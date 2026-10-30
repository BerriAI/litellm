import hashlib
import json
from dataclasses import dataclass
from typing import Final, Protocol

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport

_JSON: Final = TypeAdapter[JsonValue](JsonValue)
_CLAIM: Final = """
local current = redis.call('GET', KEYS[1])
if current == false then
  redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
  return ARGV[1]
end
local decoded, stored = pcall(cjson.decode, current)
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
"""


def configuration_fingerprint(config_json: str) -> str:
    return hashlib.sha256(config_json.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class TierModelIdentity:
    model_name: str
    configuration: str
    caller: str | None
    session_id: str
    tier: str
    replay_fallback: str | None = None


def affinity_key(identity: TierModelIdentity) -> str:
    parts: Final = (identity.model_name, identity.configuration, identity.caller, identity.session_id, identity.tier)
    cache_identity: Final = (
        (*parts, ("replay_fallback", identity.replay_fallback)) if identity.replay_fallback is not None else parts
    )
    return "complexity_router_tier_model_affinity:v1:" + hashlib.sha256(json.dumps(cache_identity).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class TierModelPin:
    value: object


@dataclass(frozen=True, slots=True)
class UnconfirmedTierModelPin:
    pass


class ComplexityTierModelAffinityStore(Protocol):
    async def claim(
        self, identity: TierModelIdentity, model: str, eligible_models: tuple[str, ...], *, ttl_seconds: int
    ) -> TierModelPin | UnconfirmedTierModelPin | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportComplexityTierModelAffinityStore:
    transport: RedisTransport

    async def claim(
        self, identity: TierModelIdentity, model: str, eligible_models: tuple[str, ...], *, ttl_seconds: int
    ) -> TierModelPin | UnconfirmedTierModelPin | RedisFailure:
        raw: Final = await self.transport.eval(
            _CLAIM,
            (affinity_key(identity),),
            (
                json.dumps({"model": model}),
                int(ttl_seconds),
                json.dumps(tuple({"model": value} for value in eligible_models)),
            ),
        )
        if isinstance(raw, RedisFailure):
            return raw
        if not isinstance(raw, bytes):
            return UnconfirmedTierModelPin()
        try:
            decoded: Final = raw.decode("utf-8")
        except UnicodeError as error:
            return RedisFailure(kind="decode", message=str(error))
        try:
            return TierModelPin(_JSON.validate_json(decoded))
        except ValidationError:
            return TierModelPin(decoded)
