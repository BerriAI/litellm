"""Atomic affinity claims shared by deployment and tier-model selection."""

import json
from collections.abc import Mapping
from typing import (
    Final,
    cast,  # noqa: TID251  # Redis script results are narrowed only to object, then validated
)

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache

_PIN_JSON_ADAPTER: Final = TypeAdapter[JsonValue](JsonValue)

_CLAIM_PIN_SCRIPT: Final = """
local current = redis.call('GET', KEYS[1])
if current == false then
  redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
  return ARGV[1]
end
if ARGV[3] then
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
end
if current == ARGV[1] then
  redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return current
"""


def set_local_affinity_pin(cache: DualCache, cache_key: str, value: object, ttl_seconds: int) -> None:
    """Replace the entry because InMemoryCache.set_cache preserves a live key's expiry."""
    cache.in_memory_cache.delete_cache(cache_key)
    cache.in_memory_cache.set_cache(cache_key, value, ttl=ttl_seconds)


def _legacy_pin_matches(stored: object, pin_value: Mapping[str, str]) -> bool:
    if isinstance(stored, dict):
        return all(stored.get(key) is not None and str(stored[key]) == value for key, value in pin_value.items())
    return isinstance(stored, str) and len(pin_value) == 1 and stored in pin_value.values()


def claim_affinity_pin_in_memory(
    cache: DualCache,
    cache_key: str,
    pin_value: Mapping[str, str],
    ttl_seconds: int,
    *,
    eligible_values: tuple[Mapping[str, str], ...] | None = None,
) -> object:
    """No await between read and write, so same-loop claims agree during a Redis outage."""
    existing: Final[object] = cache.in_memory_cache.get_cache(cache_key)
    if existing is not None and eligible_values is None:
        if _legacy_pin_matches(existing, pin_value):
            set_local_affinity_pin(cache, cache_key, pin_value, ttl_seconds)
        return existing
    winner: Final = existing if existing is not None and existing in (eligible_values or ()) else pin_value
    set_local_affinity_pin(cache, cache_key, winner, ttl_seconds)
    return winner


def _decode_pin(value: str) -> object:
    try:
        return _PIN_JSON_ADAPTER.validate_json(value)
    except ValidationError:
        return value


async def claim_affinity_pin(
    cache: DualCache,
    cache_key: str,
    pin_value: Mapping[str, str],
    ttl_seconds: int,
    *,
    eligible_values: tuple[Mapping[str, str], ...] | None = None,
) -> object:
    """Return the authoritative first writer, replacing it only when it becomes ineligible.

    Eligible claims refresh the returned winner. Legacy deployment claims only refresh
    a matching candidate. Resolve Redis per call because the proxy attaches it lazily.
    """
    redis_cache: Final = cache.redis_cache
    if redis_cache is not None:
        try:
            claim_script: Final = redis_cache.async_register_script(_CLAIM_PIN_SCRIPT)
            args: Final = (
                json.dumps(dict(pin_value)),  # mutable-ok: JSON serialization requires dict, not a generic Mapping
                int(ttl_seconds),
                *(
                    (json.dumps(tuple(dict(value) for value in eligible_values)),)  # mutable-ok: JSON requires dict
                    if eligible_values is not None
                    else ()
                ),
            )
            raw: Final = cast(  # cast-ok: Redis scripts return heterogeneous values; only object is asserted here
                object, await claim_script(keys=(cache_key,), args=args)
            )
            decoded: Final = raw.decode("utf-8") if isinstance(raw, bytes) else raw
            if not isinstance(decoded, str):
                return pin_value
            winner: Final = _decode_pin(decoded)
            set_local_affinity_pin(cache, cache_key, winner, ttl_seconds)
            return winner
        except Exception as error:  # noqa: BLE001  # Redis/Lua faults retain same-pod affinity through local claims
            verbose_router_logger.debug("Affinity cache: Redis claim failed, using pod-local claim. error=%s", error)
    return claim_affinity_pin_in_memory(cache, cache_key, pin_value, ttl_seconds, eligible_values=eligible_values)
