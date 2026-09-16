from __future__ import annotations

import atexit
import functools
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

from provider_cache import LIFETIME_SECONDS, CacheBusy, CacheEdge, CacheHit, CacheLookup, CacheUnavailable, CaptureLease
from pydantic import TypeAdapter, ValidationError
from redis import Redis
from redis.exceptions import RedisError

REDIS_ARRAY: Final[TypeAdapter[list[bytes]]] = TypeAdapter(list[bytes])

LOOKUP: Final = """
local clock = redis.call('TIME')
local now = clock[1] * 1000 + math.floor(clock[2] / 1000)
local row = redis.call('HMGET', KEYS[1], 'captured', 'expires', 'payload')
if row[3] then
  local captured = tonumber(row[1])
  local expires = tonumber(row[2])
  if captured and expires and captured <= now and expires > now
     and expires - captured == tonumber(ARGV[2]) then
    return {'hit', row[3], tostring(expires - now)}
  end
  redis.call('DEL', KEYS[1])
end
if redis.call('SET', KEYS[2], ARGV[1], 'NX', 'PX', ARGV[3]) then
  return {'lease', tostring(now), tostring(now + tonumber(ARGV[2]))}
end
return {'busy'}
"""

PUBLISH: Final = """
if redis.call('GET', KEYS[2]) ~= ARGV[1] then return 0 end
local clock = redis.call('TIME')
local now = clock[1] * 1000 + math.floor(clock[2] / 1000)
local captured = tonumber(ARGV[2])
local expires = tonumber(ARGV[3])
if captured > now or expires <= now or expires - captured ~= tonumber(ARGV[5]) then return 0 end
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
redis.call('HSET', KEYS[1], 'captured', ARGV[2], 'expires', ARGV[3], 'payload', ARGV[4])
redis.call('PEXPIREAT', KEYS[1], expires)
redis.call('DEL', KEYS[2])
return 1
"""

RELEASE: Final = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
return redis.call('DEL', KEYS[1])
"""

DISCARD: Final = """
if redis.call('HGET', KEYS[1], 'payload') ~= ARGV[1] then return 0 end
return redis.call('DEL', KEYS[1])
"""


class RedisCommands(Protocol):
    def eval(self, script: str, numkeys: int, *args: str | bytes | int) -> object: ...


@dataclass(frozen=True, slots=True)
class RedisResponseStore:
    client: RedisCommands
    namespace: str
    lifetime_ms: int = LIFETIME_SECONDS * 1000
    lease_ms: int = 120_000

    def keys(self, key: str) -> tuple[str, str]:
        prefix: Final = f"e2e-provider-cache:v1:{self.namespace}:{{{key}}}"
        return prefix + ":response", prefix + ":lease"

    def lookup(self, key: str) -> CacheLookup:
        token: Final = uuid.uuid4().hex
        started: Final = time.monotonic()
        try:
            result: Final = self.client.eval(LOOKUP, 2, *self.keys(key), token, self.lifetime_ms, self.lease_ms)
        except (RedisError, OSError):
            return CacheUnavailable()
        try:
            parts: Final = tuple(REDIS_ARRAY.validate_python(result, strict=True))
        except ValidationError:
            return CacheUnavailable()
        if len(parts) == 3 and parts[0] == b"hit" and parts[2].isdigit():
            return CacheHit(parts[1], started + int(parts[2]) / 1000)
        if len(parts) == 3 and parts[0] == b"lease" and parts[1].isdigit() and parts[2].isdigit():
            return CaptureLease(token, int(parts[1]), int(parts[2]))
        if parts == (b"busy",):
            return CacheBusy()
        return CacheUnavailable()

    def publish(self, key: str, lease: CaptureLease, payload: bytes) -> bool:
        try:
            result: Final = self.client.eval(
                PUBLISH, 2, *self.keys(key), lease.token, lease.captured_at_ms, lease.expires_at_ms, payload, self.lifetime_ms,
            )
        except (RedisError, OSError):
            return False
        return result == 1

    def release(self, key: str, lease: CaptureLease) -> bool:
        try:
            result: Final = self.client.eval(RELEASE, 1, self.keys(key)[1], lease.token)
        except (RedisError, OSError):
            return False
        return result == 1

    def discard(self, key: str, payload: bytes) -> bool:
        try:
            result: Final = self.client.eval(DISCARD, 1, self.keys(key)[0], payload)
        except (RedisError, OSError):
            return False
        return result == 1


def redis_store(url: str, namespace: str) -> RedisResponseStore:
    client: Final = Redis.from_url(url, socket_timeout=0.25, socket_connect_timeout=0.25, decode_responses=False)
    return RedisResponseStore(cast(RedisCommands, client), namespace)


def write_metrics(cache: CacheEdge) -> None:
    report: Final = json.dumps({"provider_cache": dict(cache.counters.counts)})
    directory: Final = os.environ.get("E2E_PROVIDER_CACHE_METRICS_DIR")
    if directory:
        try:
            root: Final = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            (root / f"{os.getpid()}.json").write_text(report + "\n")
        except OSError:
            logging.getLogger(__name__).warning("provider cache metrics artifact unavailable")
    logging.getLogger(__name__).info("%s", report)


@functools.lru_cache(maxsize=1)
def configured_cache() -> CacheEdge | None:
    if os.environ.get("E2E_PROVIDER_CACHE", "0") == "0":
        return None
    if os.environ.get("E2E_PROVIDER_CACHE") != "1":
        raise ValueError("E2E_PROVIDER_CACHE must be 0 or 1")
    secret: Final = os.environ.get("E2E_PROVIDER_CACHE_HMAC_KEY", "").encode()
    namespace: Final = os.environ.get("E2E_PROVIDER_CACHE_NAMESPACE", "")
    if len(secret) < 32 or re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", namespace) is None:
        raise ValueError("provider cache requires a dedicated key and namespace")
    cache: Final = CacheEdge(redis_store(os.environ["E2E_PROVIDER_CACHE_REDIS_URL"], namespace), secret)
    atexit.register(write_metrics, cache)
    return cache
