import time
from collections.abc import Awaitable, Mapping
from functools import lru_cache
from typing import Callable

from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_router_logger
from litellm.caching.redis_cache import RedisCache
from litellm.router_strategy.complexity_router.cache_warming.types import (
    CACHE_WARMING_RECORD_SCHEMA_VERSION,
    CacheWarmingAttribution,
    CacheWarmingRecord,
)

_WARMTH_KEY_PREFIX = "complexity_router_cache_warmth:v1"

_CAPTURE_SCRIPT = """
local sessions_key = KEYS[1]
local index_key = KEYS[2]
local member = ARGV[1]
local record_json = ARGV[2]
local now = tonumber(ARGV[3])
local expires_at = tonumber(ARGV[4])
local max_sessions = tonumber(ARGV[5])
local expired = redis.call('ZRANGEBYSCORE', index_key, 0, now)
if #expired > 0 then
    redis.call('HDEL', sessions_key, unpack(expired))
    redis.call('ZREMRANGEBYSCORE', index_key, 0, now)
end
if not redis.call('ZSCORE', index_key, member) and redis.call('ZCARD', index_key) >= max_sessions then
    return 0
end
redis.call('HSET', sessions_key, member, record_json)
redis.call('ZADD', index_key, expires_at, member)
redis.call('EXPIREAT', sessions_key, math.ceil(expires_at))
redis.call('EXPIREAT', index_key, math.ceil(expires_at))
return 1
"""

_LIST_LIVE_SESSIONS_SCRIPT = """
local index_key = KEYS[1]
local now = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
return redis.call('ZRANGEBYSCORE', index_key, '(' .. now, '+inf', 'LIMIT', 0, limit)
"""

_GET_RECORD_SCRIPT = """
return redis.call('HGET', KEYS[1], ARGV[1])
"""

_MEMBERS_ADAPTER: TypeAdapter[tuple[str | bytes, ...]] = TypeAdapter(tuple[str | bytes, ...])


@lru_cache(maxsize=64)
def _warn_redis_missing(auto_router_model_name: str) -> None:
    verbose_router_logger.warning(
        "cache_warming is enabled for auto-router %s but the router cache has no Redis; "
        "cache warming is inactive until Redis is configured",
        auto_router_model_name,
    )


@lru_cache(maxsize=64)
def _warn_session_cap_reached(auto_router_model_name: str) -> None:
    verbose_router_logger.warning(
        "cache_warming: auto-router %s reached max_sessions; new sessions are not captured until "
        "existing records expire or go idle",
        auto_router_model_name,
    )


def _parse_record(raw: object) -> CacheWarmingRecord | None:
    if not isinstance(raw, (str, bytes)):
        return None
    try:
        record = CacheWarmingRecord.model_validate_json(raw)
    except ValidationError:
        return None
    if record.schema_version != CACHE_WARMING_RECORD_SCHEMA_VERSION:
        return None
    return record


def _parse_warmth(raw: object) -> float | None:
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, (str, bytes)):
        try:
            return float(raw)
        except ValueError:
            return None
    return None


class CacheWarmingStore:
    """Per-session capture state for provider prompt-cache warming.

    Session records and their expiry index live in two hash-tagged keys on one
    Redis Cluster slot, so every capture is a single atomic Lua operation that
    prunes expired sessions, enforces the max_sessions cap exactly, and writes
    the record all-or-nothing. No partial write states exist, so no
    compensation or fencing is needed anywhere, and the cap bounds the slot's
    footprint by construction. Warmth stamps are plain single-writer keys with
    their own TTL. Records are last-writer-wins by design: the latest turn is
    the correct replay payload. A script fault raises and fails closed, never
    an empty result mistaken for capacity."""

    def __init__(self, redis_cache: RedisCache | None, auto_router_model_name: str) -> None:
        self.redis_cache = redis_cache
        self.auto_router_model_name = auto_router_model_name
        register = redis_cache.async_register_script if redis_cache is not None else None
        self._capture: Callable[..., Awaitable[object]] | None = register(_CAPTURE_SCRIPT) if register else None
        self._list_live: Callable[..., Awaitable[object]] | None = (
            register(_LIST_LIVE_SESSIONS_SCRIPT) if register else None
        )
        self._get: Callable[..., Awaitable[object]] | None = register(_GET_RECORD_SCRIPT) if register else None

    @staticmethod
    def record_key(auto_router_model_name: str, caller_scope: str, session_id: str) -> str:
        return f"{caller_scope}:{session_id}"

    @staticmethod
    def warmth_key(record_key: str, model_group: str) -> str:
        return f"{_WARMTH_KEY_PREFIX}:{record_key}:{model_group}"

    def sessions_key(self) -> str:
        return f"{{cache_warm:v1:{self.auto_router_model_name}}}:sessions"

    def index_key(self) -> str:
        return f"{{cache_warm:v1:{self.auto_router_model_name}}}:index"

    def _require_redis(self) -> RedisCache | None:
        if self.redis_cache is None:
            _warn_redis_missing(self.auto_router_model_name)
            return None
        return self.redis_cache

    async def get_record(self, key: str) -> CacheWarmingRecord | None:
        if self._require_redis() is None or self._get is None:
            return None
        raw = await self._get(keys=[self.sessions_key()], args=[key])
        return _parse_record(raw)

    async def upsert_session(
        self,
        *,
        caller_scope: str,
        session_id: str,
        payload_compressed: str,
        payload_sha256: str,
        token_estimate: int,
        served_model: str,
        attribution: CacheWarmingAttribution,
        ttl_seconds: int,
        max_sessions: int,
    ) -> None:
        redis_cache = self._require_redis()
        if redis_cache is None or self._capture is None:
            return
        key = self.record_key(self.auto_router_model_name, caller_scope, session_id)
        now = time.time()
        record = CacheWarmingRecord(
            schema_version=CACHE_WARMING_RECORD_SCHEMA_VERSION,
            payload_compressed=payload_compressed,
            payload_sha256=payload_sha256,
            token_estimate=token_estimate,
            last_activity=now,
            served_model=served_model,
            attribution=attribution,
            auto_router_model_name=self.auto_router_model_name,
        )
        try:
            admitted = await self._capture(
                keys=[self.sessions_key(), self.index_key()],
                args=[key, record.model_dump_json(), now, now + ttl_seconds, max_sessions],
            )
        except Exception:  # noqa: BLE001  # a capture fault fails closed: no capture beats an uncapped write
            verbose_router_logger.warning("cache_warming capture script failed; skipping capture", exc_info=True)
            return
        if admitted != 1:
            _warn_session_cap_reached(self.auto_router_model_name)
            return
        await self.mark_warm_attempt(key, served_model, attempted_at=now, ttl_seconds=ttl_seconds)

    async def mark_warm_attempt(self, key: str, model_group: str, attempted_at: float, ttl_seconds: int) -> None:
        redis_cache = self._require_redis()
        if redis_cache is None:
            return
        await redis_cache.async_set_cache(  # pyright: ignore[reportUnknownMemberType]  # RedisCache is legacy-untyped
            key=self.warmth_key(key, model_group), value=attempted_at, ttl=ttl_seconds
        )

    async def get_warmth(self, key: str, model_groups: tuple[str, ...]) -> Mapping[str, float]:
        redis_cache = self._require_redis()
        if redis_cache is None:
            return {}  # mutable-ok: fresh per-call result, not shared state
        return {  # mutable-ok: fresh per-call result, not shared state
            model_group: stamp
            for model_group in model_groups
            if (
                stamp := _parse_warmth(
                    await redis_cache.async_get_cache(self.warmth_key(key, model_group))  # pyright: ignore[reportUnknownMemberType]  # RedisCache is legacy-untyped
                )
            )
            is not None
        }

    async def list_session_keys(self, max_sessions: int) -> tuple[str, ...]:
        if self._require_redis() is None or self._list_live is None:
            return ()
        members = _MEMBERS_ADAPTER.validate_python(
            await self._list_live(keys=[self.index_key()], args=[time.time(), max_sessions])
        )
        return tuple(member.decode("utf-8") if isinstance(member, bytes) else member for member in members)
