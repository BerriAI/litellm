"""
Max Iterations Limiter for LiteLLM Proxy.

Enforces a per-agent, per-session cap on the number of LLM calls an agentic loop can make.
Callers send a `session_id` with each request (via `x-litellm-session-id` header
or `metadata.session_id`), and this hook counts calls per session. When the count
exceeds `max_iterations` (configured in agent litellm_params or key metadata), returns 429.

Works across multiple proxy instances via DualCache (in-memory + Redis).
Follows the same pattern as parallel_request_limiter_v3.py.
"""

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Final, cast

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.exceptions import RateLimitType
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit

if TYPE_CHECKING:
    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    InternalUsageCache = _InternalUsageCache
else:
    InternalUsageCache = Any


# Redis Lua scripts keep the legacy aggregate and new per-agent counters in
# sync. All keys use the legacy session hash tag so this also works on Redis
# Cluster. Old proxy instances continue to update the aggregate key.
MAX_ITERATIONS_INCREMENT_SCRIPT: Final = """
local legacy_key = KEYS[1]
if #KEYS == 1 then
    local current = redis.call('INCR', legacy_key)
    if current == 1 then
        redis.call('EXPIRE', legacy_key, tonumber(ARGV[1]))
    end
    return current
end

local agent_scope_key = KEYS[2]
local ttl = tonumber(ARGV[1])
local agent_field = ARGV[2]
if redis.call('EXISTS', agent_scope_key) == 0 then
    if redis.call('EXISTS', legacy_key) == 0 then
        redis.call('SET', legacy_key, '0')
        redis.call('PEXPIRE', legacy_key, ttl * 1000)
    end

    local legacy_ttl = redis.call('PTTL', legacy_key)
    if legacy_ttl == -2 then
        return redis.error_reply('legacy session count expired during migration')
    end
    if legacy_ttl >= 0 and legacy_ttl <= 1 then
        return redis.error_reply('legacy session count is expiring before agent scope')
    end

    redis.call('HSET', agent_scope_key, '__total_new', '0')
    if legacy_ttl >= 0 then
        redis.call('PEXPIRE', agent_scope_key, legacy_ttl - 1)
    end
end

if redis.call('EXISTS', legacy_key) == 0 then
    return redis.error_reply('legacy session count expired before agent scope')
end

local total_new_raw = redis.call('HGET', agent_scope_key, '__total_new')
if total_new_raw == false then
    return redis.error_reply('agent session scope is missing its migration total')
end
if string.match(total_new_raw, '^%d+$') == nil then
    return redis.error_reply('agent session migration total is not an integer')
end
local validated_total_new = tonumber(total_new_raw)
if validated_total_new == nil or validated_total_new >= 9223372036854774784 then
    return redis.error_reply('agent session migration total is out of range')
end

local agent_raw = redis.call('HGET', agent_scope_key, agent_field)
if agent_raw ~= false and string.match(agent_raw, '^%d+$') == nil then
    return redis.error_reply('agent session counter is not an integer')
end
local validated_agent_value = tonumber(agent_raw or '0')
if validated_agent_value >= 9223372036854774784 then
    return redis.error_reply('agent session counter is out of range')
end

local legacy_value = redis.call('INCR', legacy_key)
local total_new_value = redis.call('HINCRBY', agent_scope_key, '__total_new', 1)
local agent_value = redis.call('HINCRBY', agent_scope_key, agent_field, 1)

return math.max(legacy_value - total_new_value, 0) + agent_value
"""

# Default TTL for session iteration counters (1 hour)
DEFAULT_MAX_ITERATIONS_TTL: Final = 3600


class _PROXY_MaxIterationsHandler(CustomLogger):
    """
    Pre-call hook that enforces max_iterations per session.

    Configuration:
        - max_iterations: set in agent litellm_params (preferred)
          e.g. litellm_params={"max_iterations": 25}
          Falls back to key metadata max_iterations for backwards compatibility.
        - session_id: sent by caller via x-litellm-session-id header or
          metadata.session_id in request body

    Cache key pattern:
        {session_iterations:<session_id>}:agent:<agent_id>:count
        Without an agent, retains {session_iterations:<session_id>}:count.

    Multi-instance support:
        Uses Redis Lua scripts for atomic increments when Redis is configured.
        Uses process-local memory only when Redis is not configured; Redis
        errors propagate so a failed shared counter cannot silently bypass the
        limit.
    """

    def __init__(self, internal_usage_cache: InternalUsageCache):
        self.internal_usage_cache = internal_usage_cache
        self._local_lock = asyncio.Lock()
        self.increment_script: Callable[..., Awaitable[object]] | None = None
        self._registered_redis_cache: object | None = None
        self.ttl = int(os.getenv("LITELLM_MAX_ITERATIONS_TTL", DEFAULT_MAX_ITERATIONS_TTL))

        self._ensure_redis_scripts()

    def _ensure_redis_scripts(self) -> None:
        redis_cache = self.internal_usage_cache.dual_cache.redis_cache
        if redis_cache is None:
            self.increment_script = None
            self._registered_redis_cache = None
            return
        if redis_cache is self._registered_redis_cache:
            return

        self.increment_script = cast(
            Callable[..., Awaitable[object]],
            redis_cache.async_register_script(MAX_ITERATIONS_INCREMENT_SCRIPT),
        )
        self._registered_redis_cache = redis_cache

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Exception | str | dict | None:
        """
        Check session iteration count before making the API call.

        Extracts session_id from request metadata and max_iterations from
        agent litellm_params. If the session has exceeded max_iterations, raises 429.
        """
        # Extract session_id from request data
        session_id: Final = self._get_session_id(data)
        if session_id is None:
            return None

        max_iterations: Final = self._get_max_iterations(user_api_key_dict)
        if max_iterations is None:
            return None

        verbose_proxy_logger.debug(
            "MaxIterationsHandler: session_id=%s, max_iterations=%s",
            session_id,
            max_iterations,
        )

        # Increment and check
        if user_api_key_dict.agent_id is None:
            current_count = await self._increment_legacy_and_get(self._make_legacy_cache_key(session_id))
        else:
            current_count = await self._increment_agent_and_get(session_id, user_api_key_dict.agent_id)

        if current_count > max_iterations:
            resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(data.get("model") if data else None)
            raise ProxyRateLimitError(
                detail=(
                    f"Max iterations exceeded for session {session_id}. "
                    f"Current count: {current_count}, max_iterations: {max_iterations}."
                ),
                rate_limit_type=RateLimitType.MAX_ITERATIONS,
                model=resolved_model,
                llm_provider=llm_provider,
            )

        verbose_proxy_logger.debug(
            "MaxIterationsHandler: session_id=%s, count=%s/%s",
            session_id,
            current_count,
            max_iterations,
        )

        return None

    def _get_session_id(self, data: dict) -> str | None:
        """Extract session_id from request metadata."""
        metadata: Final = data.get("metadata") or {}
        session_id = metadata.get("session_id")
        if session_id is not None:
            return str(session_id)

        # Also check litellm_metadata (used for /thread and /assistant endpoints)
        litellm_metadata: Final = data.get("litellm_metadata") or {}
        session_id = litellm_metadata.get("session_id")
        if session_id is not None:
            return str(session_id)

        return None

    def _get_max_iterations(self, user_api_key_dict: UserAPIKeyAuth) -> int | None:
        """Extract max_iterations from agent litellm_params, with fallback to key metadata."""
        # Try agent litellm_params first
        agent_id: Final = user_api_key_dict.agent_id
        if agent_id is not None:
            from litellm.proxy.agent_endpoints.agent_registry import (
                global_agent_registry,
            )

            agent: Final = global_agent_registry.get_agent_by_id(agent_id=agent_id)
            if agent is not None:
                litellm_params: Final = agent.litellm_params or {}
                max_iterations = litellm_params.get("max_iterations")
                if max_iterations is not None:
                    return int(max_iterations)

        # Fallback to key metadata for backwards compatibility
        metadata: Final = user_api_key_dict.metadata or {}
        max_iterations = metadata.get("max_iterations")
        if max_iterations is not None:
            return int(max_iterations)
        return None

    def _make_legacy_cache_key(self, session_id: str) -> str:
        return f"{{session_iterations:{session_id}}}:count"

    def _make_agent_scope_cache_key(self, session_id: str) -> str:
        return f"{{session_iterations:{session_id}}}:agent-scope"

    def _make_agent_scope_field(self, agent_id: str) -> str:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        stable_agent_id: Final = json.dumps(global_agent_registry.stable_agent_id(agent_id), separators=(",", ":"))
        return f"agent:{stable_agent_id}"

    async def _get_local_scope(self, cache_key: str) -> dict[str, object] | None:
        result: Final[object | None] = cast(
            object | None,
            await self.internal_usage_cache.async_get_cache(
                key=cache_key,
                litellm_parent_otel_span=None,
                local_only=True,
            ),
        )
        if result is None:
            return None
        if isinstance(result, dict) and all(isinstance(key, str) for key in result):
            return cast(dict[str, object], result)
        raise RuntimeError("Agent session scope cache has an invalid value")

    async def _get_local_count(self, cache_key: str) -> int | None:
        local_result: Final[object | None] = cast(
            object | None,
            await self.internal_usage_cache.async_get_cache(
                key=cache_key,
                litellm_parent_otel_span=None,
                local_only=True,
            ),
        )
        if isinstance(local_result, (int, float, str, bytes)):
            return int(local_result)
        return None

    async def _increment_legacy_and_get(self, cache_key: str) -> int:
        self._ensure_redis_scripts()
        if self.increment_script is not None:
            result: Final[object] = await self.increment_script(
                keys=[cache_key],
                args=[self.ttl],
            )
            if isinstance(result, (int, float, str, bytes)):
                return int(result)
            raise TypeError(f"Unexpected Redis iteration result: {type(result).__name__}")

        async with self._local_lock:
            current: Final = await self._get_local_count(cache_key)
            new_value: Final = (current or 0) + 1
            await self.internal_usage_cache.async_set_cache(
                key=cache_key,
                value=new_value,
                ttl=self.ttl,
                litellm_parent_otel_span=None,
                local_only=True,
            )
            return new_value

    async def _increment_agent_and_get(self, session_id: str, agent_id: str) -> int:
        legacy_key: Final = self._make_legacy_cache_key(session_id)
        agent_scope_key: Final = self._make_agent_scope_cache_key(session_id)
        agent_field: Final = self._make_agent_scope_field(agent_id)
        self._ensure_redis_scripts()
        if self.increment_script is not None:
            try:
                result: Final[object] = await self.increment_script(
                    keys=[legacy_key, agent_scope_key],
                    args=[self.ttl, agent_field],
                )
                if isinstance(result, (int, float, str, bytes)):
                    return int(result)
                raise TypeError(f"Unexpected Redis iteration result: {type(result).__name__}")
            except Exception as e:
                verbose_proxy_logger.warning(
                    "MaxIterationsHandler: Redis migration increment failed; refusing an unsafe retry: %s",
                    str(e),
                )
                raise

        async with self._local_lock:
            legacy_value = await self._get_local_count(legacy_key)
            agent_scope = await self._get_local_scope(agent_scope_key)
            if legacy_value is None:
                if agent_scope is not None:
                    raise RuntimeError("Legacy session count expired before agent scope")
                legacy_value = 0
            total_new_value = 0
            agent_value = 0
            if agent_scope is not None:
                raw_total_new = agent_scope.get("__total_new")
                if not isinstance(raw_total_new, (int, float, str, bytes)):
                    raise RuntimeError("Agent session scope is missing its migration total")
                raw_agent_value = agent_scope.get(agent_field, 0)
                if not isinstance(raw_agent_value, (int, float, str, bytes)):
                    raise RuntimeError("Agent session scope has an invalid counter")
                total_new_value = int(raw_total_new)
                agent_value = int(raw_agent_value)
            new_legacy: Final = legacy_value + 1
            new_total_new: Final = total_new_value + 1
            new_agent: Final = agent_value + 1
            await self.internal_usage_cache.async_set_cache(
                key=legacy_key,
                value=new_legacy,
                ttl=self.ttl,
                litellm_parent_otel_span=None,
                local_only=True,
            )
            await self.internal_usage_cache.async_set_cache(
                key=agent_scope_key,
                value={
                    **(agent_scope or {}),
                    "__total_new": new_total_new,
                    agent_field: new_agent,
                },
                # Keep this map on a shorter rolling TTL than the aggregate fallback.
                ttl=max(self.ttl - 1, 0),
                litellm_parent_otel_span=None,
                local_only=True,
            )
            return max(new_legacy - new_total_new, 0) + new_agent
