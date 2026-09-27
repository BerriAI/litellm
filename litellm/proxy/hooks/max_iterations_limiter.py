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

local total_new_key = KEYS[2]
local agent_key = KEYS[3]
local ttl = tonumber(ARGV[1])
if redis.call('EXISTS', total_new_key) == 0 then
    if redis.call('EXISTS', agent_key) == 1 then
        return redis.error_reply('agent session count exists without migration total')
    end

    if redis.call('EXISTS', legacy_key) == 0 then
        redis.call('SET', legacy_key, '0')
        redis.call('PEXPIRE', legacy_key, ttl * 1000)
    end

    local legacy_ttl = redis.call('PTTL', legacy_key)
    if legacy_ttl == -2 then
        return redis.error_reply('legacy session count expired during migration')
    end

    redis.call('SET', total_new_key, '0')
    if legacy_ttl >= 0 then
        redis.call('PEXPIRE', total_new_key, legacy_ttl + 1000)
    end
end

if redis.call('EXISTS', legacy_key) == 0 then
    return redis.error_reply('legacy session count expired before agent scope')
end

local legacy_value = redis.call('INCR', legacy_key)
local total_new_value = redis.call('INCR', total_new_key)
local agent_existed = redis.call('EXISTS', agent_key)
local agent_value = redis.call('INCR', agent_key)
if agent_existed == 0 then
    local migration_ttl = redis.call('PTTL', total_new_key)
    if migration_ttl >= 0 then
        redis.call('PEXPIRE', agent_key, migration_ttl + 1000)
    end
end

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
        self.ttl = int(os.getenv("LITELLM_MAX_ITERATIONS_TTL", DEFAULT_MAX_ITERATIONS_TTL))

        # Register Lua script with Redis if available (same pattern as v3 limiter)
        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            self.increment_script = cast(
                Callable[..., Awaitable[object]],
                self.internal_usage_cache.dual_cache.redis_cache.async_register_script(MAX_ITERATIONS_INCREMENT_SCRIPT),
            )

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

    def _make_cache_key(self, session_id: str, agent_id: str | None = None) -> str:
        """
        Create cache key for session iteration counter.

        Agent-scoped counters share the legacy session hash tag so migration
        scripts can atomically update both scopes on Redis Cluster.
        """
        if agent_id is None:
            return f"{{session_iterations:{session_id}}}:count"
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        stable_agent_id: Final = json.dumps(global_agent_registry.stable_agent_id(agent_id), separators=(",", ":"))
        return f"{{session_iterations:{session_id}}}:agent:{stable_agent_id}:count"

    def _make_legacy_cache_key(self, session_id: str) -> str:
        return f"{{session_iterations:{session_id}}}:count"

    def _make_total_new_cache_key(self, session_id: str) -> str:
        return f"{{session_iterations:{session_id}}}:agent-scope-total"

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
                ttl=self.ttl if current is None else None,
                litellm_parent_otel_span=None,
                local_only=True,
            )
            return new_value

    async def _increment_agent_and_get(self, session_id: str, agent_id: str) -> int:
        legacy_key: Final = self._make_legacy_cache_key(session_id)
        total_new_key: Final = self._make_total_new_cache_key(session_id)
        agent_key: Final = self._make_cache_key(session_id, agent_id)
        if self.increment_script is not None:
            try:
                result: Final[object] = await self.increment_script(
                    keys=[legacy_key, total_new_key, agent_key],
                    args=[self.ttl],
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
            total_new_value = await self._get_local_count(total_new_key)
            agent_value = await self._get_local_count(agent_key)
            if legacy_value is None:
                if total_new_value is not None:
                    raise RuntimeError("Legacy session count expired before agent scope")
                legacy_value = 0
            total_new_value = total_new_value or 0
            agent_value = agent_value or 0
            new_legacy: Final = legacy_value + 1
            new_total_new: Final = total_new_value + 1
            new_agent: Final = agent_value + 1
            await self.internal_usage_cache.async_set_cache(
                key=legacy_key,
                value=new_legacy,
                ttl=self.ttl if legacy_value == 0 else None,
                litellm_parent_otel_span=None,
                local_only=True,
            )
            await self.internal_usage_cache.async_set_cache(
                key=total_new_key,
                value=new_total_new,
                ttl=self.ttl + 1 if total_new_value == 0 else None,
                litellm_parent_otel_span=None,
                local_only=True,
            )
            await self.internal_usage_cache.async_set_cache(
                key=agent_key,
                value=new_agent,
                ttl=self.ttl + 1 if agent_value == 0 else None,
                litellm_parent_otel_span=None,
                local_only=True,
            )
            return max(new_legacy - new_total_new, 0) + new_agent
