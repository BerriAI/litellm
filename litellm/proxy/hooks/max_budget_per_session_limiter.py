"""
Per-Session Budget Limiter for LiteLLM Proxy.

Enforces a dollar-amount cap per agent and session (identified by `session_id` /
`x-litellm-trace-id`). After each successful LLM call the response cost is
accumulated against the session. When the accumulated spend exceeds
`max_budget_per_session` (configured in agent litellm_params), subsequent
requests for that session receive a 429.

Note: trace-id enforcement (require_trace_id_on_calls_by_agent) is handled
separately in auth_checks.py at the agent level, not in this hook.

Works across multiple proxy instances via DualCache (in-memory + Redis).
Follows the same pattern as max_iterations_limiter.py.
"""

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Final, cast

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_cache import log_redis_failure
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
MAX_BUDGET_SESSION_INCREMENT_SCRIPT: Final = """
local legacy_key = KEYS[1]
local agent_scope_key = KEYS[2]
local amount = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local agent_field = ARGV[3]
if amount == nil or amount <= 0 or amount ~= amount or math.abs(amount) == math.huge then
    return redis.error_reply('agent session spend increment is invalid')
end

if redis.call('EXISTS', agent_scope_key) == 0 then
    if redis.call('EXISTS', legacy_key) == 0 then
        redis.call('SET', legacy_key, '0')
        redis.call('PEXPIRE', legacy_key, ttl * 1000)
    end

    local legacy_ttl = redis.call('PTTL', legacy_key)
    if legacy_ttl == -2 then
        return redis.error_reply('legacy session spend expired during migration')
    end
    if legacy_ttl >= 0 and legacy_ttl <= 1 then
        return redis.error_reply('legacy session spend is expiring before agent scope')
    end

    redis.call('HSET', agent_scope_key, '__total_new', '0')
    if legacy_ttl >= 0 then
        redis.call('PEXPIRE', agent_scope_key, legacy_ttl - 1)
    end
end

if redis.call('EXISTS', legacy_key) == 0 then
    return redis.error_reply('legacy session spend expired before agent scope')
end

local total_new_raw = redis.call('HGET', agent_scope_key, '__total_new')
if total_new_raw == false then
    return redis.error_reply('agent session scope is missing its migration total')
end
local total_new_value = tonumber(total_new_raw)
if total_new_value == nil or total_new_value ~= total_new_value or math.abs(total_new_value) == math.huge then
    return redis.error_reply('agent session migration total is not numeric')
end
local agent_raw = redis.call('HGET', agent_scope_key, agent_field)
local agent_value = tonumber(agent_raw or '0')
if agent_value == nil or agent_value ~= agent_value or math.abs(agent_value) == math.huge then
    return redis.error_reply('agent session counter is not numeric')
end
if math.abs(total_new_value + amount) == math.huge or math.abs(agent_value + amount) == math.huge then
    return redis.error_reply('agent session spend increment is out of range')
end

local legacy_value = tonumber(redis.call('INCRBYFLOAT', legacy_key, amount))
local next_total_new = tonumber(redis.call('HINCRBYFLOAT', agent_scope_key, '__total_new', amount))
local next_agent_value = tonumber(redis.call('HINCRBYFLOAT', agent_scope_key, agent_field, amount))

return tostring(math.max(legacy_value - next_total_new, 0) + next_agent_value)
"""

MAX_BUDGET_SESSION_GET_AGENT_SPEND_SCRIPT: Final = """
local legacy_key = KEYS[1]
local agent_scope_key = KEYS[2]
local legacy_value = tonumber(redis.call('GET', legacy_key)) or 0

if redis.call('EXISTS', agent_scope_key) == 0 then
    return tostring(legacy_value)
end

if redis.call('EXISTS', legacy_key) == 0 then
    return redis.error_reply('legacy session spend expired before agent scope')
end

local total_new_raw = redis.call('HGET', agent_scope_key, '__total_new')
if total_new_raw == false then
    return redis.error_reply('agent session scope is missing its migration total')
end
local total_new_value = tonumber(total_new_raw)
if total_new_value == nil or total_new_value ~= total_new_value or math.abs(total_new_value) == math.huge then
    return redis.error_reply('agent session migration total is not numeric')
end
local agent_raw = redis.call('HGET', agent_scope_key, ARGV[1])
local agent_value = 0
if agent_raw ~= false then
    agent_value = tonumber(agent_raw)
    if agent_value == nil or agent_value ~= agent_value or math.abs(agent_value) == math.huge then
        return redis.error_reply('agent session counter is not numeric')
    end
end

return tostring(math.max(legacy_value - total_new_value, 0) + agent_value)
"""

# Default TTL for session budget counters (1 hour)
DEFAULT_MAX_BUDGET_PER_SESSION_TTL: Final = 3600


class _PROXY_MaxBudgetPerSessionHandler(CustomLogger):
    """
    Pre-call hook that enforces max_budget_per_session.

    Configuration (set in agent litellm_params):
        - max_budget_per_session: dollar cap per agent and session_id

    Cache key pattern:
        {session_budget:<session_id>}:agent:<agent_id>:spend
    """

    def __init__(self, internal_usage_cache: InternalUsageCache):
        self.internal_usage_cache = internal_usage_cache
        self._local_lock = asyncio.Lock()
        self.increment_script: Callable[..., Awaitable[object]] | None = None
        self.get_agent_spend_script: Callable[..., Awaitable[object]] | None = None
        self._registered_redis_cache: object | None = None
        self.ttl = int(
            os.getenv(
                "LITELLM_MAX_BUDGET_PER_SESSION_TTL",
                DEFAULT_MAX_BUDGET_PER_SESSION_TTL,
            )
        )

        self._ensure_redis_scripts()

    def _ensure_redis_scripts(self) -> None:
        redis_cache = self.internal_usage_cache.dual_cache.redis_cache
        if redis_cache is None:
            self.increment_script = None
            self.get_agent_spend_script = None
            self._registered_redis_cache = None
            return
        if redis_cache is self._registered_redis_cache:
            return

        self.increment_script = cast(  # cast-ok: Redis registration returns the callable invoked below
            Callable[..., Awaitable[object]],
            redis_cache.async_register_script(MAX_BUDGET_SESSION_INCREMENT_SCRIPT),
        )
        self.get_agent_spend_script = cast(  # cast-ok: Redis registration returns the callable invoked below
            Callable[..., Awaitable[object]],
            redis_cache.async_register_script(MAX_BUDGET_SESSION_GET_AGENT_SPEND_SCRIPT),
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
        Before each LLM call, check if max_budget_per_session is set and
        whether accumulated spend exceeds the budget (429 if so).
        """
        max_budget = self._get_max_budget_per_session(user_api_key_dict)

        session_id: Final = self._get_session_id(data)
        agent_id: Final = user_api_key_dict.agent_id

        if max_budget is None or session_id is None or agent_id is None:
            return None

        max_budget = float(max_budget)
        current_spend: Final = await self._get_agent_spend(session_id, agent_id)

        verbose_proxy_logger.debug(
            "MaxBudgetPerSessionHandler: session_id=%s, spend=%.4f, max=%.2f",
            session_id,
            current_spend,
            max_budget,
        )

        if current_spend >= max_budget:
            resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(data.get("model") if data else None)
            raise ProxyRateLimitError(
                detail=(
                    f"Session budget exceeded for session {session_id}. "
                    f"Current spend: ${current_spend:.4f}, "
                    f"max_budget_per_session: ${max_budget:.2f}."
                ),
                rate_limit_type=RateLimitType.BUDGET,
                model=resolved_model,
                llm_provider=llm_provider,
            )

        return None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Record every successful agent call so limits added later still see the
        session's accumulated spend. The pre-call hook enforces a cap only when
        one is configured.
        """
        try:
            litellm_params: Final = kwargs.get("litellm_params") or {}
            metadata: Final = litellm_params.get("metadata") or {}
            session_id: Final = metadata.get("session_id")
            if session_id is None:
                return

            agent_id: Final = metadata.get("agent_id")
            if agent_id is None:
                return

            from litellm.proxy.agent_endpoints.agent_registry import (
                global_agent_registry,
            )

            agent: Final = global_agent_registry.get_agent_by_id(agent_id=str(agent_id))
            if agent is None:
                return

            response_cost: Final = kwargs.get("response_cost") or 0.0
            if response_cost <= 0:
                return

            await self._increment_agent_spend(str(session_id), agent.agent_id, float(response_cost))

            verbose_proxy_logger.debug(
                "MaxBudgetPerSessionHandler: incremented session %s spend by %.6f",
                session_id,
                response_cost,
            )
        except Exception as e:
            verbose_proxy_logger.warning(
                "MaxBudgetPerSessionHandler: error in async_log_success_event: %s",
                str(e),
            )
            raise

    def _get_session_id(self, data: dict) -> str | None:
        """Extract session_id from request metadata."""
        metadata: Final = data.get("metadata") or {}
        session_id = metadata.get("session_id")
        if session_id is not None:
            return str(session_id)

        litellm_metadata: Final = data.get("litellm_metadata") or {}
        session_id = litellm_metadata.get("session_id")
        if session_id is not None:
            return str(session_id)

        return None

    def _get_max_budget_per_session(self, user_api_key_dict: UserAPIKeyAuth) -> float | None:
        """Extract max_budget_per_session from agent litellm_params."""
        agent_id: Final = user_api_key_dict.agent_id
        if agent_id is None:
            return None

        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        agent: Final = global_agent_registry.get_agent_by_id(agent_id=agent_id)
        if agent is None:
            return None

        litellm_params: Final = agent.litellm_params or {}
        max_budget: Final = litellm_params.get("max_budget_per_session")
        if max_budget is not None:
            return float(max_budget)
        return None

    def _make_agent_scope_cache_key(self, session_id: str) -> str:
        return f"{{session_budget:{session_id}}}:agent-scope"

    def _make_agent_scope_field(self, agent_id: str) -> str:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        stable_agent_id: Final = json.dumps(global_agent_registry.stable_agent_id(agent_id), separators=(",", ":"))
        return f"agent:{stable_agent_id}"

    def _make_legacy_cache_key(self, session_id: str) -> str:
        return f"{{session_budget:{session_id}}}:spend"

    async def _get_local_scope(self, cache_key: str) -> dict[str, object] | None:
        result: Final[object | None] = cast(  # cast-ok: cache API returns Any; validate value before use
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
            return cast(dict[str, object], result)  # cast-ok: keys are checked and values remain opaque
        raise RuntimeError("Agent session scope cache has an invalid value")

    @staticmethod
    def _get_scope_spend(scope: dict[str, object], field: str) -> float:
        value = scope.get(field)
        if value is None:
            return 0.0
        if isinstance(value, (int, float, str, bytes)):
            return float(value)
        raise RuntimeError("Agent session scope has an invalid counter")

    async def _get_agent_spend(self, session_id: str, agent_id: str) -> float:
        legacy_key: Final = self._make_legacy_cache_key(session_id)
        agent_scope_key: Final = self._make_agent_scope_cache_key(session_id)
        agent_field: Final = self._make_agent_scope_field(agent_id)
        self._ensure_redis_scripts()
        if self.get_agent_spend_script is not None:
            try:
                result: Final[object] = await self.get_agent_spend_script(
                    keys=[legacy_key, agent_scope_key],
                    args=[agent_field],
                )
                if isinstance(result, (int, float, str, bytes)):
                    return float(result)
                raise TypeError(f"Unexpected Redis spend result: {type(result).__name__}")
            except Exception as e:
                log_redis_failure(
                    verbose_proxy_logger,
                    logging.WARNING,
                    "MaxBudgetPerSessionHandler: Redis agent spend read failed",
                    e,
                )
                raise

        legacy_value: Final = await self._get_local_spend(legacy_key)
        agent_scope: Final = await self._get_local_scope(agent_scope_key)
        if legacy_value is None:
            if agent_scope is not None:
                raise RuntimeError("Legacy session spend expired before agent scope")
            return 0.0
        if agent_scope is None:
            return float(legacy_value)
        raw_total_new = agent_scope.get("__total_new")
        if not isinstance(raw_total_new, (int, float, str, bytes)):
            raise RuntimeError("Agent session scope is missing its migration total")
        total_new_value = float(raw_total_new)
        agent_value = self._get_scope_spend(agent_scope, agent_field)
        return max(float(legacy_value) - total_new_value, 0.0) + agent_value

    async def _get_local_spend(self, cache_key: str) -> float | None:
        result: Final[object | None] = cast(  # cast-ok: cache API returns Any; validate value before use
            object | None,
            await self.internal_usage_cache.async_get_cache(
                key=cache_key,
                litellm_parent_otel_span=None,
                local_only=True,
            ),
        )
        if isinstance(result, (int, float, str, bytes)):
            return float(result)
        return None

    async def _increment_agent_spend(self, session_id: str, agent_id: str, amount: float) -> float:
        legacy_key: Final = self._make_legacy_cache_key(session_id)
        agent_scope_key: Final = self._make_agent_scope_cache_key(session_id)
        agent_field: Final = self._make_agent_scope_field(agent_id)
        self._ensure_redis_scripts()
        if self.increment_script is not None:
            try:
                result: Final[object] = await self.increment_script(
                    keys=[legacy_key, agent_scope_key],
                    args=[str(amount), self.ttl, agent_field],
                )
                if isinstance(result, (int, float, str, bytes)):
                    return float(result)
                raise TypeError(f"Unexpected Redis spend result: {type(result).__name__}")
            except Exception as e:
                log_redis_failure(
                    verbose_proxy_logger,
                    logging.WARNING,
                    "MaxBudgetPerSessionHandler: Redis migration increment failed; refusing an unsafe retry",
                    e,
                )
                raise

        async with self._local_lock:
            legacy_value = await self._get_local_spend(legacy_key)
            agent_scope = await self._get_local_scope(agent_scope_key)
            if legacy_value is None:
                if agent_scope is not None:
                    raise RuntimeError("Legacy session spend expired before agent scope")
                legacy_value = 0.0
            if agent_scope is None:
                total_new_value = 0.0
                agent_value = 0.0
            else:
                raw_total_new = agent_scope.get("__total_new")
                if not isinstance(raw_total_new, (int, float, str, bytes)):
                    raise RuntimeError("Agent session scope is missing its migration total")
                total_new_value = float(raw_total_new)
                agent_value = self._get_scope_spend(agent_scope, agent_field)
            new_legacy: Final = float(legacy_value) + amount
            new_total_new: Final = float(total_new_value) + amount
            new_agent: Final = float(agent_value) + amount
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
            return max(new_legacy - new_total_new, 0.0) + new_agent
