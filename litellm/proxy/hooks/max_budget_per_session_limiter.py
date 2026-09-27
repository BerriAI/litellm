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
local total_new_key = KEYS[2]
local agent_key = KEYS[3]
local amount = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])

if redis.call('EXISTS', total_new_key) == 0 then
    if redis.call('EXISTS', agent_key) == 1 then
        return redis.error_reply('agent session spend exists without migration total')
    end

    if redis.call('EXISTS', legacy_key) == 0 then
        redis.call('SET', legacy_key, '0')
        redis.call('PEXPIRE', legacy_key, ttl * 1000)
    end

    local legacy_ttl = redis.call('PTTL', legacy_key)
    if legacy_ttl == -2 then
        return redis.error_reply('legacy session spend expired during migration')
    end

    redis.call('SET', total_new_key, '0')
    if legacy_ttl >= 0 then
        redis.call('PEXPIRE', total_new_key, legacy_ttl + 1000)
    end
end

if redis.call('EXISTS', legacy_key) == 0 then
    return redis.error_reply('legacy session spend expired before agent scope')
end

local legacy_value = tonumber(redis.call('INCRBYFLOAT', legacy_key, amount))
local total_new_value = tonumber(redis.call('INCRBYFLOAT', total_new_key, amount))
local agent_existed = redis.call('EXISTS', agent_key)
local agent_value = tonumber(redis.call('INCRBYFLOAT', agent_key, amount))
if agent_existed == 0 then
    local migration_ttl = redis.call('PTTL', total_new_key)
    if migration_ttl >= 0 then
        redis.call('PEXPIRE', agent_key, migration_ttl + 1000)
    end
end

return tostring(math.max(legacy_value - total_new_value, 0) + agent_value)
"""

MAX_BUDGET_SESSION_GET_AGENT_SPEND_SCRIPT: Final = """
local legacy_value = tonumber(redis.call('GET', KEYS[1])) or 0
local total_new_value = tonumber(redis.call('GET', KEYS[2]))
local agent_value = tonumber(redis.call('GET', KEYS[3])) or 0

if total_new_value == nil then
    if redis.call('EXISTS', KEYS[3]) == 1 then
        return redis.error_reply('agent session spend exists without migration total')
    end
    return tostring(math.max(legacy_value, agent_value))
end
if redis.call('EXISTS', KEYS[1]) == 0 then
    return redis.error_reply('legacy session spend expired before agent scope')
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
        self.ttl = int(
            os.getenv(
                "LITELLM_MAX_BUDGET_PER_SESSION_TTL",
                DEFAULT_MAX_BUDGET_PER_SESSION_TTL,
            )
        )

        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            self.increment_script = cast(
                Callable[..., Awaitable[object]],
                self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                    MAX_BUDGET_SESSION_INCREMENT_SCRIPT
                ),
            )
            self.get_agent_spend_script = cast(
                Callable[..., Awaitable[object]],
                self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                    MAX_BUDGET_SESSION_GET_AGENT_SPEND_SCRIPT
                ),
            )

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

    def _make_cache_key(self, session_id: str, agent_id: str) -> str:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        stable_agent_id: Final = json.dumps(global_agent_registry.stable_agent_id(agent_id), separators=(",", ":"))
        return f"{{session_budget:{session_id}}}:agent:{stable_agent_id}:spend"

    def _make_legacy_cache_key(self, session_id: str) -> str:
        return f"{{session_budget:{session_id}}}:spend"

    def _make_total_new_cache_key(self, session_id: str) -> str:
        return f"{{session_budget:{session_id}}}:agent-scope-total"

    async def _get_agent_spend(self, session_id: str, agent_id: str) -> float:
        legacy_key: Final = self._make_legacy_cache_key(session_id)
        total_new_key: Final = self._make_total_new_cache_key(session_id)
        agent_key: Final = self._make_cache_key(session_id, agent_id)
        if self.get_agent_spend_script is not None:
            try:
                result: Final[object] = await self.get_agent_spend_script(
                    keys=[legacy_key, total_new_key, agent_key],
                    args=[],
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
        total_new_value: Final = await self._get_local_spend(total_new_key)
        agent_value: Final = await self._get_local_spend(agent_key)
        if legacy_value is None:
            if total_new_value is not None:
                raise RuntimeError("Legacy session spend expired before agent scope")
            return float(agent_value or 0.0)
        if total_new_value is None:
            return max(float(legacy_value), float(agent_value or 0.0))
        return max(float(legacy_value) - float(total_new_value), 0.0) + float(agent_value or 0.0)

    async def _get_local_spend(self, cache_key: str) -> float | None:
        result: Final[object | None] = cast(
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
        total_new_key: Final = self._make_total_new_cache_key(session_id)
        agent_key: Final = self._make_cache_key(session_id, agent_id)
        if self.increment_script is not None:
            try:
                result: Final[object] = await self.increment_script(
                    keys=[legacy_key, total_new_key, agent_key],
                    args=[str(amount), self.ttl],
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
            total_new_value = await self._get_local_spend(total_new_key)
            agent_value = await self._get_local_spend(agent_key)
            if legacy_value is None:
                if total_new_value is not None:
                    raise RuntimeError("Legacy session spend expired before agent scope")
                legacy_value = 0.0
            total_new_value = total_new_value or 0.0
            agent_value = agent_value or 0.0
            new_legacy: Final = float(legacy_value) + amount
            new_total_new: Final = float(total_new_value) + amount
            new_agent: Final = float(agent_value) + amount
            await self.internal_usage_cache.async_set_cache(
                key=legacy_key,
                value=new_legacy,
                ttl=self.ttl if legacy_value == 0.0 else None,
                litellm_parent_otel_span=None,
                local_only=True,
            )
            await self.internal_usage_cache.async_set_cache(
                key=total_new_key,
                value=new_total_new,
                ttl=self.ttl + 1 if total_new_value == 0.0 else None,
                litellm_parent_otel_span=None,
                local_only=True,
            )
            await self.internal_usage_cache.async_set_cache(
                key=agent_key,
                value=new_agent,
                ttl=self.ttl + 1 if agent_value == 0.0 else None,
                litellm_parent_otel_span=None,
                local_only=True,
            )
            return max(new_legacy - new_total_new, 0.0) + new_agent
