"""
Per-Session Budget Limiter for LiteLLM Proxy.

Enforces a dollar-amount cap per session (identified by `session_id` /
`x-litellm-trace-id`). After each successful LLM call the response cost is
accumulated against the session. When the accumulated spend exceeds
`max_budget_per_session` (configured in agent litellm_params), subsequent
requests for that session receive a 429.

Note: trace-id enforcement (require_trace_id_on_calls_by_agent) is handled
separately in auth_checks.py at the agent level, not in this hook.

Works across multiple proxy instances via DualCache (in-memory + Redis).
Follows the same pattern as max_iterations_limiter.py.
"""

import os
from typing import TYPE_CHECKING, Any, Optional, Union

from fastapi import HTTPException

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

if TYPE_CHECKING:
    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    InternalUsageCache = _InternalUsageCache
else:
    InternalUsageCache = Any


# Redis Lua script for atomic float increment with TTL.
# INCRBYFLOAT returns the new value as a string.
# Only sets EXPIRE on first call (when prior value was nil).
MAX_BUDGET_SESSION_INCREMENT_SCRIPT = """
local key = KEYS[1]
local amount = ARGV[1]
local ttl = tonumber(ARGV[2])

local existed = redis.call('EXISTS', key)
local new_val = redis.call('INCRBYFLOAT', key, amount)
if existed == 0 then
    redis.call('EXPIRE', key, ttl)
end

return new_val
"""

# Default TTL for session budget counters (1 hour)
DEFAULT_MAX_BUDGET_PER_SESSION_TTL = 3600


class _PROXY_MaxBudgetPerSessionHandler(CustomLogger):
    """
    Pre-call hook that enforces max_budget_per_session.

    Configuration (set in agent litellm_params):
        - max_budget_per_session: dollar cap per session_id

    Cache key pattern:
        {session_budget:<session_id>}:spend
    """

    def __init__(self, internal_usage_cache: InternalUsageCache):
        self.internal_usage_cache = internal_usage_cache
        self.ttl = int(
            os.getenv(
                "LITELLM_MAX_BUDGET_PER_SESSION_TTL",
                DEFAULT_MAX_BUDGET_PER_SESSION_TTL,
            )
        )

        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            self.increment_script = (
                self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                    MAX_BUDGET_SESSION_INCREMENT_SCRIPT
                )
            )
        else:
            self.increment_script = None

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Before each LLM call, check if max_budget_per_session is set and
        whether accumulated spend exceeds the budget (429 if so).
        """
        max_budget = self._get_max_budget_per_session(user_api_key_dict)

        session_id = self._get_session_id(data)

        if max_budget is None or session_id is None:
            return None

        max_budget = float(max_budget)
        cache_key = self._make_cache_key(session_id)
        current_spend = await self._get_current_spend(cache_key)

        verbose_proxy_logger.debug(
            "MaxBudgetPerSessionHandler: session_id=%s, spend=%.4f, max=%.2f",
            session_id,
            current_spend,
            max_budget,
        )

        if current_spend >= max_budget:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Session budget exceeded for session {session_id}. "
                    f"Current spend: ${current_spend:.4f}, "
                    f"max_budget_per_session: ${max_budget:.2f}."
                ),
            )

        return None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        After a successful LLM call, increment the session spend by the response cost.
        """
        try:
            litellm_params = kwargs.get("litellm_params") or {}
            metadata = (
                litellm_params.get("metadata")
                or litellm_params.get("litellm_metadata")
                or {}
            )
            session_id = metadata.get("session_id")
            if session_id is None:
                return

            agent_id = metadata.get("agent_id")
            if agent_id is None:
                return

            from litellm.proxy.agent_endpoints.agent_registry import (
                global_agent_registry,
            )

            agent = global_agent_registry.get_agent_by_id(agent_id=str(agent_id))
            if agent is None:
                return

            agent_litellm_params = agent.litellm_params or {}
            max_budget = agent_litellm_params.get("max_budget_per_session")
            if max_budget is None:
                return

            response_cost = kwargs.get("response_cost") or 0.0
            if response_cost <= 0:
                return

            cache_key = self._make_cache_key(str(session_id))
            await self._increment_spend(cache_key, float(response_cost))

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

    def _get_session_id(self, data: dict) -> Optional[str]:
        """Extract session_id from request metadata."""
        metadata = data.get("metadata") or {}
        session_id = metadata.get("session_id")
        if session_id is not None:
            return str(session_id)

        litellm_metadata = data.get("litellm_metadata") or {}
        session_id = litellm_metadata.get("session_id")
        if session_id is not None:
            return str(session_id)

        return None

    def _get_max_budget_per_session(
        self, user_api_key_dict: UserAPIKeyAuth
    ) -> Optional[float]:
        """Extract max_budget_per_session from agent litellm_params."""
        agent_id = user_api_key_dict.agent_id
        if agent_id is None:
            return None

        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        agent = global_agent_registry.get_agent_by_id(agent_id=agent_id)
        if agent is None:
            return None

        litellm_params = agent.litellm_params or {}
        max_budget = litellm_params.get("max_budget_per_session")
        if max_budget is not None:
            return float(max_budget)
        return None

    def _make_cache_key(self, session_id: str) -> str:
        return f"{{session_budget:{session_id}}}:spend"

    async def _get_current_spend(self, cache_key: str) -> float:
        """Read current accumulated spend for a session."""
        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            try:
                result = await self.internal_usage_cache.dual_cache.redis_cache.async_get_cache(
                    key=cache_key
                )
                if result is not None:
                    return float(result)
                return 0.0
            except Exception as e:
                verbose_proxy_logger.warning(
                    "MaxBudgetPerSessionHandler: Redis GET failed, "
                    "falling back to in-memory: %s",
                    str(e),
                )

        result = await self.internal_usage_cache.async_get_cache(
            key=cache_key,
            litellm_parent_otel_span=None,
            local_only=True,
        )
        if result is not None:
            return float(result)
        return 0.0

    async def _increment_spend(self, cache_key: str, amount: float) -> float:
        """Atomically increment the session spend and return the new value."""
        if self.increment_script is not None:
            try:
                result = await self.increment_script(
                    keys=[cache_key],
                    args=[str(amount), self.ttl],
                )
                return float(result)
            except Exception as e:
                verbose_proxy_logger.warning(
                    "MaxBudgetPerSessionHandler: Redis INCRBYFLOAT failed, "
                    "falling back to in-memory: %s",
                    str(e),
                )

        return await self._in_memory_increment_spend(cache_key, amount)

    async def _in_memory_increment_spend(self, cache_key: str, amount: float) -> float:
        current = await self.internal_usage_cache.async_get_cache(
            key=cache_key,
            litellm_parent_otel_span=None,
            local_only=True,
        )
        new_value = (float(current) if current is not None else 0.0) + amount
        await self.internal_usage_cache.async_set_cache(
            key=cache_key,
            value=new_value,
            ttl=self.ttl,
            litellm_parent_otel_span=None,
            local_only=True,
        )
        return new_value
