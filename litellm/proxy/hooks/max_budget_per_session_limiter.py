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
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final

from pydantic import ConfigDict, TypeAdapter

from litellm import DualCache
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.exceptions import RateLimitType
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit
from litellm.redis.stores.session_budget import SessionBudgetStore, TransportSessionBudgetStore, spend_key
from litellm.redis.transport.contract import RedisFailure

if TYPE_CHECKING:
    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    InternalUsageCache = _InternalUsageCache
else:
    InternalUsageCache = Any


_OPTIONAL_CACHE_AMOUNT_ADAPTER: Final[TypeAdapter[str | bytes | int | float | None]] = TypeAdapter(
    str | bytes | int | float | None, config=ConfigDict(strict=True, hide_input_in_errors=True)
)

# Default TTL for session budget counters (1 hour)
DEFAULT_MAX_BUDGET_PER_SESSION_TTL: Final = 3600


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

        redis_cache: Final = self.internal_usage_cache.dual_cache.redis_cache
        self._store: Final[SessionBudgetStore | None] = (
            TransportSessionBudgetStore(redis_cache.transport()) if redis_cache is not None else None
        )

    @with_service_target("session_budgets")
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

        if max_budget is None or session_id is None:
            return None

        max_budget = float(max_budget)
        current_spend: Final = await self._get_current_spend(session_id)

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

    @with_service_target("session_budgets")
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        """
        After a successful LLM call, increment the session spend by the response cost.
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

            agent_litellm_params: Final = agent.litellm_params or {}
            max_budget: Final = agent_litellm_params.get("max_budget_per_session")
            if max_budget is None:
                return

            response_cost: Final = kwargs.get("response_cost") or 0.0
            if response_cost <= 0:
                return

            await self._increment_spend(str(session_id), float(response_cost))

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

    def _make_cache_key(self, session_id: str) -> str:
        return spend_key(session_id)

    @with_service_target("session_budgets")
    async def _get_current_spend(self, session_id: str) -> float:
        """Read current accumulated spend for a session."""
        if self._store is not None:
            match await self._store.spend(session_id):
                case None | RedisFailure(kind="transport" | "server"):
                    return 0.0
                case RedisFailure():
                    pass
                case stored:
                    return stored

        cache_result: Final[str | bytes | int | float | None] = _OPTIONAL_CACHE_AMOUNT_ADAPTER.validate_python(
            await self.internal_usage_cache.async_get_cache(
                key=self._make_cache_key(session_id),
                litellm_parent_otel_span=None,
                local_only=True,
            )
        )
        if cache_result is not None:
            return float(cache_result)
        return 0.0

    async def _increment_spend(self, session_id: str, amount: float) -> float:
        """Atomically increment the session spend and return the new value."""
        if self._store is not None:
            added: Final = await self._store.add_spend(session_id, amount, ttl=timedelta(seconds=self.ttl))
            if not isinstance(added, RedisFailure):
                return added

        return await self._in_memory_increment_spend(self._make_cache_key(session_id), amount)

    async def _in_memory_increment_spend(self, cache_key: str, amount: float) -> float:
        current: Final[str | bytes | int | float | None] = _OPTIONAL_CACHE_AMOUNT_ADAPTER.validate_python(
            await self.internal_usage_cache.async_get_cache(
                key=cache_key,
                litellm_parent_otel_span=None,
                local_only=True,
            )
        )
        current_spend: Final = float(current) if current is not None else 0.0
        new_value: Final = current_spend + amount
        await self.internal_usage_cache.async_set_cache(
            key=cache_key,
            value=new_value,
            ttl=self.ttl,
            litellm_parent_otel_span=None,
            local_only=True,
        )
        return new_value


PROXY_MaxBudgetPerSessionHandler: Final = _PROXY_MaxBudgetPerSessionHandler
