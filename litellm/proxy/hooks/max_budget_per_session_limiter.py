"""
Per-Session Budget Limiter for LiteLLM Proxy.

Enforces a dollar-amount cap per session (identified by `session_id` /
`x-litellm-trace-id`). Configured via `max_budget_per_session` in agent
litellm_params. When a session's spend would exceed the cap, requests receive
a 429.

Admission reserves each request's estimated max cost against the session
counter before the call runs (an atomic Redis INCRBYFLOAT, same idea as the
key/team optimistic reservation in budget_reservation.py), so two concurrent
requests for one session can't both read the same below-budget value and both
slip past the gate before either records its cost. After the call the
reservation is reconciled to the actual response cost; on failure it is
refunded. When the request's cost can't be estimated (model not in the cost
map) the hook falls back to read-time enforcement only.

Note: trace-id enforcement (require_trace_id_on_calls_by_agent) is handled
separately in auth_checks.py at the agent level, not in this hook.

Works across multiple proxy instances via DualCache (in-memory + Redis).
"""

import os
import secrets
from typing import TYPE_CHECKING, Any, Optional, Union

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.exceptions import RateLimitType
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit

if TYPE_CHECKING:
    from fastapi import HTTPException

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

_RESERVATION_KEY = "_litellm_session_budget_reservation"
_RESERVATION_TOKEN_FIELD = "token"
_RESERVATION_SESSION_FIELD = "session_id"
_RESERVATION_COST_FIELD = "reserved_cost"
_RESERVATION_RELEASED_FIELD = "released"

_EPSILON = 1e-12


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
        self._reservation_token = secrets.token_hex(16)
        self.ttl = int(
            os.getenv(
                "LITELLM_MAX_BUDGET_PER_SESSION_TTL",
                DEFAULT_MAX_BUDGET_PER_SESSION_TTL,
            )
        )

        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            self.increment_script = self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                MAX_BUDGET_SESSION_INCREMENT_SCRIPT
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
        Reserve this request's estimated max cost against the session counter
        and reject (429) if the reservation would exceed max_budget_per_session.
        Falls back to a read-time spend check when the cost can't be estimated.
        """
        max_budget = self._get_max_budget_per_session(user_api_key_dict)
        session_id = self._get_session_id(data)

        if max_budget is None or session_id is None:
            return None

        max_budget = float(max_budget)
        cache_key = self._make_cache_key(session_id)

        reservation_cost = self._estimate_reservation_cost(data=data, call_type=call_type)
        if reservation_cost is None or reservation_cost <= 0:
            await self._enforce_read_time(data=data, session_id=session_id, cache_key=cache_key, max_budget=max_budget)
            return None

        await self._reserve_and_enforce(
            data=data,
            session_id=session_id,
            cache_key=cache_key,
            max_budget=max_budget,
            reservation_cost=reservation_cost,
        )
        return None

    async def _enforce_read_time(self, data: dict, session_id: str, cache_key: str, max_budget: float) -> None:
        """Read-time only enforcement, used when the request cost is unknown."""
        current_spend = await self._get_current_spend(cache_key)

        verbose_proxy_logger.debug(
            "MaxBudgetPerSessionHandler: session_id=%s, spend=%.4f, max=%.2f (read-time)",
            session_id,
            current_spend,
            max_budget,
        )

        if current_spend >= max_budget:
            raise self._budget_exceeded_error(
                data=data,
                session_id=session_id,
                current_spend=current_spend,
                max_budget=max_budget,
            )

    async def _reserve_and_enforce(
        self,
        data: dict,
        session_id: str,
        cache_key: str,
        max_budget: float,
        reservation_cost: float,
    ) -> None:
        """
        Atomically reserve `reservation_cost` against the session counter, then
        decide admission from the post-increment value so concurrent requests
        serialize on the counter instead of racing a stale read.
        """
        new_total = await self._increment_spend(cache_key, reservation_cost)
        spend_before = new_total - reservation_cost

        if new_total > max_budget:
            remaining_before = max_budget - spend_before
            if remaining_before > _EPSILON:
                await self._increment_spend(cache_key, remaining_before - reservation_cost)
                self._stash_reservation(data=data, session_id=session_id, reserved_cost=remaining_before)
                return

            await self._increment_spend(cache_key, -reservation_cost)
            raise self._budget_exceeded_error(
                data=data,
                session_id=session_id,
                current_spend=spend_before,
                max_budget=max_budget,
            )

        self._stash_reservation(data=data, session_id=session_id, reserved_cost=reservation_cost)

        verbose_proxy_logger.debug(
            "MaxBudgetPerSessionHandler: reserved %.6f for session %s, spend=%.4f/%.2f",
            reservation_cost,
            session_id,
            new_total,
            max_budget,
        )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Reconcile the admission reservation to the call's actual cost. When no
        reservation was made (cost couldn't be estimated at admission), fall
        back to incrementing the session spend by the full response cost.
        """
        try:
            response_cost = float(kwargs.get("response_cost") or 0.0)

            if await self._reconcile_reservation(container=kwargs, actual_cost=response_cost):
                return

            if response_cost <= 0:
                return

            litellm_params = kwargs.get("litellm_params") or {}
            metadata = litellm_params.get("metadata") or {}
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
            if agent_litellm_params.get("max_budget_per_session") is None:
                return

            cache_key = self._make_cache_key(str(session_id))
            await self._increment_spend(cache_key, response_cost)

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

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Refund the admission reservation when the LLM call fails."""
        await self._reconcile_reservation(container=kwargs, actual_cost=0.0)

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        """
        Refund the reservation for a request rejected after admission but before
        the LLM call ran (e.g. a downstream guardrail raised). async_log_failure_event
        is a completion-level callback and never fires for these proxy-side
        rejections, so the reservation would otherwise stay pinned until its TTL.
        """
        await self._reconcile_reservation(container=request_data, actual_cost=0.0)

    async def _reconcile_reservation(self, container: Any, actual_cost: float) -> bool:
        """
        Adjust the session counter from the reserved amount to `actual_cost`.

        Only reservations this handler created are honored: the stored token
        must match this instance's secret, so caller-supplied metadata can't
        forge a reservation and drive a negative adjustment on another session.
        Returns True when such a reservation existed, so the success path knows
        to skip the legacy full-cost increment. Idempotent via the released
        marker on the reservation record.
        """
        reservation = self._load_active_reservation(container)
        if reservation is None:
            return False

        reservation[_RESERVATION_RELEASED_FIELD] = True
        session_id = str(reservation[_RESERVATION_SESSION_FIELD])
        reserved_cost = float(reservation[_RESERVATION_COST_FIELD])
        adjustment = actual_cost - reserved_cost
        if adjustment != 0:
            await self._increment_spend(self._make_cache_key(session_id), adjustment)
        return True

    def _estimate_reservation_cost(self, data: dict, call_type: str) -> float | None:
        try:
            from litellm.proxy.proxy_server import llm_router
            from litellm.proxy.spend_tracking.budget_reservation import (
                estimate_request_max_cost,
            )

            return estimate_request_max_cost(request_body=data, route=call_type or "", llm_router=llm_router)
        except Exception as e:
            verbose_proxy_logger.debug(
                "MaxBudgetPerSessionHandler: could not estimate request cost, falling back to read-time check: %s",
                str(e),
            )
            return None

    def _budget_exceeded_error(
        self,
        data: dict,
        session_id: str,
        current_spend: float,
        max_budget: float,
    ) -> "HTTPException":
        resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(data.get("model") if data else None)
        return ProxyRateLimitError(
            detail=(
                f"Session budget exceeded for session {session_id}. "
                f"Current spend: ${current_spend:.4f}, "
                f"max_budget_per_session: ${max_budget:.2f}."
            ),
            rate_limit_type=RateLimitType.BUDGET,
            model=resolved_model,
            llm_provider=llm_provider,
        )

    def _stash_reservation(self, data: dict, session_id: str, reserved_cost: float) -> None:
        reservation = {
            _RESERVATION_TOKEN_FIELD: self._reservation_token,
            _RESERVATION_SESSION_FIELD: session_id,
            _RESERVATION_COST_FIELD: reserved_cost,
            _RESERVATION_RELEASED_FIELD: False,
        }
        channels = self._metadata_channels(data)
        if not channels:
            data["metadata"] = {}
            channels = (data["metadata"],)
        for channel in channels:
            channel[_RESERVATION_KEY] = reservation

    def _load_active_reservation(self, container: Any) -> dict | None:
        """
        Return the reservation record this handler stamped on the request, or
        None. A record is honored only when its token matches this instance's
        secret (proving provenance) and it has not already been released.
        """
        for channel in self._metadata_channels(container):
            reservation = channel.get(_RESERVATION_KEY)
            if not isinstance(reservation, dict):
                continue
            if reservation.get(_RESERVATION_TOKEN_FIELD) != self._reservation_token:
                continue
            if reservation.get(_RESERVATION_RELEASED_FIELD) is True:
                return None
            session_id = reservation.get(_RESERVATION_SESSION_FIELD)
            reserved_cost = reservation.get(_RESERVATION_COST_FIELD)
            if isinstance(session_id, str) and isinstance(reserved_cost, (int, float)):
                return reservation
        return None

    @staticmethod
    def _metadata_channels(container: Any) -> "tuple[dict, ...]":
        if not isinstance(container, dict):
            return ()
        litellm_params = container.get("litellm_params")
        candidates = (
            container.get("metadata"),
            container.get("litellm_metadata"),
            litellm_params.get("metadata") if isinstance(litellm_params, dict) else None,
        )
        return tuple(channel for channel in candidates if isinstance(channel, dict))

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

    def _get_max_budget_per_session(self, user_api_key_dict: UserAPIKeyAuth) -> Optional[float]:
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
                result = await self.internal_usage_cache.dual_cache.redis_cache.async_get_cache(key=cache_key)
                if result is not None:
                    return float(result)
                return 0.0
            except Exception as e:
                verbose_proxy_logger.warning(
                    "MaxBudgetPerSessionHandler: Redis GET failed, falling back to in-memory: %s",
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
                    "MaxBudgetPerSessionHandler: Redis INCRBYFLOAT failed, falling back to in-memory: %s",
                    str(e),
                )

        return await self._in_memory_increment_spend(cache_key, amount)

    async def _in_memory_increment_spend(self, cache_key: str, amount: float) -> float:
        new_value = await self.internal_usage_cache.async_increment_cache(
            key=cache_key,
            value=amount,
            litellm_parent_otel_span=None,
            local_only=True,
            ttl=self.ttl,
        )
        return float(new_value) if new_value is not None else amount
