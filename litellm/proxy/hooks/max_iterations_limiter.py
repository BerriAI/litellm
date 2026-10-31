"""
Max Iterations Limiter for LiteLLM Proxy.

Enforces a per-session cap on the number of LLM calls an agentic loop can make.
Callers send a `session_id` with each request (via `x-litellm-session-id` header
or `metadata.session_id`), and this hook counts calls per session. When the count
exceeds `max_iterations` (configured in agent litellm_params or key metadata), returns 429.

Works across multiple proxy instances via DualCache (in-memory + Redis).
Follows the same pattern as parallel_request_limiter_v3.py.
"""

import os
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final

from litellm import DualCache
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.exceptions import RateLimitType
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit
from litellm.redis.stores.iteration_count import IterationCountStore, TransportIterationCountStore, count_key
from litellm.redis.transport.contract import RedisFailure

if TYPE_CHECKING:
    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    InternalUsageCache = _InternalUsageCache
else:
    InternalUsageCache = Any


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
        {session_iterations:<session_id>}:count

    Multi-instance support:
        Counts in Redis through the iteration count store, falls back to
        in-memory cache when Redis is unavailable.
    """

    def __init__(self, internal_usage_cache: InternalUsageCache):
        self.internal_usage_cache = internal_usage_cache
        self.ttl = int(os.getenv("LITELLM_MAX_ITERATIONS_TTL", DEFAULT_MAX_ITERATIONS_TTL))

        redis_cache: Final = self.internal_usage_cache.dual_cache.redis_cache
        self._store: Final[IterationCountStore | None] = (
            TransportIterationCountStore(redis_cache.transport()) if redis_cache is not None else None
        )

    @with_service_target("session_iterations")
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
        current_count: Final = await self._increment_and_get(session_id)

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

    def _make_cache_key(self, session_id: str) -> str:
        return count_key(session_id)

    async def _increment_and_get(self, session_id: str) -> int:
        """Increment the session counter in Redis, or in memory when Redis is unavailable."""
        if self._store is not None:
            count: Final = await self._store.increment(session_id, ttl=timedelta(seconds=self.ttl))
            if not isinstance(count, RedisFailure):
                return count

        return await self._in_memory_increment(self._make_cache_key(session_id))

    async def _in_memory_increment(self, cache_key: str) -> int:
        """Increment counter in in-memory cache with TTL."""
        current: Final = await self.internal_usage_cache.async_get_cache(
            key=cache_key,
            litellm_parent_otel_span=None,
            local_only=True,
        )
        new_value: Final = (int(current) if current is not None else 0) + 1
        await self.internal_usage_cache.async_set_cache(
            key=cache_key,
            value=new_value,
            ttl=self.ttl,
            litellm_parent_otel_span=None,
            local_only=True,
        )
        return new_value


PROXY_MaxIterationsHandler: Final = _PROXY_MaxIterationsHandler
