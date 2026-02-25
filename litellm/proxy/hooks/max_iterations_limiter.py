"""
Max Iterations Limiter for LiteLLM Proxy.

Enforces a per-session cap on the number of LLM calls an agentic loop can make.
Callers send a `session_id` with each request (via `x-litellm-session-id` header
or `metadata.session_id`), and this hook counts calls per session. When the count
exceeds `max_iterations` (configured in key/team metadata), returns 429.

Works across multiple proxy instances via DualCache (in-memory + Redis).
Follows the same pattern as parallel_request_limiter_v3.py.
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


# Redis Lua script for atomic increment with TTL.
# Returns the new count after increment.
# Only sets EXPIRE on first increment (when count becomes 1).
MAX_ITERATIONS_INCREMENT_SCRIPT = """
local key = KEYS[1]
local ttl = tonumber(ARGV[1])

local current = redis.call('INCR', key)
if current == 1 then
    redis.call('EXPIRE', key, ttl)
end

return current
"""

# Default TTL for session iteration counters (1 hour)
DEFAULT_MAX_ITERATIONS_TTL = 3600


class _PROXY_MaxIterationsHandler(CustomLogger):
    """
    Pre-call hook that enforces max_iterations per session.

    Configuration:
        - max_iterations: set in key metadata via /key/generate or /key/update
          e.g. metadata={"max_iterations": 25}
        - session_id: sent by caller via x-litellm-session-id header or
          metadata.session_id in request body

    Cache key pattern:
        {session_iterations:<session_id>}:count

    Multi-instance support:
        Uses Redis Lua script for atomic increment (same pattern as
        parallel_request_limiter_v3). Falls back to in-memory cache
        when Redis is unavailable.
    """

    def __init__(self, internal_usage_cache: InternalUsageCache):
        self.internal_usage_cache = internal_usage_cache
        self.ttl = int(
            os.getenv("LITELLM_MAX_ITERATIONS_TTL", DEFAULT_MAX_ITERATIONS_TTL)
        )

        # Register Lua script with Redis if available (same pattern as v3 limiter)
        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            self.increment_script = (
                self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                    MAX_ITERATIONS_INCREMENT_SCRIPT
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
        Check session iteration count before making the API call.

        Extracts session_id from request metadata and max_iterations from
        key metadata. If the session has exceeded max_iterations, raises 429.
        """
        # Extract session_id from request data
        session_id = self._get_session_id(data)
        if session_id is None:
            return None

        # Extract max_iterations from key metadata
        max_iterations = self._get_max_iterations(user_api_key_dict)
        if max_iterations is None:
            return None

        verbose_proxy_logger.debug(
            "MaxIterationsHandler: session_id=%s, max_iterations=%s",
            session_id,
            max_iterations,
        )

        # Increment and check
        cache_key = self._make_cache_key(session_id)
        current_count = await self._increment_and_get(cache_key)

        if current_count > max_iterations:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Max iterations exceeded for session {session_id}. "
                    f"Current count: {current_count}, max_iterations: {max_iterations}."
                ),
            )

        verbose_proxy_logger.debug(
            "MaxIterationsHandler: session_id=%s, count=%s/%s",
            session_id,
            current_count,
            max_iterations,
        )

        return None

    def _get_session_id(self, data: dict) -> Optional[str]:
        """Extract session_id from request metadata."""
        metadata = data.get("metadata") or {}
        session_id = metadata.get("session_id")
        if session_id is not None:
            return str(session_id)

        # Also check litellm_metadata (used for /thread and /assistant endpoints)
        litellm_metadata = data.get("litellm_metadata") or {}
        session_id = litellm_metadata.get("session_id")
        if session_id is not None:
            return str(session_id)

        return None

    def _get_max_iterations(
        self, user_api_key_dict: UserAPIKeyAuth
    ) -> Optional[int]:
        """Extract max_iterations from key metadata."""
        metadata = user_api_key_dict.metadata or {}
        max_iterations = metadata.get("max_iterations")
        if max_iterations is not None:
            return int(max_iterations)
        return None

    def _make_cache_key(self, session_id: str) -> str:
        """
        Create cache key for session iteration counter.

        Uses Redis hash-tag pattern {session_iterations:<session_id>} so all
        keys for a session land on the same Redis Cluster slot.
        """
        return f"{{session_iterations:{session_id}}}:count"

    async def _increment_and_get(self, cache_key: str) -> int:
        """
        Atomically increment the session counter and return the new value.

        Tries Redis first (via registered Lua script for atomicity across
        instances), falls back to in-memory cache.
        """
        if self.increment_script is not None:
            try:
                result = await self.increment_script(
                    keys=[cache_key],
                    args=[self.ttl],
                )
                return int(result)
            except Exception as e:
                verbose_proxy_logger.warning(
                    "MaxIterationsHandler: Redis failed, falling back to in-memory: %s",
                    str(e),
                )

        # Fallback: in-memory cache
        return await self._in_memory_increment(cache_key)

    async def _in_memory_increment(self, cache_key: str) -> int:
        """Increment counter in in-memory cache with TTL."""
        current = await self.internal_usage_cache.async_get_cache(
            key=cache_key,
            litellm_parent_otel_span=None,
            local_only=True,
        )
        new_value = (int(current) if current is not None else 0) + 1
        await self.internal_usage_cache.async_set_cache(
            key=cache_key,
            value=new_value,
            ttl=self.ttl,
            litellm_parent_otel_span=None,
            local_only=True,
        )
        return new_value
