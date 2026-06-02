"""
Sensitive Data Routing Hook for LiteLLM Proxy.

When a guardrail detects sensitive data and is configured with on_sensitive_data='route',
this hook manages:
1. Storing the routing decision (session_id -> model) in cache
2. Checking incoming requests for existing routing overrides
3. Applying sticky routing so all subsequent requests in a session go to the same model

Works across multiple proxy instances via DualCache (in-memory + Redis).
"""

import os
from typing import TYPE_CHECKING, Any, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import get_session_id_from_request_data
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

if TYPE_CHECKING:
    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    InternalUsageCache = _InternalUsageCache
else:
    InternalUsageCache = Any


SENSITIVE_ROUTING_CACHE_PREFIX = "sensitive_route"
DEFAULT_SENSITIVE_ROUTING_TTL = 3600


class _PROXY_SensitiveDataRoutingHandler(CustomLogger):
    """
    Pre-call hook that checks for existing sensitive data routing overrides
    and applies them to incoming requests.

    This hook runs early in the pre-call chain and modifies the request's
    model field if a routing override exists for the session.
    """

    def __init__(self, internal_usage_cache: InternalUsageCache):
        self.internal_usage_cache = internal_usage_cache
        self.ttl = int(
            os.getenv(
                "LITELLM_SENSITIVE_ROUTING_TTL",
                str(DEFAULT_SENSITIVE_ROUTING_TTL),
            )
        )

    def _make_cache_key(self, session_id: str, tenant: str) -> str:
        return f"{{{SENSITIVE_ROUTING_CACHE_PREFIX}:{tenant}:{session_id}}}:model"

    @staticmethod
    def _resolve_tenant(user_api_key_dict: Optional[UserAPIKeyAuth]) -> str:
        """
        Identify the authenticated principal the routing override belongs to.

        API-key auth is scoped by the hashed key. JWT (and other keyless) auth
        has no api_key, so fall back to a stable identity claim. Without this,
        every keyless caller would share the ``default`` namespace and could read
        or overwrite another principal's session routing.
        """
        if user_api_key_dict is None:
            return "default"
        if user_api_key_dict.api_key:
            return user_api_key_dict.api_key
        principal = [
            f"{label}:{value}"
            for label, value in (
                ("user", user_api_key_dict.user_id),
                ("team", user_api_key_dict.team_id),
                ("org", user_api_key_dict.org_id),
            )
            if value
        ]
        return "|".join(principal) if principal else "default"

    async def _get_routed_model(
        self, session_id: str, user_api_key_dict: Optional[UserAPIKeyAuth]
    ) -> Optional[str]:
        """Get the model this session should be routed to, if any."""
        cache_key = self._make_cache_key(
            session_id, self._resolve_tenant(user_api_key_dict)
        )

        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            try:
                result = await self.internal_usage_cache.dual_cache.redis_cache.async_get_cache(
                    key=cache_key
                )
                if result is not None:
                    routed_model = str(result)
                    remaining_ttl = await self.internal_usage_cache.dual_cache.redis_cache.async_get_ttl(
                        key=cache_key
                    )
                    await self.internal_usage_cache.async_set_cache(
                        key=cache_key,
                        value=routed_model,
                        ttl=remaining_ttl if remaining_ttl is not None else self.ttl,
                        litellm_parent_otel_span=None,
                        local_only=True,
                    )
                    return routed_model
            except Exception as e:
                verbose_proxy_logger.warning(
                    "SensitiveDataRoutingHandler: Redis GET failed, falling back to in-memory: %s",
                    str(e),
                )

        result = await self.internal_usage_cache.async_get_cache(
            key=cache_key,
            litellm_parent_otel_span=None,
            local_only=True,
        )
        if result is not None:
            return str(result)
        return None

    async def set_session_routing(
        self,
        session_id: str,
        model: str,
        user_api_key_dict: Optional[UserAPIKeyAuth] = None,
        guardrail_name: Optional[str] = None,
    ) -> None:
        """
        Store a routing override for a session.

        Called by guardrails when they detect sensitive data and want to
        route the session to a specific model. The override is scoped to the
        requesting principal so sessions from different tenants cannot collide.
        """
        cache_key = self._make_cache_key(
            session_id, self._resolve_tenant(user_api_key_dict)
        )

        verbose_proxy_logger.info(
            "SensitiveDataRoutingHandler: Setting session routing session_id=%s model=%s guardrail=%s ttl=%s",
            session_id,
            model,
            guardrail_name,
            self.ttl,
        )

        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            try:
                await self.internal_usage_cache.dual_cache.redis_cache.async_set_cache(
                    key=cache_key,
                    value=model,
                    ttl=self.ttl,
                )
            except Exception as e:
                verbose_proxy_logger.warning(
                    "SensitiveDataRoutingHandler: Redis SET failed, falling back to in-memory: %s",
                    str(e),
                )

        await self.internal_usage_cache.async_set_cache(
            key=cache_key,
            value=model,
            ttl=self.ttl,
            litellm_parent_otel_span=None,
            local_only=True,
        )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Before each LLM call, check if this session has a routing override.
        If so, modify the request's model field.
        """
        session_id = get_session_id_from_request_data(data)
        if session_id is None:
            return None

        routed_model = await self._get_routed_model(session_id, user_api_key_dict)
        if routed_model is None:
            return None

        original_model = data.get("model")
        if original_model == routed_model:
            return None

        verbose_proxy_logger.info(
            "SensitiveDataRoutingHandler: Applying session routing override "
            "session_id=%s original_model=%s routed_model=%s",
            session_id,
            original_model,
            routed_model,
        )

        data["model"] = routed_model

        metadata = data.get("metadata") or {}
        metadata["sensitive_data_routing_applied"] = True
        metadata["sensitive_data_routing_original_model"] = original_model
        data["metadata"] = metadata

        return data
