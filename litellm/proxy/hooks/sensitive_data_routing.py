"""
Sensitive Data Routing Hook for LiteLLM Proxy.

When a guardrail detects sensitive data and is configured with on_sensitive_data='route',
this hook manages:
1. Storing the routing decision (session_id -> model) in cache
2. Checking incoming requests for existing routing overrides
3. Applying sticky routing so all subsequent requests in a session go to the same model

Works across multiple proxy instances via the native sensitive route pin store (in-memory + Redis).
"""

import os
from typing import TYPE_CHECKING, Any, Final

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import get_session_id_from_request_data
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.redis.stores.connection import connect
from litellm.redis.stores.sensitive_route_pins import SensitiveRoutePinStore

if TYPE_CHECKING:
    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    InternalUsageCache = _InternalUsageCache
else:
    InternalUsageCache = Any


DEFAULT_SENSITIVE_ROUTING_TTL: Final = 3600


class _PROXY_SensitiveDataRoutingHandler(CustomLogger):
    """
    Pre-call hook that checks for existing sensitive data routing overrides
    and applies them to incoming requests.

    This hook runs early in the pre-call chain and modifies the request's
    model field if a routing override exists for the session.
    """

    def __init__(self, internal_usage_cache: InternalUsageCache, route_pins: SensitiveRoutePinStore | None = None):
        self.route_pins: Final = route_pins or SensitiveRoutePinStore(
            connect(internal_usage_cache.dual_cache.redis_cache).sensitive_route_pins()
        )
        self.ttl = int(
            os.getenv(
                "LITELLM_SENSITIVE_ROUTING_TTL",
                str(DEFAULT_SENSITIVE_ROUTING_TTL),
            )
        )

    @staticmethod
    def _resolve_tenant(user_api_key_dict: UserAPIKeyAuth | None) -> str:
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
        principal: Final = [
            f"{label}:{value}"
            for label, value in (
                ("user", user_api_key_dict.user_id),
                ("team", user_api_key_dict.team_id),
                ("org", user_api_key_dict.org_id),
            )
            if value
        ]
        return "|".join(principal) if principal else "default"

    async def _get_routed_model(self, session_id: str, user_api_key_dict: UserAPIKeyAuth | None) -> str | None:
        """Get the model this session should be routed to, if any."""
        return await self.route_pins.get(self._resolve_tenant(user_api_key_dict), session_id, self.ttl)

    async def set_session_routing(
        self,
        session_id: str,
        model: str,
        user_api_key_dict: UserAPIKeyAuth | None = None,
        guardrail_name: str | None = None,
    ) -> None:
        """
        Store a routing override for a session.

        Called by guardrails when they detect sensitive data and want to
        route the session to a specific model. The override is scoped to the
        requesting principal so sessions from different tenants cannot collide.
        """
        verbose_proxy_logger.info(
            "SensitiveDataRoutingHandler: Setting session routing session_id=%s model=%s guardrail=%s ttl=%s",
            session_id,
            model,
            guardrail_name,
            self.ttl,
        )
        await self.route_pins.set(self._resolve_tenant(user_api_key_dict), session_id, model, self.ttl)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Exception | str | dict | None:
        """
        Before each LLM call, check if this session has a routing override.
        If so, modify the request's model field.
        """
        session_id: Final = get_session_id_from_request_data(data)
        if session_id is None:
            return None

        routed_model: Final = await self._get_routed_model(session_id, user_api_key_dict)
        if routed_model is None:
            return None

        original_model: Final = data.get("model")
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

        metadata: Final = data.get("metadata") or {}
        metadata["sensitive_data_routing_applied"] = True
        metadata["sensitive_data_routing_original_model"] = original_model
        data["metadata"] = metadata

        return data


PROXY_SensitiveDataRoutingHandler: Final = _PROXY_SensitiveDataRoutingHandler
