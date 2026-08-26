"""
A2A provider configuration for IBM watsonx Orchestrate (WXO).
"""

from collections.abc import AsyncIterator
from typing import Any, Final

from litellm.a2a_protocol.providers.base import BaseA2AProviderConfig
from litellm.a2a_protocol.providers.watsonx_orchestrate.handler import (
    WatsonxOrchestrateHandler,
)
from litellm.interactions.agents.utils import merge_agent_headers


class WatsonxOrchestrateA2AConfig(BaseA2AProviderConfig):
    """A2A bridge for IBM watsonx Orchestrate (REST runs API + poll/SSE)."""

    async def handle_non_streaming(
        self,
        request_id: str,
        params: dict[str, Any],
        api_base: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Handle a non-streaming A2A request via WXO runs API."""
        litellm_params: Final = kwargs.get("litellm_params")
        if not litellm_params:
            raise ValueError(
                "litellm_params is required for WatsonxOrchestrateA2AConfig "
                "(must contain cp4d_host, instance_id, wxo_agent_id, api_key)"
            )
        forwarded_headers: Final = merge_agent_headers(
            dynamic_headers=kwargs.get("agent_extra_headers"),
            static_headers=kwargs.get("agent_static_headers"),
        )
        return await WatsonxOrchestrateHandler.handle_non_streaming(
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            static_headers=forwarded_headers,
            timeout=kwargs.get("timeout"),
        )

    async def handle_streaming(
        self,
        request_id: str,
        params: dict[str, Any],
        api_base: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Handle a streaming A2A request via WXO streaming runs API."""
        litellm_params: Final = kwargs.get("litellm_params")
        if not litellm_params:
            raise ValueError(
                "litellm_params is required for WatsonxOrchestrateA2AConfig "
                "(must contain cp4d_host, instance_id, wxo_agent_id, api_key)"
            )
        forwarded_headers: Final = merge_agent_headers(
            dynamic_headers=kwargs.get("agent_extra_headers"),
            static_headers=kwargs.get("agent_static_headers"),
        )
        async for chunk in WatsonxOrchestrateHandler.handle_streaming(
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            static_headers=forwarded_headers,
            timeout=kwargs.get("timeout"),
        ):
            yield chunk
