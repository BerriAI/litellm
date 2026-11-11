"""
A2A provider configuration for IBM watsonx Orchestrate (WXO).
"""

from collections.abc import AsyncIterator

from litellm.a2a_protocol.providers.base import BaseA2AProviderConfig
from litellm.a2a_protocol.providers.watsonx_orchestrate.handler import (
    WatsonxOrchestrateHandler,
    WXOLitellmParams,
)


class WatsonxOrchestrateA2AConfig(BaseA2AProviderConfig):
    """A2A bridge for IBM watsonx Orchestrate (REST runs API + poll/SSE)."""

    async def handle_non_streaming(
        self,
        request_id: str,
        params: dict[str, object],
        api_base: str | None = None,
        *,
        litellm_params: WXOLitellmParams | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        """Handle a non-streaming A2A request via WXO runs API."""
        if not litellm_params:
            raise ValueError(
                "litellm_params is required for WatsonxOrchestrateA2AConfig "
                "(must contain cp4d_host, instance_id, wxo_agent_id, api_key)"
            )
        return await WatsonxOrchestrateHandler.handle_non_streaming(
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
        )

    async def handle_streaming(
        self,
        request_id: str,
        params: dict[str, object],
        api_base: str | None = None,
        *,
        litellm_params: WXOLitellmParams | None = None,
        **kwargs: object,
    ) -> AsyncIterator[dict[str, object]]:
        """Handle a streaming A2A request via WXO streaming runs API."""
        if not litellm_params:
            raise ValueError(
                "litellm_params is required for WatsonxOrchestrateA2AConfig "
                "(must contain cp4d_host, instance_id, wxo_agent_id, api_key)"
            )
        async for chunk in WatsonxOrchestrateHandler.handle_streaming(
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
        ):
            yield chunk
