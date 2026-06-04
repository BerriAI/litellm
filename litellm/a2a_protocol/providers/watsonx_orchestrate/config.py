"""
A2A provider configuration for IBM watsonx Orchestrate (WXO).
"""

from typing import Any, AsyncIterator, Dict, Optional

from litellm.a2a_protocol.providers.base import BaseA2AProviderConfig
from litellm.a2a_protocol.providers.watsonx_orchestrate.handler import (
    WatsonxOrchestrateHandler,
)


class WatsonxOrchestrateA2AConfig(BaseA2AProviderConfig):
    """A2A bridge for IBM watsonx Orchestrate (REST runs API + poll/SSE)."""

    async def handle_non_streaming(
        self,
        request_id: str,
        params: Dict[str, Any],
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Handle a non-streaming A2A request via WXO runs API."""
        litellm_params = kwargs.get("litellm_params")
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
        params: Dict[str, Any],
        api_base: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle a streaming A2A request via WXO streaming runs API."""
        litellm_params = kwargs.get("litellm_params")
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
