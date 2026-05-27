from typing import Any, AsyncIterator, Dict, Optional

from litellm.a2a_protocol.litellm_completion_bridge.handler import (
    A2ACompletionBridgeHandler,
)
from litellm.a2a_protocol.providers.base import BaseA2AProviderConfig
from litellm.llms.langflow.a2a import merge_a2a_session_into_litellm_params


class LangFlowA2AConfig(BaseA2AProviderConfig):
    """A2A bridge for LangFlow: maps contextId to session_id, then uses completion."""

    async def handle_non_streaming(
        self,
        request_id: str,
        params: Dict[str, Any],
        api_base: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        litellm_params = merge_a2a_session_into_litellm_params(
            kwargs.get("litellm_params") or {},
            params,
        )
        return await A2ACompletionBridgeHandler.handle_non_streaming(
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            api_base=api_base,
            _skip_a2a_provider_routing=True,
        )

    async def handle_streaming(
        self,
        request_id: str,
        params: Dict[str, Any],
        api_base: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        litellm_params = merge_a2a_session_into_litellm_params(
            kwargs.get("litellm_params") or {},
            params,
        )
        async for chunk in A2ACompletionBridgeHandler.handle_streaming(
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            api_base=api_base,
            _skip_a2a_provider_routing=True,
        ):
            yield chunk
