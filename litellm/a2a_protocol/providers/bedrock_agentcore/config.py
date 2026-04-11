"""
Bedrock AgentCore A2A provider configuration.
"""

from typing import Any, AsyncIterator, Dict, Optional

from litellm.a2a_protocol.providers.base import BaseA2AProviderConfig
from litellm.a2a_protocol.providers.bedrock_agentcore.handler import (
    BedrockAgentCoreA2AHandler,
)


class BedrockAgentCoreA2AConfig(BaseA2AProviderConfig):
    """
    Provider configuration for Bedrock AgentCore A2A-native agents.

    AgentCore agents that speak A2A natively expect the full JSON-RPC envelope.
    This config bypasses the completion bridge and forwards requests directly,
    deriving the endpoint URL from the model ARN and signing with SigV4/JWT.
    """

    async def handle_non_streaming(
        self,
        request_id: str,
        params: Dict[str, Any],
        api_base: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Handle non-streaming request to AgentCore A2A agent."""
        litellm_params = kwargs.get("litellm_params")
        if not litellm_params:
            raise ValueError(
                "litellm_params is required for BedrockAgentCoreA2AConfig "
                "(must contain model with AgentCore ARN)"
            )
        return await BedrockAgentCoreA2AHandler.handle_non_streaming(
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
        )

    async def handle_streaming(
        self,
        request_id: str,
        params: Dict[str, Any],
        api_base: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle streaming request to AgentCore A2A agent."""
        litellm_params = kwargs.get("litellm_params")
        if not litellm_params:
            raise ValueError(
                "litellm_params is required for BedrockAgentCoreA2AConfig "
                "(must contain model with AgentCore ARN)"
            )
        async for chunk in BedrockAgentCoreA2AHandler.handle_streaming(
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
        ):
            yield chunk
