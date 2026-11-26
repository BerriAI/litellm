"""
Bedrock AgentCore A2A provider configuration.
"""

from collections.abc import AsyncIterator, Mapping
from typing import Final

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

    def supports_jsonrpc(self) -> bool:
        """Return True because AgentCore agents speak A2A JSON-RPC natively."""
        return True

    async def handle_jsonrpc(
        self,
        method: str,
        request_id: str | int | None,
        params: Mapping[str, object],
        api_base: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        agent_extra_headers: Mapping[str, str] | None = None,
    ) -> Mapping[str, object]:
        """Handle a non-streaming A2A JSON-RPC method for an AgentCore agent."""
        if not litellm_params:
            raise ValueError(
                "litellm_params is required for BedrockAgentCoreA2AConfig (must contain model with AgentCore ARN)"
            )
        return await BedrockAgentCoreA2AHandler.handle_jsonrpc(
            method=method,
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            agent_extra_headers=agent_extra_headers,
        )

    async def handle_jsonrpc_streaming(
        self,
        method: str,
        request_id: str | int | None,
        params: Mapping[str, object],
        api_base: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        agent_extra_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[Mapping[str, object]]:
        """Handle a streaming A2A JSON-RPC method for an AgentCore agent."""
        if not litellm_params:
            raise ValueError(
                "litellm_params is required for BedrockAgentCoreA2AConfig (must contain model with AgentCore ARN)"
            )
        async for chunk in BedrockAgentCoreA2AHandler.handle_jsonrpc_streaming(
            method=method,
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            agent_extra_headers=agent_extra_headers,
        ):
            yield chunk

    async def handle_non_streaming(
        self,
        request_id: str,
        params: dict[str, object],
        api_base: str | None = None,
        **kwargs,
    ) -> dict[str, object]:
        """Handle non-streaming request to AgentCore A2A agent."""
        litellm_params: Final = kwargs.get("litellm_params")
        if not litellm_params:
            raise ValueError(
                "litellm_params is required for BedrockAgentCoreA2AConfig (must contain model with AgentCore ARN)"
            )
        return await BedrockAgentCoreA2AHandler.handle_non_streaming(
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            agent_extra_headers=kwargs.get("agent_extra_headers"),
        )

    async def handle_streaming(
        self,
        request_id: str,
        params: dict[str, object],
        api_base: str | None = None,
        **kwargs,
    ) -> AsyncIterator[dict[str, object]]:
        """Handle streaming request to AgentCore A2A agent."""
        litellm_params: Final = kwargs.get("litellm_params")
        if not litellm_params:
            raise ValueError(
                "litellm_params is required for BedrockAgentCoreA2AConfig (must contain model with AgentCore ARN)"
            )
        async for chunk in BedrockAgentCoreA2AHandler.handle_streaming(
            request_id=request_id,
            params=params,
            litellm_params=litellm_params,
            agent_extra_headers=kwargs.get("agent_extra_headers"),
        ):
            yield chunk
