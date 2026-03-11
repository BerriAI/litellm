"""
Pydantic AI provider configuration.
"""

from typing import Any, AsyncIterator, Dict

from litellm.a2a_protocol.providers.base import BaseA2AProviderConfig
from litellm.a2a_protocol.providers.pydantic_ai_agents.handler import PydanticAIHandler


class PydanticAIProviderConfig(BaseA2AProviderConfig):
    """
    Provider configuration for Pydantic AI agents.
    
    Pydantic AI agents follow A2A protocol but don't support streaming natively.
    This config provides fake streaming by converting non-streaming responses into streaming chunks.
    """

    async def handle_non_streaming(
        self,
        request_id: str,
        params: Dict[str, Any],
        api_base: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Handle non-streaming request to Pydantic AI agent."""
        return await PydanticAIHandler.handle_non_streaming(
            request_id=request_id,
            params=params,
            api_base=api_base,
            timeout=kwargs.get("timeout", 60.0),
        )

    async def handle_streaming(
        self,
        request_id: str,
        params: Dict[str, Any],
        api_base: str,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Handle streaming request with fake streaming."""
        async for chunk in PydanticAIHandler.handle_streaming(
            request_id=request_id,
            params=params,
            api_base=api_base,
            timeout=kwargs.get("timeout", 60.0),
            chunk_size=kwargs.get("chunk_size", 50),
            delay_ms=kwargs.get("delay_ms", 10),
        ):
            yield chunk

