"""
Handler for Pydantic AI agents.

Pydantic AI agents follow A2A protocol but don't support streaming natively.
This handler provides fake streaming by converting non-streaming responses into streaming chunks.
"""

from typing import Any, AsyncIterator, Dict

from litellm._logging import verbose_logger
from litellm.a2a_protocol.providers.pydantic_ai_agents.transformation import (
    PydanticAITransformation,
)


class PydanticAIHandler:
    """
    Handler for Pydantic AI agent requests.
    
    Provides:
    - Direct non-streaming requests to Pydantic AI agents
    - Fake streaming by converting non-streaming responses into streaming chunks
    """

    @staticmethod
    async def handle_non_streaming(
        request_id: str,
        params: Dict[str, Any],
        api_base: str,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Handle non-streaming request to Pydantic AI agent.

        Args:
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message
            api_base: Base URL of the Pydantic AI agent
            timeout: Request timeout in seconds

        Returns:
            A2A SendMessageResponse dict
        """
        verbose_logger.info(
            f"Pydantic AI: Routing to Pydantic AI agent at {api_base}"
        )

        # Send request directly to Pydantic AI agent
        response_data = await PydanticAITransformation.send_non_streaming_request(
            api_base=api_base,
            request_id=request_id,
            params=params,
            timeout=timeout,
        )

        return response_data

    @staticmethod
    async def handle_streaming(
        request_id: str,
        params: Dict[str, Any],
        api_base: str,
        timeout: float = 60.0,
        chunk_size: int = 50,
        delay_ms: int = 10,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Handle streaming request to Pydantic AI agent with fake streaming.

        Since Pydantic AI agents don't support streaming natively, this method:
        1. Makes a non-streaming request
        2. Converts the response into streaming chunks

        Args:
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message
            api_base: Base URL of the Pydantic AI agent
            timeout: Request timeout in seconds
            chunk_size: Number of characters per chunk
            delay_ms: Delay between chunks in milliseconds

        Yields:
            A2A streaming response events
        """
        verbose_logger.info(
            f"Pydantic AI: Faking streaming for Pydantic AI agent at {api_base}"
        )

        # Get raw task response first (not the transformed A2A format)
        raw_response = await PydanticAITransformation.send_and_get_raw_response(
            api_base=api_base,
            request_id=request_id,
            params=params,
            timeout=timeout,
        )

        # Convert raw task response to fake streaming chunks
        async for chunk in PydanticAITransformation.fake_streaming_from_response(
            response_data=raw_response,
            request_id=request_id,
            chunk_size=chunk_size,
            delay_ms=delay_ms,
        ):
            yield chunk


