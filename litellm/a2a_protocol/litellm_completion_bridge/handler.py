"""
Handler for A2A to LiteLLM completion bridge.

Routes A2A requests through litellm.acompletion based on custom_llm_provider.
"""

from typing import Any, AsyncIterator, Dict, Optional

import litellm
from litellm._logging import verbose_logger
from litellm.a2a_protocol.litellm_completion_bridge.transformation import (
    A2ACompletionBridgeTransformation,
)


class A2ACompletionBridgeHandler:
    """
    Static methods for handling A2A requests via LiteLLM completion.
    """

    @staticmethod
    async def handle_non_streaming(
        request_id: str,
        params: Dict[str, Any],
        litellm_params: Dict[str, Any],
        api_base: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle non-streaming A2A request via litellm.acompletion.

        Args:
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message
            litellm_params: Agent's litellm_params (custom_llm_provider, model, etc.)
            api_base: API base URL from agent_card_params

        Returns:
            A2A SendMessageResponse dict
        """
        # Extract message from params
        message = params.get("message", {})

        # Transform A2A message to OpenAI format
        openai_messages = A2ACompletionBridgeTransformation.a2a_message_to_openai_messages(
            message
        )

        # Get completion params
        custom_llm_provider = litellm_params.get("custom_llm_provider")
        model = litellm_params.get("model", "agent")

        # Build full model string if provider specified
        if custom_llm_provider:
            full_model = f"{custom_llm_provider}/{model}"
        else:
            full_model = model

        verbose_logger.info(
            f"A2A completion bridge: model={full_model}, api_base={api_base}"
        )

        # Call litellm.acompletion
        response = await litellm.acompletion(
            model=full_model,
            messages=openai_messages,
            api_base=api_base,
            stream=False,
        )

        # Transform response to A2A format
        a2a_response = A2ACompletionBridgeTransformation.openai_response_to_a2a_response(
            response=response,
            request_id=request_id,
        )

        verbose_logger.info(f"A2A completion bridge completed: request_id={request_id}")

        return a2a_response

    @staticmethod
    async def handle_streaming(
        request_id: str,
        params: Dict[str, Any],
        litellm_params: Dict[str, Any],
        api_base: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Handle streaming A2A request via litellm.acompletion with stream=True.

        Args:
            request_id: A2A JSON-RPC request ID
            params: A2A MessageSendParams containing the message
            litellm_params: Agent's litellm_params (custom_llm_provider, model, etc.)
            api_base: API base URL from agent_card_params

        Yields:
            A2A streaming response chunks
        """
        # Extract message from params
        message = params.get("message", {})

        # Transform A2A message to OpenAI format
        openai_messages = A2ACompletionBridgeTransformation.a2a_message_to_openai_messages(
            message
        )

        # Get completion params
        custom_llm_provider = litellm_params.get("custom_llm_provider")
        model = litellm_params.get("model", "agent")

        # Build full model string if provider specified
        if custom_llm_provider:
            full_model = f"{custom_llm_provider}/{model}"
        else:
            full_model = model

        verbose_logger.info(
            f"A2A completion bridge streaming: model={full_model}, api_base={api_base}"
        )

        # Call litellm.acompletion with streaming
        response = await litellm.acompletion(
            model=full_model,
            messages=openai_messages,
            api_base=api_base,
            stream=True,
        )

        chunk_count = 0
        async for chunk in response:  # type: ignore[union-attr]
            chunk_count += 1
            a2a_chunk = A2ACompletionBridgeTransformation.openai_chunk_to_a2a_chunk(
                chunk=chunk,
                request_id=request_id,
                is_final=False,
            )
            if a2a_chunk:
                yield a2a_chunk

        # Send final chunk
        final_chunk = A2ACompletionBridgeTransformation.openai_chunk_to_a2a_chunk(
            chunk=None,
            request_id=request_id,
            is_final=True,
        )
        if final_chunk:
            # Clear content for final chunk
            final_chunk["result"]["message"]["parts"][0]["text"] = ""
            yield final_chunk

        verbose_logger.info(
            f"A2A completion bridge streaming completed: request_id={request_id}, chunks={chunk_count}"
        )


# Convenience functions that delegate to the class methods
async def handle_a2a_completion(
    request_id: str,
    params: Dict[str, Any],
    litellm_params: Dict[str, Any],
    api_base: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function for non-streaming A2A completion."""
    return await A2ACompletionBridgeHandler.handle_non_streaming(
        request_id=request_id,
        params=params,
        litellm_params=litellm_params,
        api_base=api_base,
    )


async def handle_a2a_completion_streaming(
    request_id: str,
    params: Dict[str, Any],
    litellm_params: Dict[str, Any],
    api_base: Optional[str] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Convenience function for streaming A2A completion."""
    async for chunk in A2ACompletionBridgeHandler.handle_streaming(
        request_id=request_id,
        params=params,
        litellm_params=litellm_params,
        api_base=api_base,
    ):
        yield chunk
