"""
Transformation utilities for A2A <-> OpenAI message format conversion.

A2A Message Format:
{
    "role": "user",
    "parts": [{"kind": "text", "text": "Hello!"}],
    "messageId": "abc123"
}

OpenAI Message Format:
{"role": "user", "content": "Hello!"}
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from litellm._logging import verbose_logger


class A2ACompletionBridgeTransformation:
    """
    Static methods for transforming between A2A and OpenAI message formats.
    """

    @staticmethod
    def a2a_message_to_openai_messages(
        a2a_message: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Transform an A2A message to OpenAI message format.

        Args:
            a2a_message: A2A message with role, parts, and messageId

        Returns:
            List of OpenAI-format messages
        """
        role = a2a_message.get("role", "user")
        parts = a2a_message.get("parts", [])

        # Map A2A roles to OpenAI roles
        openai_role = role
        if role == "user":
            openai_role = "user"
        elif role == "assistant":
            openai_role = "assistant"
        elif role == "system":
            openai_role = "system"

        # Extract text content from parts
        content_parts = []
        for part in parts:
            kind = part.get("kind", "")
            if kind == "text":
                text = part.get("text", "")
                content_parts.append(text)

        content = "\n".join(content_parts) if content_parts else ""

        verbose_logger.debug(
            f"A2A -> OpenAI transform: role={role} -> {openai_role}, content_length={len(content)}"
        )

        return [{"role": openai_role, "content": content}]

    @staticmethod
    def openai_response_to_a2a_response(
        response: Any,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transform a LiteLLM ModelResponse to A2A SendMessageResponse format.

        Args:
            response: LiteLLM ModelResponse object
            request_id: Original A2A request ID

        Returns:
            A2A SendMessageResponse dict
        """
        # Extract content from response
        content = ""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and choice.message:
                content = choice.message.content or ""

        # Build A2A message
        a2a_message = {
            "role": "agent",
            "parts": [{"kind": "text", "text": content}],
            "messageId": uuid4().hex,
        }

        # Build A2A response
        a2a_response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "message": a2a_message,
            },
        }

        verbose_logger.debug(
            f"OpenAI -> A2A transform: content_length={len(content)}"
        )

        return a2a_response

    @staticmethod
    def openai_chunk_to_a2a_chunk(
        chunk: Any,
        request_id: Optional[str] = None,
        is_final: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Transform a LiteLLM streaming chunk to A2A streaming format.

        Args:
            chunk: LiteLLM ModelResponse chunk
            request_id: Original A2A request ID
            is_final: Whether this is the final chunk

        Returns:
            A2A streaming chunk dict or None if no content
        """
        # Extract delta content
        content = ""
        if chunk is not None and hasattr(chunk, "choices") and chunk.choices:
            choice = chunk.choices[0]
            if hasattr(choice, "delta") and choice.delta:
                content = choice.delta.content or ""

        if not content and not is_final:
            return None

        # Build A2A streaming chunk
        a2a_chunk = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "message": {
                    "role": "agent",
                    "parts": [{"kind": "text", "text": content}],
                    "messageId": uuid4().hex,
                },
                "final": is_final,
            },
        }

        return a2a_chunk
