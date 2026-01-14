"""
Utility functions for A2A protocol.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import litellm
from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from a2a.types import SendMessageRequest, SendStreamingMessageRequest


class A2ARequestUtils:
    """Utility class for A2A request/response processing."""

    @staticmethod
    def extract_text_from_message(message: Any) -> str:
        """
        Extract text content from A2A message parts.

        Args:
            message: A2A message dict or object with 'parts' containing text parts

        Returns:
            Concatenated text from all text parts
        """
        if message is None:
            return ""

        # Handle both dict and object access
        if isinstance(message, dict):
            parts = message.get("parts", [])
        else:
            parts = getattr(message, "parts", []) or []

        text_parts: List[str] = []
        for part in parts:
            if isinstance(part, dict):
                if part.get("kind") == "text":
                    text_parts.append(part.get("text", ""))
            else:
                if getattr(part, "kind", None) == "text":
                    text_parts.append(getattr(part, "text", ""))

        return " ".join(text_parts)

    @staticmethod
    def extract_text_from_response(response_dict: Dict[str, Any]) -> str:
        """
        Extract text content from A2A response result.

        Args:
            response_dict: A2A response dict with 'result' containing message

        Returns:
            Text from response message parts
        """
        result = response_dict.get("result", {})
        if not isinstance(result, dict):
            return ""

        message = result.get("message", {})
        return A2ARequestUtils.extract_text_from_message(message)

    @staticmethod
    def get_input_message_from_request(
        request: "Union[SendMessageRequest, SendStreamingMessageRequest]",
    ) -> Any:
        """
        Extract the input message from an A2A request.

        Args:
            request: The A2A SendMessageRequest or SendStreamingMessageRequest

        Returns:
            The message object/dict or None
        """
        params = getattr(request, "params", None)
        if params is None:
            return None
        return getattr(params, "message", None)

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Count tokens in text using litellm.token_counter.

        Args:
            text: Text to count tokens for

        Returns:
            Token count, or 0 if counting fails
        """
        if not text:
            return 0
        try:
            return litellm.token_counter(text=text)
        except Exception:
            verbose_logger.debug("Failed to count tokens")
            return 0

    @staticmethod
    def calculate_usage_from_request_response(
        request: "Union[SendMessageRequest, SendStreamingMessageRequest]",
        response_dict: Dict[str, Any],
    ) -> Tuple[int, int, int]:
        """
        Calculate token usage from A2A request and response.

        Args:
            request: The A2A SendMessageRequest or SendStreamingMessageRequest
            response_dict: The A2A response as a dict

        Returns:
            Tuple of (prompt_tokens, completion_tokens, total_tokens)
        """
        # Count input tokens
        input_message = A2ARequestUtils.get_input_message_from_request(request)
        input_text = A2ARequestUtils.extract_text_from_message(input_message)
        prompt_tokens = A2ARequestUtils.count_tokens(input_text)

        # Count output tokens
        output_text = A2ARequestUtils.extract_text_from_response(response_dict)
        completion_tokens = A2ARequestUtils.count_tokens(output_text)

        total_tokens = prompt_tokens + completion_tokens

        return prompt_tokens, completion_tokens, total_tokens


# Backwards compatibility aliases
def extract_text_from_a2a_message(message: Any) -> str:
    return A2ARequestUtils.extract_text_from_message(message)


def extract_text_from_a2a_response(response_dict: Dict[str, Any]) -> str:
    return A2ARequestUtils.extract_text_from_response(response_dict)
