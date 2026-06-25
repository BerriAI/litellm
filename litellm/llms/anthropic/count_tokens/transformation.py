"""
Anthropic CountTokens API transformation logic.

This module handles the transformation of requests to Anthropic's CountTokens API format.
"""

from typing import Any, Dict, List, Optional

from litellm.constants import ANTHROPIC_TOKEN_COUNTING_BETA_VERSION


class AnthropicCountTokensConfig:
    """
    Configuration and transformation logic for Anthropic CountTokens API.

    Anthropic CountTokens API Specification:
    - Endpoint: POST https://api.anthropic.com/v1/messages/count_tokens
    - Beta header required: anthropic-beta: token-counting-2024-11-01
    - Response: {"input_tokens": <number>}
    """

    _COUNT_TOKENS_PATH: str = "/v1/messages/count_tokens"
    _DEFAULT_ENDPOINT: str = f"https://api.anthropic.com{_COUNT_TOKENS_PATH}"

    def _build_count_tokens_url(self, api_base: Optional[str]) -> str:
        """Return the full count_tokens URL, appending the path when a custom base is provided."""
        if not api_base:
            return self._DEFAULT_ENDPOINT
        base = api_base.rstrip("/")
        if base.endswith(self._COUNT_TOKENS_PATH):
            return base
        messages_prefix = self._COUNT_TOKENS_PATH.rsplit("/count_tokens", 1)[0]
        if base.endswith(messages_prefix):
            return f"{base}/count_tokens"
        v1_prefix = messages_prefix.rsplit("/messages", 1)[0]
        if base.endswith(v1_prefix):
            return f"{base}/messages/count_tokens"
        return f"{base}{self._COUNT_TOKENS_PATH}"

    def transform_request_to_count_tokens(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Transform request to Anthropic CountTokens format.

        Includes optional system and tools fields for accurate token counting.
        """
        request: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }

        if system is not None:
            request["system"] = system

        if tools is not None:
            request["tools"] = tools

        return request

    def get_required_headers(self, api_key: str) -> Dict[str, str]:
        """
        Get the required headers for the CountTokens API.

        Args:
            api_key: The Anthropic API key

        Returns:
            Dictionary of required headers
        """
        from litellm.llms.anthropic.common_utils import (
            optionally_handle_anthropic_oauth,
        )

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ANTHROPIC_TOKEN_COUNTING_BETA_VERSION,
        }
        headers, _ = optionally_handle_anthropic_oauth(headers=headers, api_key=api_key)
        return headers

    def validate_request(self, model: str, messages: List[Dict[str, Any]]) -> None:
        """
        Validate the incoming count tokens request.

        Args:
            model: The model name
            messages: The messages to count tokens for

        Raises:
            ValueError: If the request is invalid
        """
        if not model:
            raise ValueError("model parameter is required")

        if not messages:
            raise ValueError("messages parameter is required")

        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"Message {i} must be a dictionary")

            if "role" not in message:
                raise ValueError(f"Message {i} must have a 'role' field")

            if "content" not in message:
                raise ValueError(f"Message {i} must have a 'content' field")
