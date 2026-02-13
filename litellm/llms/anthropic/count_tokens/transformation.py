"""
Anthropic CountTokens API transformation logic.

This module handles the transformation of requests to Anthropic's CountTokens API format.
"""

from typing import Any, Dict, List

from litellm.constants import ANTHROPIC_TOKEN_COUNTING_BETA_VERSION


class AnthropicCountTokensConfig:
    """
    Configuration and transformation logic for Anthropic CountTokens API.

    Anthropic CountTokens API Specification:
    - Endpoint: POST https://api.anthropic.com/v1/messages/count_tokens
    - Beta header required: anthropic-beta: token-counting-2024-11-01
    - Response: {"input_tokens": <number>}
    """

    def get_anthropic_count_tokens_endpoint(self) -> str:
        """
        Get the Anthropic CountTokens API endpoint.

        Returns:
            The endpoint URL for the CountTokens API
        """
        return "https://api.anthropic.com/v1/messages/count_tokens"

    def transform_request_to_count_tokens(
        self,
        model: str,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Transform request to Anthropic CountTokens format.

        Input:
        {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello!"}]
        }

        Output (Anthropic CountTokens format):
        {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello!"}]
        }
        """
        return {
            "model": model,
            "messages": messages,
        }

    def get_required_headers(self, api_key: str) -> Dict[str, str]:
        """
        Get the required headers for the CountTokens API.

        Args:
            api_key: The Anthropic API key

        Returns:
            Dictionary of required headers
        """
        return {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ANTHROPIC_TOKEN_COUNTING_BETA_VERSION,
        }

    def validate_request(
        self, model: str, messages: List[Dict[str, Any]]
    ) -> None:
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
