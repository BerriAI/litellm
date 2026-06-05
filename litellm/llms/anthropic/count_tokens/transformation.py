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

    def get_anthropic_count_tokens_endpoint(
        self, api_base: Optional[str] = None
    ) -> str:
        """
        Get the Anthropic CountTokens API endpoint.

        Mirrors how /v1/messages resolves its URL: if a custom ``api_base``
        is configured, append the ``/v1/messages/count_tokens`` path when
        it's not already there. This is what makes self-hosted vLLM /
        air-gapped Anthropic-compatible backends work — without it the
        handler hit the hardcoded ``api.anthropic.com`` even when an
        ``api_base`` was set on the deployment (#29764).

        Args:
            api_base: Optional custom API base from the deployment's
                ``litellm_params.api_base``.

        Returns:
            The endpoint URL for the CountTokens API.
        """
        if not api_base:
            return "https://api.anthropic.com/v1/messages/count_tokens"
        api_base = api_base.rstrip("/")
        if api_base.endswith("/v1/messages/count_tokens"):
            return api_base
        if api_base.endswith("/v1/messages"):
            return f"{api_base}/count_tokens"
        if api_base.endswith("/v1"):
            return f"{api_base}/messages/count_tokens"
        return f"{api_base}/v1/messages/count_tokens"

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
