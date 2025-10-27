"""Databricks Native Anthropic Messages API Configuration."""

from typing import Any, Dict, List, Optional, Tuple

import httpx

from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.router import GenericLiteLLMParams


class DatabricksAnthropicMessagesConfig(BaseAnthropicMessagesConfig):
    """Config for Databricks Native Anthropic Messages API."""

    def get_supported_anthropic_messages_params(self, model: str) -> list:
        """Get supported parameters for Anthropic Messages API."""
        return [
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "tools",
            "tool_choice",
            "system",
            "metadata",
            "stop_sequences",
        ]

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """Get complete URL - Databricks native endpoint doesn't need /v1/messages suffix."""
        return api_base or ""

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        """Set Bearer token authentication for Databricks."""
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers, api_base

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """Transform request - pass through as-is for native endpoint."""
        return anthropic_messages_optional_request_params

    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> AnthropicMessagesResponse:
        """Transform response - pass through native Anthropic response."""
        return AnthropicMessagesResponse(**raw_response.json())
