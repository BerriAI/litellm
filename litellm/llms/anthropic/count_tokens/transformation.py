"""
Anthropic CountTokens API transformation logic.

This module handles the transformation of requests to Anthropic's CountTokens API format.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter

from litellm.constants import ANTHROPIC_TOKEN_COUNTING_BETA_VERSION
from litellm.types.llms.openai import ChatCompletionImageObject

_COUNT_REQUEST: Final = TypeAdapter(dict[str, JsonValue])
_IMAGE_BLOCK: Final = TypeAdapter(ChatCompletionImageObject)
COUNT_TOKEN_OPTION_NAMES: Final = ("thinking", "tool_choice", "output_config")


def _count_image(block: JsonValue) -> JsonValue:
    if not isinstance(block, dict) or block.get("type") != "image_url":
        return block
    from litellm.litellm_core_utils.prompt_templates.factory import convert_to_anthropic_image_obj

    image_block: Final = _IMAGE_BLOCK.validate_python(block)
    image_url: Final = image_block["image_url"]
    source: Final = convert_to_anthropic_image_obj(
        openai_image_url=image_url if isinstance(image_url, str) else image_url["url"],
        format=image_url.get("format") if isinstance(image_url, dict) else None,
    )
    image: Final = _COUNT_REQUEST.validate_python({"type": "image", "source": source})
    return {**{key: value for key, value in block.items() if key not in {"type", "image_url"}}, **image}


def _count_block(block: JsonValue) -> JsonValue:
    if not isinstance(block, dict) or block.get("type") != "tool_result":
        return _count_image(block)
    content: Final = block.get("content")
    if not isinstance(content, list):
        return block
    return {**block, "content": [_count_image(part) for part in content]}


def _count_content(content: JsonValue) -> JsonValue:
    return [_count_block(block) for block in content] if isinstance(content, list) else content


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
        messages: list[dict[str, JsonValue]],
        tools: list[dict[str, JsonValue]] | None = None,
        system: JsonValue = None,
        optional_params: Mapping[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:  # mutable-ok: provider transport requires JSON dictionaries
        """
        Transform request to Anthropic CountTokens format.

        Includes optional system and tools fields for accurate token counting.
        """
        options: Final[Mapping[str, JsonValue]] = optional_params or MappingProxyType({})
        return _COUNT_REQUEST.validate_python(
            MappingProxyType(
                {
                    "model": model,
                    "messages": [{**message, "content": _count_content(message["content"])} for message in messages],
                    **MappingProxyType(
                        {key: value for key, value in (("system", system), ("tools", tools)) if value is not None}
                    ),
                    **MappingProxyType(
                        {key: value for key, value in options.items() if key in COUNT_TOKEN_OPTION_NAMES}
                    ),
                }
            )
        )

    def get_required_headers(self, api_key: str) -> dict[str, str]:
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

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ANTHROPIC_TOKEN_COUNTING_BETA_VERSION,
        }
        headers, _ = optionally_handle_anthropic_oauth(headers=headers, api_key=api_key)
        return headers

    def validate_request(
        self,
        model: str,
        messages: Sequence[Mapping[str, JsonValue]],
        *,
        system: JsonValue = None,
        tools: list[dict[str, JsonValue]] | None = None,
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

        if not messages and not system and not tools:
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
