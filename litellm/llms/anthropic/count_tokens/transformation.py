"""
Anthropic CountTokens API transformation logic.

This module handles the transformation of requests to Anthropic's CountTokens API format.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter

from litellm.constants import ANTHROPIC_TOKEN_COUNTING_BETA_VERSION
from litellm.llms.anthropic.common_utils import merge_anthropic_beta_headers
from litellm.llms.anthropic.wif import resolve_anthropic_base
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

    def get_anthropic_count_tokens_endpoint(self, api_base: str | None = None) -> str:
        """
        Get the Anthropic CountTokens API endpoint.

        Args:
            api_base: The deployment's api_base, which names the chat surface (a host, or a
                base already carrying ``/v1`` or ``/v1/messages``); the count-tokens path is
                appended to it, so it is never the full count-tokens URL. Unset or empty falls
                back to ``ANTHROPIC_API_BASE`` / ``ANTHROPIC_BASE_URL`` and then Anthropic's
                host, the same resolution chat and the federated exchange use

        Returns:
            The endpoint URL for the CountTokens API
        """
        return resolve_anthropic_base(api_base) + "/v1/messages/count_tokens"

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

    def get_count_tokens_headers(self, auth_header: Mapping[str, str]) -> dict[str, str]:
        """The count-tokens headers around a resolved Anthropic auth header
        (``AnthropicModelInfo.get_auth_header``): x-api-key for a static key, an Authorization
        bearer for ``ANTHROPIC_AUTH_TOKEN`` and for sk-ant-oat tokens, whose mandatory oauth beta
        merges with the token-counting beta instead of replacing it."""
        return {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            **auth_header,
            "anthropic-beta": merge_anthropic_beta_headers(
                auth_header.get("anthropic-beta"), ANTHROPIC_TOKEN_COUNTING_BETA_VERSION
            ),
        }

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
