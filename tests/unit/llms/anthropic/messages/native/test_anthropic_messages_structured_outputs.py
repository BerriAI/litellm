"""
Tests for structured outputs support in Anthropic /v1/messages endpoint.
"""

from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.llms.anthropic.messages.native.transformation import (
    AnthropicMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams


def test_output_format_supported_and_transforms_correctly():
    """Test that output_format is supported and properly transformed with beta header."""
    config = AnthropicMessagesConfig()

    # 1. Verify it's in supported parameters
    supported_params = config.get_supported_anthropic_messages_params("claude-sonnet-4-5")
    assert "output_format" in supported_params

    # 2. Verify transformation preserves output_format and adds beta header
    output_format = {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"result": {"type": "string"}}},
    }

    optional_params = {"max_tokens": 1024, "output_format": output_format}
    headers = {}

    # Transform request
    result = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "test"}],
        anthropic_messages_optional_request_params=optional_params.copy(),
        litellm_params={},
        headers=headers,
    )

    # Update headers
    headers = config._update_headers_with_anthropic_beta(headers, optional_params)

    # Verify output_format preserved in request body
    assert "output_format" in result
    assert result["output_format"]["type"] == "json_schema"

    # Verify beta header added
    assert "anthropic-beta" in headers
    assert "structured-outputs-2025-11-13" in headers["anthropic-beta"]


def test_output_config_format_supported_and_transforms_correctly():
    """Test that output_config.format is preserved and adds the structured-output beta."""
    config = AnthropicMessagesConfig()

    supported_params = config.get_supported_anthropic_messages_params("claude-opus-4-7")
    assert "output_config" in supported_params

    output_format = {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"result": {"type": "string"}}},
    }
    optional_params = {
        "max_tokens": 1024,
        "output_config": {"format": output_format, "effort": "xhigh"},
    }
    headers = {}

    result = config.transform_anthropic_messages_request(
        model="claude-opus-4-7",
        messages=[{"role": "user", "content": "test"}],
        anthropic_messages_optional_request_params=optional_params.copy(),
        litellm_params={},
        headers=headers,
    )

    headers = config._update_headers_with_anthropic_beta(headers, optional_params)

    assert result["output_config"]["format"] == output_format
    assert result["output_config"]["effort"] == "xhigh"
    assert "anthropic-beta" in headers
    assert "structured-outputs-2025-11-13" in headers["anthropic-beta"]


def test_output_format_works_with_bedrock_and_azure():
    """Test that output_format works with Bedrock and Azure Foundry models."""
    config = AnthropicMessagesConfig()

    output_format = {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {}},
    }
    optional_params = {"max_tokens": 1024, "output_format": output_format}
    messages = [{"role": "user", "content": "test"}]

    # Test Bedrock
    bedrock_result = config.transform_anthropic_messages_request(
        model="bedrock/anthropic.claude-sonnet-4-5-v2:0",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params.copy(),
        litellm_params={},
        headers={},
    )
    assert "output_format" in bedrock_result

    # Test Azure Foundry
    azure_result = config.transform_anthropic_messages_request(
        model="azure_ai/claude-sonnet-4-5",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params.copy(),
        litellm_params={},
        headers={},
    )
    assert "output_format" in azure_result


_OBJECT_OUTPUT_FORMAT: Final = {"type": "json_schema", "schema": {"type": "object", "properties": {}}}


def _request_with_output_format(
    output_format: object, litellm_params: GenericLiteLLMParams
) -> Mapping[str, object]:
    return AnthropicMessagesConfig().transform_anthropic_messages_request(
        model="claude-haiku-5-5",
        messages=[{"role": "user", "content": "What is 6 times 7?"}],
        anthropic_messages_optional_request_params={"max_tokens": 1024, "output_format": output_format},
        litellm_params=litellm_params,
        headers={},
    )


@pytest.mark.parametrize("output_format", ["json", ["json_schema"], 42])
def test_malformed_output_format_is_dropped_under_drop_params(output_format: object) -> None:
    result: Final = _request_with_output_format(output_format, GenericLiteLLMParams(drop_params=True))

    assert "output_format" not in result


def test_malformed_output_format_is_dropped_under_the_global_drop_params_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", True)

    result: Final = _request_with_output_format("json", GenericLiteLLMParams())

    assert "output_format" not in result


def test_object_output_format_is_kept_under_drop_params() -> None:
    result: Final = _request_with_output_format(_OBJECT_OUTPUT_FORMAT, GenericLiteLLMParams(drop_params=True))

    assert result["output_format"] == _OBJECT_OUTPUT_FORMAT


def test_malformed_output_format_is_forwarded_for_the_provider_to_reject_without_drop_params() -> None:
    result: Final = _request_with_output_format("json", GenericLiteLLMParams())

    assert result["output_format"] == "json"
