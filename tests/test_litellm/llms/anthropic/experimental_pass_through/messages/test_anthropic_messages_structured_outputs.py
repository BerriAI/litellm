"""
Tests for structured outputs support in Anthropic /v1/messages endpoint.
"""
import pytest
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)


def test_output_format_supported_and_transforms_correctly():
    """Test that output_format is supported and properly transformed with beta header."""
    config = AnthropicMessagesConfig()

    # 1. Verify it's in supported parameters
    supported_params = config.get_supported_anthropic_messages_params("claude-sonnet-4-5")
    assert "output_format" in supported_params

    # 2. Verify transformation preserves output_format and adds beta header
    output_format = {
        "type": "json_schema",
        "schema": {"type": "object", "properties": {"result": {"type": "string"}}}
    }

    optional_params = {"max_tokens": 1024, "output_format": output_format}
    headers = {}

    # Transform request
    result = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "test"}],
        anthropic_messages_optional_request_params=optional_params.copy(),
        litellm_params={},
        headers=headers
    )

    # Update headers
    headers = config._update_headers_with_anthropic_beta(headers, optional_params)

    # Verify output_format preserved in request body
    assert "output_format" in result
    assert result["output_format"]["type"] == "json_schema"

    # Verify beta header added
    assert "anthropic-beta" in headers
    assert "structured-outputs-2025-11-13" in headers["anthropic-beta"]


def test_output_format_works_with_bedrock_and_azure():
    """Test that output_format works with Bedrock and Azure Foundry models."""
    config = AnthropicMessagesConfig()

    output_format = {"type": "json_schema", "schema": {"type": "object", "properties": {}}}
    optional_params = {"max_tokens": 1024, "output_format": output_format}
    messages = [{"role": "user", "content": "test"}]

    # Test Bedrock
    bedrock_result = config.transform_anthropic_messages_request(
        model="bedrock/anthropic.claude-sonnet-4-5-v2:0",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params.copy(),
        litellm_params={},
        headers={}
    )
    assert "output_format" in bedrock_result

    # Test Azure Foundry
    azure_result = config.transform_anthropic_messages_request(
        model="azure_ai/claude-sonnet-4-5",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params.copy(),
        litellm_params={},
        headers={}
    )
    assert "output_format" in azure_result


def test_output_config_stripped_for_bedrock_provider():
    """Test that Bedrock requests do not include unsupported output_config."""
    config = AnthropicMessagesConfig()

    optional_params = {
        "max_tokens": 1024,
        "output_config": {"effort": "medium"},
        "output_format": {"type": "json_schema", "schema": {"type": "object"}},
    }

    result = config.transform_anthropic_messages_request(
        model="bedrock-nova-pro",
        messages=[{"role": "user", "content": "Say hello"}],
        anthropic_messages_optional_request_params=optional_params.copy(),
        litellm_params={"custom_llm_provider": "bedrock"},
        headers={},
    )

    assert "output_config" not in result
    assert "output_format" in result
