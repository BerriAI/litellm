"""
Tests for structured outputs support in Anthropic /v1/messages endpoint.

This tests the fix for the issue where output_format parameter was not being
properly handled in the /v1/messages endpoint, causing structured outputs to
return markdown text instead of JSON.
"""
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.llms.anthropic import AnthropicOutputSchema


def test_output_format_in_supported_params():
    """Test that output_format is included in supported parameters list."""
    config = AnthropicMessagesConfig()
    supported_params = config.get_supported_anthropic_messages_params(
        model="claude-sonnet-4-5-20250929"
    )

    assert "output_format" in supported_params, \
        "output_format should be in supported parameters for /v1/messages endpoint"


def test_transform_anthropic_messages_request_with_output_format():
    """Test that output_format is preserved during request transformation."""
    config = AnthropicMessagesConfig()

    output_format: AnthropicOutputSchema = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "plan_interest": {"type": "string"},
                "demo_requested": {"type": "boolean"}
            },
            "required": ["name", "email", "plan_interest", "demo_requested"],
            "additionalProperties": False
        }
    }

    messages = [
        {
            "role": "user",
            "content": "Extract the key information from this email: John Smith (john@example.com) is interested in our Enterprise plan and wants to schedule a demo for next Tuesday at 2pm."
        }
    ]

    optional_params = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "output_format": output_format
    }

    result = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-5-20250929",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params,
        litellm_params={},
        headers={}
    )

    assert "output_format" in result, \
        "output_format should be in transformed request"
    assert result["output_format"]["type"] == "json_schema", \
        "output_format.type should be 'json_schema'"
    assert "schema" in result["output_format"], \
        "output_format should contain schema"
    assert result["output_format"]["schema"] == output_format["schema"], \
        "output_format schema should be preserved exactly"


def test_structured_outputs_beta_header_added():
    """Test that structured-outputs beta header is automatically added when output_format is used."""
    config = AnthropicMessagesConfig()

    output_format: AnthropicOutputSchema = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "result": {"type": "string"}
            }
        }
    }

    optional_params = {
        "output_format": output_format
    }

    headers = {}

    result_headers = config._update_headers_with_anthropic_beta(
        headers=headers,
        optional_params=optional_params
    )

    assert "anthropic-beta" in result_headers, \
        "anthropic-beta header should be added when output_format is present"
    assert "structured-outputs-2025-11-13" in result_headers["anthropic-beta"], \
        "structured-outputs-2025-11-13 should be in anthropic-beta header"


def test_structured_outputs_beta_header_merges_with_existing():
    """Test that structured-outputs beta header merges correctly with existing beta headers."""
    config = AnthropicMessagesConfig()

    output_format: AnthropicOutputSchema = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "result": {"type": "string"}
            }
        }
    }

    optional_params = {
        "output_format": output_format,
        "context_management": {"type": "ephemeral"}
    }

    # Start with an existing beta header
    headers = {
        "anthropic-beta": "custom-beta-feature"
    }

    result_headers = config._update_headers_with_anthropic_beta(
        headers=headers,
        optional_params=optional_params
    )

    beta_value = result_headers["anthropic-beta"]
    assert "structured-outputs-2025-11-13" in beta_value, \
        "structured-outputs-2025-11-13 should be in merged beta header"
    assert "context-management-2025-06-27" in beta_value, \
        "context-management-2025-06-27 should be in merged beta header"
    assert "custom-beta-feature" in beta_value, \
        "custom-beta-feature should be preserved in merged beta header"


@pytest.mark.asyncio
async def test_anthropic_messages_with_output_format_makes_correct_request():
    """
    Integration test that verifies output_format is correctly passed to the Anthropic API.
    This test mocks the HTTP client to verify the request structure.
    """
    from litellm.anthropic_interface import messages

    client = AsyncHTTPHandler()

    with patch.object(client, "post") as mock_post:
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.text = "mock response"
        mock_response.json.return_value = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": '{"name": "John Smith", "email": "john@example.com", "plan_interest": "Enterprise plan", "demo_requested": true}'
                }
            ],
            "model": "claude-sonnet-4-5-20250929",
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 44,
                "output_tokens": 30
            }
        }
        mock_post.return_value = mock_response

        output_format = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                    "plan_interest": {"type": "string"},
                    "demo_requested": {"type": "boolean"}
                },
                "required": ["name", "email", "plan_interest", "demo_requested"],
                "additionalProperties": False
            }
        }

        try:
            await messages.acreate(
                client=client,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": "Extract the key information from this email: John Smith (john@example.com) is interested in our Enterprise plan."
                    }
                ],
                model="anthropic/claude-sonnet-4-5-20250929",
                output_format=output_format,
            )
        except Exception as e:
            print(f"Test error (expected due to mock): {e}")

        # Verify the request was made
        mock_post.assert_called_once()

        # Extract the request data
        call_kwargs = mock_post.call_args.kwargs
        json_data = call_kwargs.get("json") or json.loads(call_kwargs.get("data", "{}"))

        # Verify output_format is in the request body
        assert "output_format" in json_data, \
            "output_format should be in the request body sent to Anthropic API"
        assert json_data["output_format"]["type"] == "json_schema", \
            "output_format.type should be 'json_schema'"
        assert "schema" in json_data["output_format"], \
            "output_format should contain schema"

        # Verify the structured-outputs beta header is set
        headers = call_kwargs.get("headers", {})
        assert "anthropic-beta" in headers, \
            "anthropic-beta header should be set in request headers"
        assert "structured-outputs-2025-11-13" in headers["anthropic-beta"], \
            "structured-outputs-2025-11-13 should be in anthropic-beta header"


def test_bedrock_and_foundry_models_with_output_format():
    """
    Test that output_format works correctly with Bedrock and Azure Foundry models.
    This is the specific use case mentioned in the GitHub issue.
    """
    config = AnthropicMessagesConfig()

    output_format: AnthropicOutputSchema = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            }
        }
    }

    messages = [{"role": "user", "content": "Extract name and email"}]

    # Test with Bedrock model
    bedrock_params = {
        "max_tokens": 1024,
        "output_format": output_format
    }

    bedrock_result = config.transform_anthropic_messages_request(
        model="bedrock/anthropic.claude-sonnet-4-5-v2:0",
        messages=messages,
        anthropic_messages_optional_request_params=bedrock_params,
        litellm_params={},
        headers={}
    )

    assert "output_format" in bedrock_result, \
        "output_format should work with Bedrock models"

    # Test with Azure Foundry model
    foundry_params = {
        "max_tokens": 1024,
        "output_format": output_format
    }

    foundry_result = config.transform_anthropic_messages_request(
        model="azure_ai/claude-sonnet-4-5",
        messages=messages,
        anthropic_messages_optional_request_params=foundry_params,
        litellm_params={},
        headers={}
    )

    assert "output_format" in foundry_result, \
        "output_format should work with Azure Foundry models"
