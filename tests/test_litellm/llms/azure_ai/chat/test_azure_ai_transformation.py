import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))  # Adds the parent directory to the system path
from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.llms.azure_ai.chat.transformation import AzureAIStudioConfig


@pytest.mark.asyncio
async def test_get_openai_compatible_provider_info():
    """
    Test that Azure AI requests are formatted correctly with the proper endpoint and parameters
    for both synchronous and asynchronous calls
    """
    config = AzureAIStudioConfig()

    (
        api_base,
        dynamic_api_key,
        custom_llm_provider,
    ) = config._get_openai_compatible_provider_info(
        model="azure_ai/gpt-4o-mini",
        api_base="https://my-base",
        api_key="my-key",
        custom_llm_provider="azure_ai",
    )

    assert custom_llm_provider == "azure"


def test_azure_ai_validate_environment():
    config = AzureAIStudioConfig()
    headers = config.validate_environment(
        headers={},
        model="azure_ai/gpt-4o-mini",
        messages=[],
        optional_params={},
        litellm_params={},
    )
    assert headers["Content-Type"] == "application/json"


def test_azure_ai_validate_environment_with_api_key():
    """
    Test that when api_key is provided, it is set in the api-key header
    for Azure Foundry endpoints (.services.ai.azure.com).
    """
    config = AzureAIStudioConfig()
    headers = config.validate_environment(
        headers={},
        model="Kimi-K2.5",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-api-key",
        api_base="https://my-endpoint.services.ai.azure.com",
    )
    assert headers["api-key"] == "test-api-key"
    assert headers["Content-Type"] == "application/json"


def test_azure_ai_validate_environment_with_azure_ad_token():
    """
    Test that when no api_key is provided but Azure AD credentials are available,
    the Authorization header is set with a Bearer token.

    Regression test for https://github.com/BerriAI/litellm/issues/20759
    """
    import litellm

    config = AzureAIStudioConfig()
    with (
        patch(
            "litellm.llms.azure.common_utils.get_azure_ad_token",
            return_value="fake-azure-ad-token",
        ),
        patch(
            "litellm.llms.azure.common_utils.get_secret_str",
            return_value=None,
        ),
        patch.object(litellm, "api_key", None),
        patch.object(litellm, "azure_key", None),
    ):
        headers = config.validate_environment(
            headers={},
            model="Kimi-K2.5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base="https://my-endpoint.services.ai.azure.com",
        )
    assert headers.get("Authorization") == "Bearer fake-azure-ad-token"
    assert "api-key" not in headers
    assert headers["Content-Type"] == "application/json"


def test_azure_ai_grok_stop_parameter_handling():
    """
    Test that Grok models properly handle stop parameter filtering in Azure AI Studio.
    """
    config = AzureAIStudioConfig()

    # Test Grok model detection
    assert config._supports_stop_reason("grok-4-fast") == False
    assert config._supports_stop_reason("grok-4") == False
    assert config._supports_stop_reason("grok-3-mini") == False
    assert config._supports_stop_reason("grok-code-fast") == False
    assert config._supports_stop_reason("gpt-4") == True

    # Test supported parameters for Grok models
    grok_params = config.get_supported_openai_params("grok-4-fast")
    assert "stop" not in grok_params, "Grok models should not support stop parameter"

    # Test supported parameters for non-Grok models
    gpt_params = config.get_supported_openai_params("gpt-4")
    assert "stop" in gpt_params, "GPT models should support stop parameter"


class TestAzureAIStructuredOutput:
    """Tests for structured output (response_format) handling in Azure AI Foundry."""

    SAMPLE_JSON_SCHEMA = {
        "type": "json_schema",
        "json_schema": {
            "name": "UserProfile",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            },
        },
    }

    def test_should_pass_through_response_format_for_supported_model(self):
        """
        When a model supports native structured output (supports_response_schema=True),
        response_format should be passed through as-is without tool conversion.
        """
        config = AzureAIStudioConfig()

        with patch(
            "litellm.llms.azure_ai.chat.transformation.supports_response_schema",
            return_value=True,
        ):
            result = config.map_openai_params(
                non_default_params={"response_format": self.SAMPLE_JSON_SCHEMA},
                optional_params={},
                model="claude-sonnet-4-5",
                drop_params=False,
            )

        assert "response_format" in result, "response_format should be passed through for supported models"
        assert result["response_format"] == self.SAMPLE_JSON_SCHEMA
        assert "tools" not in result, "No tool conversion should occur for supported models"
        assert "json_mode" not in result

    def test_should_convert_to_tool_call_for_unsupported_model(self):
        """
        When a model does NOT support native structured output (supports_response_schema=False),
        response_format should be converted to a forced tool call with the JSON schema.
        """
        config = AzureAIStudioConfig()

        with patch(
            "litellm.llms.azure_ai.chat.transformation.supports_response_schema",
            return_value=False,
        ):
            result = config.map_openai_params(
                non_default_params={"response_format": self.SAMPLE_JSON_SCHEMA},
                optional_params={},
                model="Mistral-7B",
                drop_params=False,
            )

        assert "response_format" not in result, "response_format should NOT be in params when using tool fallback"
        assert "tools" in result, "A tool should be injected for schema enforcement"
        assert len(result["tools"]) == 1

        tool_function = result["tools"][0]["function"]
        assert tool_function["name"] == RESPONSE_FORMAT_TOOL_NAME
        assert tool_function["parameters"] == self.SAMPLE_JSON_SCHEMA["json_schema"]["schema"]

        assert "tool_choice" in result, "tool_choice should be forced"
        assert result["tool_choice"]["function"]["name"] == RESPONSE_FORMAT_TOOL_NAME

        assert result.get("json_mode") is True

    def test_should_pass_through_json_object_format_without_schema(self):
        """
        response_format={"type": "json_object"} (no schema) should always pass
        through directly, regardless of model support for structured output.
        """
        config = AzureAIStudioConfig()

        json_object_format = {"type": "json_object"}

        with patch(
            "litellm.llms.azure_ai.chat.transformation.supports_response_schema",
            return_value=False,
        ):
            result = config.map_openai_params(
                non_default_params={"response_format": json_object_format},
                optional_params={},
                model="Mistral-7B",
                drop_params=False,
            )

        assert result["response_format"] == json_object_format
        assert "tools" not in result, "json_object mode should not trigger tool conversion"
        assert "json_mode" not in result

    def test_should_preserve_other_params_alongside_response_format(self):
        """
        Other params (temperature, max_tokens, etc.) should still be mapped
        correctly when response_format is also present.
        """
        config = AzureAIStudioConfig()

        with patch(
            "litellm.llms.azure_ai.chat.transformation.supports_response_schema",
            return_value=True,
        ):
            result = config.map_openai_params(
                non_default_params={
                    "temperature": 0.7,
                    "max_tokens": 500,
                    "response_format": self.SAMPLE_JSON_SCHEMA,
                },
                optional_params={},
                model="claude-sonnet-4-5",
                drop_params=False,
            )

        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 500
        assert result["response_format"] == self.SAMPLE_JSON_SCHEMA

    def test_should_handle_response_schema_key_variant(self):
        """
        Some callers use 'response_schema' instead of 'json_schema' inside
        the response_format dict. The tool fallback should handle both variants.
        """
        config = AzureAIStudioConfig()

        response_format_with_response_schema = {
            "type": "json_schema",
            "response_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }

        with patch(
            "litellm.llms.azure_ai.chat.transformation.supports_response_schema",
            return_value=False,
        ):
            result = config.map_openai_params(
                non_default_params={"response_format": response_format_with_response_schema},
                optional_params={},
                model="Phi-4",
                drop_params=False,
            )

        assert "tools" in result, "Tool fallback should work with response_schema key"
        tool_function = result["tools"][0]["function"]
        assert tool_function["name"] == RESPONSE_FORMAT_TOOL_NAME
        assert tool_function["parameters"] == response_format_with_response_schema["response_schema"]

    def test_should_append_to_existing_tools(self):
        """
        When the user already provides tools and the model does not support
        structured output, the schema tool should be appended to the existing
        tools list (not replace it).
        """
        config = AzureAIStudioConfig()

        existing_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }
        ]

        with patch(
            "litellm.llms.azure_ai.chat.transformation.supports_response_schema",
            return_value=False,
        ):
            result = config.map_openai_params(
                non_default_params={
                    "tools": existing_tools,
                    "response_format": self.SAMPLE_JSON_SCHEMA,
                },
                optional_params={},
                model="Mistral-7B",
                drop_params=False,
            )

        assert len(result["tools"]) == 2, "Should have both the user's tool and the schema tool"
        tool_names = [t["function"]["name"] for t in result["tools"]]
        assert "get_weather" in tool_names
        assert RESPONSE_FORMAT_TOOL_NAME in tool_names
