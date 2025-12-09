import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.azure_ai.anthropic.transformation import AzureAnthropicConfig
from litellm.types.router import GenericLiteLLMParams


class TestAzureAnthropicConfig:
    def test_custom_llm_provider(self):
        """Test that custom_llm_provider returns 'azure_ai'"""
        config = AzureAnthropicConfig()
        assert config.custom_llm_provider == "azure_ai"

    def test_validate_environment_with_dict_litellm_params(self):
        """Test validate_environment with dict litellm_params"""
        config = AzureAnthropicConfig()
        headers = {}
        model = "claude-sonnet-4-5"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {}
        litellm_params = {"api_key": "test-api-key"}
        api_key = "test-api-key"

        with patch(
            "litellm.llms.azure.common_utils.BaseAzureLLM._base_validate_azure_environment"
        ) as mock_validate:
            mock_validate.return_value = {"api-key": "test-api-key"}
            result = config.validate_environment(
                headers=headers,
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                api_key=api_key,
            )

            # Verify that dict was converted to GenericLiteLLMParams
            call_args = mock_validate.call_args
            assert isinstance(call_args[1]["litellm_params"], GenericLiteLLMParams)
            assert call_args[1]["litellm_params"].api_key == "test-api-key"
            assert "anthropic-version" in result

    def test_validate_environment_with_generic_litellm_params(self):
        """Test validate_environment with GenericLiteLLMParams object"""
        config = AzureAnthropicConfig()
        headers = {}
        model = "claude-sonnet-4-5"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {}
        litellm_params = GenericLiteLLMParams(api_key="test-api-key")
        api_key = "test-api-key"

        with patch(
            "litellm.llms.azure.common_utils.BaseAzureLLM._base_validate_azure_environment"
        ) as mock_validate:
            mock_validate.return_value = {"api-key": "test-api-key"}
            result = config.validate_environment(
                headers=headers,
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                api_key=api_key,
            )

            # Verify that GenericLiteLLMParams was passed through
            call_args = mock_validate.call_args
            assert isinstance(call_args[1]["litellm_params"], GenericLiteLLMParams)
            assert "anthropic-version" in result

    def test_validate_environment_sets_api_key_in_litellm_params(self):
        """Test that api_key parameter is set in litellm_params if provided"""
        config = AzureAnthropicConfig()
        headers = {}
        model = "claude-sonnet-4-5"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {}
        litellm_params = {}  # Empty dict, no api_key
        api_key = "provided-api-key"

        with patch(
            "litellm.llms.azure.common_utils.BaseAzureLLM._base_validate_azure_environment"
        ) as mock_validate:
            mock_validate.return_value = {"api-key": "provided-api-key"}
            config.validate_environment(
                headers=headers,
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                api_key=api_key,
            )

            # Verify that api_key was set in litellm_params
            call_args = mock_validate.call_args
            assert call_args[1]["litellm_params"].api_key == "provided-api-key"

    def test_validate_environment_preserves_api_key_header(self):
        """Test that api-key header is preserved as-is (Azure handles the header internally)"""
        config = AzureAnthropicConfig()
        headers = {}
        model = "claude-sonnet-4-5"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {}
        litellm_params = {"api_key": "test-api-key"}

        with patch(
            "litellm.llms.azure.common_utils.BaseAzureLLM._base_validate_azure_environment"
        ) as mock_validate:
            mock_validate.return_value = {"api-key": "test-api-key"}
            with patch.object(
                config, "get_anthropic_headers", return_value={}
            ):
                result = config.validate_environment(
                    headers=headers,
                    model=model,
                    messages=messages,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                )

                # Verify api-key header is preserved as-is
                assert "api-key" in result
                assert result["api-key"] == "test-api-key"

    def test_validate_environment_sets_anthropic_version(self):
        """Test that anthropic-version header is set"""
        config = AzureAnthropicConfig()
        headers = {}
        model = "claude-sonnet-4-5"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {}
        litellm_params = {"api_key": "test-api-key"}

        with patch(
            "litellm.llms.azure.common_utils.BaseAzureLLM._base_validate_azure_environment"
        ) as mock_validate:
            mock_validate.return_value = {"api-key": "test-api-key"}
            with patch.object(config, "get_anthropic_headers", return_value={}):
                result = config.validate_environment(
                    headers=headers,
                    model=model,
                    messages=messages,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                )

                assert result["anthropic-version"] == "2023-06-01"

    def test_validate_environment_preserves_existing_anthropic_version(self):
        """Test that existing anthropic-version header is preserved"""
        config = AzureAnthropicConfig()
        headers = {"anthropic-version": "2024-01-01"}
        model = "claude-sonnet-4-5"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {}
        litellm_params = {"api_key": "test-api-key"}

        with patch(
            "litellm.llms.azure.common_utils.BaseAzureLLM._base_validate_azure_environment"
        ) as mock_validate:
            mock_validate.return_value = {"api-key": "test-api-key", "anthropic-version": "2024-01-01"}
            with patch.object(config, "get_anthropic_headers", return_value={"anthropic-version": "2024-01-01"}):
                result = config.validate_environment(
                    headers=headers,
                    model=model,
                    messages=messages,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                )

                assert result["anthropic-version"] == "2024-01-01"

    def test_inherits_anthropic_config_methods(self):
        """Test that AzureAnthropicConfig inherits methods from AnthropicConfig"""
        config = AzureAnthropicConfig()

        # Test that it has AnthropicConfig methods
        assert hasattr(config, "get_anthropic_headers")
        assert hasattr(config, "is_cache_control_set")
        assert hasattr(config, "is_computer_tool_used")
        assert hasattr(config, "transform_request")
        assert hasattr(config, "transform_response")

    def test_transform_request_removes_max_retries(self):
        """Test that max_retries is removed from optional_params before transformation.

        Azure AI Anthropic API doesn't accept max_retries parameter and returns:
        {"type":"error","error":{"type":"invalid_request_error","message":"max_retries: Extra inputs are not permitted"}}
        """
        config = AzureAnthropicConfig()
        model = "claude-haiku-4-5"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {"max_tokens": 100, "max_retries": 3}
        litellm_params = {"api_key": "test-key"}
        headers = {"api-key": "test-key", "anthropic-version": "2023-06-01"}

        data = config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # max_retries should NOT be in the request data
        assert "max_retries" not in data
        # max_tokens should still be there
        assert data.get("max_tokens") == 100

    def test_transform_request_adds_custom_type_to_tools(self):
        """Test that user-defined tools get type='custom' added.

        Azure AI Anthropic requires explicit "type": "custom" for user-defined tools,
        while regular Anthropic API allows tools without a type field.
        """
        config = AzureAnthropicConfig()
        model = "claude-haiku-4-5"
        messages = [{"role": "user", "content": "Hello"}]
        # Tool in Anthropic format (after map_openai_params transformation) but without type
        optional_params = {
            "max_tokens": 100,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"],
                    },
                }
            ],
        }
        litellm_params = {"api_key": "test-key"}
        headers = {"api-key": "test-key", "anthropic-version": "2023-06-01"}

        data = config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Tool should now have type="custom"
        assert len(data["tools"]) == 1
        assert data["tools"][0]["type"] == "custom"
        assert data["tools"][0]["name"] == "get_weather"

    def test_transform_request_preserves_builtin_tool_types(self):
        """Test that built-in tools with existing type are not modified.

        Built-in tools like web_search_20250305, bash_20250124, etc. should keep
        their original type and not be changed to 'custom'.
        """
        config = AzureAnthropicConfig()
        model = "claude-haiku-4-5"
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "max_tokens": 100,
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                },
                {
                    "name": "my_custom_tool",
                    "description": "A custom tool",
                    "input_schema": {"type": "object", "properties": {}},
                },
            ],
        }
        litellm_params = {"api_key": "test-key"}
        headers = {"api-key": "test-key", "anthropic-version": "2023-06-01"}

        data = config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # web_search tool should keep its original type
        web_search_tool = next(t for t in data["tools"] if t["name"] == "web_search")
        assert web_search_tool["type"] == "web_search_20250305"

        # Custom tool should get type="custom" added
        custom_tool = next(t for t in data["tools"] if t["name"] == "my_custom_tool")
        assert custom_tool["type"] == "custom"

