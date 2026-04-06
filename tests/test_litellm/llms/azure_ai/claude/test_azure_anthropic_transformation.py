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

    def test_transform_request_removes_unsupported_params(self):
        """Test that transform_request removes max_retries, stream_options, and extra_body.

        These parameters are LiteLLM-internal and not supported by Azure AI Anthropic endpoint.
        See: https://github.com/BerriAI/litellm/issues/XXXX
        """
        config = AzureAnthropicConfig()

        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "max_tokens": 100,
        }
        litellm_params = {"api_key": "test-key"}
        headers = {"api-key": "test-key", "anthropic-version": "2023-06-01"}

        with patch.object(
            config.__class__.__bases__[0],  # AnthropicConfig
            "transform_request",
            return_value={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
                "max_tokens": 100,
                "max_retries": 3,  # Should be removed
                "stream_options": {"include_usage": True},  # Should be removed
                "extra_body": {"custom": "param"},  # Should be removed
            },
        ):
            result = config.transform_request(
                model="claude-sonnet-4-5",
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )

        # Verify unsupported params are removed
        assert "max_retries" not in result
        assert "stream_options" not in result
        assert "extra_body" not in result

        # Verify supported params are preserved
        assert result["model"] == "claude-sonnet-4-5"
        assert result["max_tokens"] == 100
        assert "messages" in result

    def test_context_management_compact_beta_header(self):
        """Test that context_management with compact adds the correct beta header for Azure AI"""
        config = AzureAnthropicConfig()
        
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "context_management": {
                "edits": [
                    {
                        "type": "compact_20260112"
                    }
                ]
            },
            "max_tokens": 100
        }
        litellm_params = {"api_key": "test-key"}
        headers = {"api-key": "test-key"}
        
        with patch(
            "litellm.llms.azure.common_utils.BaseAzureLLM._base_validate_azure_environment"
        ) as mock_validate:
            mock_validate.return_value = {"api-key": "test-key"}
            result = config.transform_request(
                model="claude-opus-4-6",
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        
        # Verify context_management is included
        assert "context_management" in result
        assert result["context_management"]["edits"][0]["type"] == "compact_20260112"

    def test_context_management_compact_beta_header_in_headers(self):
        """Test that compact beta header is added to headers for Azure AI"""
        config = AzureAnthropicConfig()
        
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "context_management": {
                "edits": [
                    {
                        "type": "compact_20260112"
                    }
                ]
            },
            "max_tokens": 100
        }
        
        # Test that the parent's update_headers_with_optional_anthropic_beta is called
        # which should add the compact beta header
        headers = {}
        headers = config.update_headers_with_optional_anthropic_beta(
            headers=headers,
            optional_params=optional_params
        )
        
        # Verify compact beta header is present
        assert "anthropic-beta" in headers
        assert "compact-2026-01-12" in headers["anthropic-beta"]

    def test_context_management_mixed_edits_beta_headers(self):
        """Test that context_management with both compact and other edits adds both beta headers"""
        config = AzureAnthropicConfig()
        
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "context_management": {
                "edits": [
                    {
                        "type": "compact_20260112"
                    },
                    {
                        "type": "replace",
                        "message_id": "msg_123",
                        "content": "new content"
                    }
                ]
            },
            "max_tokens": 100
        }
        
        headers = {}
        headers = config.update_headers_with_optional_anthropic_beta(
            headers=headers,
            optional_params=optional_params
        )
        
        # Verify both beta headers are present
        assert "anthropic-beta" in headers
        assert "compact-2026-01-12" in headers["anthropic-beta"]
        assert "context-management-2025-06-27" in headers["anthropic-beta"]

