"""
Tests for Azure AI Anthropic CountTokens transformation.

Verifies that the CountTokens API uses the correct authentication headers.
"""
import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path


from litellm.llms.azure_ai.anthropic.count_tokens.transformation import (
    AzureAIAnthropicCountTokensConfig,
)


class TestAzureAIAnthropicCountTokensConfig:
    """Test Azure AI Anthropic CountTokens configuration and headers."""

    def test_get_required_headers_includes_x_api_key(self):
        """
        Test that get_required_headers includes x-api-key header.

        Azure AI Anthropic uses Anthropic's native API format which requires
        the x-api-key header for authentication (not just Azure's api-key).
        """
        config = AzureAIAnthropicCountTokensConfig()
        api_key = "test-api-key-12345"

        headers = config.get_required_headers(api_key=api_key)

        # Verify x-api-key header is set
        assert "x-api-key" in headers
        assert headers["x-api-key"] == api_key

        # Verify base headers are present
        assert headers["Content-Type"] == "application/json"
        assert headers["anthropic-version"] == "2023-06-01"
        assert "anthropic-beta" in headers

    def test_get_required_headers_includes_azure_api_key(self):
        """
        Test that get_required_headers includes Azure api-key header.

        Both x-api-key and api-key headers should be present.
        """
        config = AzureAIAnthropicCountTokensConfig()
        api_key = "test-azure-key-67890"

        headers = config.get_required_headers(api_key=api_key)

        # Verify both authentication headers are set
        assert "x-api-key" in headers
        assert "api-key" in headers
        assert headers["x-api-key"] == api_key
        assert headers["api-key"] == api_key

    def test_get_required_headers_with_litellm_params(self):
        """
        Test that get_required_headers works with litellm_params.
        """
        config = AzureAIAnthropicCountTokensConfig()
        api_key = "test-key"
        litellm_params = {"api_key": "param-key", "custom_field": "value"}

        headers = config.get_required_headers(
            api_key=api_key, litellm_params=litellm_params
        )

        # x-api-key should use the direct api_key parameter
        assert headers["x-api-key"] == api_key
        # Azure api-key should come from litellm_params
        assert headers["api-key"] == "param-key"

    def test_get_count_tokens_endpoint_with_base_url(self):
        """Test endpoint generation from base URL."""
        config = AzureAIAnthropicCountTokensConfig()

        api_base = "https://my-resource.services.ai.azure.com"
        endpoint = config.get_count_tokens_endpoint(api_base)

        assert (
            endpoint
            == "https://my-resource.services.ai.azure.com/anthropic/v1/messages/count_tokens"
        )

    def test_get_count_tokens_endpoint_with_anthropic_path(self):
        """Test endpoint generation when base URL already includes /anthropic."""
        config = AzureAIAnthropicCountTokensConfig()

        api_base = "https://my-resource.services.ai.azure.com/anthropic"
        endpoint = config.get_count_tokens_endpoint(api_base)

        assert (
            endpoint
            == "https://my-resource.services.ai.azure.com/anthropic/v1/messages/count_tokens"
        )

    def test_get_count_tokens_endpoint_with_trailing_slash(self):
        """Test endpoint generation with trailing slash in base URL."""
        config = AzureAIAnthropicCountTokensConfig()

        api_base = "https://my-resource.services.ai.azure.com/"
        endpoint = config.get_count_tokens_endpoint(api_base)

        assert (
            endpoint
            == "https://my-resource.services.ai.azure.com/anthropic/v1/messages/count_tokens"
        )
