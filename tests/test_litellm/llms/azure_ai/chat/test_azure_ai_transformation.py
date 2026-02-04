import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
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


class TestAzureAIStudioConfigGetCompleteUrl:
    """Tests for AzureAIStudioConfig.get_complete_url method.

    These tests verify that URLs with existing endpoint paths are preserved,
    and URLs without endpoint paths get the appropriate path appended.

    Related issue: https://github.com/BerriAI/litellm/issues/7275
    """

    def test_get_complete_url_base_url_services_ai_azure(self):
        """Test that base URL gets /models/chat/completions appended for services.ai.azure.com"""
        config = AzureAIStudioConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com",
            api_key="test-key",
            model="gpt-4",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://myresource.services.ai.azure.com/models/chat/completions"

    def test_get_complete_url_base_url_other_domain(self):
        """Test that base URL gets /chat/completions appended for other domains"""
        config = AzureAIStudioConfig()
        url = config.get_complete_url(
            api_base="https://myresource.eastus.models.ai.azure.com",
            api_key="test-key",
            model="gpt-4",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://myresource.eastus.models.ai.azure.com/chat/completions"

    def test_get_complete_url_preserves_anthropic_v1_messages(self):
        """Test that URL with /v1/messages is preserved (Anthropic endpoint)"""
        config = AzureAIStudioConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com/anthropic/v1/messages",
            api_key="test-key",
            model="claude-3-sonnet",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://myresource.services.ai.azure.com/anthropic/v1/messages"

    def test_get_complete_url_preserves_chat_completions(self):
        """Test that URL with /chat/completions is preserved"""
        config = AzureAIStudioConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com/openai/chat/completions",
            api_key="test-key",
            model="gpt-4",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://myresource.services.ai.azure.com/openai/chat/completions"

    def test_get_complete_url_preserves_models_chat_completions(self):
        """Test that URL with /models/chat/completions is preserved"""
        config = AzureAIStudioConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com/models/chat/completions",
            api_key="test-key",
            model="gpt-4",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://myresource.services.ai.azure.com/models/chat/completions"

    def test_get_complete_url_preserves_completions(self):
        """Test that URL with /completions is preserved"""
        config = AzureAIStudioConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com/openai/completions",
            api_key="test-key",
            model="davinci",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://myresource.services.ai.azure.com/openai/completions"

    def test_get_complete_url_preserves_v1_complete(self):
        """Test that URL with /v1/complete is preserved (Cohere endpoint)"""
        config = AzureAIStudioConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com/cohere/v1/complete",
            api_key="test-key",
            model="command-r",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://myresource.services.ai.azure.com/cohere/v1/complete"

    def test_get_complete_url_preserves_embeddings(self):
        """Test that URL with /embeddings is preserved"""
        config = AzureAIStudioConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com/openai/embeddings",
            api_key="test-key",
            model="text-embedding-ada-002",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://myresource.services.ai.azure.com/openai/embeddings"

    def test_get_complete_url_adds_api_version(self):
        """Test that api_version is added as query parameter"""
        config = AzureAIStudioConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com",
            api_key="test-key",
            model="gpt-4",
            optional_params={},
            litellm_params={"api_version": "2024-05-01-preview"},
        )
        assert "api-version=2024-05-01-preview" in url

    def test_get_complete_url_preserves_existing_query_params(self):
        """Test that existing query parameters are preserved"""
        config = AzureAIStudioConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com/anthropic/v1/messages?custom=param",
            api_key="test-key",
            model="claude-3-sonnet",
            optional_params={},
            litellm_params={},
        )
        assert "custom=param" in url
        assert "/anthropic/v1/messages" in url

    def test_get_complete_url_does_not_append_to_anthropic_path(self):
        """Test that /models/chat/completions is NOT appended to URL with /anthropic path

        This is the core fix for issue #7275 - URLs with provider-specific paths
        should not get the default endpoint path appended.
        """
        config = AzureAIStudioConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com/anthropic",
            api_key="test-key",
            model="some-model",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://myresource.services.ai.azure.com/anthropic"

    def test_get_complete_url_does_not_append_to_deepseek_path(self):
        """Test that /models/chat/completions is NOT appended to URL with /deepseek path"""
        config = AzureAIStudioConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com/deepseek",
            api_key="test-key",
            model="deepseek-chat",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://myresource.services.ai.azure.com/deepseek"

    def test_get_complete_url_does_not_append_to_cohere_path(self):
        """Test that /models/chat/completions is NOT appended to URL with /cohere path"""
        config = AzureAIStudioConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com/cohere",
            api_key="test-key",
            model="command-r",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://myresource.services.ai.azure.com/cohere"

    def test_get_complete_url_does_not_append_to_mistral_path(self):
        """Test that /models/chat/completions is NOT appended to URL with /mistral path"""
        config = AzureAIStudioConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com/mistral",
            api_key="test-key",
            model="mistral-large",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://myresource.services.ai.azure.com/mistral"


class TestHasCompleteEndpointPath:
    """Tests for AzureAIStudioConfig._has_complete_endpoint_path method."""

    def test_detects_v1_messages(self):
        """Test detection of /v1/messages endpoint"""
        config = AzureAIStudioConfig()
        assert config._has_complete_endpoint_path("/anthropic/v1/messages") == True

    def test_detects_chat_completions(self):
        """Test detection of /chat/completions endpoint"""
        config = AzureAIStudioConfig()
        assert config._has_complete_endpoint_path("/models/chat/completions") == True
        assert config._has_complete_endpoint_path("/openai/chat/completions") == True

    def test_detects_completions(self):
        """Test detection of /completions endpoint"""
        config = AzureAIStudioConfig()
        assert config._has_complete_endpoint_path("/openai/completions") == True

    def test_detects_v1_complete(self):
        """Test detection of /v1/complete endpoint (Cohere)"""
        config = AzureAIStudioConfig()
        assert config._has_complete_endpoint_path("/cohere/v1/complete") == True

    def test_detects_embeddings(self):
        """Test detection of /embeddings endpoint"""
        config = AzureAIStudioConfig()
        assert config._has_complete_endpoint_path("/v1/embeddings") == True
        assert config._has_complete_endpoint_path("/openai/embeddings") == True

    def test_detects_provider_specific_paths(self):
        """Test detection of provider-specific paths"""
        config = AzureAIStudioConfig()
        assert config._has_complete_endpoint_path("/anthropic") == True
        assert config._has_complete_endpoint_path("/cohere") == True
        assert config._has_complete_endpoint_path("/mistral") == True
        assert config._has_complete_endpoint_path("/meta") == True
        assert config._has_complete_endpoint_path("/ai21") == True
        assert config._has_complete_endpoint_path("/deepseek") == True

    def test_does_not_detect_base_paths(self):
        """Test that generic base paths are not detected as complete endpoints"""
        config = AzureAIStudioConfig()
        assert config._has_complete_endpoint_path("") == False
        assert config._has_complete_endpoint_path("/") == False
        assert config._has_complete_endpoint_path("/models") == False
        assert config._has_complete_endpoint_path("/openai") == False
        assert config._has_complete_endpoint_path("/v1") == False
