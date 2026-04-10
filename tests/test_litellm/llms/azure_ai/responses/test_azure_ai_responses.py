"""
Tests for azure_ai Responses API support.

Verifies that when api_base contains /projects/, litellm.responses()
routes to the real upstream /responses endpoint instead of falling
back to the completions-style bridge.

Ref: https://github.com/BerriAI/litellm/issues/25407
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.azure_ai.responses.transformation import (
    AzureAIResponsesAPIConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def _make_mock_responses_api_response(content: str = "Hello!") -> dict:
    return {
        "id": "resp-test-azure-ai",
        "object": "response",
        "created_at": 1234567890,
        "model": "DeepSeek-R1",
        "output": [
            {
                "type": "message",
                "id": "msg-test-azure-ai",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        ],
        "status": "completed",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
        },
    }


def _make_mock_http_client(response_body: dict) -> MagicMock:
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = response_body
    mock_response.text = json.dumps(response_body)
    mock_client.post.return_value = mock_response
    return mock_client


class TestAzureAIResponsesProviderConfig:
    """Test that ProviderConfigManager returns AzureAIResponsesAPIConfig for azure_ai."""

    def test_provider_config_registration(self):
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="DeepSeek-R1",
            provider=LlmProviders.AZURE_AI,
        )
        assert config is not None
        assert isinstance(config, AzureAIResponsesAPIConfig)
        assert config.custom_llm_provider == LlmProviders.AZURE_AI


class TestAzureAIResponsesURL:
    """Test get_complete_url() constructs the correct URL for various api_base patterns."""

    def test_project_based_url(self):
        """When api_base contains /projects/, append /openai/v1/responses."""
        config = AzureAIResponsesAPIConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com/api/projects/my-project",
            litellm_params={},
        )
        assert url == (
            "https://myresource.services.ai.azure.com/api/projects/my-project"
            "/openai/v1/responses"
        )

    def test_project_based_url_with_api_version(self):
        """api-version query param should be appended."""
        config = AzureAIResponsesAPIConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com/api/projects/my-project",
            litellm_params={"api_version": "2025-03-01-preview"},
        )
        assert "api-version=2025-03-01-preview" in url
        assert "/openai/v1/responses" in url

    def test_services_ai_azure_url_without_projects(self):
        """Standard services.ai.azure.com base (no /projects/) uses /models/responses."""
        config = AzureAIResponsesAPIConfig()
        url = config.get_complete_url(
            api_base="https://myresource.services.ai.azure.com",
            litellm_params={},
        )
        assert "/models/responses" in url

    def test_generic_api_base(self):
        """Non-Azure-Foundry base uses /v1/responses."""
        config = AzureAIResponsesAPIConfig()
        url = config.get_complete_url(
            api_base="https://custom-proxy.example.com",
            litellm_params={},
        )
        assert url == "https://custom-proxy.example.com/v1/responses"

    def test_missing_api_base_raises(self):
        """ValueError raised when api_base is None and no env var set."""
        config = AzureAIResponsesAPIConfig()
        with patch.dict(os.environ, {}, clear=False):
            # Ensure env var is not set
            os.environ.pop("AZURE_AI_API_BASE", None)
            with pytest.raises(ValueError, match="api_base is required"):
                config.get_complete_url(api_base=None, litellm_params={})


class TestAzureAIResponsesAuth:
    """Test validate_environment() sets correct auth headers."""

    def test_api_key_header_for_azure_foundry_host(self):
        """*.services.ai.azure.com should use api-key header."""
        config = AzureAIResponsesAPIConfig()
        headers = config.validate_environment(
            headers={},
            model="DeepSeek-R1",
            litellm_params=GenericLiteLLMParams(
                api_key="test-key-123",
                api_base="https://myresource.services.ai.azure.com/api/projects/proj",
            ),
        )
        assert headers.get("api-key") == "test-key-123"
        assert "Authorization" not in headers

    def test_bearer_auth_for_non_azure_host(self):
        """Non-Azure hosts should use Bearer auth."""
        config = AzureAIResponsesAPIConfig()
        headers = config.validate_environment(
            headers={},
            model="DeepSeek-R1",
            litellm_params=GenericLiteLLMParams(
                api_key="test-key-456",
                api_base="https://custom-proxy.example.com",
            ),
        )
        assert headers.get("Authorization") == "Bearer test-key-456"
        assert "api-key" not in headers

    def test_api_key_header_for_openai_azure_host(self):
        """*.openai.azure.com should also use api-key header."""
        config = AzureAIResponsesAPIConfig()
        headers = config.validate_environment(
            headers={},
            model="gpt-4",
            litellm_params=GenericLiteLLMParams(
                api_key="test-key-789",
                api_base="https://myresource.openai.azure.com",
            ),
        )
        assert headers.get("api-key") == "test-key-789"


class TestAzureAIResponsesE2E:
    """End-to-end test with mocked HTTP client."""

    def test_responses_create_routes_to_native_endpoint(self):
        """
        Verify litellm.responses() uses native /responses routing for azure_ai
        with a project-based api_base, rather than the completions bridge.
        """
        mock_client = _make_mock_http_client(
            _make_mock_responses_api_response("Hello from Azure AI Foundry!")
        )

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=mock_client,
        ):
            response = litellm.responses(
                model="azure_ai/DeepSeek-R1",
                input="Hello, how are you?",
                api_base="https://myresource.services.ai.azure.com/api/projects/my-project",
                api_key="test-key",
            )

        from litellm.types.llms.openai import ResponsesAPIResponse

        assert response is not None
        assert isinstance(response, ResponsesAPIResponse)
        assert len(response.output) > 0
        output_message = response.output[0]
        assert output_message.role == "assistant"
        assert "Azure AI Foundry" in output_message.content[0].text

        # Verify the URL used contains /responses (native endpoint)
        call_args = mock_client.post.call_args
        called_url = call_args[1].get("url", call_args[0][0] if call_args[0] else "")
        assert "/responses" in str(called_url)
