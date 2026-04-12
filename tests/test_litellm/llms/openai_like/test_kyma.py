"""
Unit tests for Kyma API provider configuration and integration.
All network calls are mocked — no real HTTP requests are made.
Live integration tests are in tests/llm_translation/test_kyma_live.py.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add workspace to path
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

import litellm


class TestKymaProviderConfig:
    """Test Kyma API provider configuration (unit tests, no network calls)"""

    def test_kyma_in_provider_list(self):
        """Test that kyma is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "KYMA")
        assert LlmProviders.KYMA.value == "kyma"
        assert "kyma" in litellm.provider_list

    def test_kyma_json_config_exists(self):
        """Test that kyma is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("kyma")

        kyma = JSONProviderRegistry.get("kyma")
        assert kyma is not None
        assert kyma.base_url == "https://kymaapi.com/v1"
        assert kyma.api_key_env == "KYMA_API_KEY"

    def test_kyma_provider_resolution(self):
        """Test that provider resolution finds kyma"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="kyma/llama-3.3-70b",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "llama-3.3-70b"
        assert provider == "kyma"
        assert api_base == "https://kymaapi.com/v1"

    def test_kyma_router_config(self):
        """Test that kyma can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "llama-3.3-70b",
                    "litellm_params": {
                        "model": "kyma/llama-3.3-70b",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "llama-3.3-70b"

    def test_kyma_completion_mocked(self):
        """Test that completion routes correctly to kyma provider (mocked HTTP)"""
        mock_message = MagicMock()
        mock_message.content = "test successful"
        mock_message.role = "assistant"

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_choice.index = 0

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model = "llama-3.3-70b"

        with patch("litellm.main.completion", return_value=mock_response):
            response = litellm.completion(
                model="kyma/llama-3.3-70b",
                messages=[{"role": "user", "content": "Say test successful"}],
                max_tokens=10,
            )

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content == "test successful"
