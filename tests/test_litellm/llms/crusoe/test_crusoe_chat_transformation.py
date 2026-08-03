"""
Unit tests for Crusoe Cloud configuration.

These tests validate the CrusoeConfig class which extends OpenAIGPTConfig.
Crusoe Cloud Managed Inference is an OpenAI-compatible provider with minor customizations.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import pytest

import litellm
from litellm import completion
from litellm.llms.crusoe.chat.transformation import CrusoeConfig


class TestCrusoeConfig:
    """Test class for Crusoe Cloud functionality"""

    def test_default_api_base(self):
        """Test that default API base is used when none is provided"""
        config = CrusoeConfig()
        headers = {}
        api_key = "fake-crusoe-key"

        # Call validate_environment without specifying api_base
        result = config.validate_environment(
            headers=headers,
            model="crusoe/meta-llama/Llama-3.3-70B-Instruct",
            messages=[{"role": "user", "content": "Hey"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base=None,  # Not providing api_base
        )

        # Verify headers are still set correctly
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    def test_map_openai_params_max_completion_tokens(self):
        """Test that max_completion_tokens is mapped to max_tokens"""
        config = CrusoeConfig()
        optional_params = config.map_openai_params(
            non_default_params={"max_completion_tokens": 1024, "temperature": 0.5},
            optional_params={},
            model="meta-llama/Llama-3.3-70B-Instruct",
            drop_params=False,
        )

        assert optional_params["max_tokens"] == 1024
        assert "max_completion_tokens" not in optional_params
        assert optional_params["temperature"] == 0.5

    @pytest.mark.respx()
    def test_crusoe_completion_mock(self, respx_mock):
        """
        Mock test for Crusoe Cloud completion using the model format from docs.
        This test mocks the actual HTTP request to test the integration properly.
        """

        litellm.disable_aiohttp_transport = (
            True  # since this uses respx, we need to set use_aiohttp_transport to False
        )

        # Set up environment variables for the test
        api_key = "fake-crusoe-key"
        api_base = "https://api.inference.crusoecloud.com/v1"
        model = "crusoe/meta-llama/Llama-3.3-70B-Instruct"
        model_name = "Llama-3.3-70B-Instruct"  # The actual model name without provider prefix

        # Mock the HTTP request to the Crusoe Managed Inference API
        respx_mock.post(f"{api_base}/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": '```python\nprint("Hey from LiteLLM!")\n```\n\nThis simple Python code prints a greeting message from LiteLLM.',
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
            status_code=200,
        )

        # Make the actual API call through LiteLLM
        response = completion(
            model=model,
            messages=[
                {"role": "user", "content": "write code for saying hey from LiteLLM"}
            ],
            api_key=api_key,
            api_base=api_base,
        )

        # Verify response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "content")
        assert response.choices[0].message.content is not None

        # Check for specific content in the response
        assert "```python" in response.choices[0].message.content
        assert "Hey from LiteLLM" in response.choices[0].message.content


class TestCrusoeUtils:
    """Test class for Crusoe branches in litellm.utils"""

    def test_get_optional_params(self):
        """Test that get_optional_params routes crusoe to CrusoeConfig param mapping"""
        optional_params = litellm.get_optional_params(
            model="meta-llama/Llama-3.3-70B-Instruct",
            custom_llm_provider="crusoe",
            max_completion_tokens=1024,
            temperature=0.5,
        )

        assert optional_params["max_tokens"] == 1024
        assert "max_completion_tokens" not in optional_params
        assert optional_params["temperature"] == 0.5

    def test_get_api_key(self, monkeypatch):
        """Test that get_api_key resolves the Crusoe key from the environment"""
        monkeypatch.setenv("CRUSOE_API_KEY", "fake-crusoe-key")
        monkeypatch.setattr(litellm, "crusoe_key", None)

        api_key = litellm.utils.get_api_key(
            llm_provider="crusoe", dynamic_api_key=None
        )

        assert api_key == "fake-crusoe-key"

    def test_validate_environment(self, monkeypatch):
        """Test that validate_environment reports CRUSOE_API_KEY presence"""
        monkeypatch.setenv("CRUSOE_API_KEY", "fake-crusoe-key")
        result = litellm.validate_environment(
            model="crusoe/meta-llama/Llama-3.3-70B-Instruct"
        )
        assert result["keys_in_environment"] is True
        assert result["missing_keys"] == []

        monkeypatch.delenv("CRUSOE_API_KEY")
        result = litellm.validate_environment(
            model="crusoe/meta-llama/Llama-3.3-70B-Instruct"
        )
        assert result["keys_in_environment"] is False
        assert "CRUSOE_API_KEY" in result["missing_keys"]

    def test_provider_config_manager(self):
        """Test that ProviderConfigManager returns CrusoeConfig for crusoe"""
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(
            model="meta-llama/Llama-3.3-70B-Instruct",
            provider=LlmProviders.CRUSOE,
        )
        assert isinstance(config, CrusoeConfig)
