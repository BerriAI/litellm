"""
Unit tests for Venice AI configuration.

These tests validate the VeniceAIChatConfig class which extends OpenAILikeChatConfig.
Venice AI is an OpenAI-compatible provider that requires custom parameters to be nested
in a `venice_parameters` object.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.venice_ai.chat.transformation import VeniceAIChatConfig


class TestVeniceAIChatConfig:
    """Test class for VeniceAIChatConfig functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = VeniceAIChatConfig()

    def test_inheritance(self):
        """Test proper inheritance from OpenAILikeChatConfig"""
        from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig

        assert isinstance(self.config, OpenAILikeChatConfig)

    def test_get_openai_compatible_provider_info_default(self):
        """Test that _get_openai_compatible_provider_info returns default values"""
        api_base, api_key = self.config._get_openai_compatible_provider_info(
            api_base=None, api_key=None
        )

        assert api_base == "https://api.venice.ai/api/v1"
        assert api_key is None

    def test_get_openai_compatible_provider_info_with_env(self):
        """Test that _get_openai_compatible_provider_info uses environment variables"""
        with patch(
            "litellm.llms.venice_ai.chat.transformation.get_secret_str",
            side_effect=lambda key: {
                "VENICE_AI_API_BASE": "https://custom.venice.ai/api/v1",
                "VENICE_AI_API_KEY": "env-api-key",
            }.get(key),
        ):
            api_base, api_key = self.config._get_openai_compatible_provider_info(
                api_base=None, api_key=None
            )

            assert api_base == "https://custom.venice.ai/api/v1"
            assert api_key == "env-api-key"

    def test_get_openai_compatible_provider_info_with_params(self):
        """Test that _get_openai_compatible_provider_info uses provided parameters"""
        api_base, api_key = self.config._get_openai_compatible_provider_info(
            api_base="https://custom.venice.ai/api/v1", api_key="custom-key"
        )

        assert api_base == "https://custom.venice.ai/api/v1"
        assert api_key == "custom-key"

    def test_transform_request_with_direct_venice_params(self):
        """Test transform_request with Venice parameters passed directly"""
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "enable_web_search": "auto",
            "enable_web_citations": True,
            "temperature": 0.7,
            "max_tokens": 100,
        }
        litellm_params = {}
        headers = {}

        result = self.config.transform_request(
            model="qwen3-235b",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Verify venice_parameters are nested in extra_body
        assert "extra_body" in result
        assert "venice_parameters" in result["extra_body"]
        assert result["extra_body"]["venice_parameters"]["enable_web_search"] == "auto"
        assert result["extra_body"]["venice_parameters"]["enable_web_citations"] is True

        # Verify Venice params are removed from top level
        assert "enable_web_search" not in result
        assert "enable_web_citations" not in result

        # Verify OpenAI params remain at top level
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100
        assert "model" in result
        assert "messages" in result

    def test_transform_request_with_nested_venice_params(self):
        """Test transform_request with venice_parameters as nested dict"""
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "venice_parameters": {
                "enable_web_search": "on",
                "character_slug": "test-character",
            },
            "temperature": 0.7,
        }
        litellm_params = {}
        headers = {}

        result = self.config.transform_request(
            model="qwen3-235b",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Verify venice_parameters are nested correctly in extra_body
        assert "extra_body" in result
        assert "venice_parameters" in result["extra_body"]
        assert result["extra_body"]["venice_parameters"]["enable_web_search"] == "on"
        assert result["extra_body"]["venice_parameters"]["character_slug"] == "test-character"

        # Verify OpenAI params remain at top level
        assert result["temperature"] == 0.7

    def test_transform_request_with_mixed_venice_params(self):
        """Test transform_request with both nested and direct Venice parameters"""
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "venice_parameters": {
                "enable_web_search": "on",
            },
            "enable_web_citations": True,  # Direct param
            "temperature": 0.7,
        }
        litellm_params = {}
        headers = {}

        result = self.config.transform_request(
            model="qwen3-235b",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Verify both nested and direct params are merged in extra_body
        assert "extra_body" in result
        assert "venice_parameters" in result["extra_body"]
        assert result["extra_body"]["venice_parameters"]["enable_web_search"] == "on"
        assert result["extra_body"]["venice_parameters"]["enable_web_citations"] is True

    def test_transform_request_with_extra_body_venice_params(self):
        """Test transform_request with Venice parameters in extra_body"""
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "extra_body": {
                "venice_parameters": {
                    "enable_web_search": "auto",
                },
                "enable_web_scraping": True,
                "other_param": "value",
            },
            "temperature": 0.7,
        }
        litellm_params = {}
        headers = {}

        result = self.config.transform_request(
            model="qwen3-235b",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Verify venice_parameters are extracted and nested in extra_body
        assert "extra_body" in result
        assert "venice_parameters" in result["extra_body"]
        assert result["extra_body"]["venice_parameters"]["enable_web_search"] == "auto"
        assert result["extra_body"]["venice_parameters"]["enable_web_scraping"] is True

        # Verify other extra_body params remain in extra_body
        assert result["extra_body"]["other_param"] == "value"

    def test_transform_request_with_all_venice_params(self):
        """Test transform_request with all Venice AI parameters"""
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "character_slug": "test-character",
            "strip_thinking_response": True,
            "disable_thinking": False,
            "enable_web_search": "auto",
            "enable_web_scraping": True,
            "enable_web_citations": True,
            "include_search_results_in_stream": False,
            "return_search_results_as_documents": True,
            "include_venice_system_prompt": False,
            "temperature": 0.7,
        }
        litellm_params = {}
        headers = {}

        result = self.config.transform_request(
            model="qwen3-235b",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Verify all Venice params are nested in extra_body
        assert "extra_body" in result
        assert "venice_parameters" in result["extra_body"]
        venice_params = result["extra_body"]["venice_parameters"]
        assert venice_params["character_slug"] == "test-character"
        assert venice_params["strip_thinking_response"] is True
        assert venice_params["disable_thinking"] is False
        assert venice_params["enable_web_search"] == "auto"
        assert venice_params["enable_web_scraping"] is True
        assert venice_params["enable_web_citations"] is True
        assert venice_params["include_search_results_in_stream"] is False
        assert venice_params["return_search_results_as_documents"] is True
        assert venice_params["include_venice_system_prompt"] is False

        # Verify OpenAI params remain at top level
        assert result["temperature"] == 0.7

    def test_transform_request_without_venice_params(self):
        """Test transform_request without any Venice parameters"""
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "temperature": 0.7,
            "max_tokens": 100,
        }
        litellm_params = {}
        headers = {}

        result = self.config.transform_request(
            model="qwen3-235b",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Verify venice_parameters is not added when not needed
        # extra_body might exist but should not contain venice_parameters
        if "extra_body" in result:
            assert "venice_parameters" not in result["extra_body"]

        # Verify OpenAI params are present
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100

    def test_transform_request_with_none_extra_body(self):
        """Test transform_request preserves None extra_body"""
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "extra_body": None,
            "temperature": 0.7,
        }
        litellm_params = {}
        headers = {}

        result = self.config.transform_request(
            model="qwen3-235b",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Verify None extra_body is preserved (passed to parent)
        # The parent transform_request should handle None appropriately
        # We verify that the request completes without error
        assert result["temperature"] == 0.7

    def test_transform_request_with_non_dict_extra_body(self):
        """Test transform_request preserves non-dict extra_body values"""
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "extra_body": "some_string_value",
            "temperature": 0.7,
        }
        litellm_params = {}
        headers = {}

        result = self.config.transform_request(
            model="qwen3-235b",
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Verify non-dict extra_body is preserved in optional_params
        # (it will be passed to parent transform_request)
        assert optional_params.get("extra_body") == "some_string_value"
        assert result["temperature"] == 0.7

    def test_transform_request_venice_params_set(self):
        """Test that VENICE_PARAMS set contains all expected parameters"""
        expected_params = {
            "character_slug",
            "strip_thinking_response",
            "disable_thinking",
            "enable_web_search",
            "enable_web_scraping",
            "enable_web_citations",
            "include_search_results_in_stream",
            "return_search_results_as_documents",
            "include_venice_system_prompt",
        }

        assert self.config.VENICE_PARAMS == expected_params

    def test_validate_environment(self):
        """Test that validate_environment adds correct headers"""
        headers = {}
        api_key = "fake-venice-key"

        result = self.config.validate_environment(
            headers=headers,
            model="qwen3-235b",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base="https://api.venice.ai/api/v1",
        )

        # Verify headers
        assert result["Authorization"] == f"Bearer {api_key}"
        assert result["Content-Type"] == "application/json"

    def test_get_supported_openai_params(self):
        """Test that get_supported_openai_params returns correct params"""
        supported_params = self.config.get_supported_openai_params(model="qwen3-235b")

        # Should inherit from OpenAILikeChatConfig which inherits from OpenAIGPTConfig
        assert "tools" in supported_params
        assert "tool_choice" in supported_params
        assert "temperature" in supported_params
        assert "max_tokens" in supported_params
        assert "stream" in supported_params

    def test_venice_ai_completion_mock(self, respx_mock):
        """
        Mock test for Venice AI completion using the model format from docs.
        This test mocks the actual HTTP request to test the integration properly.
        """
        import litellm

        litellm.disable_aiohttp_transport = (
            True  # since this uses respx, we need to set use_aiohttp_transport to False
        )
        from litellm import completion

        # Set up environment variables for the test
        api_key = "fake-venice-key"
        api_base = "https://api.venice.ai/api/v1"
        model = "venice_ai/qwen3-235b"
        model_name = "qwen3-235b"  # The actual model name without provider prefix

        # Mock the HTTP request to the Venice AI API
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
                            "content": "Hello! How can I help you today?",
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

        # Make the actual API call through LiteLLM with Venice parameters
        response = completion(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            api_key=api_key,
            api_base=api_base,
            enable_web_search="auto",
            enable_web_citations=True,
        )

        # Verify response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "content")
        assert response.choices[0].message.content is not None

        # Verify the request was made with venice_parameters
        # Note: The OpenAI SDK merges extra_body into the request body, so venice_parameters
        # appears at the top level in the actual HTTP request (which is what Venice AI expects)
        request = respx_mock.calls[0].request
        import json

        request_body = json.loads(request.content)
        assert "venice_parameters" in request_body
        assert request_body["venice_parameters"]["enable_web_search"] == "auto"
        assert request_body["venice_parameters"]["enable_web_citations"] is True
