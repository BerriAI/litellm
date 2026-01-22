"""
Unit tests for Firmware.ai provider integration.

Firmware.ai is an OpenAI-compatible provider that uses the openai_like providers.json
configuration. These tests validate that:
1. The provider is properly registered
2. Model routing works correctly
3. API key handling works
4. Basic completion calls are properly formatted
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path


def get_repo_root():
    """Get the repository root directory."""
    from pathlib import Path
    # tests/test_litellm/llms/firmware/test_firmware_completion.py -> go up 4 levels
    return Path(__file__).parents[4]


class TestFirmwareProviderConfig:
    """Test class for Firmware.ai provider configuration"""

    def test_firmware_provider_in_providers_json(self):
        """Test that firmware is registered in providers.json"""
        import json

        providers_json_path = get_repo_root() / "litellm" / "llms" / "openai_like" / "providers.json"

        with open(providers_json_path, "r") as f:
            providers = json.load(f)

        assert "firmware" in providers
        assert providers["firmware"]["base_url"] == "https://app.firmware.ai/api/v1"
        assert providers["firmware"]["api_key_env"] == "FIRMWARE_API_KEY"

    def test_firmware_models_in_pricing(self):
        """Test that firmware models are registered in model pricing json file"""
        import json

        pricing_json_path = get_repo_root() / "model_prices_and_context_window.json"

        with open(pricing_json_path, "r") as f:
            model_cost = json.load(f)

        # Check that firmware models are available
        firmware_models = [
            key for key in model_cost.keys()
            if key.startswith("firmware/")
        ]

        assert len(firmware_models) > 0

        # Check specific models are present
        expected_models = [
            "firmware/openai/gpt-4o",
            "firmware/anthropic/claude-opus-4-5",
            "firmware/google/gemini-2.5-pro",
            "firmware/xai/grok-4-fast-reasoning",
            "firmware/deepseek/deepseek-chat",
            "firmware/cerebras/gpt-oss-120b",
        ]

        for model in expected_models:
            assert model in model_cost, f"Model {model} not found in model_cost"

    def test_firmware_model_provider_attribute(self):
        """Test that firmware models have correct provider attribute"""
        import json

        pricing_json_path = get_repo_root() / "model_prices_and_context_window.json"

        with open(pricing_json_path, "r") as f:
            model_cost = json.load(f)

        # Check a few firmware models have the correct provider
        test_models = [
            "firmware/openai/gpt-4o",
            "firmware/anthropic/claude-opus-4-5",
        ]

        for model in test_models:
            model_info = model_cost.get(model)
            assert model_info is not None, f"Model {model} not found"
            assert model_info.get("litellm_provider") == "firmware"


class TestFirmwareCompletion:
    """Test class for Firmware.ai completion functionality"""

    def test_firmware_completion_mock(self, respx_mock):
        """
        Mock test for Firmware.ai completion.
        This test mocks the HTTP request to validate the integration.
        """
        import litellm

        litellm.disable_aiohttp_transport = True
        from litellm import completion

        api_key = "fw_api_test_key"
        api_base = "https://app.firmware.ai/api/v1"
        model = "firmware/openai/gpt-4o"

        # Mock the HTTP request to Firmware.ai API
        respx_mock.post(f"{api_base}/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "openai/gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! I'm an AI assistant.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 8,
                    "total_tokens": 17,
                },
            },
            status_code=200,
        )

        response = completion(
            model=model,
            messages=[{"role": "user", "content": "Hello!"}],
            api_key=api_key,
        )

        # Verify response structure
        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert response.choices[0].message.content == "Hello! I'm an AI assistant."

    def test_firmware_streaming_completion_mock(self, respx_mock):
        """
        Mock test for Firmware.ai streaming completion.
        """
        import litellm

        litellm.disable_aiohttp_transport = True
        from litellm import completion

        api_key = "fw_api_test_key"
        api_base = "https://app.firmware.ai/api/v1"
        model = "firmware/anthropic/claude-opus-4-5"

        # Mock streaming response
        streaming_response = """data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"anthropic/claude-opus-4-5","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"anthropic/claude-opus-4-5","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"anthropic/claude-opus-4-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
"""

        respx_mock.post(f"{api_base}/chat/completions").respond(
            content=streaming_response,
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
        )

        response = completion(
            model=model,
            messages=[{"role": "user", "content": "Hello!"}],
            api_key=api_key,
            stream=True,
        )

        # Collect streaming chunks
        chunks = list(response)

        assert len(chunks) > 0


class TestFirmwareProviderEndpoints:
    """Test class for Firmware.ai provider endpoints support"""

    def test_firmware_in_endpoint_support(self):
        """Test that firmware is in provider_endpoints_support.json"""
        import json

        endpoints_json_path = get_repo_root() / "provider_endpoints_support.json"

        with open(endpoints_json_path, "r") as f:
            endpoints = json.load(f)

        providers = endpoints.get("providers", {})
        assert "firmware" in providers

        firmware_config = providers["firmware"]
        assert firmware_config["display_name"] == "Firmware.ai (`firmware`)"
        assert firmware_config["endpoints"]["chat_completions"] is True
        # messages and responses are false - firmware.ai only supports OpenAI chat completions natively
        assert firmware_config["endpoints"]["messages"] is False
        assert firmware_config["endpoints"]["responses"] is False
